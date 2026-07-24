# Rejected-Edit / Experience Buffer 设计 v2

2026-07-24 修订。v1 的缺陷(用户指正):把机械记录当成了设计主体,而两个
参照系的核心价值在**评测后的 LLM 归因**——SkillOpt 的 reflect 步骤、Shinka
的 MetaSummarizer 都是让 LLM 读结果写教训,机械台账只是它们的证据底座。
v2 以"证据 → 归因 → 巩固"三层重构;已实现的清单 1–3(RejectionEvent +
schema v2)降级为 L0 证据层,不废弃。

## 1. 参照系精读结论(v2 依据)

**SkillOpt 主环**(论文 + `skillopt/gradient/reflect.py`):
rollout → **reflect** → aggregate → apply → gate。reflect 是 LLM 分析师
读失败/成功轨迹的 minibatch,产出 `{patch:{reasoning, edits},
failure_summary}`——归因和编辑方向由 LLM 写出。`step_buffer`(被拒编辑
摘要)只是注入 reflect 的辅助上下文,防重复,不是学习的主体。

**SAR(blog Part II)**:反思时把**当前 skill 放进上下文**,先解读
"这次失败是 skill 缺陷还是执行失误",再路由编辑方向:
Skill Defect → 改正文;Execution Lapse → 只进受保护附录当提醒。
归因分类决定编辑去向,这是"反思"高于"记录"的地方。

**巩固是硬约束,不是锦上添花**(blog 2.1/2.2 实测):
未巩固的宽反思记忆(Success-and-Failure SAR, no consolidation)让
gpt-5.4-mini 回归 -11.4%(SearchQA)/-13.8%(SpreadsheetBench);
加 Consolidation(20 条阈值合并去重)后 70.87%→80.52%、47.14%→62.14%,
基本收回回归。教训:**反思记忆必须带预算和去重,累积越多越伤弱模型。**

**Shinka MetaSummarizer**(`shinka/core/summarizer.py`):评测后每
`meta_rec_interval=10` 个程序跑三步 LLM 蒸馏:①逐程序摘要 →
②全局 scratchpad(Successful Patterns / Ineffective Approaches /
Implementation Insights 分区,喂入上一版=隐式巩固)→ ③编号建议;
注入时**随机抽 1 条**(sample_single_meta_rec,并行任务间保多样)。
预评估拒绝(novelty/patch 失败)只进 attempt_log 审计,从不回灌。

## 2. 三层架构

```
L0 证据层(机械,无 LLM,无损)          ← 清单 1–3 已实现
   evaluated / rejected_novelty / proposal_failed 条目
   Δfitness, diff 摘要, redemption 回写
        │ 每 K 个 graded 候选攒一批
        ▼
L1 归因层(新核心:批量 LLM 反思)
   每条产 lesson{verdict, why, advice},写回条目
        │ 每次 L1 后增量更新;lessons 超阈值合并去重
        ▼
L2 巩固层(scratchpad,硬预算)
   三分区,2000 chars 上限;Consolidation 阈值 40
        │
        ▼
注入:按失败类别检索 lessons + scratchpad 抽 1 条建议
```

### 2a. L0 证据层(已落地,角色重定位)

作用:给 L1 供料;LLM 断供/预算耗尽时降级为 v1 的机械渲染(E3r/E4a 臂
保留为消融基线)。不再是注入主体。

### 2b. L1 归因层(MutationReflector,新)

- **触发**:LoopObserver,攒 K=8 个新 graded 条目(或 run 结束 flush)
  批量一次 LLM 调用(对齐 Shinka interval=10 的量级;150 evals/run
  ≈ 19 次调用,用 haiku 级模型,单 run 反思成本 << $1)。
- **输入**(每条):diff 摘要、Δfitness、父/子代表性失败例
  (structured_feedback 的 expected/predicted + grader_critique
  摘录,经 sanitize 出口)、以及 **pass_vector hamming**(父子
  BehaviorSignature 距离,现成)。
- **输出**(每条 lesson,写回 ExperienceEntry.lesson 字段):

```json
{"child_id": "...", "verdict": "improved|regressed|noise",
 "why": "一句归因", "advice": "对下次同类失败模式的可执行建议"}
```

- **noise verdict 是 Execution Lapse 的对位**,也是对 backlog#5
  (12 题 val、单次评估、val→test 0.595→0.389)的 prompt 侧治理:
  机械先验 |Δ| ≤ 1/n_items 且 hamming ≤ 1 时,prompt 明示"考虑判为
  noise";判为 noise 的条目**不进负经验注入、不写进 scratchpad**,
  防止把评估噪声固化成教训。
- **SAR 对位**:反思 prompt 里放入父代当前策略代码的骨架摘要
  (当前 skill 的对位),要求归因区分"策略缺陷"(→ advice 指向代码
  方向)与"评估噪声/个例"(→ noise)。
- rejected_novelty / proposal_failed 不逐条反思:压成批次统计行进
  L2 输入("本窗口 4/10 提案因近重复被拒——方向多样性不足")。
- **lesson 附带 `tags`(开放词表打标,泛化关键)**:reflector 输出
  `tags: ["assumes-injectivity", ...]`(小写 slug)。这是非结构化域的
  检索索引来源——类别在反思时涌现,不要求域适配器预定义税则。

### 2b'. error_category 依赖已废除(2026-07-24 决定,已落码)

`error_category` / `category_delta` 是 IMO 域特产(grade.py 有
`_error_category`);多数域只有自然语言批语。v1 里类别是唯一检索索引
(承重墙),v2 有 L1 后它只是对 LLM 所读原料的有损预压缩——**整个废除**,
不保留"可选加速器"双路径:

1. 检索脊柱 = L1 lesson 的 `tags`(单一路径)。同义词漂移治理:
   reflector prompt 展示已有词表要求先复用;consolidation 合并同义标签。
2. L0 的机械检索(E3r 臂)退化为全局近期 wins/losses 清单——
   约等于上游式基线,当消融臂更干净;L1 断供时同样降级至此。
3. "修了什么/引入了什么"由 lesson.why 从批语归纳。
4. **pass_vector 与 category 无关且普适**:逐题 pass/fail 几乎所有域
   都有,hamming + noise 先验(|Δ| ≤ 1/n_items 且 hamming ≤ 1)不受
   废除影响。
5. `feedback.py` 的 StructuredFeedback/error_histogram 不动
   (FeedbackContributor 和 IMO grade.py 仍在用)。

### 2c. L2 巩固层(scratchpad)

- 三分区:Successful patterns / Ineffective approaches /
  Unexplored directions(Shinka 三分区 + 我们把 Insights 换成
  Unexplored,服务探索)。
- 每次 L1 批后增量更新:输入 = 上一版 scratchpad + 本批 lessons +
  拒绝统计行 + 当前 best 摘要;硬预算 2000 chars(有限长即隐式巩固,
  Shinka 同款)。
- **Consolidation**:store 中带 lesson 的条目超 40 条时,触发一次
  合并去重(重叠 advice 合并、被 redeem 的 regressed lesson 改写,
  见 §3),对齐 SkillOpt 阈值巩固,防 blog 2.1 的宽记忆回归。

### 2d. Redemption 语义上移

L0 的 redeemed_by 回写保留;L1/L2 消费它:被 redeem 的 regressed 条目
lesson 在下次巩固时改写为 stepping-stone("此方向短期回落但导向 g8 型
突破,可继续但控制幅度"),而不是从记忆里消失。比 v1 的"移出负经验区"
更进一步:回落→突破本身成为一条正向教训。

## 3. 注入策略(Contributor v2)

1. **检索式 lessons**(主,差异化保留):按 lesson `tags` 与父代
   失败摘要匹配检索(不再渲染原始 diff 行),wins 的 advice 在后
   (recency bias),noise 条目排除;冷启动回退到全局近期清单。
2. **scratchpad 抽 1 条**:从 Unexplored/建议里随机抽 1 条注入
   (Shinka sample-1,避免所有提案挤同一方向)。
3. 负经验区(v1 §5b)保留但降级:只在 lessons 尚未覆盖的窗口内条目
   上用机械渲染兜底;措辞仍是 "avoid repeating without variation"。
4. 预算:lessons 区 2048 B、scratchpad 条目 ~300 B、兜底区 1024 B。

## 4. 与 v1 已实现件的关系(清单 1–3 不废弃)

- schema v2 条目 = L1 的输入记录格式;新增 `lesson: dict | None` 字段。
- RejectionEvent 采集 = L2 的拒绝统计原料(多样性诊断信号)。
- redemption 回写 = lesson 改写的触发器。
- ExperienceContributor 的 retrieval/`retrieval+rejected` 模式 =
  L1 消融基线(证明"LLM 归因比机械记录多赚多少")。

## 5. 实验矩阵(修订)

| 臂 | 注入内容 | 说明 |
|---|---|---|
| E3r | 机械近期清单(v1,无索引) | L0 基线 |
| E4a | + 机械负经验区 | v1 方案 |
| E5r | L1 lessons 检索 | 归因层主张 |
| E5s | E5r + scratchpad 抽 1 条 | 完整 v2 |

验收:E5r vs E3r 看 val 提升与跨岛重复率;E5s vs E5r 看 Unexplored
注入是否提高提案多样性(novelty 拒绝率下降);弱优化器模型上复核
blog 2.1 的"宽记忆回归"是否被 40 条巩固阈值挡住。

## 6. 实施清单(续 v1 的 1–5)

6. ✅ `evoplus/experience.py`:ExperienceEntry 加 `lesson: dict | None`;
   store 加 `pending_reflection()` / `attach_lesson()`(JSONL 追加
   lesson 行,last-write-wins,与 redemption 同机制)。
7. ✅ `evoplus/reflection.py`(新):`MutationReflector`
   (LoopObserver;攒批、构反思 prompt、解析 lessons、更新 scratchpad;
   state()/set_state() 进 checkpoint;LLM 失败时跳过本批不中断 run)。
8. ✅ Contributor v2(2026-07-24):`lessons` / `lessons+scratchpad` 模式。
   检索键 = 父代自身条目的 lesson tags(`store.parent_tags`);noise
   排除;已 lesson 的回落从机械负经验区移除(去重);scratchpad 解析
   bullet、优先 Unexplored 分区、按 (generation, parent_id) 种子确定性
   抽 1 条;冷启动回退机械渲染。测试在 test_evoplus.py L1/v2 段。
9. ✅ Consolidation(2026-07-24):`consolidated_at` 水位线进 checkpoint;
   触发在 attach_lesson **之后**(否则永远晚一批);tag_map 合并同义
   标签,rewrites **只作用于 redeemed 条目**(防幻觉守卫);失败不推
   水位线、下批重试;`_charge` 把反思+巩固花费接进 BudgetMeter。
10. ✅ `evolution.py` 接线(2026-07-24):mode="lessons+scratchpad",
    observers=[store, reflector],reflector 复用 optimizer_client +
    budget(独立 cheap model 等 spec 加字段)。test_imo_proof 端到端绿。
    ⚠️ 接线时顺带修了 agent 包 WIP 的循环导入:conversation/
    session_proposer 的 `from .. import NativeToolAgentBackend` 改
    `from .runtime import`;tools/preflight.py 对 feedback 的导入推迟
    到方法体(feedback→tools→preflight→feedback 环)。
