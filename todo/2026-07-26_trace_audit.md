# 运行轨迹审计:框架缺陷清单(2026-07-26)

> **历史证据文档**(2026-08-04 标注):所列缺陷已在 07-26 至 08-03 间陆续
> 修复(算子可见性、novelty 挂载、预算记账、超时挂死等);同方法后续又
> 捕获扰动闸误杀、外层冻结互杀、pgrep 转义陷阱、新颖性闸判父代等缺陷。
> 方法论沉淀在 scripts/audit.py 与"活性审计"纪律里。数字与推理保留原样。

对象:`results/e5s_r2`(lessons+scratchpad,15 代)、`results/e3r_r2`
(retrieval 基线,13 代)、`results/e5s_s0`、`results/imo_multi_island`。
证据来自 `run.db` / `metrics.jsonl` / `experience.jsonl` /
`gen_*/evaluation.json` / `agent_sessions/*/`。

**总判**:三个 run 的全部进化增益是 **+0.083 / +0.095 / +0.179**,而单次
评估的标准误是 **0.14**。**三分之二的"进化成果"落在噪声带以内。** 但在
动噪声之前,先有两个更致命的缺陷 —— 它们在污染信号本身。

---

## 🔴 缺陷 1:预算耗尽被当作候选质量记账(27% 的候选)

`SolverBudget.max_calls_per_problem = 8`。候选程序每题超过 8 次 LLM 调用
就抛 `CandidateExecutionError: solver call limit exceeded`,该题记 0 分,
且**照常进入 fitness**。

`e5s_r2` 逐题失败统计:

```
gen  6   9/12 失败  fit 0.2500  Re-plan during revision loop
gen  7   9/12 失败  fit 0.2500  Increase revision limit from 3 to 5
gen  9   9/12 失败  fit 0.2500  Plan-then-prove with iterative revision
gen 12   8/12 失败  fit 0.2619  Plan-validate-revise with conditional replanning
e3r_r2 gen 12  12/12 失败  fit 0.0000
```

**4/15 = 27% 的候选如此。全域共 47 次逐题预算耗尽,失败原因 100% 是
"solver call limit exceeded"。**

三个结构性后果:

1. **偏差有方向**。被杀的全是"多迭代 / 重规划"这一族 —— 恰恰是这个任务
   最自然的改进方向。框架系统性地惩罚更彻底的策略。
2. **只杀难题**。简单题早停(调用少)得以计分,难题触发迭代→撞上限→记 0。
   **分数最依赖的那些题被选择性清零。**
3. **约束不可见**。seed workspace 一共 103 行
   (`experiments/imo_proof/seed_agent/`),**没有任何地方提到 8 次上限**
   —— 没有 README、没有 docstring、没有 preflight 校验。优化器被邀请去改
   revision loop,却不知道 >6 轮的循环在结构上不可计分。

最干净的证据:`solver.py +1/-1 | added: max_revisions = 5` —— **改一个
字符**,fitness_delta **−0.19**。这个数量级不可能来自解题质量。

---

## 🔴 缺陷 2:L1 归因编造因果,并把编造固化成"经验"

缺陷 1 的失败原因在到达反思器之前就被丢掉了。`grade.py:_error_category`:

```python
return "exec:" + item.failure.split(":", 1)[0].strip()
# "CandidateExecutionError: solver call limit exceeded" -> "exec:CandidateExecutionError"
```

**"solver call limit exceeded" 这句话从不出现在优化器或反思器看到的任何
地方。** 于是 LLM 只能编一个像样的数学故事。`e5s_r2` 实录:

> child `66f3906f4eea`(= `max_revisions = 5`)
> verdict: **regressed**
> why: "Increasing the revision limit to 5 allowed the model to **over-edit
> valid solutions, introducing new errors through excessive self-correction**"
> advice: "**Cap revision loops at 3 iterations** to prevent degradation
> from over-refinement and token waste."

真相是 9/12 道题**根本没被评分**。没有任何"过度编辑"发生。

**这条编造出来的 advice 随后作为 lesson 注入后续 prompt。** 完整链条:

```
评估缺陷 → 原因被 error_category 截断 → LLM 编造归因
        → 写入 experience buffer → 注入后代 prompt → 搜索被永久推离该区域
```

**经验层在评估信号不可信时不是增益,是放大器。** 这是"LLM 归因比机械记录
多赚"这个论点的反面:机械记录只会沉默,归因会**自信地说错**。

---

## 🟠 缺陷 3:选择压力失效 —— 亲本单一化

| run | 被反复选中的亲本 | 次数 | 该岛总代数 |
|---|---|---|---|
| e5s_r2 | `a77e4693037e` (fit .4405) | **6** | 7 |
| e3r_r2 | `740ec0a5276b` (fit .3571,**是 seed**) | **6** | 7 |
| e5s_s0 | `3e53637f5be1` (fit .4524) | **5** | 7 |

`e5s_r2` 岛 1 连续 6 次选同一个亲本,子代 fitness 依次
`.3452 / .4286 / .25 / .25 / .3214 / .2976` —— **五次回落**。
`RegressionSoftPenalty`(decay 0.6,floor 0.1)确实在算权重,但**没能改变
结果**:被罚到地板 0.1 之后它仍是岛上最优,其余候选权重更低。

**软门的地板设计在小岛上失效** —— floor 0.1 是相对权重,当只有 7 个候选、
且被罚者仍是最优时,0.1 仍然赢。

`e3r_r2` 更糟:被选 6 次的是 **seed 本身**,13 代里岛 1 从未离开原点。

---

## 🟠 缺陷 4:岛屿从不迁移 —— 这不是岛模型,是两条独立链

三个 run,**跨岛亲本选择 = 0**。15 代里迁移一次都没发生。

实际形态:2 个岛 × 每岛约 7 个候选,严格交替(isl 1,0,1,0,…),
**每代只产 1 个候选**。所谓"种群"是两条长度 7 的链。这个规模下:

- 归档 5 个 / 共 16 个,但归档对选择的影响被单一化淹没;
- 任何基于种群多样性的机制(novelty、bandit、迁移)都没有作用空间。

---

## 🟡 缺陷 5:40–88% 的 lesson 是"这是噪声,忽略"

| run | improved | noise | regressed |
|---|---|---|---|
| e5s_r2 | 2 | **5** | 5 |
| e5s_s0 | 2 | **6** | 4 |
| e5s_verify | 1 | **7** | 0 |

判据是 `abs(fitness_delta) <= 1/n_items and flips <= 1`(即 |Δ|≤0.083)。
lesson 原文反复出现 "within the established noise floor (~0.083)"。

问题不是判据太松 —— 真实 sem 是 0.14,**0.083 还偏紧**。问题是:
**这个判据一旦命中,这次 LLM 调用就产出零信息**,而它命中了近一半。
且 "noise" 判定用的是**对称的 hamming**,分不清 `+1 −1`(纯搅动)和
`+0 −2`(实打实退步)—— 见 `p01_noise_plan.md` §3。

---

## 🟡 缺陷 6:行为去重被缺陷 1 污染

`e5s_r2` 三个被预算饿死的候选(gen 6/7/9)产出**逐位相同**的行为向量
`000000110001`,于是全部被 `BehavioralNoveltyPolicy` 标记为
`behavior_duplicate` —— 被逐出归档并把选择权重乘 0.25。

**16 个候选里 11 个被标为重复(69%)**;`e3r_r2` 是 5/14;`e5s_s0` 是
**0/15**。同一个策略、同一个臂,0 vs 11 —— 这个机制的行为不可预测,
而它对选择施加的压力很大。

同时 `novelty_rejections_total` 在 15 步里**恒为 0**:identity 模式下
novelty 门自换用后一次都没触发,现在是纯开销。

---

## 🟡 缺陷 7:成本可见性完全为零

`sys/llm_cost_total` 与 `sys/eval_cost_total` **全程 0.0**,每个 agent
session 的 `cost_usd` 也是 0.0 —— 定价字段为 null,`BudgetMeter` 空转。

**实际成本结构(e5s_r2,已推算)**:

| | 值 | 占比 |
|---|---|---|
| 墙钟总时长 | 4.33 h | 100% |
| 优化器(14 个 session) | 1152 s | **7.4%** |
| 评估 | ~14450 s | **92.6%** |
| 优化器 prompt tokens | 1.46 M | ~18% |
| 评估 solver tokens | ~6.5 M | ~82% |

**结论**:框架成本被评估完全主导。这意味着 ——
① 任何"买第二意见"的抗噪方案花的是最贵的那种钱;
② 经验/反思/prompt 层的改进几乎免费,是应该先榨干的地方。

---

## 🟡 缺陷 8:manifest 记不全,导致 run 之间不可比

`experiment_manifest.json` 的 `actual` 只记了 proposal 侧配置和三个开关,
**完全没有记录搜索配置**:岛数、每代候选数、archive_size、
hamming_threshold、duplicate_penalty、迁移间隔、RegressionSoftPenalty 参数。

更实际的问题:**`e5s_s0` 的 `protocol_fingerprint` 与 `e5s_r2` / `e3r_r2`
不同**(`1d97890…` vs `93649c7…`)—— 题目划分就不一样,**不能放在一起
比较**。`e5s_s0` 还早于 `code_version` 字段,连代码版本都不知道。

顺带:同一 run 内 `model_name` 同时出现 `qwen3.7-plus` 和
`openai/qwen3.7-plus`,按模型做的任何统计都会被劈成两半。

---

## 🟢 缺陷 9:两个开关从未被打开,三个工具从未被调用

- `lesson_directive: false`、`operator_bandit: false` —— **所有已记录的
  run 都是 false**。这两个机制写完了、测试绿了、**从未在真实 run 里跑过**。
- 工具调用统计(14 个 session,173 次):

```
workspace_read 82   inspect_parent_eval 33   workspace_edit 20
inspect_candidate 16   run_preflight 14   workspace_write 6   workspace_glob 2
run 0   workspace_grep 0   workspace_delete 0
```

  渐进披露的两个工具(`inspect_parent_eval` 2.4 次/session、
  `inspect_candidate` 1.1 次/session)**确实活着**,这是好消息。
  但 `run` / `workspace_grep` / `workspace_delete` **零调用**,而每个工具
  定义在每一轮都要占 prompt。

- `repair_rounds` 全 0、`attempts` 全 1、`turns` 均值 11.4(上限 20)、
  `tool_calls` 均值 12.4(上限 40)—— **限额都没有被逼近**,不是瓶颈。

---

## 修复优先级

| | 项 | 理由 |
|---|---|---|
| **1** | **预算耗尽必须与"答错"分开记账** | 27% 的候选被错记;它污染 fitness、lesson、去重三处 |
| **2** | **把约束写进 workspace,并在 preflight 里查** | 优化器不知道 8 次上限;告诉它比惩罚它便宜得多 |
| **3** | **`_error_category` 保留失败原句** | 断了这一条,L1 只能编 |
| **4** | **反思器缺证据时必须允许"不知道"** | 现在的 schema 逼它在 improved/regressed/noise 里三选一 |
| 5 | 软门地板改成绝对判据(连败 N 次后**换岛/换族**,而非只降权重) | 小岛上相对权重罚不动 |
| 6 | manifest 记全搜索配置 + 统一 model_name | 否则跨 run 结论都不可信 |
| 7 | 摘掉零调用工具 / 让 novelty 门要么起作用要么下线 | 纯开销 |
| 8 | 打开 `lesson_directive` / `operator_bandit` 各跑一臂 | 已建成但从未验证 |

**顺序上的判断**:1–4 全部关于**信号纯度**,而且都在便宜的那一侧
(优化器只占 7% 墙钟)。`p01_noise_plan.md` 里的抗噪工作应该排在它们
**之后** —— 在信号被系统性污染时降噪,降的是错误信号的方差。
