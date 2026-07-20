# Coding 任务域接入计划（模式 A → B → C）

> 状态：Draft v1.1（2026-07-14 D7 升格；v1 = 2026-07-09）
> **D7 注**：本计划从"接口预留"升格为 **WS-4 泛化主线第一站（T1）**——
> 对照 HyperAgents `domains/polyglot`（同为确定性单测型 coding 域）。
> 模式 B 与 WS-3 M3（AgentSessionProposer）同物，优先级同步上调；
> 模式 C 即 HyperAgents 本体路线，仍按 P3 门槛（多域可信基线后）。
> **模式 A.5（2026-07-14 新增）**= WS-3 M2.5：多文件单发变异（`### FILE:` 全文块
> 格式 + with_files 透镜），介于 A（单文件单发）与 B（agent 会话）之间——
> 本任务域的仓库型题目在 M3 之前即可用 A.5 进化。
> 前置阅读：`docs/architecture.md`（分层与 Grader 契约）、`docs/naming_map.md`
> 结论先行：**先做模式 A 的自包含任务版**（单文件代码 + 自带测试，不碰 Docker），
> 复用现有 SearchLoop / StagedGrader 分级思想 / C1-C3 全套改造；
> 模式 B（agent 作为变异算子）与模式 C（进化 agent harness 本身，DGM 路线）作为后续阶段，
> 本文只锁定 A 的实现细节与 B/C 的接口预留。

---

## 0. 三种模式定义（决策记录）

| 模式 | 候选（Candidate.code）是什么 | 变异方式 | 评估方式 | 状态 |
|---|---|---|---|---|
| **A** | 解题程序（单文件 Python） | 单次 LLM 调用（现有 PromptBuilder + PatchEngine） | 跑测试集 → pass rate | **本计划实施** |
| **B** | 解题补丁（可多文件） | agentic 会话产出 diff（AgentClient） | 同 A / 仓库级测试 | 接口预留 |
| **C** | coding agent 的 harness 代码 | agent 改自己的 harness | 固定 SWE 子集分数 | 远期，只记录约束 |

选 A 先行的理由：
- engine 零改动——只新增 `tasks/coding/` 实现 `evocore.interfaces.Grader`；
- 自包含任务（每题测试自带、无第三方依赖）让现有 `evoguard.Sandbox`（rlimit + subprocess）够用，Docker 化推迟；
- 逐测试 pass/fail 向量直接充当 C2 的 `BehaviorSignature.pass_vector`，C1 的错误类别直方图也天然成立。

## 1. 数据集选型（第 1 步，先于写代码）

| 候选 | 优点 | 缺点 | 决定 |
|---|---|---|---|
| **LiveCodeBench 子集** | 题目持续更新（防污染）、难度分层、测试自带 | 需要写题目适配器 | **首选**，取 medium/hard 各 ~20 题 |
| HumanEval+ | 接入最快、测试增强版 | 污染严重，fitness 天花板太低（区分度差） | 备用冒烟集（L1 用） |
| KernelBench | 贴数学/物理叙事（数值 kernel 优化），fitness 连续 | 需要 GPU / torch 环境，评估慢 | 二期，作为第二个 coding 任务 |

产出物：
- [ ] `tasks/coding/datasets/` 下落地 train/holdout 切分（JSONL，固定 seed）
- [ ] 每题 schema：`{id, statement, starter_code, tests: [{input, expected} | assert_src], difficulty, tags}`
- [ ] holdout 集路径进 `AntiHackScanner` 的禁读名单

## 2. tasks/coding/ 模块设计

```text
tasks/coding/
  __init__.py
  dataset.py        # 题目加载、train/holdout 切分、按 seed 抽子集
  runner.py         # 单题执行：sandbox 内跑候选程序 + 测试，产出 ItemResult
  taxonomy.py       # 错误分类器（见 §4）
  grader.py         # CodingGrader：实现 Grader 协议，内嵌分级早退（见 §3）
  seeds/            # 初始种子程序（朴素解 / starter code 直传）
```

依赖方向不变：`tasks/coding → evocore.interfaces + evoguard`，engine 不 import tasks。

### CodingGrader 契约

```python
class CodingGrader:
    """实现 evocore.interfaces.Grader。
    fitness = pass_rate(train) − λ·cost_norm(runtime)   # λ 起步 0.1，消融可调
    structured_feedback = 逐题 ItemResult + error_histogram(taxonomy 类别)
    """
    def __init__(self, dataset: CodingDataset, sandbox: Sandbox,
                 scanner: AntiHackScanner, budget: BudgetLike,
                 screen_margin: float = 0.05): ...
    def grade(self, cand: Candidate, workdir: Path) -> EvalReport: ...
```

## 3. 分级早退映射（对齐 StagedGrader 概念，coding 版）

| 级 | 内容 | 早退条件 | 成本 |
|---|---|---|---|
| **L0** | AntiHackScanner 扫描 + `ast.parse` + EDIT-REGION 标记校验 | 命中扫描规则 / 语法错 → `passed=False, fault=...` | 零 |
| **L1** | 冒烟：2 道最简单题（固定） | 全挂（含崩溃/超时）→ 早退 | 秒级 |
| **L2** | 筛选子集：train 的固定 seed 30% | fitness < parent.fitness − screen_margin → 早退，`stage_reached=2` | ~1/3 |
| **L3** | train 全量 → 正式 fitness + StructuredFeedback + 行为签名 | — | 全量 |

沿用 architecture.md §5.2 的约束：L2 早退候选进种群不进 archive；A3 对照实验时降级为直通 L3。

## 4. 错误分类 taxonomy（C1 反馈质量的关键）

`taxonomy.py` 把每题失败归入固定类别（顺序即优先级，一题一类）：

| 类别 | 判定依据 |
|---|---|
| `syntax_error` | L0 已拦，理论上不出现在逐题层 |
| `runtime_exception:<type>` | stderr 中的异常类型（KeyError/IndexError/...，取 type 名） |
| `timeout` | SandboxResult 超时标记 |
| `wrong_answer` | 输出与 expected 不匹配（диff 摘要进 ItemResult.predicted/expected） |
| `partial_output` | 输出格式对但不完整（可选，二期） |
| `memory_exceeded` | rlimit 触发 |

要求：类别名稳定（进 `BehaviorSignature.error_histogram` 与 C3 检索 key），新增类别走追加不重命名。

## 5. AntiHackScanner 新增 coding 规则

- [ ] 禁读 holdout / tests 数据路径（现有规则复用，加 coding 数据集路径）
- [ ] 检测"打表"：候选中出现对 expected 输出的大段字面量硬编码（启发式：超长字符串/字典字面量 + 与测试期望高重合，先做保守版只报 Finding 不判死）
- [ ] 禁 `sys.modules` 篡改、`unittest.mock` 导入（模式 A 阶段测试由 runner 侧执行，候选程序本身接触不到测试代码——runner 设计上保证候选进程与断言进程分离）

**runner 关键设计**：候选程序以子进程方式接收 stdin/args、产出 stdout，断言在 runner（沙箱外的受信进程）中做。候选代码永远见不到测试内容 → 从结构上消除改测试/mock 类攻击面，扫描规则只是纵深防御。

## 6. 复用/不动清单

| 组件 | 处置 |
|---|---|
| SearchLoop / selection / novelty / routing | 不动 |
| PatchEngine（SEARCH/REPLACE 单文件） | 不动（模式 A 单文件够用） |
| PromptBuilder | 不动；`task_sys_msg` 换 coding 版系统提示（解题目标 + 输出格式 + EDIT-REGION 约定） |
| C1 FeedbackContributor | 不动（structured_feedback schema 兼容） |
| C2 BehavioralNoveltyPolicy | 不动（pass_vector 语义一致） |
| C3 ExperienceStore | 不动（检索 key = taxonomy 类别） |
| recipes/e0–e3r | 加 `--task coding` 装配分支（common.py 里按 task 名选 Grader + 数据集） |

## 7. 模式 B/C 的接口预留（本期只留口子，不实现）

- [ ] `evocore/llm.py` 的调用点抽一个最小协议 `ProposalClient`（`query(system, user, model) -> (text, cost)`），
      `LLMClient` 是其一个实现 → 未来 `AgentClient`（headless agent 会话 → diff + session cost）可平替。
      改动控制在一个协议声明 + 类型标注，不动行为。
- [ ] `evoguard/sandbox.py` 的 `Sandbox` 上提一个抽象基类，现实现更名为 `LocalSandbox`（对外别名保持 `Sandbox` 兼容）；
      预留 `ContainerSandbox` 占位（仓库级任务 / SWE-bench 需要 per-instance 镜像）。
- [ ] trace 落盘：模式 B 的 agent session 中间步骤是诊断层最有价值数据，本期先保证
      `EvalReport.stdout_log/stderr_log` + workdir 归档路径可回放，schema 版本化。
- 模式 C 约束备忘：元沙箱（沙箱里跑会开沙箱的 agent）、固定小题集 + 激进 L2 早退、
  AntiHackScanner 升级为 harness 行为审计。**不排期**。

## 8. 里程碑与验收

| 里程碑 | 内容 | 验收标准 |
|---|---|---|
| **M1** 数据集 | LiveCodeBench 子集落地 + schema + 切分 | `dataset.py` 单测通过；train/holdout 无泄漏 |
| **M2** Runner + taxonomy | 单题执行 + 错误分类 | 对 5 个手工构造的坏程序（语法错/超时/WA/异常/打表）分类全对 |
| **M3** CodingGrader | 分级早退 + EvalReport 产出 | seed 程序 grade 出合理 fitness；L0–L2 早退路径有单测 |
| **M4** 端到端 | recipes 装配 + E0 组小规模跑通（~30 代） | fitness 曲线上升；`results/` 出 run 报告 |
| **M5** 消融 | E0 vs E1 vs E2 在 coding 任务上对照 | 报告含 fitness 曲线 + 行为覆盖率 + 成本，结论可写进对外文档 |

预估：M1–M3 约一周，M4–M5 视预算一周内。KernelBench（二期）在 M5 后另开计划。

## 9. 风险

- **数据污染**：LiveCodeBench 选题时间窗要晚于所用模型的训练截止；报告里注明窗口。
- **fitness 天花板**：题太易 → 前 10 代打满、进化无区分度。选题时用 baseline 单次调用先测通过率，目标落在 30–70% 区间。
- **评估成本**：coding 全量 L3 比 equational 慢（每题多 case）。用 `execution_time` 记账，必要时 L3 也抽样（记 deviation）。
- **打表检测误报**：保守起步（只报不杀），累积 Finding 样本后再收紧。
