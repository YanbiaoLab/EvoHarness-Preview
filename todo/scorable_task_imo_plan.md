# ScorableTask + IMO Proof 主计划

> 状态：执行中（2026-07-20）  
> 产品边界：用户提供 `grade_func + seed_agent`；EvoHarness 负责候选复制、Agent 修改、
> 工作区恢复、执行评分、失败归类、选择、归档和 transcript。  
> 首个验收域：IMO Proof。modmul 不作为本阶段设计约束或迁移门槛。

## 1. 目标 API

```python
task = ScorableTask.from_directory(
    seed_dir=HERE / "seed_agent",
    grade_func=grade_func,
    main_file="solver.py",
    include_files=("solver.py", "prompts.py", "policy.py"),
    task_sys_msg=TASK_SYSTEM_PROMPT,
)
```

任务侧保留：

- `seed_agent/`：允许 Agent 修改的候选程序。
- `grade.py`：把候选工作区映射为 `GradeValue`。
- 数据集、judge、隐藏答案、预算和 admission 配置：只读评测依赖，不进入候选工作区。
- `task.py` 或薄组装函数：绑定依赖并调用 `ScorableTask.from_directory()`。

框架侧统一承担：

- 文本工作区读取、安全路径检查、Git 基因型和 patch lineage。
- `GradeContext` 构造、`GradeValue → EvalReport`、异常与 infra failure 分类。
- `TaskBundle` 兼容、SearchLoop、AgentSessionProposer、preflight、工具与 transcript。

## 2. 阶段与状态

### 阶段 1：冻结公共任务契约 — 已完成

- [x] 新增顶层 `evoharness.task`，避免任务 API 下沉到搜索引擎内部。
- [x] 导出 `ScorableTask`、`WorkspaceGradeFn`、`WorkspaceGradeFnGrader`。
- [x] `recipes.common.TaskBundle` 保留为 `ScorableTask` 兼容别名，旧任务无需同步迁移。
- [x] 保持依赖方向：框架不 import `tasks/`、`recipes/` 或 `experiments/`。

验收：现有 TaskBundle 调用不变；新任务可直接 `from evoharness import ScorableTask`。

### 阶段 2：目录种子成为一等工作区 — 已完成

- [x] `GitWorkspace.from_directory()` 从 `seed_agent/` 冻结初始基因型。
- [x] 支持 `main_file` 和显式 `include_files`，IMO 直接复用冻结的 `mutable_files`。
- [x] 拒绝绝对路径、`..`、symlink、超大文件和候选内二进制文件。
- [x] 忽略 `.git/` 与 `__pycache__/`，避免本地运行缓存污染候选。
- [x] 缺少主入口或白名单文件时在搜索前失败。

验收：IMO 的 `solver.py/prompts.py/policy.py` 成为同一个 GitWorkspace，其他文件不进入基因型。

### 阶段 3：通用工作区评分 adapter — 已完成

- [x] `WorkspaceGradeFn(candidate_dir, GradeContext) -> GradeValue` 成为多文件评分边界。
- [x] adapter 在独立目录物化候选，并补齐 candidate id、generation、operator 和评测 workdir。
- [x] 普通异常转为失败 `EvalReport` 并保留 traceback。
- [x] `InfraError` 转为 `EvalInfraError`，不把供应商/评测基础设施故障记成候选零分。
- [x] 保留 `adapt_source_grade_fn()` 与 `ScorableTask.from_source()`，但它们只是单文件兼容层。

验收：任务作者不实现框架 `Grader`；单文件和多文件评分走同一个报告转换路径。

### 阶段 4：IMO Proof v1 迁移 — 已完成

- [x] 新增 `experiments/imo_proof/grade.py`，任务评分以 `grade_func` 表达。
- [x] `IMOEvaluator.evaluate_directory()` 直接消费已物化候选，避免重复构造 Workspace。
- [x] 删除 EvoHarness adapter 内的任务专有 `IMOProofGrader`。
- [x] `make_task()` 使用 `ScorableTask.from_directory(seed_agent, grade_func, ...)`。
- [x] train 评分、admission failure、solver/grader usage、结构 DOA 和 infra failure 语义保持不变。
- [x] `CandidateEvaluation` 收敛为唯一任务评测 IR；本地引擎、独立 worker 和 HTTP backend 不再定义平行结果模型。
- [x] `evaluation/` 独立拥有 dataset、candidate executor、judge、聚合、worker 与可选 HTTP 服务。
- [x] `grade.py` 只依赖 `evaluation.contract` 并执行 `CandidateEvaluation → Grade`；协议/基础设施故障均不记为候选零分。
- [x] train/validation/test 可路由到三个 profile 固定、凭据隔离的评测服务；完整候选工作区按内容寻址和幂等复用。
- [x] 原生 SearchLoop + AgentSessionProposer 的离线 IMO 端到端回归通过。

验收证据：`tests/test_imo_proof.py` 覆盖唯一 IR、进程 worker、HTTP API、内容缓存、
profile 路由和协议失败分类；排除仍引用已删除 `modmul` 包的两份旧测试后，仓库其余
428 项测试通过。

### 阶段 5：DryRunReport — 下一步

- [ ] `task.dry_run()` 至少评估一次冻结 seed，验证分数有限、入口可加载、依赖可用。
- [ ] 输出候选文件清单、seed hash、耗时、调用/token/cost、admission 与失败类别。
- [ ] 随机任务重复 2–3 次，仅报告观测方差；本阶段不自动推断统计结论。
- [ ] dry-run 失败时不创建正式 population/archive。

### 阶段 6：ScoreContract v0

- [ ] 冻结方向（max/min）、失败分、主指标、随机性来源和最低复评样本数。
- [ ] 记录 task/grader/dataset/seed 指纹，禁止不同评分契约的候选静默混排。
- [ ] IMO 映射现有 BenchmarkSpec：points percentage、train/validation/test、grader/dataset hash。
- [ ] ScoreContract 保持纯 IR，不负责 prompt 渲染或执行。

### 阶段 7：更薄的任务启动入口

- [ ] 将通用 run 组装从 IMO adapter 中抽到框架入口，任务侧不再手写 RecipeContext。
- [ ] 支持 `evoharness run path/to/task.py` 或等价 Python API。
- [ ] CLI manifest 自动记录 ScorableTask、ScoreContract、模型、预算和 proposal 配置。
- [ ] IMO 实验只保留任务专有的 validation 选择、一次性 test 和结果报告。

### 阶段 8：可信评分与跨任务验证

- [ ] train/search 候选按 ScoreContract 重采样并产生 `mean/std/n/lcb`。
- [ ] validation 只负责选择，held-out test 只在最终候选上执行一次。
- [ ] 用第二个非 IMO 多文件任务验证 API，而不是回头让 modmul 约束公共设计。
- [ ] 对 Agentic 与 `tools=[]` 做相同模型/预算的 DOA、有效候选率、成本和提升率对照。

## 3. IMO 目录职责

| 文件/目录 | 保留理由 | 是否候选可变 |
|---|---|---|
| `seed_agent/` | 初始 solver agent 基因型 | 是，仅白名单文件 |
| `grade.py` | `CandidateEvaluation → Grade` 的 EvoHarness 薄适配 | 否 |
| `evaluation/contract.py` | 唯一评测 IR 与 backend 协议 | 否 |
| `evaluation/engine.py` | admission、solver 执行、judge 和账务 | 否 |
| `evaluation/worker.py` | 独立 evaluator/candidate 进程入口 | 否 |
| `evaluation/service.py` | 可选 HTTP 服务与 client backend | 否 |
| `protocol.py` + `benchmark.v1.json` | 冻结公平比较协议 | 否 |
| `evolution.py` | ScorableTask、grade_func 与原生 SearchLoop 接入 | 否 |
| `result.py` | 单次实验 run manifest | 否 |
| `assets/` | 本实验冻结的数据集和 grader prompt | 否 |

## 4. 不做

- 不把 IMO 的 BenchmarkSpec、ProofDataset 或 judge 协议塞进 `evoharness/`。
- 不让 `grade_func` 自己实现 SearchLoop、Agent tools、workspace patch 或 transcript。
- 不把 provider outage、judge failure、dataset missing 记成候选失败。
- 不让候选读取数据集、参考解、grading guidelines 或 held-out。
- 不为兼容 modmul 扭曲多文件任务 API；旧单文件任务使用兼容 adapter 即可。
- 不在 DryRun/ScoreContract 之前进入 ResearchIdea、Breakthrough Ledger 或 Flat-UCB。

## 5. 下一步最短路径

只进入阶段 5：定义纯数据 `DryRunReport`，为 `ScorableTask.dry_run()` 写 IMO 离线验收，
先保证失败不会污染正式 run，再讨论 ScoreContract 字段。
