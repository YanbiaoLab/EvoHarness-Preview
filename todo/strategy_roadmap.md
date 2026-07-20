# 战略路线图：泛化任务基座 → 评估资产 → 产品外衣

> 状态：Draft v2（2026-07-14 方向修订，见 D7；v1 = 2026-07-10）
> 前置阅读：`docs/eval_protocol.md`（远程评估协议）、`docs/architecture.md`、`todo/coding_task_plan.md`（coding 任务域，模式 B 与本文 M3 同物）、`todo/hypothesis_generation_product.md`（Discovery 产品决策）
> 结论先行：**价值排序 C > B > A 不变；执行顺序由 A → C → B 修订为
> B（WS-3 agentic 基质）+ WS-4（多域任务接入）并行主线，C 持续，A 冻结存档（D7）**。
> A = 单任务比赛战绩（SAIR，已冻结，资产保留为第一个全链路任务案例）；
> B = Claude Code 式持续进化产品表面；C = 评估协议 + 编排经济层（护城河）。
> 新检验题：**任务泛化性**——参照 HyperAgents `domains/` 谱系
> （polyglot / imo / paper_review / search_arena / balrog / genesis），
> 一个新域接入的边际成本是否持续下降、引擎是否零改动。原检验题仍成立：
> grade_fn 生态、裁定护栏、失败分类、预算经济不随 agent 供应商更替贬值。

---

## 0. 决策记录（2026-07-10 对话沉淀）

| 决策 | 内容 | 理由 |
|---|---|---|
| D1 | 价值 C>B>A，执行 A→C→B | 模型变强 → 瓶颈移向验证；scaffolding 的苦涩教训侵蚀 A 的 prompt 装配层 |
| D2 | 基因型升级走 git 基质（Candidate.ref = commit SHA） | 血统=DAG、去重=tree hash（直接充当协议幂等键）、物化=worktree |
| D3 | agent 会话作为重型变异算子，与 single-shot 混合调度 | 成本 10–100×，"一次变异多聪明"本身是消融维度（Proposer 头注既定方向） |
| D4 | modmul 不等 agentic 改造：v0 用现框架单文件构造器模式 | 两线并行，M3 落地时 modmul 升级为旗舰案例 |
| D5 | 加仓：协议 v2 / 裁定图审计 / 预算经济 / HITL；维持不加仓：PatchEngine / prompt 装配细节 / 选择器变体 | 后者被模型进步稀释 |
| D6 | Hypothesis Generation 独立产品入口、共享 EvoHarness 平台；先内部模块验证闭环再拆 SKU | 假设发现与实验执行的用户旅程不同，但拆后端会切断证据—实验—结果回流；不改变 A→C→B 近期顺序 |
| D7 | **（2026-07-14）废止"近期一切服务于 A（打赢 SAIR）"**：近期服务 HyperAgents 示例域式的多任务泛化（WS-4），WS-1 冻结存档；WS-3 升为最高优先 | 目标从单任务战绩转向框架泛化性；SAIR 资产（grade_fn、镜像裁判方法论、evoserve 部署经验）保留为任务接入范例；modmul 实测教训（E-σ/E-DOA/E-fit 等）是引擎层的，不随任务退役 |

---

## WS-1 modmul / SAIR（已冻结存档，D7 2026-07-14）

> **冻结说明**：下列未完项（计时 harness、神经性裁定 v1/v2、官方提交闭环、E 矩阵
> 放量、GPU 部署调参）不再推进。已完成部分作为**第一个全链路任务案例**保留：
> grade_fn 契约实践、镜像裁判忠实度验证方法论、sys_msg/research_brief 双文档模式、
> Round 0 基线纪律——这些是 WS-4 每个新域接入时的模板。

- [x] **grade_fn v0+v0.5**（2026-07-10 完成，`modmul/grade.py` + 5 测试）：官方 `check_source` 裁定 + 官方 `decode_answer` + `public_benchmark` tier 1–3 各 30 题；`train()` 钩子走 `Sandbox` 子进程硬超时
  - [x] 官方仓库摸底：提交=model.py+manifest+权重；11 tiers；等权平均计分；静态指纹禁 `int*int%int` 与 3 参 pow（int() 本身合法）
  - [x] 候选契约 = 官方契约：`MANIFEST` dict + `ModularMultiplicationModel` 子类
  - [x] 种子 `seeds/serial_ar.py`（closed_genome 赢家假设改造）+ 契约测试 3 绿
  - [x] **Round 0 基线**（500 步 MPS 快速轮，111s）：fitness **0.2556**，t1=63.3% / t2=10% / t3=3.3%，零 malformed（输出契约全学会）；完整 3000 步基线待 GPU
  - 注意：Sandbox rlimit 内存上限可能干扰 CUDA，GPU 机需可放宽；`baselines/closed_genome.py` 为无 LLM 对照臂
- [x] **背景资产撰写**（2026-07-10 完成 v1）
  - [x] 规则转译 → `modmul/sys_msg.md`：契约 + 四层裁定 + "溯源非架构"边界 + 反馈类别解读（来源：官方 rules/overview+evaluation.md 原文，比外网采集更权威）
  - [x] 方法级领域简报 → `modmul/research_msg.md`：地形与 Round 0 基线 / NeuralHorner 方案形态（含合法性走位与 Fermat 弱点）/ 文献蒸馏（grokking 警示、跨素数迁移失败、数据分布杠杆、重复采样、位置鲁棒）/ 6 条变异公理
  - [x] 版本化：两文件头部绑定 task_version，改动升版并记 manifest
- [ ] **计时 harness**：镜像 1100 题 / 5 分钟，在评估机上测；计时噪声（重评方差）先测量再定对策
- [ ] **神经性裁定 v1**：predict 路径禁 `pow`/`%`/大整数乘 + 数值抽查；v2 计算图算子白名单审计（跟 Zulip 判例对齐）
- [ ] **官方提交闭环**：最优候选 → 官方提交格式导出；定期官方服务器校准镜像裁判偏差
- [ ] **E 矩阵试跑**：小预算第一轮 → 校准 fitness 定义（bit-size 档加权、时间罚项 λ）→ 放量
- [ ] 部署：GPU 机 evoserve、torch 环境、候选构建子进程隔离、Bearer token、max_workers 调参

## WS-2 评估协议与编排经济（形态 C：护城河）

- [x] **批式代**（2026-07-10）：`SearchConfig.eval_batch_size`（默认 1=串行 parity）+ `_run_batch_generation`——线程只跑 `grader.grade()`，预算/种群/路由主线程串行吸收；Barrier 并发证明测试 + infra 熔断语义保持
- [x] **首次真实进化 run**（2026-07-10，`results/modmul_run1`）：e1 + qwen3-coder-plus（DashScope compat）+ 批 2 + 快速种子，10 评估：**best 0.267 > seed 0.256**（gen1 revise，t1 63%→80%）；gen3 出现 t3 方向改进（3.3%→6.7%）；坏变异被 train-failed/裁定正确拦截、repair 算子实战触发。修了一个真 bug：相对 run-dir + Sandbox chdir 的路径二次拼接（回归测试已锁）。已知项：compat transport 不计价 → llm_cost=0
- [ ] **协议 v2：工作区上线**：git bundle / blob 端点；幂等键升级为 `tree_hash + task_version`；`grade_fn(workdir, ctx)` 签名；wire report / 失败分类 / 202 语义不动
- [ ] **预算经济可观测化**：每美元 fitness 提升、缓存命中率、算子成本分布——做成 run 报告一级指标
- [ ] `evowire` 抽包触发条件盯守（已记录：第二个外部评估实现出现 / schema 频繁升版 / 协议出仓库）
- [ ] 可选：作业表落盘（服务重启失忆；客户端已自愈，低优先级）
- [ ] 里程碑验收：**一个外部任务方仅凭 docs/eval_protocol.md 接入成功**（不 import 本仓库）

## WS-3 Agentic 工作区进化（形态 B；**D7 后最高优先**）

> 与 `todo/coding_task_plan.md` 模式 B 同物；两轴正交：轴A=变异算子智能度，轴B=基因型。
> D7 注：HyperAgents 示例域的候选都是多文件 agent 仓库，Workspace/git 基质（M1）与
> 提案通道升级（M2.5/M3）从"产品表面"升为泛化目标的**结构前置**。
> M1 现状（2026-07-14）：workspace.py + Candidate/store 接线 + 全部 `.code` 消费点
> 替换完成（173 测试全绿）。M2 复核提醒：WS-2 的"协议 v2 工作区上线"仍是未完项，
> M2.5/M3 走本地评估可先行，远程工作区评估前须补 M2。
> **排序决策（2026-07-14 晚）**：多文件变异不等 agent——M2.5 先于 M3 独立交付；
> 两条提案通道会师于 `Proposal.workspace` 双通道接口（code 永远=主文件文本，
> workspace 可选=完整子代基因组），M3 用 `git diff` 收口，M2.5 用 edits dict 收口。

- [x] **M1** `Workspace` 抽象 + git 种群基质；`FileWorkspace` 保持现行为——纯 parity 重构，**验收：全量测试不红**（2026-07-14 完成）
- [x] **M2** = WS-2 协议 v2（同一事项，勿重复做）
- [x] **M2.5 多文件变异（先于 agent，新增；2026-07-15 完成，184 测试全绿，冒烟=test_multi_file_evolution_smoke）**：
  ① `Proposal.workspace` 双通道 + 透镜原语 `with_main_text`/`with_files`
  （含塌缩 bug 修复：子代继承 `workspace_kind`；FileWorkspace 拒绝长新文件）——进行中；
  ② 工作区渲染进 prompt（小仓库全量文件，大仓库是 M3 的地盘）；
  ③ 多文件输出格式 `### FILE: path` 全文块 + 解析（不让 LLM 手写 unified diff，
  补丁由 git 计算——把容易错的活从 LLM 手里收回来）；
  ④ 反作弊扫描升级为全工作区（安全硬项：git 下作弊代码可藏进副文件绕过主文件扫描）。
  **验收**：git 种子任务上单发算子完成一次跨文件变异（改 N + 新建 M），谱系可重建，
  全量测试不红
- [ ] **M3** `AgentSessionProposer`（**S0-S8 框架与离线重放已完成，410 tests；仅缺外部
  付费模型 live smoke，故暂不勾选里程碑**；2026-07-16 修订：借鉴 HyperAgents/Claude Code
  的持久会话工具循环形态，但使用厂商无关的原生结构化 tool calling、可注入 transport、
  turn/tool/deadline 机械强制与成本准入上限，弃 `claude -p` 与文本标签协议；
  **P1.1 并入本项**,Conversational=tools=[] 消融臂,run 工具打 E-DOA 主因）：
  每会话工具/轮次/deadline 硬顶；成本在 provider 无服务端预算时采用下一调用准入上限并
  如实记录单次 overshoot；transcript 全量归档为一级实验产物、
  混合算子调度（agent 会话低概率 / 停滞触发）；收口走 M2.5 的 `Proposal.workspace` 通道。
  详细施工顺序、契约与验收见
  [`docs/agent_session_proposer_plan.md`](../docs/agent_session_proposer_plan.md)。
- [ ] **M4** inspiration 只读 worktrees（agent 自己 diff 精英候选）、repair 升级为真调试会话、多文件 novelty（diff embedding）
- [ ] **研究员 Copilot（外环 agent，设计已定稿 → `docs/research_copilot_design.md`）**：R1 只读 Findings（6 playbook）→ R2 提案卡+git 应用+版本强制 → R3 自主验证 run；前置：**仓库 git init**、schema 先行、playbook 配测试
  - [x] 控制台交互层（2026-07-12）：Insights 标签页（Finding 卡+证据深链芯片 / Proposal 卡+diff+版本横幅+接受/拒绝护栏）、决策 API（拒绝必填原因、L2 无版本升级不可接受、已决策不可变——服务端机械强制）、待审徽标；golden 样本 = modmul_run1 真实分析按 §3 schema 手写（3 findings + 2 proposals，证据 id 全部可点）
  - [ ] 剩余：agent runner（run 结束触发 headless 会话产出 findings）、6 playbook 实现、git 应用步（前置 git init）
- [ ] 时间窗提醒：B 的窗口估 12–24 个月（至官方 agent 产品原生支持目标函数后台迭代）
- [ ] **Hypothesis Generation 产品预留**（决策与路线见 `todo/hypothesis_generation_product.md`）：近期只在 Candidate / Finding / run schema 预留 `hypothesis_id`、`experiment_id`，并在 Research Copilot 落地后制作一条“Finding → Hypothesis → paired run → result backflow”黄金样例；黄金样例前不启动独立前端或多 Agent tournament

## WS-4 多域任务接入（D7 新增：泛化性主线）

> 参照系 = `third_party/HyperAgents/domains/` 六域谱系，按评估形态分三类：
> **确定性单测型**（polyglot）、**LLM-judge 型**（imo / paper_review / search_arena）、
> **episode 环境型**（balrog / genesis）。目标不是复刻这六个域，而是让
> TaskBundle/Grader 契约覆盖这三种评估形态，且新域边际接入成本持续下降。

- [ ] **T1 coding 域**（`todo/coding_task_plan.md` 模式 A，LiveCodeBench 子集）：
  确定性打分、接入成本最低，泛化线第一站；模式 B 与 WS-3 M3 会师
- [ ] **T2 LLM-judge 域**（paper_review / search_arena 形态取其一）：打分本身随机，
  是 `docs/EvoHarness.md` P0.1（评估重采样 + LCB 置信排序）的强制前置消费者——
  P0.1 与 T2 同批做
- [ ] **T3 episode 环境型域**（远期）：评估侧为长时环境 rollout，依赖 WS-2 协议 v2
  工作区端点 + 批式并发
- [ ] **泛化性验收**：第三个域接入时，新域代码量 ≤ 已有域中位数的 1/2，
  evocore 零改动；每域交付 = TaskBundle + grade_fn（本地或 evoserve 远程均可）+
  sys_msg/brief 双文档（modmul 模板）
- [ ] `tasks/__init__.py` 的 `equational` stub 待排期（原停车场旧 #15，D7 出栈）

## 已完成存档（防止重做）

- [x] 远程评估协议两侧全量落地 + 137 测试（evoserve 三件套 / RemoteGrader / e2e 加冕测试，2026-07-10）
- [x] SearchLoop 处理 EvalInfraError：drop + infra_streak 熔断 + 种子放行（tests/test_loop_infra.py）
- [x] 失败分类学三级完备：候选的锅 / 依赖的锅（InfraError）/ 任务作者的 bug
- [x] infra_error 不满足幂等重放；poll 404 自愈重提；双版本冻结
