# Research Layer Asset Inventory

> 状态：I-0 基线
> 日期：2026-08-12
> 范围：跨任务通用基础设施，不包含领域事故夹具。

## 能力矩阵

| 能力 | 状态 | 当前实现 | I-1～I-3 缺口 |
|---|---|---|---|
| Candidate 身份 | 已实现 | `Candidate.id` | 关联 experiment_id 和 hypothesis_id |
| Workspace 快照 | 已实现 | FileWorkspace / GitWorkspace | 纳入 TaskSpec 指纹 |
| Parent lineage | 已实现 | parent、ancestor、inspiration metadata | 关联研究对象 |
| Population / archive | 已实现 | PopulationStore | 不应承担 Research Store 职责 |
| Checkpoint / resume | 已实现 | checkpoint.json、run.db | 验证 ExperimentSpec hash |
| Config fingerprint | 已实现 | Task/Run/Search 独立 hash + checkpoint fingerprint | I-3 再加入 ExperimentSpec hash |
| Run manifest | 已实现 | manifest.json | 写入实验和评测协议身份 |
| Budget / concurrency | 已实现 | BudgetMeter、RunSpec | I-3 关联实验预算审批 |
| Preflight | 已实现 | SearchLoop 最终准入 + Agent Session 修复反馈 | I-2 将结果写入 Evidence provenance |
| EvalReport | 已实现但语义过载 | fitness、passed、fault | 拆分 EvidenceEnvelope |
| Infrastructure fault | 部分实现 | EvalInfraError | 建立统一 FaultKind |
| Coverage | 部分实现 | n_units | 缺少 planned/observed units |
| Score namespace | 缺失 | — | I-2 实现 |
| TaskSpec | 已实现 | `evoharness/contracts/task.py` | I-3 纳入 ExperimentSpec |
| RunSpec | 已实现 | `evoharness/contracts/run.py` | I-3 纳入 ExperimentSpec |
| SearchProfile | 已实现 | `evoharness/contracts/search.py` | 后续细化资源分配策略类型 |
| EvidenceEnvelope | 缺失 | — | I-2 实现 |
| ExperimentSpec | 缺失 | — | I-3 实现 |
| ExperimentOutcome | 缺失 | Manifest 和 RunReport 只承担部分职责 | I-3 实现 |
| ResearchDecision | 部分实现 | evoweb proposal decision | 升级为追加式类型化对象 |
| Hypothesis Portfolio | 缺失 | — | I-3 以后实现 |

## 分层测试边界

### Core

Core 的通用测试负责：

- 身份与指纹稳定性；
- 预算和停止原因；
- checkpoint/resume；
- Candidate lineage；
- proposer、preflight、grader 的统一生命周期；
- infrastructure fault 不污染 Population。

### Research Layer

Research Layer 的通用测试负责：

- ExperimentSpec 冻结；
- Outcome 和 Decision 追加写入；
- Evidence 的 coverage 和 provenance；
- Claim 与 Evidence capability 的匹配；
- score namespace 隔离；
-从 Candidate 反向追溯 Experiment 和 Hypothesis。

### Domain Layer

各 Domain 自行测试：

- 候选语义是否合法；
- 领域专有 Preflight；
- 领域 Grader；
- SuccessPolicy；
- 证明完整性、禁用公理、benchmark 泄漏等领域规则。

## I-0 通用不变量

1. `missing != task_failure`
2. `infra_error` 不形成对研究假设的反对证据
3. partial coverage 不得冒充完整总体
4. resume 不得改变冻结实验身份
5. 不同 score namespace 的分数不得直接比较
6. Candidate 或 Agent 自述不能建立 objective_met
7. Core 不得依赖 Research 或具体 Domain
8. Research Layer 不得修改 Candidate workspace
