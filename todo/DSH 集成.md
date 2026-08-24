# DSH 集成

**dsh 驱动会话与提案，Python 驱动搜索与裁决。**

外壳可以换，裁决链不能动：控制流的归属可以谈，证据、覆盖与判定的权力不让。

---

## 1. 执行内核：谁驱动谁

```mermaid
flowchart LR
    USER["用户"] --> DSH["dsh 宿主<br/>UI · 会话 · Job"]

    DSH -->|"启动 / 停止 / Attach<br/>带 shell 身份哈希"| PY["Python 引擎<br/>SearchLoop · 种群 · 预算 · 断路器"]

    PY -->|"反向 propose<br/>Prompt + 候选工作区"| DSH
    DSH -->|"起候选子 Agent"| SA["候选子 Agent<br/>tools.guard() 封死能力"]
    DSH -->|"工作区改动 + Usage + Session ID"| PY

    PY -->|"评测候选"| EVAL["独立评测服务<br/>evoserve"]
    EVAL -->|"EvalReport"| PY

    PY --> STATE[("Run 状态<br/>run.db · checkpoint")]
    PY --> EVIDENCE[("可信证据<br/>EvidenceEnvelope")]

    PY -.->|"进度 / 冠军 / Inbox 通知"| DSH
    SA -.-> TRACE[("会话 Trace<br/>仅供下钻，不作证据")]

    STATE -->|"Attach 后按身份指纹校验恢复"| PY

    classDef human fill:#264766,stroke:#152C42,color:#FFFFFF,stroke-width:2px
    classDef dshDark fill:#426F99,stroke:#274A6B,color:#FFFFFF,stroke-width:2px
    classDef dshLight fill:#B9D3E8,stroke:#6489A8,color:#142B3D
    classDef spec fill:#DDD4EF,stroke:#8875AC,color:#2E2541
    classDef core fill:#70538F,stroke:#44305D,color:#FFFFFF,stroke-width:2px
    classDef evidence fill:#D8EBDD,stroke:#73977C,color:#183B24
    classDef demoted fill:#EFEDE8,stroke:#A8A69E,color:#3A3A36

    class USER human
    class DSH dshDark
    class SA dshLight
    class PY core
    class EVAL,EVIDENCE evidence
    class STATE spec
    class TRACE demoted
```

**读图三条：**

- **箭头方向就是控制流归属。** dsh 只管起停与 Attach，代际循环留在
  `SearchLoop`；提案是 Python 发起的**反向请求**，dsh 是它的服务方。
  好处是 `SearchLoop.run()` 一行不动——种群、断路器、检查点、停机理由
  的权威天然只有一处，不需要跨语言协议转告。
- **候选子 Agent 单独成节点，因为守卫装在这一层。** `tools.guard()` 是
  单调守卫：在 `tools/pre-execute` 瀑布之后、工具体之前执行，返回理由
  即拒绝，监听器顺序无法把拒绝翻回允许。工具可见性过滤（`restrict`）
  只是纵深——它管不着作用域内注册的工具，不能当安全边界。
- **两个圆柱一实一虚。** `EvidenceEnvelope` 是唯一事实层；会话 Trace
  以冷副本身份归档，仅供从候选下钻回原始会话，**任何情况下不作证据**。
  dsh 的会话日志遵守"模型可见即已记录"，它是模型上下文的来源，
  不是评测判定的来源。

`evoserve` 与 Run 状态都画在 dsh 之外：前者是评估信任边界，后者是耐久性
归属。会话关掉，插件和 Job 全没了，检查点还在——Attach 就是对同一个
run 目录重新拉起并按身份指纹校验后续跑。

---

## 2. 研究闭环：判定权归谁

```mermaid
flowchart TD
    H["Human Researcher<br/>目标 · 冻结审批 · 最终拍板"]
    A["DSH Research Copilot<br/>建议，不判定"]

    G["Research Goal"]
    P["Hypothesis Portfolio<br/>可证伪表述"]
    C["Critique / Evidence Review"]
    EP["Experiment Plan"]
    EC["Experiment Compiler"]

    H -->|"提出目标 / 约束"| G
    A -.->|"澄清、扩展、结构化"| G
    G --> P
    A -.->|"生成与分解假设"| P
    P --> C
    A -.->|"批判、查漏"| C
    C --> EP
    A -.->|"设计实验"| EP
    EP --> EC
    A -.->|"协助编译规格"| EC

    EC --> T["TaskSpec<br/>任务语义与评分契约"]
    EC --> R["RunSpec<br/>资源与运行环境"]
    EC --> S["SearchProfile<br/>Basic / Evolution 搜索行为"]

    T --> FZ["ExperimentSpec 冻结<br/>三 hash · 预测 · 停止规则"]
    R --> FZ
    S --> FZ
    H -->|"审批与冻结"| FZ
    FZ -->|"verify_refs 拒跑漂移实验"| CORE

    subgraph DSH["DSH / Cordis Runtime"]
        UI["Session / UI / Jobs"]
        PL["EvoHarness Plugin<br/>提案执行器"]
        SA["DSH Subagents<br/>tools.guard() 封死能力"]
        UI --> PL
        PL --> SA
    end

    subgraph EVO["Python EvoHarness"]
        CORE["SearchLoop / Core"]
        EV["Evaluation<br/>evoserve · 独立进程"]
    end

    PL -->|"spawn / status / stop"| CORE
    CORE -->|"反向 propose"| PL
    SA -->|"candidate changes"| PL
    CORE --> EV

    EV --> O["EvidenceEnvelope<br/>覆盖 · 命名空间 · 成本"]
    O --> AG["AssessmentGuard<br/>版本化判定 · assessor_hash"]
    REF[("ReferenceStore<br/>当前冠军")] -.->|"比较基线"| AG

    AG --> U["Belief Update"]
    AG --> PROM["PromotionPolicy<br/>晋升 / 排名翻转"]
    AG --> Q["Research Inbox<br/>类型化决策卡"]
    AG --> I["Evidence Interpreter<br/>叙述，非 verdict"]

    PROM --> REF
    PROM -.->|"换冠军 → 排名翻转卡"| Q
    U --> P
    A -.->|"归纳、解释、起草 Finding"| I
    I -.->|"卡片草案"| Q

    Q -->|"approve / branch / veto / revise"| H
    H -->|"修订假设或启动新实验"| P

    classDef human fill:#264766,stroke:#152C42,color:#FFFFFF,stroke-width:2px
    classDef dshDark fill:#426F99,stroke:#274A6B,color:#FFFFFF,stroke-width:2px
    classDef dshLight fill:#B9D3E8,stroke:#6489A8,color:#142B3D
    classDef research fill:#DDEBF5,stroke:#7D9DB5,color:#183246
    classDef spec fill:#DDD4EF,stroke:#8875AC,color:#2E2541
    classDef frozen fill:#8875AC,stroke:#5A4A7A,color:#FFFFFF,stroke-width:2px
    classDef core fill:#70538F,stroke:#44305D,color:#FFFFFF,stroke-width:2px
    classDef evidence fill:#D8EBDD,stroke:#73977C,color:#183B24
    classDef verdict fill:#4E7A5A,stroke:#2F4A37,color:#FFFFFF,stroke-width:2px
    classDef review fill:#F2DFC0,stroke:#B28A4B,color:#4B3518

    class H human
    class A,UI,PL dshDark
    class SA,I dshLight
    class G,P,C,EP,EC research
    class T,R,S spec
    class FZ frozen
    class CORE core
    class EV,O,U,PROM,REF evidence
    class AG verdict
    class Q review
```

**读图四条：**

- **copilot 的虚线够不到 `AssessmentGuard`。** 它在目标澄清、假设分解、
  实验设计、规格编译、证据解读上都出力，产物是叙述、Finding 与卡片草案；
  但 supported / contradicted / unknown 三态只能由带 `assessor_hash` 的
  版本化规则产出。`Evidence Interpreter` 因此被涂成 copilot 的浅蓝色，
  而不是证据的绿色——**它是 agent 的产出，不是事实**。
  理由不是防 agent 说谎，是防判定标准悄悄漂移：LLM 每次解读的隐含阈值
  都不一样，而现行 assessor 的噪声下限有具体来历（IMO 终选在 12 题上取
  argmax，validation 到 test 掉 0.206，点估计比较会把幸运种子当成优势）。
- **信念更新与决策卡都挂在判定之后，不与判定并联。** 没有一条路径能让
  信念绕开 `AssessmentGuard` 更新。这也是 Research Layer 那条约束的
  字面意思：自动生成 Finding 和 DecisionRequest，但不绕过 AssessmentGuard
  和 Human Policy。
- **`ExperimentSpec 冻结` 单独占一格。** 三份 Spec 直接连到 SearchLoop 的话，
  "跑的和批的是同一个"没有着落。冻结把 task / run / search 三个 hash、
  预测声明与停止规则一起钉死，`verify_refs()` 靠它拒跑漂移实验——
  人审批的对象是这一格，不是三份 Spec。
- **晋升与信念是两条独立的反馈路径。** `Belief Update` 回到假设组合，
  决定人接下来相信什么、试什么；`PromotionPolicy` 回到 `ReferenceStore`，
  决定下一批候选跟谁比。后者还兼作 `AssessmentGuard` 的比较基线——
  换冠军会改变所有后续比较类判定的参照物，所以换冠军要出排名翻转卡。

---

## 3. 两图的接口

上下两层之间只有一个接口：**上层递进冻结好的 `ExperimentSpec`，下层递出
`EvidenceEnvelope`。**

中间发生了什么，治理层不需要知道；判定用什么规则，执行层无权干预。
这条接口是两张图能各自独立演进的原因——换外壳只动第一张，改判定规则
只动第二张。

> 配对文档：`todo/dsh_integration.md`（接入计划与停止线）、
> `todo/research_layer.md`（治理层契约）、`docs/eval_protocol.md`（评估信任边界）。
