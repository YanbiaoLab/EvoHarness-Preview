from dataclasses import asdict, dataclass

from evoharness.contracts import spec_hash

_DECISION_ACTIONS = {
    "approve",
    "veto",
    "branch",
    "revise",
    "request-more-evidence",
    "approve-protocol-change",
}


@dataclass(frozen=True)
class ResearchGoal:
    goal_id: str
    statement: str
    created_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.goal_id.strip() or not self.statement.strip():
            raise ValueError("goal_id and statement must be non-empty")

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_at")
        return spec_hash({
            "kind": "research_goal",
            **payload
        })


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    goal_id: str
    statement: str          # 可证伪表述
    rationale: str = ""
    created_at: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "goal_id", "statement"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_at")
        return spec_hash({"kind": "hypothesis", **payload})


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis_id: str
    goal_id: str
    task_ref: str
    run_ref: str
    search_ref: str
    intervention: str
    control: str = ""
    prediction_claims: tuple[str, ...] = ()
    falsification_conditions: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    stopping_rule: str = ""
    estimated_cost_usd: float| None = None
    approved_by: str | None = ""
    created_at: float = 0.0

    def __post_init__(self):
        for name in (
            "experiment_id", "hypothesis_id", "goal_id",
            "task_ref", "run_ref", "search_ref", "intervention",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"Experiment {name} must not be empty")

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload.pop("created_at")
        payload.pop("approved_by")
        return spec_hash({
            "kind": "experiment",
            **payload
        })

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "spec_hash": self.hash,
            **asdict(self)
        }

    @classmethod
    def from_json(cls, d: dict) -> "ExperimentSpec":
        if d.get("schema_version") != 1:
            raise ValueError("Experiment schema version must be 1")

        payload = {
            k: v for k, v in d.items()
            if k not in ("schema_version", "spec_hash")
        }

        for name in (
            "prediction_claims", "falsification_conditions", "confounders",
        ):
            payload[name] = tuple(payload.get(name, ()))

        spec = cls(**payload)

        recorded = d.get("spec_hash")
        if recorded and recorded != spec.hash:
            raise ValueError(
                f"experiment {spec.experiment_id} content does not match "
                f"its recorded spec_hash ({recorded} != {spec.hash})"
            )
        return spec




@dataclass(frozen=True)
class ExperimentOutcome:

    experiment_id: str
    spec_hash: str          # 执行时的 ExperimentSpec.hash
    run_id: str             # run 目录名
    stopped_reason: str
    generations_planned: int
    generations_completed: int
    evaluations: int
    infra_drops: int
    evidence_count: int
    best_fitness: float | None
    eval_cost_usd: float
    created_at: float

    def to_json(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_json(cls, d: dict) -> "ExperimentOutcome":
        payload = {k: v for k, v in d.items() if k != "schema_version"}
        return cls(**payload)


@dataclass(frozen=True)
class ResearchDecision:
    experiment_id: str
    action: str
    reason: str
    actor: str
    created_at: float
    request_id: str = ""    # 由 Inbox 回答产生时,指回 DecisionRequest
    #: 签字经过的界面。没有这个字段,事后审计分不清"人在终端上敲的"和
    #: "经会话签的",而这两者的可信度不同——会话签字若不是人的点击本身
    #: 构成事件,它就只是模型对一句话的解读。缺省 unknown 是为了老记录
    #: 读得出来:**不知道来源**与**来源是终端**必须可区分。
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.action not in _DECISION_ACTIONS:
            raise ValueError(
                f"unknown decision action {self.action!r}; "
                f"expected one of {sorted(_DECISION_ACTIONS)}"
            )
        for name in ("experiment_id", "reason", "actor"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    def to_json(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_json(cls, data: dict) -> "ResearchDecision":
        if data.get("schema_version") != 1:
            raise ValueError("ResearchDecision schema_version must be 1")
        payload = {
            key: value for key, value in data.items() if key != "schema_version"
        }
        return cls(**payload)



@dataclass(frozen=True)
class ProtocolChangeProposal:
    """A proposed change to the research protocol, for review and approval."""

    proposal_id: str
    experiment_id: str
    target: str
    description: str
    created_at: float

    def __post_init__(self) -> None:
        if self.target not in {
            "criterion", "measurement", "evaluator", "held-out"
        }:
            raise ValueError(f"invalid proposal target: {self.target!r}")

        for name in ("proposal_id", "experiment_id", "description"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    def to_json(self) -> dict:
        return {"schema_version": 1, **asdict(self)}


DECISION_ACTIONS = frozenset(_DECISION_ACTIONS)
