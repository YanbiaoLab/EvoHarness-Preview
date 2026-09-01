# EvoHarness original extension interfaces (no upstream counterpart).
# These are the mount points used by evoplus (C1/C2/C3) and tasks/; core
# never imports evoplus.
"""Extension protocols: Grader, PromptContributor, LoopObserver, weights, budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .population import Candidate, IslandView, PopulationStore
    from .workspace import Workspace


@dataclass
class MutationContext:
    """Everything a prompt-time plugin may want to know about the proposal."""

    parent: "Candidate"
    archive_inspirations: list["Candidate"]
    top_k_inspirations: list["Candidate"]
    operator: str
    generation: int
    inspiration_notes: dict[str, str] = field(default_factory=dict)
    #: 已试过而没有涨分的改动:(标题, 相对父本的分差)。
    #:
    #: archive / top_k 两条参考通道都按分数选,于是分数长期持平的 run 里失败
    #: 尝试对提案器不可见,同一类改动会被反复提出,每次付一整轮评测。
    #: 只带标题和分差 —— 目的是不重复,要读代码有 inspect_candidate。
    failed_attempts: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RejectionEvent:
    """A proposal discarded before evaluation (novelty gate, or the proposer
    failed to produce a valid edit). Carried to observers so negative
    experience can be recorded; core itself never interprets it."""
    kind: str
    generation: int
    operator: str
    parent: "Candidate"
    proposal_code: str | None = None
    proposal_workspace: "Workspace | None" = None
    change_title: str = ""
    failure_reason: str = ""
    max_similarity: float = 0.0
    most_similar_id: str | None = None


@runtime_checkable
class Grader(Protocol):
    """Task evaluation contract. Implemented by tasks/, wraps its own sandbox."""

    def grade(self, cand: "Candidate", workdir: Path) -> "EvalReport": ...  # noqa: F821


@runtime_checkable
class PromptContributor(Protocol):
    """Emits one section for the mutation system prompt (None = skip this time)."""

    def contribute(self, ctx: MutationContext) -> str | None: ...


@runtime_checkable
class LoopObserver(Protocol):
    """Called after each candidate is graded, before insertion into the store."""

    def on_candidate_graded(
        self, cand: "Candidate", store: "PopulationStore"
    ) -> None: ...


@runtime_checkable
class RejectionObserver(Protocol):
    """Optional hook: hears about pre-evaluation rejections. Dispatch in
    SearchLoop is duck-typed, so plain LoopObservers are unaffected."""

    def on_proposal_rejected(self, event: RejectionEvent) -> None: ...


@runtime_checkable
class SamplingWeightPolicy(Protocol):
    """Multiplicative weight adjustment for parent selection."""

    def weight_multiplier(self, cand: "Candidate") -> float: ...


@runtime_checkable
class OperatorSelector(Protocol):
    """Optional replacement for the static sample_operator draw. Implemented
    by evoplus (e.g. OperatorBandit); SearchLoop falls back to the fixed
    config probabilities when none is supplied."""

    def sample_operator(self, has_inspirations: bool, rng: object) -> str: ...


@runtime_checkable
class InspirationPolicy(Protocol):
    """Optional: pick one reference program for WHAT IT KNOWS, not where
    it ranks.

    Implemented by evoplus (ComplementaryInspiration). Returning None means
    "no intervention": the selector's fitness-ranked picks stand unchanged.
    A pick replaces one top-k slot -- never grows the prompt -- and carries
    a note telling the model why this program is worth reading.
    """

    def pick(
        self, parent: "Candidate", pool: list["Candidate"]
    ) -> "tuple[Candidate, str] | None": ...

@runtime_checkable
class MergePlanner(Protocol):
    """Optional: proposes a merge of two candidates' carried state.

    Implemented by evoplus (StateMergePlanner). Returning None means "no
    merge this time" and the loop plans an ordinary proposal instead. A merge
    costs no model call: the returned plan is turned into a candidate whose
    genome is the base's, unchanged, carrying `state_donors` for the domain.
    """

    def plan(self, island: "IslandView", rng: object) -> "MergePlanLike | None": ...  # noqa: F821


@runtime_checkable
class MergePlanLike(Protocol):
    """What SearchLoop needs from a merge plan; evoplus owns the rest."""

    @property
    def base(self) -> "Candidate": ...

    def state_donors(self) -> list[dict]: ...

    def title(self) -> str: ...

    def summary(self) -> str: ...


@runtime_checkable
class IslandHealthPolicy(Protocol):
    """Optional: revives islands that have stopped contributing.

    Called once per generation after archive refresh and migration, which is
    the only point at which the generation's results are all visible.
    """

    def maybe_restart(
        self, store: "PopulationStore", generation: int, num_islands: int
    ) -> list: ...


@runtime_checkable
class BudgetLike(Protocol):
    """Minimal budget interface; guard.BudgetMeter satisfies it."""

    def charge(self, usd: float) -> None: ...

    def should_stop(self) -> bool: ...


class NullBudget:
    """No-op budget for tests and unmetered runs."""

    def charge(self, usd: float) -> None:
        pass

    def should_stop(self) -> bool:
        return False
