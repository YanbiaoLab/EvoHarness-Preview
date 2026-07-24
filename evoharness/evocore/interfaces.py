# EvoHarness original extension interfaces (no upstream counterpart).
# These are the mount points used by evoplus (C1/C2/C3) and tasks/; evocore
# never imports evoplus.
"""Extension protocols: Grader, PromptContributor, LoopObserver, weights, budget."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .population import Candidate, PopulationStore
    from .workspace import Workspace


@dataclass
class MutationContext:
    """Everything a prompt-time plugin may want to know about the proposal."""

    parent: "Candidate"
    archive_inspirations: list["Candidate"]
    top_k_inspirations: list["Candidate"]
    operator: str
    generation: int


@dataclass(frozen=True)
class RejectionEvent:
    """A proposal discarded before evaluation (novelty gate, or the proposer
    failed to produce a valid edit). Carried to observers so negative
    experience can be recorded; evocore itself never interprets it."""
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
class BudgetLike(Protocol):
    """Minimal budget interface; evoguard.BudgetMeter satisfies it."""

    def charge(self, usd: float) -> None: ...

    def should_stop(self) -> bool: ...


class NullBudget:
    """No-op budget for tests and unmetered runs."""

    def charge(self, usd: float) -> None:
        pass

    def should_stop(self) -> bool:
        return False
