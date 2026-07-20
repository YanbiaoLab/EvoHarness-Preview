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


@dataclass
class MutationContext:
    """Everything a prompt-time plugin may want to know about the proposal."""

    parent: "Candidate"
    archive_inspirations: list["Candidate"]
    top_k_inspirations: list["Candidate"]
    operator: str
    generation: int


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
