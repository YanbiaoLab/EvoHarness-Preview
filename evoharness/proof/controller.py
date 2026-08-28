"""The loop: pick a goal, attack it, propagate, decompose when direct proving
fails.

Deliberately the dumbest thing that can work. LEAP itself runs a plain
depth-first search with backtracking over its DAG, and notes that better branch
prioritisation is future work -- and in this project the selection policy is one
of the genes P-6 would evolve. Making it clever now would lock in by hand the
thing the search is supposed to discover.

Four rules live here and nowhere else:

**Direct first, decompose on failure.** LEAP's order. A goal is attacked
directly until that stops working; only then is a decomposition asked for.

**Only a capability failure may trigger decomposition.** An infrastructure
fault, an interruption or an exhausted budget say nothing about the goal, so
they must not send the controller looking for a different route. Letting them
would let a judge outage rewrite the shape of the whole graph.

**Consecutive infrastructure faults stop the run.** Not the goal -- the run.
Retrying forever against a dead judge burns a night and produces one line of
log. `RunSpec.max_consecutive_infra_failures` does the same thing a layer down.

**The ledger is derived, never kept.** Spend comes from `store.total_cost()`,
which sums the attempts table, so a resumed controller starts with the right
number without anything having to be written twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .graph import (
    NO_VERDICT_OUTCOMES,
    CycleError,
    DecompositionStatus,
    Goal,
    GoalStatus,
    Outcome,
)
from .solver import Solver
from .store import ProofGraphStore

#: A goal whose route is already being worked is not one to attack directly.
#: This is a scheduling judgement, not graph semantics, which is why it lives
#: here rather than in `graph.py`.
_ROUTE_IN_PROGRESS = frozenset(
    {DecompositionStatus.PROPOSED, DecompositionStatus.ACCEPTED}
)

#: (identity, statement) pairs.
Subgoals = Sequence[tuple[str, str]]


@dataclass(frozen=True)
class Validation:
    """Whether a proposed decomposition may be trusted to compose back."""

    ok: bool
    reason: str = ""


class DecompositionSource(Protocol):
    """Where a route from a goal to subgoals comes from.

    P-3 hands over a hand-written table. P-4 replaces it with the dsh call that
    retrieves from Mathlib and proposes a sketch. The controller cannot tell
    the difference, which is the point.
    """

    def propose(self, goal: Goal) -> Subgoals | None:
        ...


@dataclass
class FixedDecompositions:
    """A hand-written table, keyed by goal identity. One route per goal."""

    table: Mapping[str, Subgoals]
    _used: set[str] = field(default_factory=set)

    def propose(self, goal: Goal) -> Subgoals | None:
        if goal.identity in self._used:
            return None
        subgoals = self.table.get(goal.identity)
        if not subgoals:
            return None
        self._used.add(goal.identity)
        return subgoals


#: Given a goal and its proposed subgoals, is the sketch sound? P-3 passes an
#: explicit stub; `sketch.py` will pass the real Lean check. Required rather
#: than defaulted, because "nobody validated this" must never be silent: only
#: an ACCEPTED decomposition may complete, and accepting without a check would
#: hand that gate away.
SketchValidator = Callable[[Goal, Subgoals], Validation]


def select_first_open(goals: Sequence[Goal]) -> Goal | None:
    """Oldest first. Replaceable on purpose -- see the module docstring."""

    return goals[0] if goals else None


@dataclass(frozen=True)
class SolveReport:
    root_proved: bool
    stopped_reason: str
    iterations: int
    spent: float
    attempts: int
    decompositions_accepted: int
    decompositions_rejected: int
    cycles_refused: int


class ProofController:
    def __init__(
        self,
        store: ProofGraphStore,
        solver: Solver,
        *,
        decompositions: DecompositionSource,
        validate_sketch: SketchValidator,
        max_capability_attempts: int = 3,
        max_consecutive_infra: int = 5,
        select: Callable[[Sequence[Goal]], Goal | None] = select_first_open,
        lease_ttl_s: float = 600.0,
        owner: str = "controller",
    ):
        self.store = store
        self.solver = solver
        self.decompositions = decompositions
        self.validate_sketch = validate_sketch
        self.max_capability_attempts = max_capability_attempts
        self.max_consecutive_infra = max_consecutive_infra
        self.select = select
        self.lease_ttl_s = lease_ttl_s
        self.owner = owner

    # -- scheduling -----------------------------------------------------------

    def actionable_goals(self) -> list[Goal]:
        """Open goals with no route already in progress.

        A goal whose accepted decomposition is being worked is carried by its
        subgoals; attacking it directly as well would spend twice and, because
        a live decomposition keeps a goal OPEN forever, would never stop.
        """

        actionable = []
        for goal in self.store.open_goals():
            statuses = {
                decomposition.status
                for decomposition in self.store.decompositions_of(goal.id)
            }
            if statuses & _ROUTE_IN_PROGRESS:
                continue
            actionable.append(goal)
        return actionable

    # -- the loop -------------------------------------------------------------

    def solve(
        self,
        root_goal_id: str,
        *,
        budget: float,
        max_iterations: int = 1000,
    ) -> SolveReport:
        iterations = 0
        attempts = 0
        accepted = 0
        rejected = 0
        cycles = 0
        consecutive_infra = 0
        stopped = "max_iterations"

        while iterations < max_iterations:
            if self.store.goal(root_goal_id).status is GoalStatus.PROVED:
                stopped = "proved"
                break

            spent = self.store.total_cost()
            if spent >= budget:
                stopped = "budget"
                break

            goal = self.select(self.actionable_goals())
            if goal is None:
                # Everything is either proved, exhausted, or already carried by
                # a decomposition. Nothing left this controller can do.
                stopped = "no_actionable_goal"
                break
            if not self.store.claim(
                goal.id, self.owner, ttl_s=self.lease_ttl_s
            ):
                # Another worker holds it. Single-threaded today; the path has
                # to exist before it is needed, not after.
                stopped = "goal_leased_elsewhere"
                break

            iterations += 1
            try:
                result = self.solver.attack(goal, budget=budget - spent)
                self.store.record_attempt(
                    goal.id,
                    result.outcome,
                    proof_text=result.proof_text,
                    run_dir=result.run_dir,
                    evidence_ref=result.evidence_ref,
                    cost=result.cost,
                )
                attempts += 1
                self.store.propagate(
                    goal.id,
                    max_capability_attempts=self.max_capability_attempts,
                    budget_spent=self.store.total_cost(),
                    solver_level=self.solver.level,
                )

                if result.outcome in NO_VERDICT_OUTCOMES:
                    # No verdict: do NOT decompose. The goal has said nothing
                    # about itself, and looking for another route on the
                    # strength of a judge outage is how infrastructure gets to
                    # rewrite the graph.
                    if result.outcome is Outcome.INFRA_FAILED:
                        consecutive_infra += 1
                        if consecutive_infra >= self.max_consecutive_infra:
                            stopped = "infra"
                            break
                    if result.outcome is Outcome.BUDGET_EXHAUSTED:
                        stopped = "budget"
                        break
                    continue

                consecutive_infra = 0
                if result.outcome is Outcome.PROVED:
                    continue

                outcome = self._maybe_decompose(goal)
                accepted += outcome[0]
                rejected += outcome[1]
                cycles += outcome[2]
            finally:
                self.store.release(goal.id, self.owner)

        return SolveReport(
            root_proved=(
                self.store.goal(root_goal_id).status is GoalStatus.PROVED
            ),
            stopped_reason=stopped,
            iterations=iterations,
            spent=self.store.total_cost(),
            attempts=attempts,
            decompositions_accepted=accepted,
            decompositions_rejected=rejected,
            cycles_refused=cycles,
        )

    # -- decomposition --------------------------------------------------------

    def _maybe_decompose(self, goal: Goal) -> tuple[int, int, int]:
        """Ask for a route and validate it. Returns (accepted, rejected, cycles)."""

        subgoals = self.decompositions.propose(goal)
        if not subgoals:
            return (0, 0, 0)

        try:
            decomposition = self.store.add_decomposition(goal.id, subgoals)
        except CycleError:
            # The degenerate proposal: a subgoal that restates an ancestor.
            # Refused at the store, so nothing landed and there is no row to
            # mark rejected -- only a count, so it shows up in the report
            # rather than vanishing.
            return (0, 0, 1)

        verdict = self.validate_sketch(goal, subgoals)
        self.store.set_decomposition_status(
            decomposition.id,
            (
                DecompositionStatus.ACCEPTED
                if verdict.ok
                else DecompositionStatus.REJECTED_BY_VERIFIER
            ),
            reason=verdict.reason,
        )
        self.store.propagate(
            goal.id,
            max_capability_attempts=self.max_capability_attempts,
            budget_spent=self.store.total_cost(),
            solver_level=self.solver.level,
        )
        return (1, 0, 0) if verdict.ok else (0, 1, 0)


__all__ = [
    "DecompositionSource",
    "FixedDecompositions",
    "ProofController",
    "SketchValidator",
    "SolveReport",
    "Validation",
    "select_first_open",
]
