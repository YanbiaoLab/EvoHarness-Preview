"""The loop: pick a goal, attack it, propagate, decompose when direct proving
fails.


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
from .sketch import Sketch, SketchUnavailable, Validation
from .solver import Solver
from .store import ProofGraphStore

#: A goal whose route is already being worked is not one to attack directly.
#: This is a scheduling judgement, not graph semantics, which is why it lives
#: here rather than in `graph.py`.
_ROUTE_IN_PROGRESS = frozenset(
    {DecompositionStatus.PROPOSED, DecompositionStatus.ACCEPTED}
)

class DecompositionSource(Protocol):
    """Where a route from a goal to subgoals comes from.

    P-3 hands over a hand-written table. P-4 replaces it with the dsh call that
    retrieves from Mathlib and proposes a sketch. The controller cannot tell
    the difference, which is the point.
    """

    def propose(self, goal: Goal) -> Sketch | None:
        ...


@dataclass
class FixedDecompositions:
    """A handwritten table, keyed by goal identity. One route per goal.

    Each goal is offered its route once. Offering it again after a rejection
    would loop: the source has nothing new to say, and the validator would
    reach the same verdict every time.
    """

    table: Mapping[str, Sketch]
    _used: set[str] = field(default_factory=set)

    def propose(self, goal: Goal) -> Sketch | None:
        if goal.identity in self._used:
            return None
        sketch = self.table.get(goal.identity)
        if sketch is None:
            return None
        self._used.add(goal.identity)
        return sketch


#: Given a goal and its proposed subgoals, is the sketch sound? P-3 passes an
#: explicit stub; `sketch.py` will pass the real Lean check. Required rather
#: than defaulted, because "nobody validated this" must never be silent: only
#: an ACCEPTED decomposition may complete, and accepting without a check would
#: hand that gate away.
SketchValidator = Callable[[Goal, Sketch], Validation]


def select_first_open(goals: Sequence[Goal]) -> Goal | None:
    """Oldest first. Replaceable on purpose -- see the module docstring."""

    return goals[0] if goals else None


@dataclass(frozen=True)
class DecompositionOutcome:
    """What happened to one proposed route.

    `deferred` is its own answer rather than a flavour of rejection. A sketch
    check that could not RUN has said nothing about the sketch, and reporting
    that as "rejected" would let one Lean timeout look identical to a genuinely
    unsound decomposition.
    """

    accepted: int = 0
    rejected: int = 0
    cycles: int = 0
    deferred: int = 0


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
        decompose_root_first: bool = False,
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
        # LEAP's order is direct-first, and that is the default. Inverting it
        # for the ROOT only is the "with graph" arm P-5 needs: comparing
        # decomposition against direct proving requires being able to ask for
        # decomposition even where direct proving would have worked. Scoped to
        # the root because applying it everywhere would decompose leaves that
        # have nothing left to split.
        self.decompose_root_first = decompose_root_first

    # -- scheduling -----------------------------------------------------------

    def actionable_goals(self, scope: set[str] | None = None) -> list[Goal]:
        """Open goals with no route already in progress, within `scope`.

        `scope` is the set reachable from the root being solved. Omitting it
        means the whole workspace, which is right for a reader and wrong for a
        solver: one workspace is meant to hold several problems, and "attack
        this goal" must not charge this goal's budget for another one.

        A goal whose accepted decomposition is being worked is carried by its
        subgoals; attacking it directly as well would spend twice and, because
        a live decomposition keeps a goal OPEN forever, would never stop.
        """

        actionable = []
        for goal in self.store.open_goals():
            if scope is not None and goal.id not in scope:
                continue
            statuses = {
                decomposition.status
                for decomposition in self.store.decompositions_of(goal.id)
            }
            if statuses & _ROUTE_IN_PROGRESS:
                continue
            actionable.append(goal)
        return actionable

    def recover(self, now: float | None = None) -> list[str]:
        """Turn abandoned leases into recorded interruptions.

        Called before the loop. A worker that died mid-attempt recorded
        nothing, so the goal looks untouched apart from a lease nobody will
        ever release. Writing an INTERRUPTED attempt makes the gap visible and
        frees the goal; because INTERRUPTED carries no verdict it cannot push
        the goal toward exhaustion, which is the whole point -- a crash is not
        evidence that a lemma is hard.

        It also rechecks decompositions still sitting at PROPOSED. One of those
        is a sketch whose check could not RUN -- a Lean timeout, a toolchain
        that was down -- and leaving it alone was a wedge rather than a
        kindness: PROPOSED counts as a live route, so the parent stops being
        attacked directly, while nothing was ever scheduled to reach a verdict
        on the route itself. The goal shut for good, quietly, off one timeout.

        Returns the goal ids recovered, so a resumed run can say how much it
        inherited rather than presenting itself as a fresh start.
        """

        recovered: list[str] = []
        self.revalidate_proposed()
        for goal in self.store.stale_leases(now):
            self.store.record_attempt(
                goal.id,
                Outcome.INTERRUPTED,
                note=f"lease held by {goal.lease_owner} expired without release",
            )
            self.store.force_release(goal.id)
            self.store.propagate(
                goal.id,
                max_capability_attempts=self.max_capability_attempts,
                budget_spent=self.store.total_cost(),
                solver_level=self.solver.level,
            )
            recovered.append(goal.id)
        return recovered

    def revalidate_proposed(self) -> DecompositionOutcome:
        """Reach a verdict on every decomposition still awaiting one.

        Cheap to retry and unbounded to skip: a sketch check is one compile,
        while an unresolved PROPOSED row costs the parent every future attempt.
        A check that fails again simply stays PROPOSED for the next round --
        that is the loop the deferred case always needed and never had.
        """

        totals = DecompositionOutcome()
        for decomposition in self.store.proposed_decompositions():
            sketch = self.store.sketch_of(decomposition.id)
            if sketch is None:
                # Nothing to recheck it against. Not a verdict either, but it
                # can never become one, so say so rather than retrying forever.
                self.store.set_decomposition_status(
                    decomposition.id,
                    DecompositionStatus.REJECTED_BY_VERIFIER,
                    reason="the decomposition was recorded without a sketch",
                )
                totals = DecompositionOutcome(
                    totals.accepted, totals.rejected + 1, totals.cycles,
                    totals.deferred,
                )
                continue
            goal = self.store.goal(decomposition.goal_id)
            try:
                verdict = self.validate_sketch(goal, sketch)
            except SketchUnavailable:
                totals = DecompositionOutcome(
                    totals.accepted, totals.rejected, totals.cycles,
                    totals.deferred + 1,
                )
                continue
            self.store.set_decomposition_status(
                decomposition.id,
                (
                    DecompositionStatus.ACCEPTED if verdict.ok
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
            totals = (
                DecompositionOutcome(
                    totals.accepted + 1, totals.rejected, totals.cycles,
                    totals.deferred,
                ) if verdict.ok else
                DecompositionOutcome(
                    totals.accepted, totals.rejected + 1, totals.cycles,
                    totals.deferred,
                )
            )
        return totals

    # -- the loop -------------------------------------------------------------

    def solve(
        self,
        root_goal_id: str,
        *,
        budget: float,
        max_iterations: int = 1000,
    ) -> SolveReport:
        self.recover()
        scope = self.store.reachable_from(root_goal_id)
        iterations = 0
        attempts = 0
        accepted = 0
        rejected = 0
        cycles = 0
        deferred = 0
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

            candidates = self.actionable_goals(scope)
            if not candidates:
                # Everything is either proved, exhausted, or already carried by
                # a decomposition. Nothing left this controller can do.
                stopped = "no_actionable_goal"
                break

            # Skip past goals somebody else holds rather than stopping. One
            # busy node must not halt the run: with several workers that would
            # make throughput depend on which goal happened to be picked first,
            # and after a crash it would block every resume until the dead
            # worker's lease expired.
            goal = None
            while candidates:
                choice = self.select(candidates)
                if choice is None:
                    break
                if self.store.claim(
                    choice.id, self.owner, ttl_s=self.lease_ttl_s
                ):
                    goal = choice
                    break
                candidates = [c for c in candidates if c.id != choice.id]
            if goal is None:
                stopped = "goal_leased_elsewhere"
                break

            iterations += 1
            try:
                if (
                    self.decompose_root_first
                    and goal.id == root_goal_id
                    and not self.store.decompositions_of(goal.id)
                ):
                    counted = self._maybe_decompose(goal)
                    scope = self.store.reachable_from(root_goal_id)
                    accepted += counted.accepted
                    rejected += counted.rejected
                    cycles += counted.cycles
                    deferred += counted.deferred
                    if counted.accepted:
                        continue

                result = self.solver.attack(goal, budget=budget - spent)
                self.store.record_attempt(
                    goal.id,
                    result.outcome,
                    proof_text=result.proof_text,
                    run_dir=result.run_dir,
                    evidence_ref=result.evidence_ref,
                    cost=result.cost,
                    note=result.note,
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
                # A decomposition that just landed added subgoals; they belong
                # to this root's scope or the loop would never attack them.
                scope = self.store.reachable_from(root_goal_id)
                accepted += outcome.accepted
                rejected += outcome.rejected
                cycles += outcome.cycles
                deferred += outcome.deferred
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

    def _maybe_decompose(self, goal: Goal) -> DecompositionOutcome:
        """Ask for a route and validate it. Returns (accepted, rejected, cycles)."""

        sketch = self.decompositions.propose(goal)
        if sketch is None:
            return DecompositionOutcome()
        return self.record_decomposition(goal, sketch)

    def record_decomposition(
        self, goal: Goal, sketch: Sketch
    ) -> DecompositionOutcome:
        """Validate a sketch someone else built, and record the verdict.

        Split out from `_maybe_decompose` so a caller who already has a sketch
        -- a person in a conversation, say -- goes through exactly the same
        gates as the autonomous loop. A second path that recorded
        decompositions without validating them would be a way to put an
        unchecked implication into the graph by hand.
        """

        subgoals = [(spec.identity, spec.signature) for spec in sketch.subgoals]
        try:
            decomposition = self.store.add_decomposition(
                goal.id, subgoals, sketch=sketch
            )
        except CycleError:
            # The degenerate proposal: a subgoal that restates an ancestor.
            # Refused at the store, so nothing landed and there is no row to
            # mark rejected -- only a count, so it shows up in the report
            # rather than vanishing.
            return DecompositionOutcome(cycles=1)

        try:
            verdict = self.validate_sketch(goal, sketch)
        except SketchUnavailable:
            # Not a verdict on the sketch. Marking it rejected-by-verifier is
            # terminal and never revisited, so a dead toolchain would delete a
            # viable route from the graph for good. It stays PROPOSED, and
            # `recover()` is what comes back to it -- leaving it here with
            # nothing scheduled to recheck it is what wedged the parent shut.
            return DecompositionOutcome(deferred=1)

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
        return (
            DecompositionOutcome(accepted=1) if verdict.ok
            else DecompositionOutcome(rejected=1)
        )


__all__ = [
    "DecompositionSource",
    "FixedDecompositions",
    "ProofController",
    "SketchValidator",
    "SolveReport",
    "Validation",
    "select_first_open",
]
