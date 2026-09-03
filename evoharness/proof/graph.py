"""Goal / Decomposition / Attempt: the three objects and their state machines.

An AND-OR DAG needs three kinds of thing, and squashing any two of them loses
something the search depends on:

    Goal            an OR node -- any one route that closes it is enough
      +- Decomposition   an AND node -- every subgoal must close
      |    +- Subgoal (a Goal again)
      +- Attempt      one run of a solver against this goal, directly
      +- Certification   one compile of the ASSEMBLED proof of this goal

The last is what separates two things the status machine cannot: a goal
whose route closed, and a goal whose finished proof was compiled as one
file. PROVED is derived from the first; only a certification records the
second.

Nothing here touches storage or Lean. It is the semantics alone, so the rules
that decide whether a goal is proved can be tested without a database, without
a model, and without a compiler.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


class GraphError(RuntimeError):
    """A write the graph refuses to make."""


class CycleError(GraphError):
    """A decomposition whose subgoal is one of its own ancestors."""


class GoalStatus(str, Enum):
    OPEN = "open"
    PROVED = "proved"
    EXHAUSTED = "exhausted"


class DecompositionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    # Rejection by a verifier is definitive (the sketch failed formal checking)
    # and should not be revisited. Rejection by a reviewer is a heuristic judgement
    # that may be reconsidered under different budgets or review criteria.
    REJECTED_BY_VERIFIER = "rejected-by-verifier"
    REJECTED_BY_REVIEWER = "rejected-by-reviewer"
    COMPLETED = "completed"


class Outcome(str, Enum):
    """How one attempt ended. Derived from the run, never written twice."""

    PROVED = "proved"
    TASK_FAILED = "task-failed"
    BUDGET_EXHAUSTED = "budget-exhausted"
    TIMEOUT = "timeout"
    INFRA_FAILED = "infra-failed"
    INTERRUPTED = "interrupted"


#: The only outcomes that say anything about whether this goal is hard.
#: Everything else says something about the run, and counting it toward
#: exhaustion is how "the judge was down" becomes "the model cannot do it".
CAPABILITY_OUTCOMES = frozenset({Outcome.TASK_FAILED, Outcome.TIMEOUT})

#: No verdict on the goal. Must never push a Goal toward EXHAUSTED, and must
#: never trigger a search for a different decomposition: doing either lets an
#: infrastructure fault rewrite the shape of the whole graph.
NO_VERDICT_OUTCOMES = frozenset(
    {Outcome.INFRA_FAILED, Outcome.INTERRUPTED, Outcome.BUDGET_EXHAUSTED}
)

#: The one outcome a later run may pick up where this one stopped, because the
#: run directory it left behind carries a checkpoint. `budget-exhausted` is
#: resumable only in the weaker sense of "try again with more money".
RESUMABLE_OUTCOMES = frozenset({Outcome.INTERRUPTED})

#: Settled for good. `rejected-by-reviewer` is deliberately absent: a heuristic
#: judgement has to stay revisitable.
_TERMINAL_DECOMPOSITIONS = frozenset(
    {
        DecompositionStatus.REJECTED_BY_VERIFIER,
        DecompositionStatus.COMPLETED,
    }
)

#: Still worth spending on: either awaiting validation or actively being worked.
_LIVE_DECOMPOSITIONS = frozenset(
    {DecompositionStatus.PROPOSED, DecompositionStatus.ACCEPTED}
)


@dataclass(frozen=True)
class Goal:
    """One proof obligation. An OR node: any single route closes it."""

    id: str
    #: The memoization key, and the same key the acyclicity check runs on.
    #: Two goals with one identity are one node; see `identity.py` for why the
    #: hash must fail toward "not the same".
    identity: str
    #: The Lean statement, as source text.
    statement: str
    status: GoalStatus = GoalStatus.OPEN
    #: Context for exhaustion: budget spent and solver tier when marked EXHAUSTED.
    #: Retaining this context allows a subsequent run with more budget to reopen the goal.
    exhausted_at_budget: float | None = None
    exhausted_at_solver: str | None = None
    #: Operational lease metadata for distributed scheduling, separated from
    #: the goal's semantic proof status.
    lease_owner: str | None = None
    lease_expires_at: float | None = None


@dataclass(frozen=True)
class Decomposition:
    """One proposed route from a goal to a set of subgoals. An AND node."""

    id: str
    goal_id: str
    subgoal_ids: tuple[str, ...] = ()
    status: DecompositionStatus = DecompositionStatus.PROPOSED
    rejected_reason: str = ""


@dataclass(frozen=True)
class Attempt:
    """One solver run against one goal. Mirrors exactly one `api.run()`."""

    id: str
    goal_id: str
    outcome: Outcome
    #: Absent for every outcome but PROVED -- a run that was dropped for an
    #: infrastructure fault produced no proof to carry.
    proof_text: str | None = None
    run_dir: str | None = None
    evidence_ref: str | None = None
    cost: float = 0.0
    #: The solver's own account of how this ended. Never parsed, but it is the
    #: only place an infra diagnosis survives -- without it, auditing a run
    #: that died on infrastructure shows six identical `infra-failed` rows and
    #: no way to tell what broke.
    note: str = ""
    created_at: float | None = None


@dataclass(frozen=True)
class Certification:
    """One compile of a goal's assembled proof, as one file.

    Recorded whether or not it passed. A failed certification is a finding
    about the graph -- it believed something the compiler does not -- and a
    record that kept only successes would show such a goal as merely
    "not yet certified".
    """

    id: str
    goal_id: str
    ok: bool
    #: Every axiom Lean reported the finished proof depends on. Kept even on
    #: failure, when a forbidden axiom is what failed it.
    axioms: frozenset[str] = frozenset()
    reason: str = ""
    #: Identifies the exact file that was compiled, without storing it: the
    #: text is reproducible from the graph -- given the route below -- and what
    #: an audit needs is to know whether the graph has changed since.
    text_sha256: str = ""
    #: Which decomposition the assembled file was built through. Three values,
    #: all different: a route id; `""` for a goal the solver closed directly,
    #: where there was no route to choose; and `None` for a record written
    #: before this was kept, where the rule was "the oldest completed route"
    #: and the file therefore cannot be reproduced from the id alone.
    decomposition_id: str | None = None
    created_at: float | None = None


def check_acyclic(
    ancestor_identities: Iterable[str],
    subgoal_identities: Iterable[str],
) -> None:
    """Refuse a decomposition that proposes one of its own ancestors.

    `ancestor_identities` must be the TRANSITIVE ancestor set, not just the
    parent: the degenerate case this exists to catch unfolds a definition into
    an intermediate lemma and folds it straight back, so the offending subgoal
    matches the grandparent rather than the parent.

    The check runs on identities, never on node ids. Minting a fresh id per
    proposal makes an id-based check vacuous -- every restatement is a new
    node, so nothing ever forms a cycle and the search walks in circles
    generating them. Identity is what gives acyclicity teeth.
    """

    ancestors = set(ancestor_identities)
    repeated = sorted(ancestors.intersection(subgoal_identities))
    if repeated:
        raise CycleError(
            "decomposition proposes an ancestor as a subgoal: "
            + ", ".join(repeated)
        )


def goal_status_from(
    *,
    attempt_outcomes: Sequence[Outcome],
    decomposition_statuses: Sequence[DecompositionStatus],
    max_capability_attempts: int,
) -> GoalStatus:
    """What this goal's status should be, given everything hanging off it.

    Pure, so the propagation rules can be tested without a store. Note what
    the caller still owes: when this returns EXHAUSTED, the caller must also
    record `exhausted_at_budget` and `exhausted_at_solver`. Without them a
    later run with more budget has no basis to reopen the goal.
    """

    if any(outcome is Outcome.PROVED for outcome in attempt_outcomes):
        return GoalStatus.PROVED
    if any(
        status is DecompositionStatus.COMPLETED
        for status in decomposition_statuses
    ):
        return GoalStatus.PROVED
    if any(status in _LIVE_DECOMPOSITIONS for status in decomposition_statuses):
        # A route is still open, whatever the direct attempts did.
        return GoalStatus.OPEN
    tried = sum(
        1 for outcome in attempt_outcomes if outcome in CAPABILITY_OUTCOMES
    )
    if max_capability_attempts > 0 and tried >= max_capability_attempts:
        return GoalStatus.EXHAUSTED
    return GoalStatus.OPEN


def decomposition_status_from(
    *,
    current: DecompositionStatus,
    subgoal_statuses: Sequence[GoalStatus],
) -> DecompositionStatus:
    """Advance a decomposition, and refuse the one shortcut that matters.

    Only an ACCEPTED decomposition may complete. ACCEPTED means the sketch
    compiled with `sorry` confined to the newly proposed lemmas, which is what
    makes "all subgoals proved" imply "the parent is proved". Letting a merely
    PROPOSED decomposition complete would let an unvalidated sketch mark its
    parent proved on the strength of subgoals that may not compose back into
    it -- the assembly-drift hazard, arriving through the state machine.
    """

    if current in _TERMINAL_DECOMPOSITIONS:
        return current
    if current is DecompositionStatus.REJECTED_BY_REVIEWER:
        return current
    if current is not DecompositionStatus.ACCEPTED:
        return current
    if subgoal_statuses and all(
        status is GoalStatus.PROVED for status in subgoal_statuses
    ):
        return DecompositionStatus.COMPLETED
    return current


__all__ = [
    "CAPABILITY_OUTCOMES",
    "NO_VERDICT_OUTCOMES",
    "RESUMABLE_OUTCOMES",
    "Attempt",
    "CycleError",
    "Decomposition",
    "DecompositionStatus",
    "Goal",
    "GoalStatus",
    "GraphError",
    "Outcome",
    "check_acyclic",
    "decomposition_status_from",
    "goal_status_from",
]
