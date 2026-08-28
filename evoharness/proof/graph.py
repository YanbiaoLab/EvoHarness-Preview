"""Goal / Decomposition / Attempt: the three objects and their state machines.

An AND-OR DAG needs three kinds of thing, and squashing any two of them loses
something the search depends on:

    Goal            an OR node -- any one route that closes it is enough
      +- Decomposition   an AND node -- every subgoal must close
      |    +- Subgoal (a Goal again)
      +- Attempt      one run of a solver against this goal, directly

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
    # 两种否决的认识论地位不同,压成一个 `rejected` 会让这个区别消失。
    # 验证器说草图不合法是事实,不该重访;审稿人说这个分解没用是启发式
    # 判断,可能错,换预算或换审稿人之后必须允许重来。
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
    #: 预算相对:在什么预算、什么求解器档次下判定穷尽的。不记这两样,加预算
    #: resume 之后没有依据重开它,图会永远绕着这个节点走。
    exhausted_at_budget: float | None = None
    exhausted_at_solver: str | None = None
    #: 调度状态,不是语义状态。混进 status 之后半年就会长出
    #: `in-progress-retrying` 和 `stale-in-progress`。
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
        # 可重访,但重访是显式动作(把它设回 PROPOSED),不是靠子目标状态
        # 自己漂回来。
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
