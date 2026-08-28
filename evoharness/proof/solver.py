"""The seam between the graph and everything that costs money.

The controller needs exactly one thing from the outside world: "attack this
goal, tell me how it went". Narrowing that to a single method is what lets the
same controller run against a stub in P-3, one `api.run()` in P-4, and whichever
rung of P-2's ladder turns out to be worth its cost -- **which is the only
reason that ladder is an experiment rather than a rewrite.**

Two rules live here rather than in the controller.

`AttemptResult` is the one place an outcome is derived. The plan is explicit
that the graph controller must never write its own opinion of whether a run
succeeded: two sources of truth about that would drift, and nothing would say
which to believe.

A PROVED result must carry proof text. Enforced at construction, because the
alternative is a goal marked proved with nothing to assemble -- a contradiction
that would otherwise surface at final re-verification, a long way from whatever
caused it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .graph import Goal, Outcome

#: `stopped_reason` values that mean the run itself broke. No verdict on the
#: goal: `evaluation/faults.py` calls these NO_VERDICT, and counting them as
#: capability failures is how "the judge was down" becomes "the model cannot
#: do it".
_INFRA_REASONS = frozenset({"eval_infra", "proposer_dead"})

#: Ran out of money rather than out of ideas. Kept apart from TASK_FAILED
#: because more budget may simply finish the job, while a task failure is the
#: only signal that says anything about difficulty.
_BUDGET_REASONS = frozenset({"budget", "llm_billing"})


class SolverError(RuntimeError):
    """The solver could not be driven at all -- not a verdict on the goal."""


@dataclass(frozen=True)
class AttemptResult:
    """How one solver run against one goal ended."""

    outcome: Outcome
    proof_text: str | None = None
    run_dir: str | None = None
    evidence_ref: str | None = None
    cost: float = 0.0
    #: Free-form, for the audit trail. Never parsed.
    note: str = ""

    def __post_init__(self) -> None:
        if self.outcome is Outcome.PROVED and not self.proof_text:
            raise ValueError(
                "a PROVED attempt must carry its proof text; without it the "
                "graph holds a proved goal with nothing to assemble"
            )
        if self.outcome is not Outcome.PROVED and self.proof_text:
            raise ValueError(
                f"a {self.outcome.value} attempt cannot carry proof text"
            )

    @classmethod
    def from_run_report(
        cls,
        report,
        *,
        run_dir: str | None = None,
        solved_at: float = 1.0,
        proof_text: str | None = None,
        evidence_ref: str | None = None,
    ) -> "AttemptResult":
        """Derive the outcome from a finished `api.run()`.

        Order matters. Whether the goal was PROVED is asked first, because
        reaching `solved_at` means Lean accepted a proof, and that stays true
        however the run later fell over. Asking "how did it stop" first would
        throw away a proof already in hand because the judge went down
        afterwards.

        INTERRUPTED is deliberately absent: a run that returns a report was not
        interrupted. See `interrupted()` for the case where it never returned.
        """

        reason = getattr(report, "stopped_reason", "")
        fitness = getattr(report, "best_fitness", None)
        cost = float(getattr(report, "total_llm_cost", 0.0) or 0.0)

        if fitness is not None and fitness >= solved_at:
            if not proof_text:
                raise SolverError(
                    "run reached the solved threshold but no proof text was "
                    "extracted from its best candidate"
                )
            return cls(
                outcome=Outcome.PROVED,
                proof_text=proof_text,
                run_dir=run_dir,
                evidence_ref=evidence_ref,
                cost=cost,
                note=f"stopped_reason={reason}",
            )
        if reason in _INFRA_REASONS:
            return cls(Outcome.INFRA_FAILED, run_dir=run_dir, cost=cost,
                       note=f"stopped_reason={reason}")
        if reason in _BUDGET_REASONS:
            return cls(Outcome.BUDGET_EXHAUSTED, run_dir=run_dir, cost=cost,
                       note=f"stopped_reason={reason}")
        return cls(Outcome.TASK_FAILED, run_dir=run_dir, cost=cost,
                   note=f"stopped_reason={reason} best_fitness={fitness}")

    @classmethod
    def interrupted(cls, run_dir: str | None = None) -> "AttemptResult":
        """The run never returned. The one outcome a later run may resume from,
        because the run directory it left behind carries a checkpoint.
        """

        return cls(Outcome.INTERRUPTED, run_dir=run_dir,
                   note="run did not return; manifest not finalized")


class Solver(Protocol):
    """One attempt against one goal. Nothing about choosing goals lives here."""

    #: Which rung of P-2's ladder this is. Recorded on a goal that exhausts,
    #: so "we could not do it" is always qualified by "with what".
    level: str

    def attack(self, goal: Goal, *, budget: float) -> AttemptResult:
        ...


@dataclass
class StubSolver:
    """Canned outcomes from a script. The reason P-3 is testable at all.

    The acceptance criteria that matter most -- an infrastructure fault must
    not push a goal toward exhaustion, an interrupted attempt must resume, the
    graph must survive a kill -- cannot be exercised against a real solver. You
    cannot make a judge go down on cue, at one chosen node, three times and then
    recover. A script does exactly that, in milliseconds:

        StubSolver({"sha256:left": [INFRA_FAILED, INFRA_FAILED, PROVED]})

    Paths that are never walked in a test are first walked in production, and
    they are wrong when they are.
    """

    #: identity -> the outcomes to return, in order.
    script: Mapping[str, Sequence[Outcome]]
    #: identity -> the Lean text a PROVED outcome should carry.
    proofs: Mapping[str, str] = field(default_factory=dict)
    #: What to return once a goal's script is used up, or for a goal the script
    #: never mentions. TASK_FAILED rather than PROVED: a stub that silently
    #: succeeds turns "the controller never asked" into "everything worked".
    default: Outcome = Outcome.TASK_FAILED
    cost_per_attempt: float = 1.0
    level: str = "stub"

    #: Every identity attacked, in order. Assertions about what the controller
    #: chose to do live on this.
    calls: list[str] = field(default_factory=list)
    _cursor: dict[str, int] = field(default_factory=dict)

    def attack(self, goal: Goal, *, budget: float) -> AttemptResult:
        self.calls.append(goal.identity)
        if budget < self.cost_per_attempt:
            # The seam where money runs out. Never TASK_FAILED: the goal was
            # not tried and has said nothing about how hard it is.
            return AttemptResult(
                Outcome.BUDGET_EXHAUSTED,
                cost=0.0,
                note=f"budget {budget} below {self.cost_per_attempt}",
            )

        planned = self.script.get(goal.identity, ())
        index = self._cursor.get(goal.identity, 0)
        outcome = planned[index] if index < len(planned) else self.default
        self._cursor[goal.identity] = index + 1

        proof_text = None
        if outcome is Outcome.PROVED:
            proof_text = self.proofs.get(goal.identity) or _stub_proof(goal)
        return AttemptResult(
            outcome=outcome,
            proof_text=proof_text,
            cost=self.cost_per_attempt,
            note=f"stub attempt {index + 1} for {goal.identity}",
        )


def _stub_proof(goal: Goal) -> str:
    """A placeholder body, so a test that does not care about Lean text does
    not have to supply one. Assembly tests supply real text instead."""

    return f"-- stub proof for {goal.identity}\n{goal.statement}"


__all__ = ["AttemptResult", "Solver", "SolverError", "StubSolver"]
