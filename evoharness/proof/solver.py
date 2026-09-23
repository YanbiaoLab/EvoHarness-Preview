"""The seam between the graph and solver execution.

The controller needs exactly one operation from the outside world: "attack this
goal, tell me how it went". Narrowing that to a single method allows the same
controller to run against test stubs, standalone runs, and different solver tiers
or strategies.

Two rules live here rather than in the controller:

1. `AttemptResult` is the single place where an outcome is derived. The graph
   controller does not write its own opinion of whether a run succeeded,
   preventing duplicate sources of truth from drifting.

2. A PROVED result must carry proof text. Enforced at construction, ensuring
   that a goal marked proved always has proof text available for assembly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .graph import NO_VERDICT_OUTCOMES, Goal, Outcome
from .verdict import CandidateVerdict, attempt_outcome

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .envelope import SourceEnvelope

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
    #: Helper declarations beside the proof, named under the goal; see
    #: `graph.Attempt.auxiliary_declarations`. Only a PROVED result carries any.
    auxiliary_declarations: tuple[str, ...] = ()
    #: What the verifier said about each candidate the run graded, when the
    #: run was graded by one. The outcome above is aggregated from these.
    verifications: tuple[CandidateVerdict, ...] = ()
    #: The envelopes those verifications were made from, to persist with them.
    envelopes: "tuple[SourceEnvelope, ...]" = ()

    def __post_init__(self) -> None:
        if self.outcome is Outcome.PROVED and not self.proof_text:
            raise ValueError(
                "a PROVED attempt must carry its proof text; without it the "
                "graph holds a proved goal with nothing to assemble"
            )
        if self.outcome is not Outcome.PROVED and (
            self.proof_text or self.auxiliary_declarations
        ):
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
        verifications: Sequence[CandidateVerdict] = (),
        envelopes: "Sequence[SourceEnvelope]" = (),
        winner_envelope: str | None = None,
        verified_by_verifier: bool = False,
        auxiliary_declarations: Sequence[str] = (),
    ) -> "AttemptResult":
        """Derive the outcome from a finished `api.run()`.

        Order matters. Whether the goal was PROVED is asked first, because
        reaching `solved_at` means a proof was accepted, and that stays true
        however the run later fell over. Asking "how did it stop" first would
        throw away a proof already in hand because the judge went down
        afterwards.

        When the run was graded by the verifier (`verified_by_verifier`),
        reaching `solved_at` is not enough on its own: the winning candidate --
        whose envelope hash is `winner_envelope` -- must carry a certification
        among `verifications`. The grader is the only place a certification is
        made; this is the only place one is bound to the attempt's result. A
        solved score with no bound certification is a fault in that wiring,
        and is reported as one rather than as a proof.

        Then the no-verdict rule: a run in which the verifier answered only
        with no-verdict statuses -- never once about a candidate -- ends with
        the most severe of them, not with whatever the run's stopping reason
        suggests. "The verifier never answered" must not read as "the model
        cannot do it".

        INTERRUPTED is deliberately absent: a run that returns a report was not
        interrupted. See `interrupted()` for the case where it never returned.
        """

        reason = getattr(report, "stopped_reason", "")
        fitness = getattr(report, "best_fitness", None)
        cost = float(getattr(report, "total_llm_cost", 0.0) or 0.0)
        carried = {
            "run_dir": run_dir,
            "cost": cost,
            "verifications": tuple(verifications),
            "envelopes": tuple(envelopes),
        }

        if fitness is not None and fitness >= solved_at:
            if not proof_text:
                raise SolverError(
                    "run reached the solved threshold but no proof text was "
                    "extracted from its best candidate"
                )
            if verified_by_verifier and not _bound(verifications, winner_envelope):
                return cls(
                    Outcome.CONTRACT_REJECTED,
                    note=(
                        "the run reached the solved threshold but its winning "
                        f"candidate (envelope {winner_envelope}) carries no "
                        "certification; the grader and the result disagree"
                    ),
                    **carried,
                )
            return cls(
                outcome=Outcome.PROVED,
                proof_text=proof_text,
                auxiliary_declarations=tuple(auxiliary_declarations),
                evidence_ref=evidence_ref,
                note=f"stopped_reason={reason}",
                **carried,
            )
        aggregated = attempt_outcome(verifications)
        if aggregated in NO_VERDICT_OUTCOMES:
            counts: dict[str, int] = {}
            for verdict in verifications:
                key = f"{verdict.status}/{verdict.reason}" if verdict.reason else verdict.status
                counts[key] = counts.get(key, 0) + 1
            return cls(
                aggregated,
                note=(
                    f"stopped_reason={reason}; the verifier returned no verdict "
                    "on any candidate: "
                    + ", ".join(f"{k} x{n}" for k, n in sorted(counts.items()))
                )[:500],
                **carried,
            )
        if reason in _INFRA_REASONS:
            return cls(Outcome.INFRA_FAILED, note=f"stopped_reason={reason}", **carried)
        if reason in _BUDGET_REASONS:
            return cls(Outcome.BUDGET_EXHAUSTED, note=f"stopped_reason={reason}", **carried)
        return cls(Outcome.TASK_FAILED,
                   note=f"stopped_reason={reason} best_fitness={fitness}", **carried)

    @classmethod
    def interrupted(
        cls, run_dir: str | None = None, note: str = "",
    ) -> "AttemptResult":
        """The run never returned. The one outcome a later run may resume from,
        because the run directory it left behind carries a checkpoint.

        `note` is the caller's diagnosis, and recovery has a better one than
        this constructor can guess: it knows whose lease went stale, while
        from here all that is visible is the absence of a report.
        """

        return cls(Outcome.INTERRUPTED, run_dir=run_dir,
                   note=note or "run did not return; manifest not finalized")


def _bound(verifications: Sequence[CandidateVerdict], envelope_hash: str | None) -> bool:
    """A certification made for exactly the winning candidate's envelope."""

    return envelope_hash is not None and any(
        v.status == "verified" and v.certification_id and v.envelope_hash == envelope_hash
        for v in verifications
    )


class Solver(Protocol):
    """One attempt against one goal. Nothing about choosing goals lives here."""

    #: Solver capability tier or strategy identifier. Recorded on a goal that
    #: exhausts, so exhaustion is explicitly qualified by the solver used.
    level: str

    def attack(self, goal: Goal, *, budget: float) -> AttemptResult:
        ...


@dataclass
class StubSolver:
    """Canned outcomes from a script for deterministic testing.

    Critical edge cases -- such as infrastructure faults not counting toward
    exhaustion, resuming interrupted attempts, and graph resilience under
    abrupt termination -- cannot be exercised reliably against live solvers.
    A script provides deterministic test behavior in milliseconds:

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
