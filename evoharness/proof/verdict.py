"""What one verifier answer says about a candidate, and what an attempt says about its goal.

Two levels, and conflating them is the mistake this module exists to prevent.

**Per candidate.** The verifier answers with a status and a machine-readable
reason. Some answers are verdicts on the candidate -- it did not compile, it
proved a different proposition, it rests on an axiom the graph's policy does
not accept. Others are verdicts on nothing: the request was built wrong, the
environment cannot host it, the verifier fell over. `classify` maps each
(status, reason) pair to exactly one `Outcome`, so the two sets in `graph.py`
stay plain sets of outcomes and nothing downstream ever parses a message.

**Per attempt.** One attempt is one search run, and a run grades many
candidates. "Some candidate got `goal_mismatch`" is not how the attempt ended.
`attempt_outcome` applies the aggregation rule, in this order:

1. any candidate certified -> PROVED (a proof in hand stays a proof);
2. otherwise, if the run produced no-verdict answers and no verdict on any
   candidate -> the most severe no-verdict outcome: a verifier that never
   answered about the mathematics must not read as "the model cannot do it";
3. otherwise the run's own stopping reason decides, as before.

The table is keyed by the verifier's reason vocabulary, a pinned copy of which
lives in `tests/fixtures/verification_reasons.json`. A pair the table does not
know maps to CONTRACT_REJECTED: the verifier's contract has moved past this
one, which is a fault on this side, not a verdict on the goal.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .graph import (
    CAPABILITY_OUTCOMES,
    NO_VERDICT_OUTCOMES,
    NO_VERDICT_SEVERITY,
    Outcome,
)

#: Candidate verdicts: the candidate's own doing, whatever the reason.
_BY_STATUS: dict[str, Outcome] = {
    "verified": Outcome.PROVED,
    "task_failed": Outcome.TASK_FAILED,
    "timeout": Outcome.TIMEOUT,
    # It proved something, just not this. The failure mode worth catching most:
    # a quietly weakened statement.
    "goal_mismatch": Outcome.TASK_FAILED,
    "stale_input": Outcome.STALE_INPUT,
    "environment_mismatch": Outcome.ENVIRONMENT_MISMATCH,
    "infra_failed": Outcome.INFRA_FAILED,
    "interrupted": Outcome.INTERRUPTED,
}

#: `policy_rejected` is the one status whose reasons split across both sets.
_POLICY: dict[str, Outcome] = {
    # The candidate chose to write these.
    "meta_declaration": Outcome.TASK_FAILED,
    "sorry_axiom": Outcome.TASK_FAILED,
    "forbidden_axiom": Outcome.TASK_FAILED,
    # The graph's policy is fixed by its scope, so a proof below the floor is a
    # verdict under that scope. Leaving it without one would let a model that
    # only ever reaches for native_decide under `trusted` retry forever.
    "trust_below_minimum": Outcome.TASK_FAILED,
    # Includes native_decide on the candidate's own functions, which the
    # verifier cannot recheck. Another proof exists for this goal or it does
    # not; the candidate may reach for `decide` instead.
    "native_unverified": Outcome.TASK_FAILED,
    # An `unsafe` or `partial` root: never kernel-checked, the candidate's doing.
    "root_not_replayed": Outcome.TASK_FAILED,
    "sorry_outside_subgoals": Outcome.TASK_FAILED,
    "parent_uses_sorry": Outcome.TASK_FAILED,
    "subgoal_unused": Outcome.TASK_FAILED,
    # The request this side built does not match what the file declares. Text
    # checks here are supposed to stop that before the verifier sees it.
    "root_contract": Outcome.CONTRACT_REJECTED,
    "namespace_prefix": Outcome.CONTRACT_REJECTED,
}

#: The status used for a candidate this side could not get an answer for at
#: all -- the verifier was unreachable. Kept, so it can be verified later.
UNVERIFIED = "unverified"


def classify(status: str, reason: str | None = None) -> Outcome:
    """The one outcome a verifier answer maps to."""

    if status == "policy_rejected":
        return _POLICY.get(reason or "", Outcome.CONTRACT_REJECTED)
    if status == UNVERIFIED:
        return Outcome.INFRA_FAILED
    return _BY_STATUS.get(status, Outcome.CONTRACT_REJECTED)


def is_known(status: str, reason: str | None) -> bool:
    """Whether `classify` maps this pair on purpose rather than by fallback."""

    if status == "policy_rejected":
        return (reason or "") in _POLICY
    return status in _BY_STATUS or status == UNVERIFIED


@dataclass(frozen=True)
class CandidateVerdict:
    """One candidate's verification, as recorded against its attempt."""

    candidate_id: str
    status: str
    reason: str | None = None
    job_id: str | None = None
    certification_id: str | None = None
    envelope_hash: str | None = None
    source_sha256: str | None = None
    #: The verifier's key for the goal the request was made against.
    goal_key: str | None = None
    trust: str | None = None
    #: For people. Never parsed.
    note: str = ""

    @property
    def outcome(self) -> Outcome:
        return classify(self.status, self.reason)

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reason": self.reason,
            "job_id": self.job_id,
            "certification_id": self.certification_id,
            "envelope_hash": self.envelope_hash,
            "source_sha256": self.source_sha256,
            "goal_key": self.goal_key,
            "trust": self.trust,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, data: dict) -> "CandidateVerdict":
        return cls(**{key: data.get(key) for key in (
            "candidate_id", "status", "reason", "job_id", "certification_id",
            "envelope_hash", "source_sha256", "goal_key", "trust")}, note=data.get("note") or "")


def attempt_outcome(verdicts: Iterable[CandidateVerdict]) -> Outcome | None:
    """Rules 1 and 2 of the aggregation. None means rule 3: the run decides.

    Rule 1 here only recognizes a certified candidate; whether that
    certification is bound to the winning candidate is the caller's check
    (`AttemptResult`), because only the caller knows which candidate won.
    """

    outcomes = [v.outcome for v in verdicts]
    if Outcome.PROVED in outcomes:
        return Outcome.PROVED
    if any(o in CAPABILITY_OUTCOMES for o in outcomes):
        return None
    seen = {o for o in outcomes if o in NO_VERDICT_OUTCOMES}
    for outcome in NO_VERDICT_SEVERITY:
        if outcome in seen:
            return outcome
    return None


__all__ = [
    "UNVERIFIED",
    "CandidateVerdict",
    "attempt_outcome",
    "classify",
    "is_known",
]
