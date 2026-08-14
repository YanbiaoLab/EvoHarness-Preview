"""AssessmentGuard: Evidence 能支持什么结论,由版本化规则推导。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum

from evoharness.contracts import spec_hash
from evoharness.evaluation import EvidenceEnvelope

from .claims import Claim, ClaimKind


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


# v2(2026-08-13):比较类规则引入噪声下限 floor = z*sqrt(sem_s^2+sem_r^2)。
# 动机:IMO 终选在 12 题上取 3 候选 argmax,validation→test 掉 0.206——
# 点估计比较会把幸运种子当成优势。v1 在噪声测量上会自信地支持/反驳,
# v2 把 floor 以内的差异判为 unknown(统计功效不足,补单位/种子)。
ASSESSOR_VERSION = "v2"

# 判定语义的权威摘要。改任何一条的行为,必须同步改这里的文字。
_RULE_DIGEST = {
    "candidate_valid": (
        "supported iff evaluation_valid and admissible is True; "
        "contradicted iff admissible is False; else unknown"
    ),
    "objective_met": (
        "supported iff verifier-set objective_met is True and coverage "
        "complete and admissible is True; contradicted iff objective_met "
        "is False; else unknown — passed never sets the objective"
    ),
    "has_any_gain": (
        "existence claim: supported iff both verdicts trustworthy, same "
        "namespace, subject better beyond the noise floor "
        "z*sqrt(sem_s^2+sem_r^2); never contradicted by a finite sample; "
        "else unknown"
    ),
    "better_than_reference": (
        "aggregate claim: assessable only when both coverages are "
        "complete; direction-normalized advantage compared against "
        "margin with a noise floor z*sqrt(sem_s^2+sem_r^2): supported "
        "beyond margin+floor, contradicted at or below margin-floor, "
        "unknown (underpowered) in between"
    ),
    "no_regression": (
        "universal claim: assessable only when both envelopes carry "
        "unit_results and subject re-executed every reference-passed "
        "unit; any flip contradicts; aggregate scores never support it"
    ),
}


class InsufficientEvidence(RuntimeError):
    """fail closed:自动化路径禁止把 unknown 当成 0、False 或成功。"""


@dataclass(frozen=True)
class ClaimAssessment:
    claim_hash: str
    claim_kind: str
    status: Verdict
    assessor_hash: str
    evidence_refs: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["evidence_refs"] = list(self.evidence_refs)
        payload["reasons"] = list(self.reasons)
        return {"schema_version": 1, **payload}


def require_supported(assessment: ClaimAssessment) -> None:
    if assessment.status is not Verdict.SUPPORTED:
        raise InsufficientEvidence(
            f"{assessment.claim_kind} is {assessment.status.value}: "
            + "; ".join(assessment.reasons)
        )


def _trustworthy(envelope: EvidenceEnvelope) -> bool:
    return (
        envelope.evaluation_valid
        and envelope.admissible is not False
        and envelope.fitness is not None
    )

# noinspection PyMethodMayBeStatic
class AssessmentGuard:
    def __init__(self, *, direction: str = "maximize", z: float = 1.0):
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"invalid direction: {direction!r}")
        if not isinstance(z, (int, float)) or z < 0 or not math.isfinite(z):
            raise ValueError("z must be a non-negative finite number")
        self._direction = direction
        self._z = float(z)
        self.assessor_hash = spec_hash({
            "kind": "assessor",
            "version": ASSESSOR_VERSION,
            "direction": direction,
            "z": self._z,
            "rules": _RULE_DIGEST,
        })

    # -- direction-aware comparisons ------------------------------------

    def _better(self, a: float, b: float, margin: float = 0.0) -> bool:
        if self._direction == "maximize":
            return a > b + margin
        return a < b - margin

    def _advantage(self, subject: float, reference: float) -> float:
        """方向归一的优势:正 = subject 更好。"""
        if self._direction == "maximize":
            return subject - reference
        return reference - subject

    def _noise_floor(
        self, subject: EvidenceEnvelope, reference: EvidenceEnvelope
    ) -> float:
        def sem_of(envelope: EvidenceEnvelope) -> float:
            try:
                return float(envelope.provenance.get("sem") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return self._z * math.sqrt(
            sem_of(subject) ** 2 + sem_of(reference) ** 2
        )

    # -- entry point ------------------------------------------------------

    def assess(
        self,
        claim: Claim,
        subject: EvidenceEnvelope,
        reference: EvidenceEnvelope | None = None,
    ) -> ClaimAssessment:
        if subject.candidate_id != claim.subject_id:
            raise ValueError(
                "subject evidence is about a different candidate "
                f"({subject.candidate_id} != {claim.subject_id})"
            )
        if claim.kind in (
            ClaimKind.HAS_ANY_GAIN,
            ClaimKind.BETTER_THAN_REFERENCE,
            ClaimKind.NO_REGRESSION,
        ):
            if reference is None:
                return self._result(claim, subject, None, Verdict.UNKNOWN,
                                    ["reference evidence missing"])
            if reference.candidate_id != claim.reference_id:
                raise ValueError(
                    "reference evidence is about a different candidate "
                    f"({reference.candidate_id} != {claim.reference_id})"
                )
            subject.namespace.require_comparable(reference.namespace)

        handlers: dict[
            ClaimKind, Callable[..., tuple[Verdict, list[str]]]
        ] = {
            ClaimKind.CANDIDATE_VALID: self._candidate_valid,
            ClaimKind.OBJECTIVE_MET: self._objective_met,
            ClaimKind.HAS_ANY_GAIN: self._has_any_gain,
            ClaimKind.BETTER_THAN_REFERENCE: self._better_than_reference,
            ClaimKind.NO_REGRESSION: self._no_regression,
        }
        status, reasons = handlers[claim.kind](claim, subject, reference)
        return self._result(claim, subject, reference, status, reasons)

    def _result(self, claim, subject, reference, status, reasons):
        refs = [subject.evidence_id]
        if reference is not None:
            refs.append(reference.evidence_id)
        return ClaimAssessment(
            claim_hash=claim.hash,
            claim_kind=claim.kind.value,
            status=status,
            assessor_hash=self.assessor_hash,
            evidence_refs=tuple(refs),
            reasons=tuple(reasons),
        )

    # -- rules(语义见 _RULE_DIGEST,改行为必须同步改摘要)-----------

    def _candidate_valid(self, claim, subject, _reference):
        if subject.evaluation_valid and subject.admissible is True:
            return Verdict.SUPPORTED, ["verdict evidence, admissible"]
        if subject.admissible is False:
            return Verdict.CONTRADICTED, ["adjudicated inadmissible"]
        return Verdict.UNKNOWN, ["no admissibility verdict"]

    def _objective_met(self, claim, subject, _reference):
        if subject.objective_met is False:
            return Verdict.CONTRADICTED, ["verifier rejected the objective"]
        if subject.objective_met is not True:
            return Verdict.UNKNOWN, [
                "no independent verifier verdict — passed never sets "
                "the objective"
            ]
        if not subject.coverage.complete:
            return Verdict.UNKNOWN, ["coverage incomplete"]
        if subject.admissible is not True:
            return Verdict.UNKNOWN, ["admissibility not established"]
        return Verdict.SUPPORTED, ["verifier verdict on complete coverage"]

    def _has_any_gain(self, claim, subject, reference):
        if not (_trustworthy(subject) and _trustworthy(reference)):
            return Verdict.UNKNOWN, ["needs trustworthy verdicts on both"]
        floor = self._noise_floor(subject, reference)
        advantage = self._advantage(subject.fitness, reference.fitness)
        if advantage > floor:
            reasons = ["one trustworthy improving witness observed"]
            if floor:
                reasons.append(f"beyond noise floor ±{floor:.4g} (z={self._z})")
            if not subject.coverage.complete:
                reasons.append("subject coverage partial — existence only")
            return Verdict.SUPPORTED, reasons
        if advantage > 0:
            return Verdict.UNKNOWN, [
                f"advantage {advantage:.4g} within noise floor "
                f"±{floor:.4g} — underpowered, add units or seeds"
            ]
        return Verdict.UNKNOWN, [
            "no gain in this sample; a finite sample cannot refute "
            "existence"
        ]

    def _better_than_reference(self, claim, subject, reference):
        if not (_trustworthy(subject) and _trustworthy(reference)):
            return Verdict.UNKNOWN, ["needs trustworthy verdicts on both"]
        if not (subject.coverage.complete and reference.coverage.complete):
            return Verdict.UNKNOWN, [
                "aggregate comparison requires complete coverage on both"
            ]
        floor = self._noise_floor(subject, reference)
        advantage = self._advantage(subject.fitness, reference.fitness)
        reasons = []
        if floor:
            reasons.append(f"noise floor ±{floor:.4g} (z={self._z})")
        if advantage > claim.margin + floor:
            return Verdict.SUPPORTED, reasons + [
                "complete-coverage margin met beyond noise floor"
            ]
        if advantage <= claim.margin - floor:
            return Verdict.CONTRADICTED, reasons + [
                "complete coverage shows no aggregate advantage"
            ]
        return Verdict.UNKNOWN, reasons + [
            f"advantage {advantage:.4g} statistically indistinguishable "
            "from margin — underpowered, add units or seeds"
        ]


    def _no_regression(self, claim, subject, reference):
        subject_units = subject.observations.get("unit_results")
        reference_units = reference.observations.get("unit_results")
        # 信封的 observations 经 freeze_json 冻结,是 Mapping 不是 dict。
        if not isinstance(subject_units, Mapping) or not isinstance(
            reference_units, Mapping
        ):
            return Verdict.UNKNOWN, [
                "unit-level results unavailable; aggregate scores never "
                "support a universal claim"
            ]
        reference_passed = {
            unit for unit, ok in reference_units.items() if ok
        }
        unexecuted = reference_passed - set(subject_units)
        if unexecuted:
            return Verdict.UNKNOWN, [
                f"{len(unexecuted)} reference-passed units not re-executed"
            ]
        flips = sorted(
            unit for unit in reference_passed if not subject_units[unit]
        )
        if flips:
            return Verdict.CONTRADICTED, [
                f"{len(flips)} reference-passed units regressed "
                f"(e.g. {flips[:3]})"
            ]
        return Verdict.SUPPORTED, [
            f"all {len(reference_passed)} reference-passed units held"
        ]