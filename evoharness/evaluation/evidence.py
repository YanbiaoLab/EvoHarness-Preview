"""Evidence contract: types, invariants, content-addressed identity, codec.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .faults import NO_VERDICT_FAULTS, FaultKind
from .namespace import ScoreNamespace
from .strictjson import (
    freeze_json,
    optional_bool,
    optional_finite_float,
    strict_bool,
    strict_int,
    thaw_json,
)

EVIDENCE_SCHEMA_VERSION = 1


class EvidenceProtocolError(ValueError):
    """A report or envelope contains contradictory protocol semantics."""

    def __init__(self, message: str, *, fault_kind: FaultKind | None = None) -> None:
        super().__init__(message)
        self.fault_kind = fault_kind


@dataclass(frozen=True)
class Coverage:
    planned_units: int
    executed_units: int
    trustworthy_units: int

    def __post_init__(self) -> None:
        for name in (
            "planned_units",
            "executed_units",
            "trustworthy_units",
        ):
            value = strict_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.trustworthy_units > self.executed_units:
            raise ValueError(
                "trustworthy_units cannot exceed executed_units"
            )
        if 0 < self.planned_units < self.executed_units:
            raise ValueError("executed_units cannot exceed planned_units")

    @property
    def declared(self) -> bool:
        return self.planned_units > 0

    @property
    def complete(self) -> bool:
        return (
            self.declared
            and self.executed_units == self.planned_units
            and self.trustworthy_units == self.planned_units
        )

    def to_json(self) -> dict[str, int]:
        return {
            "planned_units": self.planned_units,
            "executed_units": self.executed_units,
            "trustworthy_units": self.trustworthy_units,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Coverage":
        return decode_coverage(payload)


@dataclass(frozen=True)
class EvidenceEnvelope:
    evidence_id: str
    candidate_id: str
    namespace: ScoreNamespace
    evaluation_valid: bool
    admissible: bool | None
    fitness: float | None
    fault_kind: FaultKind | None
    coverage: Coverage
    objective_met: bool | None = None
    objective_verifier_hash: str | None = None
    observations: Mapping[str, Any] = field(default_factory=dict)
    missing_reasons: tuple[str, ...] = ()
    budget_used_usd: float | None = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    artifacts_ref: str | None = None

    def __post_init__(self) -> None:
        normalize_and_validate_evidence(self)
        assign_or_validate_evidence_id(self)

    def content_hash(self) -> str:
        return evidence_content_hash(self)

    def to_json(self) -> dict[str, Any]:
        return encode_evidence(self)

    @classmethod
    def from_json(cls, payload: dict) -> "EvidenceEnvelope":
        return decode_evidence(payload)


# --- cross-field invariants (construction AND decode both land here) -------


def normalize_and_validate_evidence(envelope: EvidenceEnvelope) -> None:
    """Normalize immutable payloads, then enforce envelope semantics."""

    if (
        not isinstance(envelope.candidate_id, str)
        or not envelope.candidate_id.strip()
    ):
        raise ValueError("candidate_id must be non-empty")
    if not isinstance(envelope.namespace, ScoreNamespace):
        raise TypeError("namespace must be ScoreNamespace")
    if not isinstance(envelope.coverage, Coverage):
        raise TypeError("coverage must be Coverage")
    strict_bool(envelope.evaluation_valid, "evaluation_valid")
    optional_bool(envelope.admissible, "admissible")
    optional_bool(envelope.objective_met, "objective_met")
    fitness = optional_finite_float(envelope.fitness, "fitness")
    object.__setattr__(envelope, "fitness", fitness)

    budget = envelope.budget_used_usd
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError("budget_used_usd must be non-negative and finite")
    object.__setattr__(envelope, "budget_used_usd", float(budget))

    if envelope.fault_kind is not None and not isinstance(
        envelope.fault_kind, FaultKind
    ):
        raise TypeError("fault_kind must be FaultKind or None")
    if envelope.artifacts_ref is not None and (
        not isinstance(envelope.artifacts_ref, str)
        or not envelope.artifacts_ref.strip()
    ):
        raise ValueError("artifacts_ref must be non-empty when present")

    reasons = tuple(envelope.missing_reasons)
    if any(
        not isinstance(reason, str) or not reason.strip()
        for reason in reasons
    ):
        raise ValueError("missing_reasons must contain non-empty strings")
    object.__setattr__(envelope, "missing_reasons", reasons)
    object.__setattr__(
        envelope,
        "observations",
        freeze_json(envelope.observations, "observations"),
    )
    object.__setattr__(
        envelope,
        "provenance",
        freeze_json(envelope.provenance, "provenance"),
    )

    if envelope.evaluation_valid:
        if envelope.admissible is None or envelope.fitness is None:
            raise EvidenceProtocolError(
                "valid evidence requires admissibility and fitness",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        if envelope.fault_kind in NO_VERDICT_FAULTS:
            raise EvidenceProtocolError(
                "no-verdict fault cannot mark evidence valid",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        expected_admissible = (
            envelope.fault_kind is not FaultKind.INVALID_CANDIDATE
        )
        if envelope.admissible is not expected_admissible:
            raise EvidenceProtocolError(
                "admissibility contradicts fault_kind",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        if envelope.missing_reasons:
            raise EvidenceProtocolError(
                "valid evidence cannot carry missing reasons",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
    else:
        if envelope.admissible is not None or envelope.fitness is not None:
            raise EvidenceProtocolError(
                "no-verdict evidence cannot carry admissibility or fitness",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        if envelope.fault_kind not in NO_VERDICT_FAULTS:
            raise EvidenceProtocolError(
                "invalid evaluation requires a no-verdict fault",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        if not envelope.missing_reasons:
            raise EvidenceProtocolError(
                "no-verdict evidence requires a reason",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
        if envelope.objective_met is not None:
            raise EvidenceProtocolError(
                "no-verdict evidence cannot assess the objective",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )

    if envelope.objective_met is not None:
        if (
            not isinstance(envelope.objective_verifier_hash, str)
            or not envelope.objective_verifier_hash.strip()
        ):
            raise EvidenceProtocolError(
                "objective assessment requires verifier identity",
                fault_kind=FaultKind.PROTOCOL_ERROR,
            )
    elif envelope.objective_verifier_hash is not None:
        raise EvidenceProtocolError(
            "verifier identity requires an objective assessment",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        )
    if envelope.objective_met is True and (
        not envelope.evaluation_valid
        or envelope.admissible is not True
        or envelope.fault_kind is not None
        or not envelope.coverage.complete
    ):
        raise EvidenceProtocolError(
            "objective_met=True requires complete admissible evidence",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        )


# --- codec + content-addressed identity ------------------------------------


def decode_coverage(payload: dict) -> Coverage:
    if not isinstance(payload, dict):
        raise TypeError("coverage must be a JSON object")
    expected = {"planned_units", "executed_units", "trustworthy_units"}
    if set(payload) != expected:
        raise ValueError("coverage fields mismatch")
    return Coverage(
        planned_units=strict_int(payload["planned_units"], "planned_units"),
        executed_units=strict_int(
            payload["executed_units"], "executed_units"
        ),
        trustworthy_units=strict_int(
            payload["trustworthy_units"], "trustworthy_units"
        ),
    )


def evidence_content_payload(envelope: EvidenceEnvelope) -> dict[str, Any]:
    """Canonical payload covered by ``evidence_id``."""

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate_id": envelope.candidate_id,
        "namespace": envelope.namespace.to_json(),
        "evaluation_valid": envelope.evaluation_valid,
        "admissible": envelope.admissible,
        "fitness": envelope.fitness,
        "fault_kind": (
            envelope.fault_kind.value if envelope.fault_kind else None
        ),
        "coverage": envelope.coverage.to_json(),
        "objective_met": envelope.objective_met,
        "objective_verifier_hash": envelope.objective_verifier_hash,
        "observations": thaw_json(envelope.observations),
        "missing_reasons": list(envelope.missing_reasons),
        "budget_used_usd": envelope.budget_used_usd,
        "provenance": thaw_json(envelope.provenance),
        "artifacts_ref": envelope.artifacts_ref,
    }


def evidence_content_hash(envelope: EvidenceEnvelope) -> str:
    encoded = json.dumps(
        evidence_content_payload(envelope),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assign_or_validate_evidence_id(envelope: EvidenceEnvelope) -> None:
    expected_id = evidence_content_hash(envelope)
    if not isinstance(envelope.evidence_id, str):
        raise TypeError("evidence_id must be a string")
    if envelope.evidence_id in {"", "auto"}:
        object.__setattr__(envelope, "evidence_id", expected_id)
        return
    if envelope.evidence_id != expected_id:
        raise EvidenceProtocolError(
            "evidence_id does not match content hash",
            fault_kind=FaultKind.PROTOCOL_ERROR,
        )


def encode_evidence(envelope: EvidenceEnvelope) -> dict[str, Any]:
    return {
        "evidence_id": envelope.evidence_id,
        **evidence_content_payload(envelope),
    }


def canonical_evidence_json(envelope: EvidenceEnvelope) -> str:
    return json.dumps(
        encode_evidence(envelope),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decode_evidence(payload: dict) -> EvidenceEnvelope:
    if not isinstance(payload, dict):
        raise TypeError("evidence envelope must be a JSON object")
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evidence schema: {payload.get('schema_version')}"
        )
    required = {
        "evidence_id",
        "candidate_id",
        "namespace",
        "evaluation_valid",
        "admissible",
        "fitness",
        "fault_kind",
        "coverage",
        "objective_met",
        "objective_verifier_hash",
        "observations",
        "missing_reasons",
        "budget_used_usd",
        "provenance",
        "artifacts_ref",
        "schema_version",
    }
    if set(payload) != required:
        raise ValueError("evidence envelope fields mismatch")
    evidence_id = payload["evidence_id"]
    if not isinstance(evidence_id, str) or not evidence_id:
        raise TypeError("evidence_id must be a non-empty string")
    fault_raw = payload["fault_kind"]
    if fault_raw is not None and not isinstance(fault_raw, str):
        raise TypeError("fault_kind must be a string or None")
    reasons = payload["missing_reasons"]
    if not isinstance(reasons, list):
        raise TypeError("missing_reasons must be a list")
    return EvidenceEnvelope(
        evidence_id=evidence_id,
        candidate_id=payload["candidate_id"],
        namespace=ScoreNamespace.from_json(payload["namespace"]),
        evaluation_valid=strict_bool(
            payload["evaluation_valid"], "evaluation_valid"
        ),
        admissible=optional_bool(payload["admissible"], "admissible"),
        fitness=optional_finite_float(payload["fitness"], "fitness"),
        fault_kind=FaultKind(fault_raw) if fault_raw is not None else None,
        coverage=decode_coverage(payload["coverage"]),
        objective_met=optional_bool(
            payload["objective_met"], "objective_met"
        ),
        objective_verifier_hash=payload["objective_verifier_hash"],
        observations=payload["observations"],
        missing_reasons=tuple(reasons),
        budget_used_usd=optional_finite_float(
            payload["budget_used_usd"], "budget_used_usd"
        ),
        provenance=payload["provenance"],
        artifacts_ref=payload["artifacts_ref"],
    )


__all__ = [
    "Coverage",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceEnvelope",
    "EvidenceProtocolError",
    "assign_or_validate_evidence_id",
    "canonical_evidence_json",
    "decode_coverage",
    "decode_evidence",
    "encode_evidence",
    "evidence_content_hash",
    "evidence_content_payload",
    "normalize_and_validate_evidence",
]
