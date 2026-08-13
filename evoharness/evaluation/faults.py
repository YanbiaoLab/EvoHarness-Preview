"""Typed evaluation failures and strict wire classification."""

from __future__ import annotations

from enum import Enum


class FaultKind(str, Enum):
    MISSING = "missing"
    INFRA_ERROR = "infra_error"
    PROTOCOL_ERROR = "protocol_error"
    TASK_FAILURE = "task_failure"
    TIMEOUT = "timeout"
    INVALID_CANDIDATE = "invalid_candidate"
    UNKNOWN = "unknown"


NO_VERDICT_FAULTS = frozenset(
    {
        FaultKind.MISSING,
        FaultKind.INFRA_ERROR,
        FaultKind.PROTOCOL_ERROR,
        FaultKind.UNKNOWN,
    }
)
VERDICT_FAULTS = frozenset(
    {
        FaultKind.TASK_FAILURE,
        FaultKind.TIMEOUT,
        FaultKind.INVALID_CANDIDATE,
    }
)
_WIRE_VOCAB = {kind.value: kind for kind in FaultKind}


def classify_fault(
    *,
    passed: bool,
    fault_kind: str | None,
) -> FaultKind | None:
    """Map a wire verdict onto the taxonomy without guessing from text.

    Legacy ``passed=False`` reports without a typed kind remain task failures.
    Unknown values and the contradictory ``passed=True`` plus a fault are
    protocol errors/no-verdict states and must fail closed downstream.
    """

    if not isinstance(passed, bool):
        raise TypeError("passed must be bool")
    if fault_kind is not None and (
        not isinstance(fault_kind, str) or not fault_kind.strip()
    ):
        raise TypeError("fault_kind must be a non-empty string or None")
    if passed:
        return None if fault_kind is None else FaultKind.PROTOCOL_ERROR
    if fault_kind is None:
        return FaultKind.TASK_FAILURE
    return _WIRE_VOCAB.get(fault_kind, FaultKind.UNKNOWN)


__all__ = [
    "FaultKind",
    "NO_VERDICT_FAULTS",
    "VERDICT_FAULTS",
    "classify_fault",
]
