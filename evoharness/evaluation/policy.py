"""唯一政策表面:证据能被搜索与承诺层怎样使用。
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import EvidenceEnvelope
from .faults import FaultKind


@dataclass(frozen=True)
class SearchUseDecision:
    recordable: bool
    selectable: bool
    archive_eligible: bool
    repairable: bool
    rankable: bool
    projected_fitness: float | None
    reason: str

    @property
    def quarantine(self) -> bool:
        """记录在案但选择/archive/修复全部排除(裁定非法、部分覆盖)。

        与 passed 正交:task_failure/timeout 走老语义(不入选、可修复),
        不隔离。Core 只认这个通用标志,不知道原因。
        """
        return self.recordable and not self.selectable and not self.repairable

    def to_json(self) -> dict:
        return {
            "recordable": self.recordable,
            "selectable": self.selectable,
            "archive_eligible": self.archive_eligible,
            "repairable": self.repairable,
            "rankable": self.rankable,
            "projected_fitness": self.projected_fitness,
            "reason": self.reason,
            "quarantine": self.quarantine,
        }


def decide_search_use(envelope: EvidenceEnvelope) -> SearchUseDecision:
    """Fail closed for declared measurements; keep the legacy path explicit.
    """

    if not envelope.evaluation_valid:
        return SearchUseDecision(
            False, False, False, False, False, None, "no-verdict"
        )
    if not envelope.admissible:
        return SearchUseDecision(
            True,
            False,
            False,
            False,
            False,
            envelope.fitness,
            "invalid-candidate",
        )
    if envelope.fault_kind in {FaultKind.TASK_FAILURE, FaultKind.TIMEOUT}:
        return SearchUseDecision(
            True,
            False,
            False,
            True,
            False,
            envelope.fitness,
            envelope.fault_kind,
        )
    if envelope.fault_kind is not None:
        return SearchUseDecision(
            False, False, False, False, False, None, "unsupported-fault"
        )
    if envelope.coverage.complete:
        return SearchUseDecision(
            True,
            True,
            True,
            False,
            True,
            envelope.fitness,
            "complete-verdict",
        )
    if not envelope.coverage.declared:
        return SearchUseDecision(
            True,
            True,
            True,
            False,
            False,
            envelope.fitness,
            "legacy-coverage-undeclared",
        )
    return SearchUseDecision(
        True,
        False,
        False,
        False,
        False,
        envelope.fitness,
        "partial-coverage",
    )


def may_enter_population(envelope: EvidenceEnvelope) -> bool:
    """Whether this evidence may be recorded in the population at all."""

    return decide_search_use(envelope).recordable


def may_rank(a: EvidenceEnvelope, b: EvidenceEnvelope) -> None:
    """Require the same namespace and complete successful measurements."""

    a.namespace.require_comparable(b.namespace)
    for envelope in (a, b):
        if not decide_search_use(envelope).rankable:
            raise ValueError(
                "cannot rank incomplete, undeclared or unsuccessful evidence"
            )


def may_support_objective(envelope: EvidenceEnvelope) -> bool:
    return (
        envelope.objective_met is True
        and bool(envelope.objective_verifier_hash)
        and envelope.evaluation_valid
        and envelope.admissible is True
        and envelope.fault_kind is None
        and envelope.coverage.complete
    )


__all__ = [
    "SearchUseDecision",
    "decide_search_use",
    "may_enter_population",
    "may_rank",
    "may_support_objective",
]
