from dataclasses import dataclass
from typing import TYPE_CHECKING

from evoharness.contracts import spec_hash
from evoharness.evaluation import EvidenceEnvelope

from .assessment import AssessmentGuard, ClaimAssessment, Verdict
from .claims import Claim, ClaimKind
from .reference import ReferenceRecord, ReferenceStore

if TYPE_CHECKING:
    from .routing import ResearchRouter


@dataclass(frozen=True)
class PromotionDecision:
    action: str                     # promote | reject | hold
    reasons: tuple[str, ...]
    assessments: tuple[ClaimAssessment, ...]
    policy_hash: str
    promoted: "ReferenceRecord | None" = None


class PromotionPolicy:
    def __init__(self,
                 guard: AssessmentGuard,
                 store: ReferenceStore,
                 *,
                 margin: float = 0.0,
                 require_no_regression: bool = False,
                 router: "ResearchRouter | None" = None,
                 experiment_id: str = "",

                 ):
        self._guard = guard
        self._store = store
        self._margin = margin
        self._require_no_regression = require_no_regression
        if router is not None and not experiment_id.strip():
            raise ValueError(
                "routing a promotion needs the experiment it belongs to"
            )
        self._router = router
        self._experiment_id = experiment_id
        # 路由器不进 policy_hash:换不换冠军由 margin 与 assessor 决定,
        # 谁被通知不改变晋升本身,不该让同一策略产生两个身份。
        self.policy_hash = spec_hash({
            "kind": "promotion_policy",
            "margin": margin,
            "require_no_regression": require_no_regression,
            "assessor_hash": guard.assessor_hash,
        })

    def consider(
        self,
        subject: EvidenceEnvelope,
        reference_evidence: EvidenceEnvelope | None,
    ) -> PromotionDecision:
        current = self._store.current()
        if current is None:
            return self._bootstrap(subject)
        if reference_evidence is None:
            return PromotionDecision(
                action="hold",
                reasons=("reference evidence unavailable — re-evaluate",),
                assessments=(),
                policy_hash=self.policy_hash,
            )

        claims = [Claim(
            kind=ClaimKind.BETTER_THAN_REFERENCE,
            subject_id=subject.candidate_id,
            reference_id=reference_evidence.candidate_id,
            margin=self._margin,
        )]
        if self._require_no_regression:
            claims.append(Claim(
                kind=ClaimKind.NO_REGRESSION,
                subject_id=subject.candidate_id,
                reference_id=reference_evidence.candidate_id,
            ))
        assessments = tuple(
            self._guard.assess(claim, subject, reference_evidence)
            for claim in claims
        )

        if any(a.status is Verdict.CONTRADICTED for a in assessments):
            return PromotionDecision(
                action="reject",
                reasons=tuple(
                    r for a in assessments
                    if a.status is Verdict.CONTRADICTED for r in a.reasons
                ),
                assessments=assessments,
                policy_hash=self.policy_hash,
            )
        if any(a.status is Verdict.UNKNOWN for a in assessments):
            # fail closed:unknown 不是 0 也不是失败,是"证据不够,去补"。
            return PromotionDecision(
                action="hold",
                reasons=tuple(
                    r for a in assessments
                    if a.status is Verdict.UNKNOWN for r in a.reasons
                ),
                assessments=assessments,
                policy_hash=self.policy_hash,
            )

        promoted = self._store.promote(
            ReferenceRecord(
                candidate_id=subject.candidate_id,
                evidence_id=subject.evidence_id,
                namespace=subject.namespace,
                fitness=subject.fitness,
                policy_hash=self.policy_hash,
                assessor_hash=self._guard.assessor_hash,
                reasons=tuple(
                    r for a in assessments for r in a.reasons
                ),
                promoted_at=0.0,
            ),
            expected_current=current.candidate_id,
        )
        self._route_flip(current, promoted, assessments)
        return PromotionDecision(
            action="promote",
            reasons=("all required claims supported",),
            assessments=assessments,
            policy_hash=self.policy_hash,
            promoted=promoted,
        )

    def _route_flip(self, previous, promoted, assessments) -> None:
        """换冠军是要被人看见的事,但晋升不等人:卡片提交后立即返回。"""
        if self._router is None:
            return
        self._router.route_promotion(
            experiment_id=self._experiment_id,
            previous=previous,
            promoted=promoted,
            reasons=tuple(r for a in assessments for r in a.reasons),
        )

    def _bootstrap(self, subject: EvidenceEnvelope) -> PromotionDecision:
        claim = Claim(
            kind=ClaimKind.CANDIDATE_VALID, subject_id=subject.candidate_id
        )
        assessment = self._guard.assess(claim, subject)
        if assessment.status is not Verdict.SUPPORTED:
            return PromotionDecision(
                action="hold",
                reasons=assessment.reasons,
                assessments=(assessment,),
                policy_hash=self.policy_hash,
            )
        promoted = self._store.promote(
            ReferenceRecord(
                candidate_id=subject.candidate_id,
                evidence_id=subject.evidence_id,
                namespace=subject.namespace,
                fitness=subject.fitness,
                policy_hash=self.policy_hash,
                assessor_hash=self._guard.assessor_hash,
                reasons=("bootstrap: first valid reference",),
                promoted_at=0.0,
            ),
            expected_current=None,
        )
        return PromotionDecision(
            action="promote",
            reasons=("bootstrap: first valid reference",),
            assessments=(assessment,),
            policy_hash=self.policy_hash,
            promoted=promoted,
        )

