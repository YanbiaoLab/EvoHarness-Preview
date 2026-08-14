"""Research Layer I-3/I-4: experiments, claims, assessment and promotion."""

from .assessment import (
    AssessmentGuard,
    ClaimAssessment,
    InsufficientEvidence,
    Verdict,
    require_supported,
)
from .claims import Claim, ClaimKind
from .migration import (
    MigrationApprovalRequired,
    MigrationPanelError,
    MigrationReport,
    compare_measurements,
    establish_baseline,
)
from .models import (
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    ProtocolChangeProposal,
    ResearchDecision,
    ResearchGoal,
)
from .promotion import PromotionDecision, PromotionPolicy
from .reference import ReferenceConflict, ReferenceRecord, ReferenceStore
from .runner import ExperimentRefMismatch, run_experiment, verify_refs
from .store import ResearchStore, ResearchStoreError

__all__ = [
    "AssessmentGuard",
    "Claim",
    "ClaimAssessment",
    "ClaimKind",
    "ExperimentOutcome",
    "ExperimentRefMismatch",
    "ExperimentSpec",
    "Hypothesis",
    "InsufficientEvidence",
    "MigrationApprovalRequired",
    "MigrationPanelError",
    "MigrationReport",
    "PromotionDecision",
    "PromotionPolicy",
    "ProtocolChangeProposal",
    "ReferenceConflict",
    "ReferenceRecord",
    "ReferenceStore",
    "ResearchDecision",
    "ResearchGoal",
    "ResearchStore",
    "ResearchStoreError",
    "Verdict",
    "compare_measurements",
    "establish_baseline",
    "require_supported",
    "run_experiment",
    "verify_refs",
]
