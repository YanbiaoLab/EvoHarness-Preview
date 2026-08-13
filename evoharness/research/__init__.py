"""Research Layer I-3: experiments as first-class frozen objects."""

from .models import (
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    ProtocolChangeProposal,
    ResearchDecision,
    ResearchGoal,
)
from .runner import ExperimentRefMismatch, run_experiment, verify_refs
from .store import ResearchStore, ResearchStoreError

__all__ = [
    "ExperimentOutcome",
    "ExperimentRefMismatch",
    "ExperimentSpec",
    "ProtocolChangeProposal",
    "ResearchDecision",
    "ResearchGoal",
    "ResearchStore",
    "ResearchStoreError",
    "run_experiment",
    "verify_refs",
]
