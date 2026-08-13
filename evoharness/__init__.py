"""EvoHarness public contracts and runtime entry objects."""

__version__ = "0.1.0"

from .contracts import (
    BasicSearchProfile,
    ComponentSpec,
    CriterionSpec,
    EvolutionSearchProfile,
    FeedbackSpec,
    MeasurementSpec,
    ProposalLimits,
    RunSpec,
    SearchProfile,
    TaskSpec,
    WorkspaceSpec,
)
from .runtime import (
    ResolvedTask,
    WorkspaceGradeFn,
    WorkspaceGradeFnGrader,
    adapt_source_grade_fn,
    compile_specs,
    spec_hashes,
)
from .api import run

__all__ = [
    "BasicSearchProfile",
    "ComponentSpec",
    "CriterionSpec",
    "EvolutionSearchProfile",
    "FeedbackSpec",
    "MeasurementSpec",
    "ProposalLimits",
    "ResolvedTask",
    "RunSpec",
    "SearchProfile",
    "TaskSpec",
    "WorkspaceGradeFn",
    "WorkspaceGradeFnGrader",
    "WorkspaceSpec",
    "adapt_source_grade_fn",
    "compile_specs",
    "spec_hashes",
    "run",
]
