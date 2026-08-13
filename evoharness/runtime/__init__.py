"""Runtime resolution and adapters for frozen public contracts."""

from .compiler import compile_specs, spec_hashes
from .grading import (
    WorkspaceGradeFn,
    WorkspaceGradeFnGrader,
    adapt_source_grade_fn,
)
from .task import ResolvedTask

__all__ = [
    "ResolvedTask",
    "WorkspaceGradeFn",
    "WorkspaceGradeFnGrader",
    "adapt_source_grade_fn",
    "compile_specs",
    "spec_hashes",
]
