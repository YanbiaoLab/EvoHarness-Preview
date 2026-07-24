"""EvoHarness: evolutionary search framework.

Framework layers live under this single top-level package:
evocore (engine), evoguard (guardrails), evoplus (research extensions),
evoserve (eval-side SDK), evoviz / evoweb (reporting & console).
Consumer packages (modmul, tasks, recipes, experiments) sit OUTSIDE and
import downward; the reverse direction is forbidden (tests/test_layering.py).
"""

__version__ = "0.1.0"

from .task import (
    ScorableTask,
    WorkspaceGradeFn,
    WorkspaceGradeFnGrader,
    adapt_source_grade_fn,
)

__all__ = [
    "ScorableTask",
    "WorkspaceGradeFn",
    "WorkspaceGradeFnGrader",
    "adapt_source_grade_fn",
]
