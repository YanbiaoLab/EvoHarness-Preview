"""Generic local grading adapters used by resolved domain tasks."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from evoharness.core.population import EvalReport
from evoharness.core.remote import EvalInfraError
from evoharness.serve import (
    GradeContext,
    GradeFn,
    GradeValue,
    InfraError,
    coerce_grade,
)

if TYPE_CHECKING:
    from evoharness.core.population import Candidate


WorkspaceGradeFn: TypeAlias = Callable[[Path, GradeContext], GradeValue]


def adapt_source_grade_fn(
    grade_fn: GradeFn,
    *,
    main_file: str = "main.py",
) -> WorkspaceGradeFn:
    @wraps(grade_fn)
    def grade_workspace(candidate_dir: Path, ctx: GradeContext) -> GradeValue:
        return grade_fn((Path(candidate_dir) / main_file).read_text(), ctx)

    return grade_workspace


class WorkspaceGradeFnGrader:
    """Materialize a candidate and call a domain-owned workspace grade fn."""

    def __init__(
        self,
        grade_func: WorkspaceGradeFn,
        lineage_dir: Path | None = None,
    ):
        self._grade_func = grade_func
        self.lineage_dir = lineage_dir

    def grade(self, cand: "Candidate", workdir: Path) -> EvalReport:
        workdir = Path(workdir).resolve()
        candidate_dir = workdir / "candidate"
        cand.workspace.materialize(candidate_dir)
        ctx = GradeContext(
            candidate_id=cand.id,
            workdir=workdir,
            operator=cand.operator,
            generation=cand.generation,
            parent_id=(
                cand.metadata.get("lineage_parent_id") or cand.parent_id
            ),
            ancestor_ids=tuple(cand.metadata.get("lineage_ancestors", ())),
            lineage_dir=self.lineage_dir,
            state_donors=tuple(cand.metadata.get("state_donors", ())),
        )
        started = time.monotonic()
        try:
            raw = self._grade_func(candidate_dir, ctx)
            report = EvalReport.from_json(coerce_grade(raw))
        except InfraError as exc:
            raise EvalInfraError(
                f"grade_func dependency failed: {exc}"
            ) from exc

        except Exception:
            report = EvalReport(
                fitness=0.0,
                passed=False,
                fault="uncaught exception in grade_func",
                stderr_log=traceback.format_exc(),
                stage_reached=0,
            )
        if not report.execution_time:
            report.execution_time = time.monotonic() - started
        return report


__all__ = [
    "WorkspaceGradeFn",
    "WorkspaceGradeFnGrader",
    "adapt_source_grade_fn",
]
