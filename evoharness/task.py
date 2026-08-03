"""Minimal public task entry point.

Task authors provide a seed workspace and a scoring function.  This module
owns the framework-specific adapters needed by the evolutionary engine.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from evoharness.evocore.agent import Runner
from evoharness.evocore.interfaces import Grader
from evoharness.evocore.llm import LLMTransport
from evoharness.evocore.population import EvalReport
from evoharness.evocore.preflight import PreflightValidator
from evoharness.evocore.remote import EvalInfraError
from evoharness.evocore.workspace import FileWorkspace, GitWorkspace, Workspace
from evoharness.evoserve import (
    GradeContext,
    GradeFn,
    GradeValue,
    InfraError,
    coerce_grade,
)

if TYPE_CHECKING:
    from evoharness.evocore.population import Candidate


WorkspaceGradeFn: TypeAlias = Callable[[Path, GradeContext], GradeValue]


def adapt_source_grade_fn(
    grade_fn: GradeFn,
    *,
    main_file: str = "main.py",
) -> WorkspaceGradeFn:
    """Lift a legacy source-string grade function into the workspace API."""

    @wraps(grade_fn)
    def grade_workspace(candidate_dir: Path, ctx: GradeContext) -> GradeValue:
        return grade_fn((Path(candidate_dir) / main_file).read_text(), ctx)

    return grade_workspace


class WorkspaceGradeFnGrader:
    """Framework Grader adapter for a user-authored workspace grade function."""

    def __init__(
        self, grade_func: WorkspaceGradeFn, lineage_dir: Path | None = None
    ):
        self._grade_func = grade_func
        # Set by whoever knows the run directory (see run_evolution). Left
        # None the domain simply gets no lineage and cold-starts, which is
        # the behaviour every existing task already has.
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
            # Redirected when the parent was a seed copy, which has an id but
            # never had anything published under it (see SearchLoop's child
            # construction). parent_id itself stays as recorded.
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


@dataclass
class ScorableTask:
    """Everything EvoHarness needs to evolve one user-defined task."""

    grader: Grader
    initial_code: str
    initial_workspace: Workspace | None = None
    task_sys_msg: str = ""
    transport: LLMTransport | None = None
    research_brief: str = ""
    # Island seeds: a whole Workspace (multi-file family) or a bare main-file
    # string, which the loop lifts into the primary workspace's kind.
    extra_seeds: list[str | Workspace] = field(default_factory=list)
    preflight_validators: tuple[PreflightValidator, ...] = ()
    runner: Runner | None = None

    @classmethod
    def _from_workspace(
        cls,
        workspace: Workspace,
        grader: Grader,
        *,
        task_sys_msg: str,
        transport: LLMTransport | None,
        research_brief: str,
        extra_seeds: Sequence[str | Workspace] | None,
        preflight_validators: Sequence[PreflightValidator],
        runner: Runner | None,
    ) -> "ScorableTask":
        return cls(
            grader=grader,
            initial_code=workspace.main_text(),
            initial_workspace=workspace,
            task_sys_msg=task_sys_msg,
            transport=transport,
            research_brief=research_brief,
            extra_seeds=list(extra_seeds or ()),
            preflight_validators=tuple(preflight_validators),
            runner=runner,
        )

    @classmethod
    def from_directory(
        cls,
        seed_dir: Path,
        grade_func: WorkspaceGradeFn,
        *,
        main_file: str = "main.py",
        include_files: Collection[str] | None = None,
        task_sys_msg: str = "",
        transport: LLMTransport | None = None,
        research_brief: str = "",
        extra_seeds: Sequence[str | Workspace] | None = None,
        preflight_validators: Sequence[PreflightValidator] = (),
        runner: Runner | None = None,
    ) -> "ScorableTask":
        workspace = GitWorkspace.from_directory(
            Path(seed_dir),
            main_file=main_file,
            include_files=include_files,
        )
        return cls._from_workspace(
            workspace,
            WorkspaceGradeFnGrader(grade_func),
            task_sys_msg=task_sys_msg,
            transport=transport,
            research_brief=research_brief,
            extra_seeds=extra_seeds,
            preflight_validators=preflight_validators,
            runner=runner,
        )

    @classmethod
    def from_source(
        cls,
        source: str,
        grade_func: GradeFn,
        *,
        filename: str = "main.py",
        task_sys_msg: str = "",
        transport: LLMTransport | None = None,
        research_brief: str = "",
        extra_seeds: Sequence[str | Workspace] | None = None,
        preflight_validators: Sequence[PreflightValidator] = (),
        runner: Runner | None = None,
    ) -> "ScorableTask":
        workspace = FileWorkspace(source, filename=filename)
        grader = WorkspaceGradeFnGrader(
            adapt_source_grade_fn(grade_func, main_file=filename)
        )
        return cls._from_workspace(
            workspace,
            grader,
            task_sys_msg=task_sys_msg,
            transport=transport,
            research_brief=research_brief,
            extra_seeds=extra_seeds,
            preflight_validators=preflight_validators,
            runner=runner,
        )


__all__ = [
    "ScorableTask",
    "WorkspaceGradeFn",
    "WorkspaceGradeFnGrader",
    "adapt_source_grade_fn",
]
