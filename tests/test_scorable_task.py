"""CD-0A acceptance tests for the minimal public task API."""

from __future__ import annotations

from pathlib import Path

import pytest

from evoharness import (
    ScorableTask,
    WorkspaceGradeFnGrader,
    adapt_source_grade_fn,
)
from evoharness.evocore import Candidate
from evoharness.evocore.remote import EvalInfraError
from evoharness.evocore.workspace import GitWorkspace, WorkspaceError
from evoharness.evoserve import Grade, GradeContext, InfraError
from recipes.common import TaskBundle


def candidate(workspace, *, candidate_id: str = "candidate-1") -> Candidate:
    return Candidate(
        id=candidate_id,
        code=workspace.serialize(),
        workspace_kind=workspace.kind,
        generation=2,
        parent_id="parent-1",
        island_idx=0,
        operator="rewrite",
    )


def test_scorable_task_from_directory_loads_multifile_seed(tmp_path):
    seed = tmp_path / "seed_agent"
    seed.mkdir()
    (seed / "main.py").write_text("from helper import answer\n")
    (seed / "helper.py").write_text("answer = 42\n")
    seen = {}

    def grade_func(candidate_dir: Path, ctx: GradeContext):
        seen["files"] = {
            path.name: path.read_text()
            for path in candidate_dir.iterdir()
            if path.is_file()
        }
        seen["ctx"] = ctx
        return Grade(fitness=1.0)

    task = ScorableTask.from_directory(
        seed,
        grade_func,
        task_sys_msg="Improve the agent.",
    )

    assert isinstance(task.initial_workspace, GitWorkspace)
    assert task.initial_code == "from helper import answer\n"
    report = task.grader.grade(
        candidate(task.initial_workspace),
        tmp_path / "evaluation",
    )
    assert report.fitness == 1.0
    assert seen["files"] == {
        "main.py": "from helper import answer\n",
        "helper.py": "answer = 42\n",
    }
    assert seen["ctx"].candidate_id == "candidate-1"
    assert seen["ctx"].generation == 2
    assert seen["ctx"].operator == "rewrite"


def test_from_directory_can_freeze_an_explicit_candidate_allowlist(tmp_path):
    seed = tmp_path / "seed_agent"
    seed.mkdir()
    (seed / "main.py").write_text("answer = 42\n")
    (seed / "local_notes.md").write_text("not part of the candidate\n")
    cache = seed / "__pycache__"
    cache.mkdir()
    (cache / "main.pyc").write_bytes(b"\x00\xff")

    task = ScorableTask.from_directory(
        seed,
        lambda _root, _ctx: 1.0,
        include_files=("main.py",),
    )

    assert task.initial_workspace.texts() == {"main.py": "answer = 42\n"}


def test_from_directory_requires_declared_main_file(tmp_path):
    seed = tmp_path / "seed_agent"
    seed.mkdir()
    (seed / "solver.py").write_text("answer = 42\n")

    with pytest.raises(WorkspaceError, match="main file"):
        ScorableTask.from_directory(seed, lambda _root, _ctx: 1.0)


def test_source_grade_adapter_reads_materialized_main_file(tmp_path):
    seen = {}

    def grade_fn(source: str, ctx: GradeContext):
        seen["source"] = source
        seen["workdir"] = ctx.workdir
        return {"fitness": 0.75, "passed": True}

    task = ScorableTask.from_source("value = 7\n", grade_fn)
    report = task.grader.grade(
        candidate(task.initial_workspace),
        tmp_path / "evaluation",
    )

    assert report.fitness == 0.75
    assert seen["source"] == "value = 7\n"
    assert seen["workdir"] == (tmp_path / "evaluation").resolve()


def test_workspace_grader_classifies_user_exception_as_failed_verdict(tmp_path):
    workspace = GitWorkspace(base_files={"main.py": "x = 1\n"})

    def broken(_candidate_dir: Path, _ctx: GradeContext):
        raise ValueError("bad candidate")

    report = WorkspaceGradeFnGrader(broken).grade(
        candidate(workspace),
        tmp_path / "evaluation",
    )

    assert report.passed is False
    assert report.stage_reached == 0
    assert report.fault == "uncaught exception in grade_func"
    assert "ValueError: bad candidate" in report.stderr_log


def test_workspace_grader_preserves_dependency_failure_as_infra_error(tmp_path):
    workspace = GitWorkspace(base_files={"main.py": "x = 1\n"})

    def unavailable(_candidate_dir: Path, _ctx: GradeContext):
        raise InfraError("judge unavailable")

    with pytest.raises(EvalInfraError, match="judge unavailable"):
        WorkspaceGradeFnGrader(unavailable).grade(
            candidate(workspace),
            tmp_path / "evaluation",
        )


def test_existing_taskbundle_name_is_compatibility_alias():
    assert TaskBundle is ScorableTask


def test_adapt_source_grade_fn_supports_non_default_main_file(tmp_path):
    workspace = GitWorkspace(
        base_files={"solver.py": "answer = 9\n"},
        main_file="solver.py",
    )
    wrapped = adapt_source_grade_fn(
        lambda source, _ctx: 1.0 if "9" in source else 0.0,
        main_file="solver.py",
    )

    report = WorkspaceGradeFnGrader(wrapped).grade(
        candidate(workspace),
        tmp_path / "evaluation",
    )
    assert report.fitness == 1.0
