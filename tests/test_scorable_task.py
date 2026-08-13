"""I-1 acceptance tests for frozen TaskSpec and ResolvedTask."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoharness import ResolvedTask, WorkspaceGradeFnGrader, adapt_source_grade_fn
from evoharness.core import Candidate
from evoharness.core.remote import EvalInfraError
from evoharness.core.workspace import FileWorkspace, GitWorkspace, WorkspaceError
from evoharness.serve import Grade, GradeContext, InfraError


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


def test_resolved_task_keeps_runtime_outside_frozen_spec(tmp_path):
    workspace = GitWorkspace(
        base_files={"main.py": "from helper import answer\n", "helper.py": "answer=42\n"}
    )
    seen = {}

    def grade_func(candidate_dir: Path, ctx: GradeContext):
        seen["files"] = {
            path.name: path.read_text()
            for path in candidate_dir.iterdir()
            if path.is_file() and path.name != ".git"
        }
        return Grade(fitness=1.0)

    grader = WorkspaceGradeFnGrader(grade_func)
    task = ResolvedTask.create(
        task_id="demo",
        version="v1",
        grader=grader,
        initial_workspace=workspace,
        domain_prompt="Improve the agent.",
    )

    assert task.grader is grader
    assert task.spec.grader.name.endswith("WorkspaceGradeFnGrader")
    assert task.spec.domain_prompt == "Improve the agent."
    assert json.loads(task.spec.initial_workspace.blob) == json.loads(
        workspace.serialize()
    )
    report = task.grader.grade(candidate(workspace), tmp_path / "evaluation")
    assert report.fitness == 1.0
    assert seen["files"]["main.py"] == "from helper import answer\n"
    assert seen["files"]["helper.py"] == "answer=42\n"


def test_workspace_hash_covers_non_main_files():
    left = GitWorkspace(base_files={"main.py": "x=1\n", "helper.py": "a=1\n"})
    right = GitWorkspace(base_files={"main.py": "x=1\n", "helper.py": "a=2\n"})
    grader = WorkspaceGradeFnGrader(lambda _root, _ctx: 1.0)
    first = ResolvedTask.create(
        task_id="demo", version="v1", grader=grader, initial_workspace=left
    )
    second = ResolvedTask.create(
        task_id="demo", version="v1", grader=grader, initial_workspace=right
    )
    assert first.spec.hash != second.spec.hash


def test_directory_workspace_can_freeze_an_explicit_allowlist(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "main.py").write_text("answer = 42\n")
    (seed / "local_notes.md").write_text("not candidate state\n")
    workspace = GitWorkspace.from_directory(seed, include_files=("main.py",))
    assert workspace.texts() == {"main.py": "answer = 42\n"}


def test_resolved_task_from_directory_builds_frozen_spec_and_grader(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "solver.py").write_text("answer = 42\n")
    (seed / "notes.md").write_text("exclude me\n")

    task = ResolvedTask.from_directory(
        seed,
        lambda root, _ctx: float("42" in (root / "solver.py").read_text()),
        task_id="directory-demo",
        version="v1",
        main_file="solver.py",
        include_files=("solver.py",),
    )

    assert task.initial_workspace.texts() == {"solver.py": "answer = 42\n"}
    assert task.spec.initial_workspace.main_file == "solver.py"
    assert task.spec.grader.config_json != "{}"


def test_directory_workspace_requires_declared_main_file(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "solver.py").write_text("answer = 42\n")
    with pytest.raises(WorkspaceError, match="main file"):
        GitWorkspace.from_directory(seed)


def test_source_grade_adapter_reads_materialized_main_file(tmp_path):
    seen = {}

    def grade_fn(source: str, ctx: GradeContext):
        seen["source"] = source
        seen["workdir"] = ctx.workdir
        return {"fitness": 0.75, "passed": True}

    workspace = FileWorkspace("value = 7\n")
    grader = WorkspaceGradeFnGrader(adapt_source_grade_fn(grade_fn))
    report = grader.grade(candidate(workspace), tmp_path / "evaluation")
    assert report.fitness == 0.75
    assert seen["source"] == "value = 7\n"
    assert seen["workdir"] == (tmp_path / "evaluation").resolve()


def test_workspace_grader_classifies_user_exception_as_failed_verdict(tmp_path):
    workspace = FileWorkspace("x = 1\n")

    def broken(_candidate_dir: Path, _ctx: GradeContext):
        raise ValueError("bad candidate")

    report = WorkspaceGradeFnGrader(broken).grade(
        candidate(workspace), tmp_path / "evaluation"
    )
    assert report.passed is False
    assert report.stage_reached == 0
    assert report.fault == "uncaught exception in grade_func"
    assert "ValueError: bad candidate" in report.stderr_log


def test_workspace_grader_preserves_dependency_failure_as_infra_error(tmp_path):
    workspace = FileWorkspace("x = 1\n")

    def unavailable(_candidate_dir: Path, _ctx: GradeContext):
        raise InfraError("judge unavailable")

    with pytest.raises(EvalInfraError, match="judge unavailable"):
        WorkspaceGradeFnGrader(unavailable).grade(
            candidate(workspace), tmp_path / "evaluation"
        )


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
        candidate(workspace), tmp_path / "evaluation"
    )
    assert report.fitness == 1.0
