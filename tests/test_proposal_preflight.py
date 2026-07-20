"""Authoritative capture and validation of materialized proposals."""

from __future__ import annotations

import pytest

from evoharness.evocore import (
    Candidate,
    PreflightContext,
    PreflightIssue,
    PreflightPipeline,
    PreflightReport,
    PreflightResult,
    ProposalCheckResult,
    ProposalPreflight,
)


class RecordingValidator:
    def __init__(self, name="compile", issues=()):
        self.name = name
        self.issues = tuple(issues)
        self.calls = 0
        self.seen_main = None

    def validate(self, ctx):
        self.calls += 1
        self.seen_main = (ctx.workdir / "main.py").read_text()
        return PreflightResult(stage=self.name, issues=self.issues)


def make_parent():
    return Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def make_context(tmp_path):
    parent = make_parent()
    workdir = parent.workspace.materialize(tmp_path / "work")
    return PreflightContext(parent, "rewrite", workdir)


def test_valid_change_returns_captured_child_and_full_report(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "main.py").write_text("x = 2\n")
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert outcome.ok
    assert outcome.child_workspace is not None
    assert outcome.child_workspace.main_text() == "x = 2\n"
    assert outcome.report.ok
    assert [result.stage for result in outcome.report.results] == [
        "workspace",
        "compile",
    ]
    assert validator.calls == 1
    assert validator.seen_main == "x = 2\n"


def test_no_changes_is_repairable_and_skips_task_validators(tmp_path):
    ctx = make_context(tmp_path)
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.child_workspace is None
    assert outcome.report.failed_stage == "workspace"
    assert outcome.report.issues[0].code == "no-changes"
    assert outcome.report.repairable
    assert validator.calls == 0


def test_deleted_main_is_workspace_invalid_and_skips_validators(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "main.py").unlink()
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.child_workspace is None
    assert outcome.report.failed_stage == "workspace"
    assert outcome.report.issues[0].code == "workspace-invalid"
    assert "main file" in outcome.report.issues[0].message
    assert validator.calls == 0


def test_binary_file_is_workspace_invalid_and_skips_validators(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "binary.dat").write_bytes(b"\xff\xfe\x00")
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.report.issues[0].code == "workspace-invalid"
    assert "binary" in outcome.report.issues[0].message
    assert validator.calls == 0


def test_symlink_is_workspace_invalid_and_skips_validators(tmp_path):
    ctx = make_context(tmp_path)
    target = tmp_path / "outside.py"
    target.write_text("secret = True\n")
    (ctx.workdir / "linked.py").symlink_to(target)
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.report.issues[0].code == "workspace-invalid"
    assert "symlink" in outcome.report.issues[0].message
    assert validator.calls == 0


def test_single_file_extra_file_is_rejected_before_validators(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "helper.py").write_text("value = 2\n")
    validator = RecordingValidator()

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.report.issues[0].code == "workspace-invalid"
    assert "extra files" in outcome.report.issues[0].message
    assert validator.calls == 0


def test_task_failure_keeps_workspace_stage_but_withholds_child(tmp_path):
    ctx = make_context(tmp_path)
    (ctx.workdir / "main.py").write_text("x =\n")
    issue = PreflightIssue(
        validator="compile",
        code="syntax-error",
        message="invalid syntax",
        path="main.py",
        line=1,
    )
    validator = RecordingValidator(issues=(issue,))

    outcome = ProposalPreflight(
        PreflightPipeline([validator])
    ).check(ctx)

    assert not outcome.ok
    assert outcome.child_workspace is None
    assert outcome.report.failed_stage == "compile"
    assert outcome.report.issues == (issue,)
    assert [result.stage for result in outcome.report.results] == [
        "workspace",
        "compile",
    ]
    assert validator.calls == 1
    assert validator.seen_main == "x =\n"


def test_workspace_stage_name_is_reserved():
    validator = RecordingValidator(name="workspace")

    with pytest.raises(ValueError, match="reserved"):
        ProposalPreflight(PreflightPipeline([validator]))


def test_check_result_rejects_inconsistent_success_state():
    with pytest.raises(ValueError, match="child_workspace"):
        ProposalCheckResult(
            child_workspace=None,
            report=PreflightReport(),
        )
