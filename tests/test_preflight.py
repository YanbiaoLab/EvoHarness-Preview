"""Ordered, fail-closed proposal preflight execution."""

import pytest

from evoharness.evocore import (
    Candidate,
    PreflightContext,
    PreflightIssue,
    PreflightPipeline,
    PreflightResult,
)


class RecordingValidator:
    def __init__(self, name, calls, issues=(), error=None):
        self.name = name
        self.calls = calls
        self.issues = tuple(issues)
        self.error = error

    def validate(self, ctx):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error
        return PreflightResult(stage=self.name, issues=self.issues)


def make_context(tmp_path):
    parent = Candidate(
        id="p",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )
    return PreflightContext(parent, "rewrite", tmp_path)


def test_pipeline_runs_in_order_and_stops_at_first_failure(tmp_path):
    calls = []
    syntax = PreflightIssue(
        validator="compile",
        code="syntax-error",
        message="invalid syntax",
        path="main.py",
        line=2,
    )
    pipeline = PreflightPipeline(
        [
            RecordingValidator("workspace", calls),
            RecordingValidator("compile", calls, issues=(syntax,)),
            RecordingValidator("shape", calls),
        ]
    )

    report = pipeline.run(make_context(tmp_path))

    assert calls == ["workspace", "compile"]
    assert not report.ok and report.repairable
    assert report.failed_stage == "compile"
    assert report.issues == (syntax,)
    assert [result.stage for result in report.results] == [
        "workspace",
        "compile",
    ]


def test_pipeline_passes_when_every_stage_passes(tmp_path):
    calls = []
    pipeline = PreflightPipeline(
        [
            RecordingValidator("workspace", calls),
            RecordingValidator("compile", calls),
            RecordingValidator("shape", calls),
        ]
    )

    report = pipeline.run(make_context(tmp_path))

    assert report.ok
    assert not report.repairable
    assert report.failed_stage is None
    assert calls == ["workspace", "compile", "shape"]
    assert report.elapsed_s >= 0


def test_validator_exception_becomes_nonrepairable_diagnostic(tmp_path):
    calls = []
    pipeline = PreflightPipeline(
        [RecordingValidator("compile", calls, error=RuntimeError("boom"))]
    )

    report = pipeline.run(make_context(tmp_path))

    assert not report.ok and not report.repairable
    assert report.failed_stage == "compile"
    assert report.issues[0].validator == "compile"
    assert report.issues[0].code == "validator-error"
    assert "RuntimeError: boom" in report.issues[0].message


def test_validator_contract_mismatch_fails_closed(tmp_path):
    class WrongStageValidator:
        name = "compile"

        def validate(self, ctx):
            return PreflightResult(stage="shape")

    report = PreflightPipeline([WrongStageValidator()]).run(
        make_context(tmp_path)
    )

    assert report.issues[0].code == "validator-error"
    assert not report.issues[0].repairable


def test_pipeline_rejects_ambiguous_validator_names():
    calls = []
    with pytest.raises(ValueError, match="unique"):
        PreflightPipeline(
            [
                RecordingValidator("compile", calls),
                RecordingValidator("compile", calls),
            ]
        )

    with pytest.raises(ValueError, match="non-empty"):
        PreflightPipeline([RecordingValidator("", calls)])


def test_empty_pipeline_is_a_valid_noop(tmp_path):
    report = PreflightPipeline().run(make_context(tmp_path))
    assert report.ok
    assert report.results == ()
