"""Agent-facing proposal preflight adapter tests."""

import json

from evoharness.evocore import (
    Candidate,
    LLMToolCall,
    PreflightIssue,
    PreflightPipeline,
    PreflightResult,
    ProposalPreflight,
)
from evoharness.evocore.agent import (
    AgentToolContext,
    AgentToolRegistry,
    RunPreflightTool,
)


class StaticValidator:
    name = "compile"

    def __init__(self, issues=(), error=None):
        self.issues = tuple(issues)
        self.error = error

    def validate(self, ctx):
        if self.error is not None:
            raise self.error
        return PreflightResult(stage=self.name, issues=self.issues)


def make_context(tmp_path, validator=None):
    parent = Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )
    workdir = parent.workspace.materialize(tmp_path / "work")
    pipeline = PreflightPipeline([validator] if validator else [])
    return AgentToolContext(
        workdir=workdir,
        parent=parent,
        operator="rewrite",
        preflight=ProposalPreflight(pipeline),
        remaining_timeout_s=30,
    )


def invoke(tool, ctx, arguments=None):
    call = LLMToolCall(
        "call-preflight",
        "run_preflight",
        arguments or {},
    )
    registry = AgentToolRegistry([tool])
    result = registry.invoke(call, ctx)
    return result, json.loads(result.content), registry, call


def test_preflight_passes_after_workspace_change(tmp_path):
    ctx = make_context(tmp_path, StaticValidator())
    (ctx.workdir / "main.py").write_text("x = 2\n")

    result, payload, registry, call = invoke(RunPreflightTool(), ctx)

    assert not result.is_error
    assert payload["ok"]
    assert payload["issues"] == []
    assert payload["summary"]["issue_count"] == 0
    assert not registry.is_concurrency_safe(call, ctx)


def test_preflight_renders_pure_issue_ir_at_tool_boundary(tmp_path):
    issue = PreflightIssue(
        validator="compile",
        code="syntax-error",
        message="invalid syntax",
        path="main.py",
        line=1,
        column=4,
        command=("python", "-m", "compileall"),
        stderr="traceback",
    )
    ctx = make_context(tmp_path, StaticValidator([issue]))
    (ctx.workdir / "main.py").write_text("x =\n")

    result, payload, _, _ = invoke(RunPreflightTool(), ctx)

    assert not result.is_error
    assert not payload["ok"]
    assert payload["summary"]["failed_stage"] == "compile"
    assert payload["summary"]["repairable"]
    assert payload["issues"][0]["command"] == [
        "python",
        "-m",
        "compileall",
    ]
    assert payload["issues"][0]["stderr"] == "traceback"


def test_preflight_no_changes_and_validator_exception_are_structured(tmp_path):
    ctx = make_context(tmp_path)
    _, no_change, _, _ = invoke(RunPreflightTool(), ctx)
    assert no_change["issues"][0]["code"] == "no-changes"

    ctx = make_context(
        tmp_path / "second",
        StaticValidator(error=RuntimeError("boom")),
    )
    (ctx.workdir / "main.py").write_text("x = 2\n")
    _, failed, _, _ = invoke(RunPreflightTool(), ctx)
    assert failed["issues"][0]["code"] == "validator-error"
    assert not failed["issues"][0]["repairable"]


def test_preflight_rejects_arguments(tmp_path):
    result, payload, _, _ = invoke(
        RunPreflightTool(),
        make_context(tmp_path),
        {"force": True},
    )

    assert result.is_error
    assert payload["error"]["code"] == "invalid-arguments"
