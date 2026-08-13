"""Controlled diagnostic command tool tests."""

import json
from dataclasses import dataclass

import pytest

from evoharness.core import (
    Candidate,
    LLMToolCall,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.core.agent import (
    AgentToolContext,
    AgentToolRegistry,
    RunTool,
)


@dataclass
class FakeResult:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    elapsed_s: float = 0.1


class FakeRunner:
    def __init__(self, result=None, error=None):
        self.result = result or FakeResult()
        self.error = error
        self.calls = []

    def run(self, cmd, workdir, timeout_s, env=None):
        self.calls.append(
            {
                "cmd": cmd,
                "workdir": workdir,
                "timeout_s": timeout_s,
                "env": env,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result


def make_context(tmp_path, remaining_timeout_s=30):
    parent = Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )
    workdir = parent.workspace.materialize(tmp_path / "work")
    return AgentToolContext(
        workdir=workdir,
        parent=parent,
        operator="rewrite",
        preflight=ProposalPreflight(PreflightPipeline()),
        remaining_timeout_s=remaining_timeout_s,
    )


def invoke(tool, ctx, **arguments):
    call = LLMToolCall("call-run", "run", arguments)
    result = AgentToolRegistry([tool]).invoke(call, ctx)
    return result, json.loads(result.content)


def test_run_reports_success_and_uses_minimum_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOHARNESS_SECRET", "must-not-leak")
    runner = FakeRunner(FakeResult(stdout="ok\n"))
    tool = RunTool(runner, runner_timeout_cap_s=10)

    result, payload = invoke(
        tool,
        make_context(tmp_path, remaining_timeout_s=3),
        argv=["python", "-m", "pytest"],
        timeout_s=20,
    )

    assert not result.is_error
    assert payload["ok"]
    assert payload["return_code"] == 0
    assert payload["timeout_s"] == 3
    assert runner.calls[0]["timeout_s"] == 3
    assert "EVOHARNESS_SECRET" not in runner.calls[0]["env"]


def test_nonzero_and_timeout_are_diagnostic_results_not_tool_errors(tmp_path):
    for fake_result in (
        FakeResult(return_code=2, stderr="failed"),
        FakeResult(return_code=-9, timed_out=True, stderr="timeout"),
    ):
        result, payload = invoke(
            RunTool(FakeRunner(fake_result)),
            make_context(tmp_path),
            argv=["test"],
            timeout_s=1,
        )
        assert not result.is_error
        assert not payload["ok"]
        assert payload["return_code"] == fake_result.return_code
        assert payload["timed_out"] == fake_result.timed_out


def test_stdout_and_stderr_are_bounded_independently(tmp_path):
    runner = FakeRunner(
        FakeResult(stdout="a" * 200, stderr="b" * 200)
    )

    _, payload = invoke(
        RunTool(runner, max_output_chars=64),
        make_context(tmp_path),
        argv=["test"],
        timeout_s=1,
    )

    assert len(payload["stdout"]) == 64
    assert len(payload["stderr"]) == 64
    assert payload["stdout_truncated"]
    assert payload["stderr_truncated"]
    assert payload["stdout"].startswith("a")
    assert payload["stdout"].endswith("a")


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"argv": [], "timeout_s": 1},
        {"argv": ["echo"], "timeout_s": 0},
        {"argv": "echo", "timeout_s": 1},
        {"argv": ["echo"], "timeout_s": 1, "shell": True},
    ],
)
def test_bad_run_arguments_return_stable_error(tmp_path, arguments):
    result, payload = invoke(
        RunTool(FakeRunner()),
        make_context(tmp_path),
        **arguments,
    )

    assert result.is_error
    assert payload["error"]["code"] == "invalid-arguments"


def test_command_start_failures_are_stable_and_run_is_always_exclusive(tmp_path):
    tool = RunTool(FakeRunner(error=FileNotFoundError("missing")))
    ctx = make_context(tmp_path)
    call = LLMToolCall(
        "call-run",
        "run",
        {"argv": ["missing"], "timeout_s": 1},
    )
    registry = AgentToolRegistry([tool])

    result = registry.invoke(call, ctx)
    payload = json.loads(result.content)

    assert result.is_error
    assert payload["error"]["code"] == "command-not-found"
    assert not registry.is_concurrency_safe(call, ctx)
