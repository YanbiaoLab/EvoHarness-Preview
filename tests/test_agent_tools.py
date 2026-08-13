"""Provider-neutral agent tool registry tests."""

import json

import pytest

from evoharness.core import (
    Candidate,
    LLMToolCall,
    LLMToolDefinition,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.core.agent import (
    AgentTool,
    AgentToolContext,
    AgentToolError,
    AgentToolRegistry,
    make_tool_result,
)


class EchoTool:
    definition = LLMToolDefinition(
        name="echo",
        description="Return the supplied text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    def is_concurrency_safe(self, call, ctx):
        return True

    def invoke(self, call, ctx):
        return make_tool_result(
            call.call_id,
            {"ok": True, "text": call.arguments["text"]},
        )


class BrokenTool:
    definition = LLMToolDefinition(
        name="broken",
        description="Always fail.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def is_concurrency_safe(self, call, ctx):
        raise RuntimeError("classifier boom")

    def invoke(self, call, ctx):
        raise RuntimeError("boom")


class BusinessErrorTool(BrokenTool):
    definition = LLMToolDefinition(
        name="business",
        description="Raise an expected tool error.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def invoke(self, call, ctx):
        raise AgentToolError("invalid-input", "bad value")


def make_context(tmp_path):
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
        remaining_timeout_s=30.0,
    )


def test_registry_dispatches_tool_and_preserves_call_id(tmp_path):
    tool = EchoTool()
    assert isinstance(tool, AgentTool)
    registry = AgentToolRegistry([tool])
    call = LLMToolCall("call-1", "echo", {"text": "hello"})

    result = registry.invoke(call, make_context(tmp_path))

    assert result.call_id == "call-1"
    assert not result.is_error
    assert json.loads(result.content) == {
        "ok": True,
        "text": "hello",
    }
    assert registry.definitions == (tool.definition,)
    assert registry.is_concurrency_safe(call, make_context(tmp_path))


def test_registry_returns_error_for_unknown_tool(tmp_path):
    registry = AgentToolRegistry([])
    call = LLMToolCall("call-1", "missing", {})

    result = registry.invoke(call, make_context(tmp_path))

    assert result.is_error
    assert json.loads(result.content)["error"]["code"] == "unknown-tool"
    assert not registry.is_concurrency_safe(call, make_context(tmp_path))


def test_registry_preserves_expected_business_error(tmp_path):
    registry = AgentToolRegistry([BusinessErrorTool()])
    call = LLMToolCall("call-1", "business", {})

    result = registry.invoke(call, make_context(tmp_path))

    payload = json.loads(result.content)
    assert result.is_error
    assert payload["error"] == {
        "code": "invalid-input",
        "message": "bad value",
    }


def test_registry_contains_unexpected_exceptions_and_classifier_fails_closed(
    tmp_path,
):
    registry = AgentToolRegistry([BrokenTool()])
    call = LLMToolCall("call-1", "broken", {})

    result = registry.invoke(call, make_context(tmp_path))

    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"]["code"] == "tool-error"
    assert "RuntimeError: boom" in payload["error"]["message"]
    assert not registry.is_concurrency_safe(call, make_context(tmp_path))


def test_registry_rejects_duplicate_names_and_incomplete_tools():
    with pytest.raises(ValueError, match="unique"):
        AgentToolRegistry([EchoTool(), EchoTool()])

    class MissingClassifier:
        definition = EchoTool.definition

        def invoke(self, call, ctx):
            return make_tool_result(call.call_id, {"ok": True})

    with pytest.raises(TypeError, match="AgentTool"):
        AgentToolRegistry([MissingClassifier()])


def test_context_requires_live_workspace_and_positive_deadline(tmp_path):
    ctx = make_context(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        AgentToolContext(
            workdir=ctx.workdir,
            parent=ctx.parent,
            operator=ctx.operator,
            preflight=ctx.preflight,
            remaining_timeout_s=0,
        )
