"""Concurrency and ordering tests for the native tool scheduler."""

import json
import threading

from evoharness.core import (
    Candidate,
    LLMToolCall,
    LLMToolDefinition,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.core.agent import (
    AgentToolContext,
    AgentToolRegistry,
    make_tool_result,
)
from evoharness.core.agent.runtime import _ToolScheduler


class TrackingTool:
    definition = LLMToolDefinition(
        name="track",
        description="Track safe and exclusive execution.",
        input_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "safe": {"type": "boolean"},
                "wait": {"type": "boolean"},
            },
            "required": ["label", "safe", "wait"],
            "additionalProperties": False,
        },
    )

    def __init__(self):
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2, timeout=2)
        self.active_safe = 0
        self.max_active_safe = 0
        self.events = []

    def is_concurrency_safe(self, call, ctx):
        return call.arguments["safe"]

    def invoke(self, call, ctx):
        label = call.arguments["label"]
        if call.arguments["safe"]:
            with self.lock:
                self.active_safe += 1
                self.max_active_safe = max(
                    self.max_active_safe,
                    self.active_safe,
                )
                self.events.append(("safe-start", label))

            if call.arguments["wait"]:
                self.barrier.wait()

            with self.lock:
                self.events.append(("safe-end", label))
                self.active_safe -= 1
        else:
            with self.lock:
                assert self.active_safe == 0
                self.events.append(("unsafe", label))

        return make_tool_result(
            call.call_id,
            {"ok": True, "label": label},
        )


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value


class AdvancingUnsafeTool:
    definition = LLMToolDefinition(
        name="advance",
        description="Advance a fake clock.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(self, clock):
        self.clock = clock

    def is_concurrency_safe(self, call, ctx):
        return False

    def invoke(self, call, ctx):
        self.clock.value += 1
        return make_tool_result(call.call_id, {"ok": True})


def make_context(tmp_path, remaining_timeout_s=10.0):
    parent = Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )
    return AgentToolContext(
        workdir=tmp_path,
        parent=parent,
        operator="rewrite",
        preflight=ProposalPreflight(PreflightPipeline()),
        remaining_timeout_s=remaining_timeout_s,
    )


def test_safe_batches_overlap_unsafe_is_exclusive_and_results_keep_order(
    tmp_path,
):
    tool = TrackingTool()
    scheduler = _ToolScheduler(
        registry=AgentToolRegistry((tool,)),
        max_workers=2,
        clock=lambda: 0.0,
    )
    calls = (
        LLMToolCall(
            "call-a",
            "track",
            {"label": "a", "safe": True, "wait": True},
        ),
        LLMToolCall(
            "call-b",
            "track",
            {"label": "b", "safe": True, "wait": True},
        ),
        LLMToolCall(
            "call-c",
            "track",
            {"label": "c", "safe": False, "wait": False},
        ),
        LLMToolCall(
            "call-d",
            "track",
            {"label": "d", "safe": True, "wait": False},
        ),
    )

    scheduled = scheduler.execute(
        calls,
        context_factory=lambda remaining_s: make_context(
            tmp_path,
            remaining_timeout_s=remaining_s,
        ),
        deadline=10.0,
    )

    assert scheduled.started_count == 4
    assert not scheduled.timed_out
    assert tool.max_active_safe == 2
    assert [
        json.loads(result.content)["label"]
        for result in scheduled.results
    ] == ["a", "b", "c", "d"]
    unsafe_index = tool.events.index(("unsafe", "c"))
    assert ("safe-end", "a") in tool.events[:unsafe_index]
    assert ("safe-end", "b") in tool.events[:unsafe_index]
    assert tool.events[-1] == ("safe-end", "d")


def test_deadline_adds_ordered_synthetic_results_for_unstarted_calls(
    tmp_path,
):
    clock = FakeClock()
    tool = AdvancingUnsafeTool(clock)
    scheduler = _ToolScheduler(
        registry=AgentToolRegistry((tool,)),
        max_workers=2,
        clock=clock,
    )
    calls = (
        LLMToolCall("call-1", "advance", {}),
        LLMToolCall("call-2", "advance", {}),
    )

    scheduled = scheduler.execute(
        calls,
        context_factory=lambda remaining_s: make_context(
            tmp_path,
            remaining_timeout_s=remaining_s,
        ),
        deadline=1.0,
    )

    assert scheduled.started_count == 1
    assert scheduled.timed_out
    assert [result.call_id for result in scheduled.results] == [
        "call-1",
        "call-2",
    ]
    assert not scheduled.results[0].is_error
    timeout_payload = json.loads(scheduled.results[1].content)
    assert timeout_payload["error"]["code"] == "timeout"
