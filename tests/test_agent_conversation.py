"""Conversational path tests for NativeToolAgentBackend."""

import json
from dataclasses import replace
from threading import get_ident

import pytest

from evoharness.evocore import (
    AgentBackend,
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionRequest,
    AgentTermination,
    Candidate,
    LLMClient,
    LLMMessage,
    LLMProtocolError,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    LLMToolDefinition,
    NativeToolAgentBackend,
    PreflightIssue,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.evocore.agent import AgentToolRegistry, make_tool_result


class QueueTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingEstimator:
    def __init__(self, tokens=10):
        self.tokens = tokens
        self.calls = []

    def __call__(self, messages, tools):
        self.calls.append((messages, tools))
        return self.tokens


class ContentEstimator:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, tools):
        self.calls.append((messages, tools))
        return sum(
            len(message.content)
            + sum(len(result.content) for result in message.tool_results)
            for message in messages
        )


class RecordingSink:
    def __init__(self, *, fail_on=None):
        self.events = []
        self.thread_ids = []
        self.fail_on = fail_on

    def emit(self, event):
        self.thread_ids.append(get_ident())
        if event.kind is self.fail_on:
            raise OSError("sink unavailable")
        self.events.append(event)


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RecordingTool:
    definition = LLMToolDefinition(
        name="record",
        description="Record one label.",
        input_schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, clock=None, advance_s=0.0):
        self.labels = []
        self.clock = clock
        self.advance_s = advance_s

    def is_concurrency_safe(self, call, ctx):
        return False

    def invoke(self, call, ctx):
        label = call.arguments["label"]
        self.labels.append(label)
        if self.clock is not None:
            self.clock.advance(self.advance_s)
        return make_tool_result(
            call.call_id,
            {"ok": True, "label": label},
        )


class LargeResultTool(RecordingTool):
    def invoke(self, call, ctx):
        label = call.arguments["label"]
        self.labels.append(label)
        return make_tool_result(
            call.call_id,
            {"ok": True, "label": label, "payload": label * 120},
        )


def make_parent():
    return Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def make_request(tmp_path, **changes):
    values = {
        "system": "Follow the repository rules.",
        "user": "Improve the candidate.",
        "parent": make_parent(),
        "operator": "rewrite",
        "workdir": tmp_path,
        "limits": AgentSessionLimits(timeout_s=30),
        "preflight": ProposalPreflight(PreflightPipeline()),
    }
    values.update(changes)
    return AgentSessionRequest(**values)


def make_backend(
    transport,
    *,
    estimator=None,
    clock=None,
    registry=None,
    max_input_tokens=1_000,
    recent_tool_results_to_keep=8,
):
    clock = clock or FakeClock()
    estimator = estimator or RecordingEstimator()
    return NativeToolAgentBackend(
        client=LLMClient(
            transport=transport,
            sleep=lambda _: None,
            clock=clock,
        ),
        model="requested-model",
        registry=registry or AgentToolRegistry(()),
        max_input_tokens=max_input_tokens,
        token_estimator=estimator,
        clock=clock,
        session_id_factory=lambda: "session-1",
        recent_tool_results_to_keep=recent_tool_results_to_keep,
    )


def test_conversational_run_completes_and_accounts_one_response(tmp_path):
    transport = QueueTransport(
        LLMResponse(
            "Implemented the requested change.",
            "provider-model",
            cost=0.25,
            prompt_tokens=11,
            completion_tokens=7,
        )
    )
    estimator = RecordingEstimator()
    backend = make_backend(transport, estimator=estimator)
    assert isinstance(backend, AgentBackend)

    initial = make_request(tmp_path)
    result = backend.run(initial)

    assert result.termination is AgentTermination.COMPLETED
    assert result.session_id == "session-1"
    assert result.final_message == "Implemented the requested change."
    assert result.model == "provider-model"
    assert result.turns == 1
    assert result.tool_calls == 0
    assert result.cost_usd == 0.25
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert [event.kind for event in result.events] == [
        AgentEventKind.SESSION_START,
        AgentEventKind.MODEL_RESPONSE,
        AgentEventKind.TERMINATION,
    ]
    assert [event.sequence for event in result.events] == [0, 1, 2]
    assert result.events[0].data["system"] == initial.system
    assert result.events[0].data["user"] == initial.user
    assert transport.calls[0]["timeout_s"] == 30
    assert transport.calls[0]["tools"] == ()
    assert estimator.calls[0][0] == (
        LLMMessage("system", "Follow the repository rules."),
        LLMMessage("user", "Improve the candidate."),
    )


def test_resume_preserves_history_and_reports_per_run_delta(tmp_path):
    transport = QueueTransport(
        LLMResponse(
            "First pass.",
            "provider-model",
            cost=0.1,
            prompt_tokens=5,
            completion_tokens=2,
        ),
        LLMResponse(
            "Repair complete.",
            "provider-model",
            cost=0.2,
            prompt_tokens=9,
            completion_tokens=3,
        ),
    )
    backend = make_backend(transport)
    initial = make_request(tmp_path)
    first = backend.run(initial)
    issue = PreflightIssue("compile", "syntax-error", "invalid syntax")

    second = backend.run(
        replace(
            initial,
            session_id=first.session_id,
            feedback=(issue,),
        )
    )

    assert second.termination is AgentTermination.COMPLETED
    assert second.turns == 1
    assert second.cost_usd == pytest.approx(0.2)
    assert second.prompt_tokens == 9
    assert second.completion_tokens == 3
    history = transport.calls[1]["messages"]
    assert [message.role for message in history] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert history[2].content == "First pass."
    assert "syntax-error" in history[3].content
    assert [event.sequence for event in second.events] == [3, 4, 5]
    assert {event.session_id for event in second.events} == {"session-1"}
    assert {event.round_index for event in second.events} == {1}
    assert json.loads(second.events[0].data["feedback"])["issues"][0][
        "code"
    ] == "syntax-error"


def test_context_limit_stops_before_transport(tmp_path):
    transport = QueueTransport(LLMResponse("unused", "model"))
    backend = make_backend(
        transport,
        estimator=RecordingEstimator(tokens=1_001),
    )

    result = backend.run(make_request(tmp_path))

    assert result.termination is AgentTermination.CONTEXT_LIMIT
    assert result.turns == 0
    assert transport.calls == []
    assert [event.kind for event in result.events] == [
        AgentEventKind.SESSION_START,
        AgentEventKind.TERMINATION,
    ]


def test_zero_cost_budget_stops_before_estimation_and_transport(tmp_path):
    transport = QueueTransport(LLMResponse("unused", "model"))
    estimator = RecordingEstimator()
    backend = make_backend(transport, estimator=estimator)

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(max_cost_usd=0),
        )
    )

    assert result.termination is AgentTermination.COST_LIMIT
    assert result.turns == 0
    assert estimator.calls == []
    assert transport.calls == []


def test_invalid_token_estimate_is_normalized_as_backend_error(tmp_path):
    transport = QueueTransport(LLMResponse("unused", "model"))
    backend = make_backend(transport, estimator=lambda messages, tools: True)

    result = backend.run(make_request(tmp_path))

    assert result.termination is AgentTermination.BACKEND_ERROR
    assert result.turns == 0
    assert "token_estimator" in result.final_message
    assert transport.calls == []


def test_deadline_exhausted_during_admission_stops_before_transport(
    tmp_path,
):
    clock = FakeClock()
    transport = QueueTransport(LLMResponse("unused", "model"))

    def slow_estimator(messages, tools):
        clock.advance(6)
        return 10

    backend = make_backend(
        transport,
        estimator=slow_estimator,
        clock=clock,
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(timeout_s=5),
        )
    )

    assert result.termination is AgentTermination.TIMEOUT
    assert result.turns == 0
    assert transport.calls == []


@pytest.mark.parametrize(
    ("stop_reason", "termination"),
    [
        (LLMStopReason.REFUSAL, AgentTermination.REFUSAL),
        (LLMStopReason.CONTENT_FILTER, AgentTermination.CONTENT_FILTER),
    ],
)
def test_terminal_model_stop_reasons_are_normalized(
    tmp_path,
    stop_reason,
    termination,
):
    backend = make_backend(
        QueueTransport(
            LLMResponse("not available", "model", stop_reason=stop_reason)
        )
    )

    result = backend.run(make_request(tmp_path))

    assert result.termination is termination
    assert result.turns == 1


def test_protocol_error_is_normalized_without_a_model_turn(tmp_path):
    backend = make_backend(
        QueueTransport(LLMProtocolError("malformed response"))
    )

    result = backend.run(make_request(tmp_path))

    assert result.termination is AgentTermination.PROTOCOL_ERROR
    assert result.turns == 0
    assert "malformed response" in result.final_message


def test_empty_completed_response_is_a_protocol_error(tmp_path):
    backend = make_backend(QueueTransport(LLMResponse("", "model")))

    result = backend.run(make_request(tmp_path))
    state = backend._sessions.get(result.session_id)

    assert result.termination is AgentTermination.PROTOCOL_ERROR
    assert result.turns == 1
    assert [message.role for message in state.messages] == ["system", "user"]


def test_unknown_tool_result_is_paired_and_model_can_self_repair(tmp_path):
    call = LLMToolCall("call-1", "workspace_read", {"path": "main.py"})
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                tool_calls=(call,),
                stop_reason=LLMStopReason.TOOL_CALLS,
            ),
            LLMResponse("Recovered after the tool error.", "model"),
        )
    )

    result = backend.run(make_request(tmp_path))
    state = backend._sessions.get(result.session_id)

    assert result.termination is AgentTermination.COMPLETED
    assert result.turns == 2
    assert result.tool_calls == 1
    assert [message.role for message in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_result = state.messages[3].tool_results[0]
    assert tool_result.call_id == "call-1"
    assert tool_result.is_error
    assert "unknown-tool" in tool_result.content


def test_tool_result_returns_to_model_before_final_response(tmp_path):
    tool = RecordingTool()
    call = LLMToolCall("call-1", "record", {"label": "alpha"})
    transport = QueueTransport(
        LLMResponse(
            "",
            "model",
            tool_calls=(call,),
            stop_reason=LLMStopReason.TOOL_CALLS,
        ),
        LLMResponse("Finished after inspecting the tool result.", "model"),
    )
    backend = make_backend(
        transport,
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(make_request(tmp_path))

    assert result.termination is AgentTermination.COMPLETED
    assert result.turns == 2
    assert result.tool_calls == 1
    assert tool.labels == ["alpha"]
    second_history = transport.calls[1]["messages"]
    assert [message.role for message in second_history] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_history[2].tool_calls == (call,)
    assert second_history[3].tool_results[0].call_id == "call-1"
    assert [event.kind for event in result.events] == [
        AgentEventKind.SESSION_START,
        AgentEventKind.MODEL_RESPONSE,
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.MODEL_RESPONSE,
        AgentEventKind.TERMINATION,
    ]


def test_tool_budget_executes_prefix_and_pairs_rejected_tail(tmp_path):
    tool = RecordingTool()
    calls = tuple(
        LLMToolCall(f"call-{index}", "record", {"label": str(index)})
        for index in range(3)
    )
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                tool_calls=calls,
                stop_reason=LLMStopReason.TOOL_CALLS,
            )
        ),
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(max_tool_calls=2),
        )
    )
    state = backend._sessions.get(result.session_id)
    tool_results = state.messages[-1].tool_results

    assert result.termination is AgentTermination.TOOL_LIMIT
    assert result.turns == 1
    assert result.tool_calls == 2
    assert tool.labels == ["0", "1"]
    assert [item.call_id for item in tool_results] == [
        "call-0",
        "call-1",
        "call-2",
    ]
    assert not tool_results[0].is_error
    assert not tool_results[1].is_error
    assert "tool-limit" in tool_results[2].content


def test_zero_tool_budget_pairs_calls_without_executing(tmp_path):
    tool = RecordingTool()
    call = LLMToolCall(
        "call-1",
        "record",
        {"label": "blocked"},
    )
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                tool_calls=(call,),
                stop_reason=LLMStopReason.TOOL_CALLS,
            )
        ),
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(max_tool_calls=0),
        )
    )
    state = backend._sessions.get(result.session_id)

    assert result.termination is AgentTermination.TOOL_LIMIT
    assert result.tool_calls == 0
    assert tool.labels == []
    assert "tool-limit" in state.messages[-1].tool_results[0].content


def test_turn_limit_stops_after_paired_tool_exchange(tmp_path):
    tool = RecordingTool()
    call = LLMToolCall("call-1", "record", {"label": "alpha"})
    transport = QueueTransport(
        LLMResponse(
            "",
            "model",
            tool_calls=(call,),
            stop_reason=LLMStopReason.TOOL_CALLS,
        )
    )
    backend = make_backend(
        transport,
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(max_turns=1),
        )
    )

    assert result.termination is AgentTermination.TURN_LIMIT
    assert result.turns == 1
    assert result.tool_calls == 1
    assert len(transport.calls) == 1
    state = backend._sessions.get(result.session_id)
    assert [message.role for message in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


def test_cost_limit_pairs_tool_calls_without_starting_them(tmp_path):
    tool = RecordingTool()
    call = LLMToolCall("call-1", "record", {"label": "alpha"})
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                cost=0.5,
                tool_calls=(call,),
                stop_reason=LLMStopReason.TOOL_CALLS,
            )
        ),
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(max_cost_usd=0.5),
        )
    )
    state = backend._sessions.get(result.session_id)

    assert result.termination is AgentTermination.COST_LIMIT
    assert result.cost_usd == 0.5
    assert result.tool_calls == 0
    assert tool.labels == []
    assert "cost-limit" in state.messages[-1].tool_results[0].content


def test_tool_deadline_pairs_unstarted_tail_with_timeout(tmp_path):
    clock = FakeClock()
    tool = RecordingTool(clock=clock, advance_s=5.0)
    calls = (
        LLMToolCall("call-1", "record", {"label": "first"}),
        LLMToolCall("call-2", "record", {"label": "second"}),
    )
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                tool_calls=calls,
                stop_reason=LLMStopReason.TOOL_CALLS,
            )
        ),
        clock=clock,
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(
        make_request(
            tmp_path,
            limits=AgentSessionLimits(timeout_s=5),
        )
    )
    state = backend._sessions.get(result.session_id)
    tool_results = state.messages[-1].tool_results

    assert result.termination is AgentTermination.TIMEOUT
    assert result.tool_calls == 1
    assert tool.labels == ["first"]
    assert not tool_results[0].is_error
    assert "timeout" in tool_results[1].content


def test_release_removes_backend_session(tmp_path):
    backend = make_backend(QueueTransport(LLMResponse("done", "model")))
    result = backend.run(make_request(tmp_path))

    backend.release(result.session_id)

    with pytest.raises(ValueError, match="unknown agent session"):
        backend._sessions.get(result.session_id)


def test_context_compacts_only_old_tool_results_and_preserves_pairing(
    tmp_path,
):
    tool = LargeResultTool()
    calls = (
        LLMToolCall("call-old", "record", {"label": "old"}),
        LLMToolCall("call-new", "record", {"label": "new"}),
    )
    transport = QueueTransport(
        LLMResponse(
            "",
            "model",
            tool_calls=calls,
            stop_reason=LLMStopReason.TOOL_CALLS,
        ),
        LLMResponse("done", "model"),
    )
    backend = make_backend(
        transport,
        estimator=ContentEstimator(),
        registry=AgentToolRegistry((tool,)),
        max_input_tokens=650,
        recent_tool_results_to_keep=1,
    )

    result = backend.run(make_request(tmp_path))
    state = backend._sessions.get(result.session_id)
    assistant_message, tool_message = state.messages[2:4]

    assert result.termination is AgentTermination.COMPLETED
    assert assistant_message.tool_calls == calls
    assert [item.call_id for item in tool_message.tool_results] == [
        "call-old",
        "call-new",
    ]
    assert '"compacted":true' in tool_message.tool_results[0].content
    assert '"compacted":true' not in tool_message.tool_results[1].content
    compact_event = next(
        event
        for event in result.events
        if event.kind is AgentEventKind.CONTEXT_COMPACT
    )
    assert compact_event.data["call_ids"] == ("call-old",)
    assert compact_event.data["estimated_tokens_released"] > 0


def test_max_tokens_continues_once_and_preserves_partial_output(tmp_path):
    transport = QueueTransport(
        LLMResponse(
            "partial response",
            "model",
            completion_tokens=4,
            stop_reason=LLMStopReason.MAX_TOKENS,
        ),
        LLMResponse(
            "completed response",
            "model",
            completion_tokens=3,
        ),
    )
    backend = make_backend(transport)

    result = backend.run(make_request(tmp_path))
    state = backend._sessions.get(result.session_id)

    assert result.termination is AgentTermination.COMPLETED
    assert result.turns == 2
    assert result.completion_tokens == 7
    assert [message.role for message in state.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert state.messages[2].content == "partial response"
    assert "Continue exactly" in state.messages[3].content
    assert sum(
        event.kind is AgentEventKind.RECOVERY
        for event in result.events
    ) == 1


def test_second_max_tokens_stops_with_output_limit(tmp_path):
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "first partial",
                "model",
                stop_reason=LLMStopReason.MAX_TOKENS,
            ),
            LLMResponse(
                "second partial",
                "model",
                stop_reason=LLMStopReason.MAX_TOKENS,
            ),
        )
    )

    result = backend.run(make_request(tmp_path))

    assert result.termination is AgentTermination.OUTPUT_LIMIT
    assert result.final_message == "second partial"
    assert result.turns == 2
    assert sum(
        event.kind is AgentEventKind.RECOVERY
        for event in result.events
    ) == 1


def test_event_sink_receives_ordered_events_and_accounting(tmp_path):
    sink = RecordingSink()
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "done",
                "provider-model",
                cost=0.25,
                prompt_tokens=11,
                completion_tokens=7,
            )
        )
    )

    result = backend.run(make_request(tmp_path, event_sink=sink))
    termination = result.events[-1]

    assert tuple(sink.events) == result.events
    assert len(set(sink.thread_ids)) == 1
    assert [event.sequence for event in result.events] == list(
        range(len(result.events))
    )
    assert termination.kind is AgentEventKind.TERMINATION
    assert termination.data["cost_usd"] == result.cost_usd
    assert termination.data["turns"] == result.turns
    assert termination.data["tool_calls"] == result.tool_calls
    assert termination.data["prompt_tokens"] == result.prompt_tokens
    assert (
        termination.data["completion_tokens"]
        == result.completion_tokens
    )
    model_events = [
        event
        for event in result.events
        if event.kind is AgentEventKind.MODEL_RESPONSE
    ]
    assert sum(event.data["cost_usd"] for event in model_events) == (
        result.cost_usd
    )
    assert sum(event.data["prompt_tokens"] for event in model_events) == (
        result.prompt_tokens
    )
    assert sum(
        event.data["completion_tokens"] for event in model_events
    ) == result.completion_tokens


def test_successful_tool_events_keep_call_id_and_sink_order(tmp_path):
    sink = RecordingSink()
    tool = RecordingTool()
    call = LLMToolCall("call-1", "record", {"label": "alpha"})
    backend = make_backend(
        QueueTransport(
            LLMResponse(
                "",
                "model",
                tool_calls=(call,),
                stop_reason=LLMStopReason.TOOL_CALLS,
            ),
            LLMResponse("done", "model"),
        ),
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(make_request(tmp_path, event_sink=sink))
    tool_events = [
        event
        for event in result.events
        if event.kind
        in {AgentEventKind.TOOL_CALL, AgentEventKind.TOOL_RESULT}
    ]

    assert result.termination is AgentTermination.COMPLETED
    assert tuple(sink.events) == result.events
    assert len(set(sink.thread_ids)) == 1
    assert [event.kind for event in tool_events] == [
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
    ]
    assert [event.call_id for event in tool_events] == [
        "call-1",
        "call-1",
    ]
    assert sum(
        event.data["executed"]
        for event in result.events
        if event.kind is AgentEventKind.TOOL_RESULT
    ) == result.tool_calls
    assert all(
        event.data["admitted"]
        for event in result.events
        if event.kind is AgentEventKind.TOOL_CALL
    )


def test_event_sink_failure_stops_before_next_model_turn(tmp_path):
    sink = RecordingSink(fail_on=AgentEventKind.TOOL_RESULT)
    tool = RecordingTool()
    call = LLMToolCall("call-1", "record", {"label": "alpha"})
    transport = QueueTransport(
        LLMResponse(
            "",
            "model",
            tool_calls=(call,),
            stop_reason=LLMStopReason.TOOL_CALLS,
        ),
        LLMResponse("must not be requested", "model"),
    )
    backend = make_backend(
        transport,
        registry=AgentToolRegistry((tool,)),
    )

    result = backend.run(make_request(tmp_path, event_sink=sink))

    assert result.termination is AgentTermination.BACKEND_ERROR
    assert "event sink failed" in result.final_message
    assert len(transport.calls) == 1
    assert [event.kind for event in result.events] == [
        AgentEventKind.SESSION_START,
        AgentEventKind.MODEL_RESPONSE,
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.TERMINATION,
    ]
