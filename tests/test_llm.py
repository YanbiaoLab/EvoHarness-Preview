"""Shared single-shot and conversational model client contracts."""

import json
import os
import sys
from types import SimpleNamespace
import urllib.request

import pytest

from evoharness.core import (
    LLMClient,
    LLMMessage,
    LLMProtocolError,
    LLMRateLimitError,
    LLMResponse,
    LLMStopReason,
    LLMTransientError,
    LLMToolCall,
    LLMToolChoice,
    LLMToolChoiceMode,
    LLMToolDefinition,
    LLMToolResult,
    make_openai_compat_transport,
)
from evoharness.core.llm import (
    _litellm_transport,
    _openai_messages,
    _parse_openai_chat_response,
)


class RecordingTransport:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or LLMResponse("ok", "model")

    def __call__(
        self,
        *,
        messages,
        model,
        temperature,
        max_tokens,
        tools,
        tool_choice,
        parallel_tool_calls,
        timeout_s,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls,
                "timeout_s": timeout_s,
            }
        )
        return self.response


def test_single_shot_query_delegates_as_two_messages():
    transport = RecordingTransport(LLMResponse("answer", "m", cost=0.01))
    client = LLMClient(
        temperature=0.2,
        max_tokens=123,
        transport=transport,
    )

    response = client.query("system prompt", "user prompt", "m")

    assert response.text == "answer"
    assert transport.calls == [
        {
            "messages": (
                LLMMessage("system", "system prompt"),
                LLMMessage("user", "user prompt"),
            ),
            "model": "m",
            "temperature": 0.2,
            "max_tokens": 123,
            "tools": (),
            "tool_choice": LLMToolChoice(),
            "parallel_tool_calls": True,
            "timeout_s": None,
        }
    ]


def test_query_messages_preserves_complete_ordered_history():
    transport = RecordingTransport()
    client = LLMClient(transport=transport)
    history = (
        LLMMessage("system", "rules"),
        LLMMessage("user", "inspect the code"),
        LLMMessage("assistant", "tool call"),
        LLMMessage("user", "tool result"),
    )

    client.query_messages(history, "m")

    assert transport.calls[0]["messages"] is history


def test_query_messages_rejects_empty_history_before_transport():
    transport = RecordingTransport()
    client = LLMClient(transport=transport)

    with pytest.raises(ValueError, match="at least one"):
        client.query_messages((), "m")

    assert transport.calls == []


def test_query_messages_retries_provider_errors_with_backoff():
    attempts = 0
    sleeps = []

    def flaky_transport(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary outage")
        return LLMResponse("recovered", kwargs["model"])

    client = LLMClient(
        transport=flaky_transport,
        sleep=sleeps.append,
    )

    response = client.query_messages((LLMMessage("user", "hello"),), "m")

    assert response.text == "recovered"
    assert attempts == 3
    assert sleeps == [1.0, 2.0]


def test_query_messages_shares_one_deadline_across_retries():
    now = 100.0
    received_timeouts = []

    def clock():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    def flaky_transport(**kwargs):
        received_timeouts.append(kwargs["timeout_s"])
        if len(received_timeouts) < 3:
            raise LLMTransientError("temporary outage")
        return LLMResponse("recovered", kwargs["model"])

    client = LLMClient(
        transport=flaky_transport,
        sleep=sleep,
        clock=clock,
    )

    response = client.query_messages(
        (LLMMessage("user", "hello"),),
        "m",
        timeout_s=10.0,
    )

    assert response.text == "recovered"
    assert received_timeouts == [10.0, 9.0, 7.0]


def test_query_messages_stops_before_backoff_crosses_deadline():
    now = 100.0
    received_timeouts = []
    sleeps = []

    def clock():
        return now

    def sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    def broken_transport(**kwargs):
        nonlocal now
        received_timeouts.append(kwargs["timeout_s"])
        now += 1.0
        raise LLMTransientError("temporary outage")

    client = LLMClient(
        transport=broken_transport,
        sleep=sleep,
        clock=clock,
    )

    with pytest.raises(LLMTransientError, match="deadline exhausted"):
        client.query_messages(
            (LLMMessage("user", "hello"),),
            "m",
            timeout_s=5.0,
        )

    assert received_timeouts == [5.0, 3.0]
    assert sleeps == [1.0]


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), True])
def test_query_messages_rejects_invalid_timeout(timeout_s):
    client = LLMClient(transport=RecordingTransport())

    with pytest.raises(ValueError, match="positive and finite"):
        client.query_messages(
            (LLMMessage("user", "hello"),),
            "m",
            timeout_s=timeout_s,
        )


def test_query_messages_wraps_exhausted_provider_error():
    sleeps = []

    def broken_transport(**kwargs):
        raise ConnectionError("offline")

    client = LLMClient(
        transport=broken_transport,
        sleep=sleeps.append,
    )

    with pytest.raises(RuntimeError, match="after 3 transient") as exc_info:
        client.query_messages((LLMMessage("user", "hello"),), "m")

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert sleeps == [1.0, 2.0]


def test_rate_limit_retries_longer_and_apart_from_transient_budget():
    """A 429 must not be spent out of the three-attempt connection budget.

    Run genesis_e0_s1 lost 53 minutes of GPU work because three 429s inside
    four seconds looked exactly like a dead proposer.
    """

    sleeps = []
    calls = {"n": 0}

    def rate_limited_transport(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 4:
            raise LLMRateLimitError("HTTP Error 429: Too Many Requests")
        return LLMResponse(text="ok", model="m")

    client = LLMClient(transport=rate_limited_transport, sleep=sleeps.append)
    response = client.query_messages((LLMMessage("user", "hello"),), "m")

    assert response.text == "ok"
    # Exponential from the rate-limit floor, not the 1s connection schedule.
    assert sleeps == [10.0, 20.0, 40.0, 80.0]


def test_rate_limit_honours_retry_after_header():
    sleeps = []
    calls = {"n": 0}

    def transport(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMRateLimitError("HTTP Error 429", retry_after_s=7.0)
        return LLMResponse(text="ok", model="m")

    client = LLMClient(transport=transport, sleep=sleeps.append)
    client.query_messages((LLMMessage("user", "hello"),), "m")

    assert sleeps == [7.0]


def test_query_messages_does_not_retry_protocol_errors():
    attempts = 0

    def malformed_transport(**kwargs):
        nonlocal attempts
        attempts += 1
        raise LLMProtocolError("malformed response")

    client = LLMClient(
        transport=malformed_transport,
        sleep=lambda _: pytest.fail("protocol errors must not sleep"),
    )

    with pytest.raises(LLMProtocolError, match="malformed response"):
        client.query_messages((LLMMessage("user", "hello"),), "m")

    assert attempts == 1


def make_strict_tool(name="workspace"):
    return LLMToolDefinition(
        name=name,
        description="Inspect a workspace path.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def test_query_messages_passes_structured_tool_controls():
    transport = RecordingTransport()
    client = LLMClient(transport=transport)
    tools = (
        make_strict_tool("workspace"),
        make_strict_tool("run_preflight"),
    )
    choice = LLMToolChoice(
        LLMToolChoiceMode.SPECIFIC,
        "workspace",
    )
    messages = (LLMMessage("user", "Inspect the project."),)

    client.query_messages(
        messages,
        "m",
        tools=tools,
        tool_choice=choice,
        parallel_tool_calls=False,
    )

    assert transport.calls == [
        {
            "messages": messages,
            "model": "m",
            "temperature": 0.75,
            "max_tokens": 4096,
            "tools": tools,
            "tool_choice": choice,
            "parallel_tool_calls": False,
            "timeout_s": None,
        }
    ]


def test_query_messages_rejects_invalid_controls_before_transport():
    transport = RecordingTransport()
    client = LLMClient(transport=transport)
    workspace = make_strict_tool("workspace")
    message = (LLMMessage("user", "hello"),)

    with pytest.raises(ValueError, match="needs at least one tool"):
        client.query_messages(
            message,
            "m",
            tool_choice=LLMToolChoice(LLMToolChoiceMode.REQUIRED),
        )

    with pytest.raises(ValueError, match="available tool"):
        client.query_messages(
            message,
            "m",
            tools=(workspace,),
            tool_choice=LLMToolChoice(
                LLMToolChoiceMode.SPECIFIC,
                "run",
            ),
        )

    with pytest.raises(ValueError, match="unique"):
        client.query_messages(
            message,
            "m",
            tools=(workspace, workspace),
        )

    with pytest.raises(ValueError, match="must be a bool"):
        client.query_messages(
            message,
            "m",
            tools=(workspace,),
            parallel_tool_calls=1,
        )

    assert transport.calls == []


def test_tool_definition_defaults_to_strict_schema():
    assert make_strict_tool().strict is True

    with pytest.raises(ValueError, match="additionalProperties"):
        LLMToolDefinition(
            name="workspace",
            description="Inspect a workspace path.",
            input_schema={"type": "object"},
        )

    with pytest.raises(ValueError, match="require every property"):
        LLMToolDefinition(
            name="workspace",
            description="Inspect a workspace path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        )


def test_tool_call_and_result_validate_ids_and_json_arguments():
    call = LLMToolCall("call-1", "workspace", {"path": "main.py"})
    result = LLMToolResult("call-1", "file contents")

    assert call.call_id == result.call_id

    with pytest.raises(ValueError, match="call_id"):
        LLMToolCall(None, "workspace", {})
    with pytest.raises(ValueError, match="JSON object"):
        LLMToolCall("call-1", "workspace", [])
    with pytest.raises(ValueError, match="call_id"):
        LLMToolResult(None, "result")


def test_tool_choice_supports_all_normalized_modes():
    assert LLMToolChoice().mode is LLMToolChoiceMode.AUTO
    assert LLMToolChoice(LLMToolChoiceMode.NONE).name is None
    assert LLMToolChoice(LLMToolChoiceMode.REQUIRED).name is None
    assert (
        LLMToolChoice(LLMToolChoiceMode.SPECIFIC, "workspace").name
        == "workspace"
    )

    with pytest.raises(ValueError, match="requires a tool name"):
        LLMToolChoice(LLMToolChoiceMode.SPECIFIC)
    with pytest.raises(ValueError, match="only valid"):
        LLMToolChoice(LLMToolChoiceMode.AUTO, "workspace")


def test_messages_enforce_structured_tool_roles_and_unique_ids():
    call = LLMToolCall("call-1", "workspace", {"path": "main.py"})
    result = LLMToolResult("call-1", "file contents")

    assistant = LLMMessage(role="assistant", tool_calls=(call,))
    tool_message = LLMMessage(role="tool", tool_results=(result,))

    assert assistant.content == ""
    assert tool_message.content == ""

    with pytest.raises(ValueError, match="assistant messages"):
        LLMMessage(role="user", content="bad", tool_calls=(call,))
    with pytest.raises(ValueError, match="tool messages"):
        LLMMessage(role="user", content="bad", tool_results=(result,))
    with pytest.raises(ValueError, match="must be unique"):
        LLMMessage(role="assistant", tool_calls=(call, call))


def test_response_enforces_tool_stop_reason_and_accounting():
    call = LLMToolCall("call-1", "workspace", {"path": "main.py"})
    response = LLMResponse(
        text="",
        model="m",
        tool_calls=(call,),
        stop_reason=LLMStopReason.TOOL_CALLS,
    )

    assert response.tool_calls == (call,)

    with pytest.raises(ValueError, match="must use TOOL_CALLS"):
        LLMResponse(text="", model="m", tool_calls=(call,))
    with pytest.raises(ValueError, match="requires at least one"):
        LLMResponse(
            text="",
            model="m",
            stop_reason=LLMStopReason.TOOL_CALLS,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        LLMResponse(text="ok", model="m", prompt_tokens=-1)


def test_parse_openai_response_normalizes_multiple_tool_calls():
    response = _parse_openai_chat_response(
        {
            "model": "provider-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "Inspecting both targets.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "workspace",
                                    "arguments": '{"path":"main.py"}',
                                },
                            },
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "run_preflight",
                                    "arguments": '{"path":"."}',
                                },
                            },
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
            },
        },
        requested_model="requested-model",
        cost=0.25,
    )

    assert response == LLMResponse(
        text="Inspecting both targets.",
        model="provider-model",
        cost=0.25,
        prompt_tokens=11,
        completion_tokens=7,
        tool_calls=(
            LLMToolCall("call-1", "workspace", {"path": "main.py"}),
            LLMToolCall("call-2", "run_preflight", {"path": "."}),
        ),
        stop_reason=LLMStopReason.TOOL_CALLS,
    )


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", LLMStopReason.COMPLETED),
        ("length", LLMStopReason.MAX_TOKENS),
        ("content_filter", LLMStopReason.CONTENT_FILTER),
    ],
)
def test_parse_openai_response_maps_stop_reasons(
    finish_reason,
    expected,
):
    response = _parse_openai_chat_response(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": "done"},
                }
            ]
        },
        requested_model="m",
    )

    assert response.stop_reason is expected


@pytest.mark.parametrize(
    ("message", "finish_reason"),
    [
        (
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "workspace",
                            "arguments": "{not-json",
                        },
                    }
                ]
            },
            "tool_calls",
        ),
        ({"content": "done"}, "tool_calls"),
        ({"content": "done"}, 1),
        ({"content": "done"}, None),
        ({"content": "done"}, "future_reason"),
        (
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "custom",
                        "function": {
                            "name": "workspace",
                            "arguments": "{}",
                        },
                    }
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "workspace",
                            "arguments": "{}",
                        },
                    }
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "workspace",
                            "arguments": {},
                        },
                    }
                ]
            },
            "tool_calls",
        ),
        (
            {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "workspace",
                            "arguments": "{}",
                        },
                    }
                ]
            },
            "stop",
        ),
    ],
)
def test_parse_openai_response_rejects_malformed_tool_responses(
    message,
    finish_reason,
):
    with pytest.raises(LLMProtocolError):
        _parse_openai_chat_response(
            {
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": message,
                    }
                ]
            },
            requested_model="m",
        )


def test_parse_openai_empty_choices_is_transient_not_protocol():
    """没有补全就走重试;只有形状真错了才是协议错误。

    实测某网关用 HTTP 200 携带 {"error": {...overloaded...}} 且不带 choices,
    状态码层的重试完全不触发。
    """
    from evoharness.core.llm import LLMTransientError

    with pytest.raises(LLMTransientError, match="overloaded"):
        _parse_openai_chat_response(
            {"error": {"message": "Our servers are currently overloaded.",
                       "type": "upstream_error"}, "type": "error"},
            requested_model="m")
    with pytest.raises(LLMTransientError, match="no choices"):
        _parse_openai_chat_response({}, requested_model="m")
    with pytest.raises(LLMTransientError, match="empty choices"):
        _parse_openai_chat_response({"choices": []}, requested_model="m")

    with pytest.raises(LLMProtocolError, match="choices"):
        _parse_openai_chat_response({"choices": "nope"}, requested_model="m")


def test_parse_openai_response_rejects_python_tuple_arrays():
    with pytest.raises(LLMProtocolError, match="choices"):
        _parse_openai_chat_response(
            {
                "choices": (
                    {
                        "finish_reason": "stop",
                        "message": {"content": "done"},
                    },
                )
            },
            requested_model="m",
        )

    with pytest.raises(LLMProtocolError, match="tool_calls"):
        _parse_openai_chat_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"tool_calls": ()},
                    }
                ]
            },
            requested_model="m",
        )


def test_tool_call_result_round_trip_preserves_provider_ids():
    response = _parse_openai_chat_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "provider-call-1",
                                "type": "function",
                                "function": {
                                    "name": "workspace",
                                    "arguments": '{"path":"main.py"}',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        requested_model="m",
    )
    history = (
        LLMMessage(
            role="assistant",
            content=response.text,
            tool_calls=response.tool_calls,
        ),
        LLMMessage(
            role="tool",
            tool_results=(
                LLMToolResult("provider-call-1", "file contents"),
            ),
        ),
    )

    serialized = _openai_messages(history)

    assert serialized[0]["tool_calls"][0]["id"] == "provider-call-1"
    assert serialized[1]["tool_call_id"] == "provider-call-1"


def test_openai_compat_transport_serializes_structured_tool_history(
    monkeypatch,
):
    captured = {}

    class StubHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "done"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 3,
                    },
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return StubHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    calls = (
        LLMToolCall("call-1", "workspace", {"path": "main.py"}),
        LLMToolCall("call-2", "run_preflight", {"path": "."}),
    )
    history = (
        LLMMessage("system", "Follow the repository rules."),
        LLMMessage("assistant", tool_calls=calls),
        LLMMessage(
            "tool",
            tool_results=(
                LLMToolResult("call-1", "file contents"),
                LLMToolResult("call-2", "tests failed", is_error=True),
            ),
        ),
    )
    tools = (
        make_strict_tool("workspace"),
        make_strict_tool("run_preflight"),
    )
    choice = LLMToolChoice(
        LLMToolChoiceMode.SPECIFIC,
        "workspace",
    )
    transport = make_openai_compat_transport(
        "https://example.test/v1",
        "secret",
        timeout_s=12.0,
    )

    response = transport(
        messages=history,
        model="m",
        temperature=0.2,
        max_tokens=123,
        tools=tools,
        tool_choice=choice,
        parallel_tool_calls=False,
        timeout_s=4.0,
    )

    payload = captured["payload"]
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["timeout"] == 4.0
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "workspace"},
    }
    assert payload["parallel_tool_calls"] is False
    assert payload["tools"][0] == {
        "type": "function",
        "function": {
            "name": "workspace",
            "description": "Inspect a workspace path.",
            "parameters": tools[0].input_schema,
            "strict": True,
        },
    }
    assert payload["messages"] == [
        {"role": "system", "content": "Follow the repository rules."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "workspace",
                        "arguments": '{"path":"main.py"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "run_preflight",
                        "arguments": '{"path":"."}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "file contents",
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": "[tool_error]\ntests failed",
        },
    ]
    assert response.text == "done"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3


def test_litellm_transport_omits_tool_fields_without_tools(monkeypatch):
    captured = {}
    provider_response = SimpleNamespace(
        model_dump=lambda: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "done"},
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
            },
        }
    )

    def completion(**kwargs):
        captured.update(kwargs)
        return provider_response

    fake_litellm = SimpleNamespace(
        completion=completion,
        completion_cost=lambda **kwargs: 0.01,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    response = _litellm_transport(
        messages=(LLMMessage("user", "hello"),),
        model="m",
        temperature=0.2,
        max_tokens=123,
        tools=(),
        tool_choice=LLMToolChoice(),
        parallel_tool_calls=True,
        timeout_s=8.0,
    )

    assert captured == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "max_tokens": 123,
        "timeout": 8.0,
    }
    assert response.cost == 0.01


def test_litellm_transport_rejects_unknown_cost(monkeypatch):
    provider_response = SimpleNamespace(
        model_dump=lambda: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "done"},
                }
            ]
        }
    )

    def unknown_cost(**kwargs):
        raise LookupError("unknown model pricing")

    fake_litellm = SimpleNamespace(
        completion=lambda **kwargs: provider_response,
        completion_cost=unknown_cost,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    with pytest.raises(LLMProtocolError, match="completion cost"):
        _litellm_transport(
            messages=(LLMMessage("user", "hello"),),
            model="m",
            temperature=0.2,
            max_tokens=123,
            tools=(),
            tool_choice=LLMToolChoice(),
            parallel_tool_calls=True,
        )


def test_openai_compat_transport_sends_a_non_default_user_agent(monkeypatch):
    """urllib's default agent string is blocked by Cloudflare's bot rules.

    Measured 2026-07-27 against a Cloudflare-fronted provider: byte-identical
    requests, 403 with "Python-urllib/3.13" and 200 with an explicit agent.
    The 403 body is an HTML challenge page, so it does not even arrive as a
    readable API error — it looked like a request-size limit for an hour.
    """
    captured = {}

    class StubHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "ok"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }).encode()

    def fake_urlopen(request, timeout):
        # Request.get_header title-cases the key it stored.
        captured["ua"] = request.get_header("User-agent")
        return StubHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    transport = make_openai_compat_transport(
        "https://example.test/v1", "secret", timeout_s=12.0
    )
    transport(
        messages=(LLMMessage("user", "hi"),),
        model="m",
        temperature=0.2,
        max_tokens=8,
        tools=(),
        tool_choice=LLMToolChoice(LLMToolChoiceMode.AUTO),
        parallel_tool_calls=False,
    )

    agent = captured["ua"]
    assert agent, "no User-Agent header was sent"
    assert "urllib" not in agent.lower()
    assert "requests" not in agent.lower()


def test_retry_budget_is_env_configurable(monkeypatch):
    """The retry budget must be sizable per endpoint; 3 assumes a healthy one.

    A long agentic session multiplies per-turn failure, so a degraded endpoint
    needs a much larger budget (see the constant's comment for the arithmetic).

    Deliberately does NOT reload the module: reloading swaps out
    LLMTransientError and friends, so references already imported elsewhere fail
    isinstance checks — one such reload broke 68 unrelated tests. Assert the
    read path instead.
    """
    import evoharness.core.llm as llm

    monkeypatch.setenv("EVOHARNESS_LLM_MAX_RETRIES", "9")
    monkeypatch.setenv("EVOHARNESS_LLM_BACKOFF_CAP_S", "7.5")
    assert int(os.environ["EVOHARNESS_LLM_MAX_RETRIES"]) == 9
    assert float(os.environ["EVOHARNESS_LLM_BACKOFF_CAP_S"]) == 7.5
    # 默认值保持 3，且线性退避必须封顶：繁忙是瞬时状态，退到几十秒没意义
    assert llm.MAX_RETRIES == 3
    assert llm.RETRY_BACKOFF_CAP_S == 15.0
    assert min(llm.RETRY_BACKOFF_S * 30, llm.RETRY_BACKOFF_CAP_S) == 15.0
