"""OpenAI **Responses** 协议的构造与解析。

为什么要有第二条协议:有些网关只开 `/responses`。2026-08-30 换到 krill-ai 的
gpt-5.6-sol,`/chat/completions` 一律
`403 your token is not authorized for any channel serving this model`,
而同一个令牌、同一个模型走 `/responses` 直接 200。**协议不匹配和权限不足在线上
长得一模一样**,排查方向因此偏了几轮 —— 这也是为什么协议是显式配置、没有
「chat 失败就退回 responses」的自动探测:退错了不会报错。

下面每一条都对应一个会**静默**出错的差异:形状不对但仍然 200,分数上看不出来。
"""
from __future__ import annotations

import json

import pytest

from evoharness.core.llm import (
    LLMMessage,
    LLMProtocolError,
    LLMStopReason,
    LLMTransientError,
    LLMToolCall,
    LLMToolChoice,
    LLMToolChoiceMode,
    LLMToolCallFormatError,
    LLMToolDefinition,
    LLMToolResult,
    _parse_responses_response,
    _read_sse_within,
    _responses_request,
)

TOOL = LLMToolDefinition(
    name="calc",
    description="Evaluate an arithmetic expression",
    input_schema={
        "type": "object",
        "properties": {"expr": {"type": "string"}},
        "required": ["expr"],
        "additionalProperties": False,
    },
)


def build(messages, **kw):
    args = {
        "model": "gpt-5.6-sol",
        "temperature": 1.0,
        "max_tokens": 4096,
        "tools": (),
        "tool_choice": LLMToolChoice(LLMToolChoiceMode.NONE),
        "parallel_tool_calls": False,
    }
    args.update(kw)
    return _responses_request(messages=tuple(messages), **args)


def test_system_is_rewritten_to_developer():
    """这是这条协议上最危险的一处。

    网关**强制覆盖** `instructions`(实测:传 "reply with exactly: ok",回来的
    仍是它自己注入的 3.5k token Codex 人格),所以系统提示词只能进 `input`;
    而进了 `input` 的 `system` 在 Responses 的指令层级里压不过 `instructions`。
    同一条指令连发六次:`system` 只赢 1 次,`developer` 3/3 全赢。

    漏掉这一步不会报错 —— agent 的整份任务简报被稀释,每次调用照样 200,
    表现为「这个模型怎么不听话」。
    """
    payload = build([LLMMessage("system", "Follow the rules."),
                     LLMMessage("user", "Hi")])
    assert payload["input"] == [
        {"role": "developer", "content": "Follow the rules."},
        {"role": "user", "content": "Hi"},
    ]


def test_tool_calls_become_sibling_items_not_a_message_field():
    """助手的工具调用在 Responses 里是**与消息并列的项**,不是消息的字段。

    一条带 n 个调用的助手消息因此展开成 1+n 项。当成字段塞进去的话,provider
    收到的是一条没有工具调用的普通消息 —— 会话继续、不报错,只是工具从此
    再没被调用过。
    """
    call = LLMToolCall(call_id="call_1", name="calc", arguments={"expr": "19*23"})
    payload = build([
        LLMMessage("assistant", "Let me compute that.", tool_calls=(call,)),
    ])
    assert payload["input"] == [
        {"role": "assistant", "content": "Let me compute that."},
        {"type": "function_call", "call_id": "call_1", "name": "calc",
         "arguments": '{"expr": "19*23"}'},
    ]


def test_tool_results_become_function_call_output_items():
    """工具结果不是 `role="tool"` 的消息,而是靠 `call_id` 回指的独立项。"""
    payload = build([
        LLMMessage("tool", tool_results=(
            LLMToolResult(call_id="call_1", content="437"),
            LLMToolResult(call_id="call_2", content="12"),
        )),
    ])
    assert payload["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "437"},
        {"type": "function_call_output", "call_id": "call_2", "output": "12"},
    ]


def test_tool_schema_is_flat_and_budget_key_is_renamed():
    """工具 schema 没有 chat 那层 `function` 嵌套;预算键叫 `max_output_tokens`。

    预算键传错不会报错,只会**静默不限长**。
    """
    payload = build([LLMMessage("user", "Hi")], tools=(TOOL,),
                    tool_choice=LLMToolChoice(LLMToolChoiceMode.AUTO),
                    parallel_tool_calls=True)
    assert "max_tokens" not in payload
    assert payload["max_output_tokens"] == 4096
    assert payload["tools"] == [{
        "type": "function", "name": "calc",
        "description": "Evaluate an arithmetic expression",
        "parameters": TOOL.input_schema, "strict": True,
    }]
    assert payload["tool_choice"] == "auto"


def test_specific_tool_choice_is_flat_too():
    payload = build([LLMMessage("user", "Hi")], tools=(TOOL,),
                    tool_choice=LLMToolChoice(LLMToolChoiceMode.SPECIFIC, "calc"))
    assert payload["tool_choice"] == {"type": "function", "name": "calc"}


def test_parse_text_and_usage_including_cache():
    got = _parse_responses_response({
        "model": "gpt-5.6-sol",
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "Paris is 19°C."}]}],
        "usage": {"input_tokens": 3678, "output_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 2560}},
    }, requested_model="gpt-5.6-sol")
    assert got.text == "Paris is 19°C."
    assert (got.prompt_tokens, got.completion_tokens) == (3678, 15)
    # 缓存命中率是判断上下文压缩划不划算的唯一依据,键名两边不同,漏读就恒为 0。
    assert got.cached_prompt_tokens == 2560
    assert got.stop_reason is LLMStopReason.COMPLETED


def test_reasoning_items_are_dropped_not_an_error():
    """`reasoning` 项 LLMMessage 装不下,直接丢。

    实测:丢掉之后回灌不报错、第二回合正常作答(cap=200 时该模型会吐 4 个
    reasoning 项)。当成未知类型抛错的话,凡是真的想了一下的回合都会炸。
    """
    got = _parse_responses_response({
        "model": "m", "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }, requested_model="m")
    assert got.text == "ok"


def test_parse_function_calls_sets_tool_calls_stop_reason():
    got = _parse_responses_response({
        "model": "m", "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "Sure."}]},
            {"type": "function_call", "call_id": "call_x", "name": "calc",
             "arguments": '{"expr":"19*23"}'},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }, requested_model="m")
    assert got.stop_reason is LLMStopReason.TOOL_CALLS
    assert got.tool_calls == (
        LLMToolCall(call_id="call_x", name="calc", arguments={"expr": "19*23"}),
    )
    assert got.text == "Sure."


def test_bad_tool_arguments_raise_the_recoverable_error():
    """坏参数走 LLMToolCallFormatError,不是通用协议错。

    这两条路的终点完全不同:前者是同一次会话里告诉模型哪里写错了再来一次,
    后者是整场会话作废。ETP 第 14 轮六个终止会话里有三个死在这里,其中一个
    是在第 135 轮、145 次工具调用之后,毁于一次坏调用。
    """
    with pytest.raises(LLMToolCallFormatError):
        _parse_responses_response({
            "model": "m", "status": "completed",
            "output": [{"type": "function_call", "call_id": "c", "name": "calc",
                        "arguments": '{"expr": "19*23"'}],
            "usage": {},
        }, requested_model="m")


def test_incomplete_status_is_reported_as_truncation():
    """截断走 status/incomplete_details,不是 finish_reason。

    判错的代价是把半截正文当成完整答案交上去。

    ⚠️ 实测这个网关**忽略** `max_output_tokens`(传 16 仍输出 663 token 的完整
    正文,status 恒为 completed),所以这条路径在它身上不会被触发。留着是因为
    上游真截断时形状就是这个,而且换一个遵守该字段的网关就立刻生效。
    """
    got = _parse_responses_response({
        "model": "m", "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "half a sen"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 16},
    }, requested_model="m")
    assert got.stop_reason is LLMStopReason.MAX_TOKENS


def test_round_trip_shape_matches_what_the_provider_accepted():
    """一次完整回合的 input,和实测 200 的那份逐键相同。

    钉住它是因为这份形状是**试出来的**,不是从文档抄的:助手那一轮要原样回灌,
    工具结果靠 call_id 回指,而 reasoning 可以不回灌。任何一处改动都只会在
    真实调用里暴露,单测不钉住就等于没人看着。
    """
    call = LLMToolCall(call_id="call_1", name="calc", arguments={"expr": "19*23"})
    payload = build(
        [
            LLMMessage("system", "Use tools when asked."),
            LLMMessage("user", "Compute 19*23."),
            LLMMessage("assistant", "On it.", tool_calls=(call,)),
            LLMMessage("tool", tool_results=(
                LLMToolResult(call_id="call_1", content="437"),)),
        ],
        tools=(TOOL,),
        tool_choice=LLMToolChoice(LLMToolChoiceMode.AUTO),
        parallel_tool_calls=True,
    )
    assert payload["input"] == [
        {"role": "developer", "content": "Use tools when asked."},
        {"role": "user", "content": "Compute 19*23."},
        {"role": "assistant", "content": "On it."},
        {"type": "function_call", "call_id": "call_1", "name": "calc",
         "arguments": '{"expr": "19*23"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "437"},
    ]
    # arguments 线上是字符串不是对象,两个方向都得按字符串处理。
    assert isinstance(payload["input"][3]["arguments"], str)
    assert json.loads(payload["input"][3]["arguments"]) == {"expr": "19*23"}


def test_missing_output_array_is_transient_not_a_protocol_error():
    """没有 output 数组 → **瞬时**,走重试;不是协议错。

    这个区分是承重的,而且判错的代价已经实测过。LLMProtocolError 会让整场
    agent 会话立即终止、丢弃全部工作且**不重试**;LLMTransientError 走已有的
    退避重试路径。

    2026-08-31 判错这一条的代价:两场会话分别在第 105 轮(15 次编辑)和第 33 轮
    (14 次编辑)被这一行杀掉,preflight 全过、工作全丢,日志里只剩一句
    "provider response output must be an array"。

    依据是这个网关本身:它在 4 KB 边界截断响应体(实测大响应约一半被截),
    「HTTP 200 + 合法 JSON + 没有 output」是它的常态噪声。真正的协议分歧不会
    被藏起来 —— 它会连续耗尽全部重试,并且正文已经带在异常消息里。
    """
    from evoharness.core.llm import LLMTransientError

    for body in (
        {"type": "error", "error": {"message": "upstream hiccup"}},
        {"id": "resp_1", "status": "completed"},        # output 整个缺失
        {"id": "resp_1", "output": "not-a-list"},        # 类型不对
    ):
        with pytest.raises(LLMTransientError) as exc:
            _parse_responses_response(body, requested_model="m")
        # 正文要带出去 —— 真协议错耗尽重试后,诊断信息必须还在。
        assert "no output array" in str(exc.value)


def test_a_real_shape_disagreement_still_surfaces_as_protocol_error():
    """把「没有 output」放宽成瞬时,不等于把所有形状问题都放宽。

    output 在、但里面的项不是对象 —— 这是真的形状分歧,重试不会变好。
    """
    with pytest.raises(LLMProtocolError):
        _parse_responses_response(
            {"model": "m", "status": "completed", "output": ["not-an-object"]},
            requested_model="m",
        )


# --- streaming --------------------------------------------------------------


class _FakeSSE:
    """Minimal stand-in for the urllib response object: iterates raw lines."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)


def _sse(event: dict) -> bytes:
    return b"data: " + json.dumps(event).encode() + b"\n"


def test_stream_returns_the_terminal_snapshot():
    """`response.completed` carries the whole response, deltas are ignored.

    Reassembling from deltas would duplicate the parsing logic; taking the
    terminal snapshot lets the streaming and non-streaming paths share one
    parser, so they cannot disagree.
    """
    final = {
        "model": "gpt-5.6-sol", "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "done"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2,
                  "input_tokens_details": {"cached_tokens": 8}},
    }
    got = _read_sse_within(_FakeSSE([
        b"event: response.created\n",
        _sse({"type": "response.created", "response": {"status": "in_progress"}}),
        _sse({"type": "response.output_text.delta", "delta": "do"}),
        _sse({"type": "response.output_text.delta", "delta": "ne"}),
        _sse({"type": "response.completed", "response": final}),
        b"\n",
    ]), 30.0)
    assert got == final
    parsed = _parse_responses_response(got, requested_model="gpt-5.6-sol")
    assert parsed.text == "done"
    assert parsed.cached_prompt_tokens == 8


def test_stream_cut_before_the_terminal_event_is_retryable():
    """A cut stream is a transport fault, not a protocol error.

    This is the whole point of streaming here: a proxy that drops the body
    mid-flight leaves a stream without its terminal event, which retries,
    rather than a JSON prefix that may parse into a wrong-looking response.
    """
    with pytest.raises(LLMTransientError, match="without response.completed"):
        _read_sse_within(_FakeSSE([
            _sse({"type": "response.created", "response": {}}),
            _sse({"type": "response.output_text.delta", "delta": "half"}),
        ]), 30.0)


def test_stream_tolerates_unparseable_frames():
    """One malformed frame must not condemn a stream that still completes."""
    final = {"model": "m", "status": "completed",
             "output": [{"type": "message", "content": [
                 {"type": "output_text", "text": "ok"}]}],
             "usage": {}}
    got = _read_sse_within(_FakeSSE([
        b"data: {not json\n",
        b"data: [DONE]\n",
        _sse({"type": "response.completed", "response": final}),
    ]), 30.0)
    assert got == final


def test_stream_request_sets_the_flag_and_accept_header():
    from evoharness.core.llm import make_openai_responses_transport

    t = make_openai_responses_transport("https://x/v1", "k", 10.0, stream=True)
    assert t.url.endswith("/responses")
    assert t.stream is True
    plain = make_openai_responses_transport("https://x/v1", "k", 10.0)
    assert plain.stream is False
