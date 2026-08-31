"""缓存命中必须被记下来 —— 否则「要不要更早压实」只能靠讲道理。

2026-08-28 实测:压实重写的是消息历史,而消息历史正是缓存的**前缀**。
对供应商实测(同前缀换后缀 → 缓存照常命中;换前缀 → 缓存全丢),所以一次压实
把一大段便宜的命中 token 换成一次全价重发。

而框架当时只记 prompt_tokens 总数。一场 110 轮的会话发了 754 万 prompt token,
我们说不出其中有多少是便宜的 —— 于是这笔交易无法评估。
"""
from __future__ import annotations

import pytest

from evoharness.core.llm import LLMResponse, LLMProtocolError


def _resp(usage):
    from evoharness.core import llm
    return llm._parse_openai_chat_response(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
         "model": "m", "usage": usage},
        requested_model="m", cost=0.0)


def test_cached_tokens_are_read_from_the_provider():
    r = _resp({"prompt_tokens": 2795, "completion_tokens": 24,
               "prompt_tokens_details": {"cached_tokens": 2048}})
    assert r.prompt_tokens == 2795
    assert r.cached_prompt_tokens == 2048


def test_absent_cache_reporting_is_zero_not_an_error():
    """这个数只用来做决策,永远不许拦住任何事。"""
    assert _resp({"prompt_tokens": 10, "completion_tokens": 1}
                 ).cached_prompt_tokens == 0
    assert _resp({"prompt_tokens": 10, "completion_tokens": 1,
                  "prompt_tokens_details": None}).cached_prompt_tokens == 0
    assert _resp({"prompt_tokens": 10, "completion_tokens": 1,
                  "prompt_tokens_details": {"cached_tokens": "x"}}
                 ).cached_prompt_tokens == 0


def test_cached_can_never_exceed_prompt():
    assert _resp({"prompt_tokens": 10, "completion_tokens": 1,
                  "prompt_tokens_details": {"cached_tokens": 999}}
                 ).cached_prompt_tokens == 10
    with pytest.raises(ValueError):
        LLMResponse(text="x", model="m", prompt_tokens=5,
                    cached_prompt_tokens=6)


def test_negative_cache_is_rejected():
    with pytest.raises(ValueError):
        LLMResponse(text="x", model="m", prompt_tokens=5,
                    cached_prompt_tokens=-1)


def test_the_agent_trace_carries_it_per_turn():
    """按轮记录,不是按会话汇总 —— 压实发生在某一轮,代价也落在那一轮。"""
    import inspect

    from evoharness.core.agent import runtime

    src = inspect.getsource(runtime.NativeToolAgentBackend._record_response
                            if hasattr(runtime.NativeToolAgentBackend,
                                       "_record_response")
                            else runtime)
    assert '"cached_prompt_tokens": response.cached_prompt_tokens' in src
