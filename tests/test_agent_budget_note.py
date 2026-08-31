"""每轮把剩余预算告诉 agent —— 但不能碰缓存前缀。

为什么要有:会话预算只在**开场**的系统提示里说一次。之后每一轮 agent 都不知道
自己用到第几轮了,得自己数。2026-08-28 ETP run18 实测:一场会话在第 109/110 轮
才启动验证探针,第 110 轮睡下等结果,然后被轮次上限切断 —— 75 分钟、18 次编辑,
交出去时从没看过它们到底有没有用。它不缺轮次,它缺的是"还剩几轮"。

为什么必须加在**尾部且不进历史**:供应商按**前缀**命中缓存(实测:同前缀换后缀
照常命中,换前缀全丢)。把逐轮计数写进系统提示 = 每一轮都丢整段缓存,比没有这个
计数更糟。而如果把尾注 append 进 state.messages,160 轮就会在历史里堆 160 条
互相矛盾的"还剩 N 轮"。
"""

from __future__ import annotations

import pytest

from evoharness.core.agent.runtime import (
    BUDGET_NOTE_PREFIX,
    NativeToolAgentBackend,
    strip_budget_note,
)
from evoharness.core.llm import LLMMessage


def _note(backend, state, request, remaining_s=600.0):
    return backend._messages_with_budget(state, request, remaining_s)


class _State:
    def __init__(self, messages, turns):
        self.messages = list(messages)
        self.lifetime_turns = turns


class _Req:
    def __init__(self, max_turns):
        from evoharness.core.agent.contracts import AgentSessionLimits

        self.limits = AgentSessionLimits(max_turns=max_turns,
                                         max_tool_calls=10, timeout_s=100.0)


def _backend():
    return NativeToolAgentBackend.__new__(NativeToolAgentBackend)


HISTORY = (
    LLMMessage(role="system", content="sys"),
    LLMMessage(role="user", content="task"),
)


def test_the_note_is_appended_at_the_tail():
    out = _note(_backend(), _State(HISTORY, 5), _Req(160))
    assert len(out) == len(HISTORY) + 1
    assert out[: len(HISTORY)] == HISTORY          # 前缀逐条不变
    assert out[-1].role == "user"
    assert out[-1].content.startswith(BUDGET_NOTE_PREFIX)


def test_the_prefix_is_byte_identical_across_turns():
    """这就是缓存安全的全部内容:两轮之间,历史部分一个字节都不能变。"""
    b = _backend()
    first = _note(b, _State(HISTORY, 5), _Req(160))
    grown = HISTORY + (LLMMessage(role="assistant", content="did a thing"),)
    second = _note(b, _State(grown, 6), _Req(160))
    assert second[: len(HISTORY)] == first[: len(HISTORY)]
    assert first[-1].content != second[-1].content   # 尾注本身每轮不同


def test_the_note_never_enters_the_history():
    """否则 160 轮会堆 160 条过时的"还剩 N 轮"。"""
    state = _State(HISTORY, 5)
    before = list(state.messages)
    _note(_backend(), state, _Req(160))
    assert state.messages == before


def test_it_reports_the_turn_the_agent_is_about_to_spend():
    out = _note(_backend(), _State(HISTORY, 40), _Req(160))
    text = out[-1].content
    assert "turn 41 of 160" in text
    assert "120 left" in text


def test_it_escalates_when_the_budget_runs_low():
    b = _backend()
    calm = _note(b, _State(HISTORY, 100), _Req(160))[-1].content
    urgent = _note(b, _State(HISTORY, 155), _Req(160))[-1].content
    assert "Reserve about" in calm and "STOP EXPLORING" not in calm
    assert "STOP EXPLORING" in urgent
    # 说清代价,而不只是催促
    assert "produces nothing" in urgent


def test_the_wall_clock_is_reported_too():
    """轮次和墙钟是两条独立的天花板,先撞哪一条都可能。"""
    out = _note(_backend(), _State(HISTORY, 10), _Req(160), remaining_s=1800.0)
    assert "30 wall-clock minutes left" in out[-1].content


def test_exhausted_budget_does_not_report_a_negative():
    out = _note(_backend(), _State(HISTORY, 200), _Req(160))
    assert "0 left" in out[-1].content
    assert "-" not in out[-1].content.split("left")[0]


# ─── strip_budget_note:凡是看"最后一条消息"的代码都得用它 ────────────────────


def test_strip_removes_only_the_note():
    tail = LLMMessage(role="user", content=f"{BUDGET_NOTE_PREFIX} turn 3 of 9")
    assert strip_budget_note(HISTORY + (tail,)) == HISTORY


def test_strip_leaves_a_normal_user_message_alone():
    tail = LLMMessage(role="user", content="please fix the syntax error")
    assert strip_budget_note(HISTORY + (tail,)) == HISTORY + (tail,)


def test_strip_is_a_no_op_when_there_is_no_note():
    assert strip_budget_note(HISTORY) == HISTORY
    assert strip_budget_note(()) == ()
