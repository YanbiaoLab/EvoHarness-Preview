"""提案的 token 上限:轮次代替不了它。

2026-08-28 ETP run18 实测:两个提案各花了 2.48M 和 2.46M token,而 110 轮 /
600 次工具调用 / 150 分钟**三条限制全部遵守** —— 没有任何东西拦得住。
`prompt_tokens` / `completion_tokens` 一直在累计,却从不参与任何判断。

为什么轮次不是代理:agentic 的一轮会重发整个上下文,而上下文随会话增长,
所以花费的增速快于轮数。`max_turns=110` 在单轮 7 万 token 时,实际是一份
7.7M 的预算,而这个数从来没被写下来过。
"""

from __future__ import annotations

import pytest

from evoharness.core.agent.contracts import (
    AgentSessionLimits,
    AgentSessionResult,
    AgentTermination,
)
from evoharness.core.agent.session_proposer import _ProposalLoopError, _ProposalUsage


def _result(turns=1, tools=0, prompt=0, completion=0):
    return AgentSessionResult(
        termination=AgentTermination.COMPLETED,
        turns=turns, tool_calls=tools, cost_usd=0.0,
        prompt_tokens=prompt, completion_tokens=completion,
    )


def _limits(**kw):
    base = dict(max_turns=110, max_tool_calls=600, timeout_s=9000.0)
    base.update(kw)
    return AgentSessionLimits(**base)


def test_no_ceiling_by_default():
    """不设就是不限 —— 老的 run 行为不变。"""
    u = _ProposalUsage()
    u.absorb(_result(prompt=5_000_000), _limits())
    left = u.remaining_limits(_limits(), remaining_timeout_s=100.0)
    assert left.max_tokens is None


def test_ceiling_stops_the_next_round():
    u = _ProposalUsage()
    u.absorb(_result(prompt=900_000, completion=150_000), _limits(max_tokens=1_000_000))
    with pytest.raises(_ProposalLoopError) as exc:
        u.remaining_limits(_limits(max_tokens=1_000_000), remaining_timeout_s=100.0)
    assert exc.value.code == "token-limit"


def test_remaining_is_carried_into_the_repair_round():
    """预算跨轮扣减,不是每轮重置 —— 否则 3 次修复就是 4 倍预算。"""
    u = _ProposalUsage()
    u.absorb(_result(prompt=300_000, completion=50_000), _limits(max_tokens=1_000_000))
    left = u.remaining_limits(_limits(max_tokens=1_000_000), remaining_timeout_s=100.0)
    assert left.max_tokens == 650_000


def test_prompt_and_completion_both_count():
    """重发的上下文是大头,只数输出会漏掉 94% 的量。"""
    u = _ProposalUsage()
    u.absorb(_result(prompt=99, completion=2), _limits(max_tokens=100))
    with pytest.raises(_ProposalLoopError):
        u.remaining_limits(_limits(max_tokens=100), remaining_timeout_s=100.0)


def test_turns_alone_would_not_have_caught_it():
    """把 run18 那次实测复现出来:轮次远没用完,token 已经烧穿。"""
    u = _ProposalUsage()
    # 38 轮,单轮约 65k 输入 —— 实测中位 73,921
    for _ in range(38):
        u.absorb(_result(turns=1, prompt=65_000, completion=2_000),
                 _limits(max_tokens=10_000_000))
    assert u.turns == 38 < 110                      # 轮次闸:远没到
    assert u.prompt_tokens + u.completion_tokens > 2_500_000
    with pytest.raises(_ProposalLoopError) as exc:
        u.remaining_limits(_limits(max_tokens=2_000_000), remaining_timeout_s=100.0)
    assert exc.value.code == "token-limit"


def test_config_knob_validates():
    from evoharness.core.config import ProposalConfig

    assert ProposalConfig().max_tokens_per_proposal is None
    assert ProposalConfig(max_tokens_per_proposal=1).max_tokens_per_proposal == 1
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ProposalConfig(max_tokens_per_proposal=bad)
