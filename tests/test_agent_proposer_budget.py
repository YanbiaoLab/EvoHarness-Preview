"""Cross-run budget accounting tests for AgentSessionProposer."""

import pytest

from evoharness.core import (
    AgentSessionLimits,
    AgentSessionResult,
    AgentTermination,
)
from evoharness.core.agent.session_proposer import (
    _ProposalLoopError,
    _ProposalUsage,
)


def make_result(
    *,
    turns=0,
    tool_calls=0,
    cost_usd=0.0,
    prompt_tokens=0,
    completion_tokens=0,
):
    return AgentSessionResult(
        termination=AgentTermination.COMPLETED,
        turns=turns,
        tool_calls=tool_calls,
        cost_usd=cost_usd,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_usage_accumulates_backend_run_deltas():
    usage = _ProposalUsage()
    limits = AgentSessionLimits(
        max_turns=5,
        max_tool_calls=6,
        timeout_s=30,
        max_cost_usd=1.0,
    )

    usage.absorb(
        make_result(
            turns=2,
            tool_calls=3,
            cost_usd=0.25,
            prompt_tokens=10,
            completion_tokens=4,
        ),
        limits,
    )
    usage.absorb(
        make_result(
            turns=1,
            tool_calls=1,
            cost_usd=0.1,
            prompt_tokens=6,
            completion_tokens=2,
        ),
        limits,
    )

    assert usage.attempts == 2
    assert usage.turns == 3
    assert usage.tool_calls == 4
    assert usage.cost_usd == pytest.approx(0.35)
    assert usage.prompt_tokens == 16
    assert usage.completion_tokens == 6


def test_remaining_limits_are_total_budget_deltas():
    usage = _ProposalUsage(
        attempts=1,
        turns=3,
        tool_calls=4,
        cost_usd=0.4,
    )
    total = AgentSessionLimits(
        max_turns=10,
        max_tool_calls=4,
        timeout_s=100,
        max_cost_usd=1.0,
    )

    remaining = usage.remaining_limits(
        total,
        remaining_timeout_s=60,
    )

    assert remaining.max_turns == 7
    assert remaining.max_tool_calls == 0
    assert remaining.timeout_s == 60
    assert remaining.max_cost_usd == pytest.approx(0.6)


def test_cost_overshoot_is_recorded_then_blocks_next_run():
    usage = _ProposalUsage()
    run_limits = AgentSessionLimits(max_cost_usd=0.1)

    usage.absorb(
        make_result(cost_usd=0.15),
        run_limits,
    )

    assert usage.cost_usd == pytest.approx(0.15)

    with pytest.raises(_ProposalLoopError, match="cost-limit"):
        usage.remaining_limits(
            AgentSessionLimits(max_cost_usd=0.1),
            remaining_timeout_s=10,
        )


@pytest.mark.parametrize(
    ("usage", "timeout_s", "expected"),
    [
        (_ProposalUsage(turns=2), 10, "turn-limit"),
        (_ProposalUsage(), 0, "timeout"),
    ],
)
def test_remaining_limits_stop_exhausted_runs(
    usage,
    timeout_s,
    expected,
):
    with pytest.raises(_ProposalLoopError, match=expected):
        usage.remaining_limits(
            AgentSessionLimits(max_turns=2),
            remaining_timeout_s=timeout_s,
        )


@pytest.mark.parametrize(
    "result",
    [
        make_result(turns=3),
        make_result(tool_calls=3),
    ],
)
def test_usage_rejects_backend_hard_limit_violations(result):
    usage = _ProposalUsage()
    run_limits = AgentSessionLimits(
        max_turns=2,
        max_tool_calls=2,
    )

    with pytest.raises(_ProposalLoopError, match="backend-error"):
        usage.absorb(result, run_limits)

    assert usage.attempts == 0
