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
    """A backend that claims a limit and passes it is misbehaving."""

    usage = _ProposalUsage()
    run_limits = AgentSessionLimits(
        max_turns=2,
        max_tool_calls=2,
    )

    with pytest.raises(_ProposalLoopError, match="backend-error"):
        usage.absorb(result, run_limits)

    # ...but the session still happened, so it is on the books. This used to
    # assert `attempts == 0`: the run was refused before a single field was
    # recorded, so the tokens were spent at the provider and counted nowhere.
    # One live run spent 16,836 input and 6,776 output tokens across five
    # rejected proposals and reported zero for every one of them.
    assert usage.attempts == 1
    assert usage.turns == result.turns
    assert usage.tool_calls == result.tool_calls


@pytest.mark.parametrize(
    "limit,result",
    [
        ("max_turns", make_result(turns=3)),
        ("max_tool_calls", make_result(tool_calls=3)),
    ],
)
def test_an_overrun_of_an_unenforceable_limit_is_recorded_not_refused(
    limit, result
):
    """dsh has no turn or tool ceiling to hand a session, so a limit the
    caller sets is a wish. Discarding a completed session for passing one
    throws away real work and buys nothing — the backend could not have
    stopped, and `timeout_s` is the bound that actually holds."""

    usage = _ProposalUsage()
    run_limits = AgentSessionLimits(max_turns=2, max_tool_calls=2)

    usage.absorb(result, run_limits, unenforceable=(limit,))

    assert usage.attempts == 1
    # Allowed, but never silent: a run whose candidates all overran has a
    # turn limit that is decoration, and that has to be visible.
    assert usage.overruns == {limit: 3}


def test_declaring_one_limit_unenforceable_does_not_excuse_the_other():
    usage = _ProposalUsage()
    result = make_result(turns=3, tool_calls=3)

    with pytest.raises(_ProposalLoopError, match="max_tool_calls"):
        usage.absorb(
            result,
            AgentSessionLimits(max_turns=2, max_tool_calls=2),
            unenforceable=("max_turns",),
        )
    assert usage.overruns == {"max_turns": 3}


def test_a_backend_that_says_nothing_is_held_to_every_limit():
    """The strict reading is the default: a backend that never considered the
    question has not been excused from anything."""

    usage = _ProposalUsage()
    with pytest.raises(_ProposalLoopError, match="backend-error"):
        usage.absorb(
            make_result(turns=3), AgentSessionLimits(max_turns=2)
        )
