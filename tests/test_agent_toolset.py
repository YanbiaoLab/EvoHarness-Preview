"""Stable default agent tool set."""

import pytest

from evoharness.core.agent import make_default_agent_tools
from evoharness.core.config import ProposalConfig
from evoharness.guard import Sandbox


def _run_tool(tools):
    return next(t for t in tools if t.definition.name == "run")


def test_run_timeout_cap_defaults_to_sixty_seconds():
    assert _run_tool(make_default_agent_tools(Sandbox())).runner_timeout_cap_s == 60.0


def test_run_timeout_cap_is_configurable():
    # ETP run 15: a self-check runs 60-150s, so against the 60s cap `wait`
    # returned "[still running]" and the agent had to poll -- and by then a
    # model call resends 130-200k tokens of context, so each poll was one of
    # the run's more expensive operations.
    tools = make_default_agent_tools(Sandbox(), run_timeout_cap_s=300.0)

    assert _run_tool(tools).runner_timeout_cap_s == 300.0


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True])
def test_proposal_config_rejects_an_unusable_run_timeout_cap(bad):
    with pytest.raises(ValueError):
        ProposalConfig(run_timeout_cap_s=bad)


def test_default_tool_set_has_stable_provider_order():
    tools = make_default_agent_tools(Sandbox())

    assert [tool.definition.name for tool in tools] == [
        "workspace_read",
        "workspace_glob",
        "workspace_grep",
        "workspace_write",
        "workspace_edit",
        "workspace_delete",
        "run",
        "run_preflight",
    ]
