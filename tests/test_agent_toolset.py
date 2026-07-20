"""Stable default agent tool set."""

from evoharness.evocore.agent import make_default_agent_tools
from evoharness.evoguard import Sandbox


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
