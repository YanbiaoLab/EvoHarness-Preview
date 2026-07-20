"""Agentic proposal session contracts and runtime components."""

from .contracts import (
    AgentBackend,
    AgentEvent,
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionRequest,
    AgentSessionResult,
    AgentTermination,
    EventSink,
    EventSinkFactory,
    ManagedEventSink,
    PreflightTraceRecord,
    ProposalTraceSummary,
)
from .conversation import ConversationalAgentBackend
from .preflight_tool import RunPreflightTool
from .run_tool import RunTool, Runner, RunnerResult
from .session_proposer import AgentSessionProposer
from .transcript import (
    JsonlEventSink,
    JsonlEventSinkFactory,
    TRANSCRIPT_SCHEMA_VERSION,
    render_workspace_patch,
)
from .tools import (
    AgentTool,
    AgentToolContext,
    AgentToolError,
    AgentToolRegistry,
    make_tool_error,
    make_tool_result,
    truncate_tool_text,
)
from .runtime import NativeToolAgentBackend, TokenEstimator
from .workspace_tools import (
    WorkspaceDeleteTool,
    WorkspaceEditTool,
    WorkspaceGlobTool,
    WorkspaceGrepTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
    resolve_workspace_path,
)


def make_default_agent_tools(runner: Runner) -> tuple[AgentTool, ...]:
    """Build the stable P1.1 tool set in provider-visible order."""

    return (
        WorkspaceReadTool(),
        WorkspaceGlobTool(),
        WorkspaceGrepTool(runner),
        WorkspaceWriteTool(),
        WorkspaceEditTool(),
        WorkspaceDeleteTool(),
        RunTool(runner),
        RunPreflightTool(),
    )

__all__ = [
    "AgentBackend",
    "AgentEvent",
    "AgentEventKind",
    "AgentSessionLimits",
    "AgentSessionProposer",
    "AgentSessionRequest",
    "AgentSessionResult",
    "AgentTermination",
    "ConversationalAgentBackend",
    "EventSink",
    "EventSinkFactory",
    "ManagedEventSink",
    "PreflightTraceRecord",
    "ProposalTraceSummary",
    "JsonlEventSink",
    "JsonlEventSinkFactory",
    "TRANSCRIPT_SCHEMA_VERSION",
    "NativeToolAgentBackend",
    "AgentTool",
    "AgentToolContext",
    "AgentToolError",
    "AgentToolRegistry",
    "RunPreflightTool",
    "RunTool",
    "Runner",
    "RunnerResult",
    "TokenEstimator",
    "WorkspaceDeleteTool",
    "WorkspaceEditTool",
    "WorkspaceGlobTool",
    "WorkspaceGrepTool",
    "WorkspaceReadTool",
    "WorkspaceWriteTool",
    "make_default_agent_tools",
    "make_tool_error",
    "make_tool_result",
    "resolve_workspace_path",
    "render_workspace_patch",
    "truncate_tool_text",
]
