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
from .dsh_backend import (
    UNSUPPORTED_LIMITS,
    DshAgentBackend,
    DshBackendError,
    DshRuntimeSpec,
)  # noqa: F401 - re-exported backend seam
from .session_proposer import AgentSessionProposer
from .transcript import (
    JsonlEventSink,
    JsonlEventSinkFactory,
    TRANSCRIPT_SCHEMA_VERSION,
    render_workspace_patch,
)
from .tools import (
    RETENTION_DURABLE,
    RETENTION_EPHEMERAL,
    AgentTool,
    AgentToolContext,
    AgentToolError,
    AgentToolRegistry,
    InspectCandidateTool,
    InspectCandidateTool,
    InspectParentEvalTool,
    RunPreflightTool,
    RunTool,
    Runner,
    RunnerResult,
    WorkspaceDeleteTool,
    WorkspaceEditTool,
    WorkspaceGlobTool,
    WorkspaceGrepTool,
    WorkspaceReadTool,
    WorkspaceWriteTool,
    make_tool_error,
    make_tool_result,
    resolve_workspace_path,
    truncate_tool_text,
)
from .runtime import NativeToolAgentBackend, TokenEstimator


def make_default_agent_tools(
    runner: Runner,
    *,
    run_timeout_cap_s: float = 60.0,
) -> tuple[AgentTool, ...]:
    """Build the stable P1.1 tool set in provider-visible order.

    `run_timeout_cap_s` is the ceiling `run` clamps requested timeouts to.
    Left at the default it is the previous hard-coded 60s; a domain whose
    verification step outlives that has to raise it or pay for the polling.
    """

    return (
        WorkspaceReadTool(),
        WorkspaceGlobTool(),
        WorkspaceGrepTool(runner),
        WorkspaceWriteTool(),
        WorkspaceEditTool(),
        WorkspaceDeleteTool(),
        RunTool(runner, runner_timeout_cap_s=run_timeout_cap_s),
        RunPreflightTool(),
    )

__all__ = [
    "RETENTION_DURABLE",
    "RETENTION_EPHEMERAL",
    "AgentBackend",
    "AgentEvent",
    "AgentEventKind",
    "AgentSessionLimits",
    "AgentSessionProposer",
    "AgentSessionRequest",
    "AgentSessionResult",
    "AgentTermination",
    "ConversationalAgentBackend",
    "DshAgentBackend",
    "DshBackendError",
    "DshRuntimeSpec",
    "UNSUPPORTED_LIMITS",
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
    "InspectCandidateTool",
    "InspectCandidateTool",
    "InspectParentEvalTool",
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
