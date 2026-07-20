"""Provider-neutral contracts for agentic proposal sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from ..population import Candidate
from ..preflight import (
    PreflightIssue,
    PreflightReport,
    ProposalPreflight,
)


class AgentTermination(str, Enum):
    """Normalized termination reasons reported by every agent backend."""

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    TURN_LIMIT = "turn_limit"
    TOOL_LIMIT = "tool_limit"
    COST_LIMIT = "cost_limit"
    CONTEXT_LIMIT = "context_limit"
    OUTPUT_LIMIT = "output_limit"
    REFUSAL = "refusal"
    CONTENT_FILTER = "content_filter"
    PROTOCOL_ERROR = "protocol_error"
    BACKEND_ERROR = "backend_error"


class AgentEventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_RESUME = "session_resume"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_COMPACT = "context_compact"
    RECOVERY = "recovery"
    TERMINATION = "termination"


@dataclass(frozen=True)
class AgentEvent:
    """One provider-neutral runtime event."""

    session_id: str
    round_index: int
    sequence: int
    kind: AgentEventKind
    turn: int
    elapsed_s: float
    call_id: str | None = None
    tool_name: str | None = None
    content: str = ""
    data: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("agent event session_id must be non-empty")
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("agent event round_index cannot be negative")
        for name in ("sequence", "turn"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"agent event {name} must be a nonnegative integer"
                )
        if (
            isinstance(self.elapsed_s, bool)
            or not isinstance(self.elapsed_s, (int, float))
            or not math.isfinite(self.elapsed_s)
            or self.elapsed_s < 0
        ):
            raise ValueError(
                "agent event elapsed_s must be nonnegative and finite"
            )
        if not isinstance(self.kind, AgentEventKind):
            raise ValueError("agent event kind must be AgentEventKind")
        for name in ("call_id", "tool_name"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"agent event {name} must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("agent event content must be text")
        if not isinstance(self.data, Mapping):
            raise TypeError("agent event data must be a mapping")


class EventSink(Protocol):
    """Consume ordered runtime events for one agent session."""

    def emit(self, event: AgentEvent) -> None: ...


@dataclass(frozen=True)
class PreflightTraceRecord:
    """One authoritative final-preflight execution in a repair series."""

    round_index: int
    session_id: str | None
    report: PreflightReport

    def __post_init__(self) -> None:
        if (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise ValueError("preflight round_index cannot be negative")
        if self.session_id is not None and (
            not isinstance(self.session_id, str)
            or not self.session_id.strip()
        ):
            raise ValueError("preflight session_id must be non-empty")
        if not isinstance(self.report, PreflightReport):
            raise TypeError("preflight trace report must be PreflightReport")


@dataclass(frozen=True)
class ProposalTraceSummary:
    """Final provider-neutral accounting and outcome for one proposal."""

    proposal_id: str
    parent_id: str
    operator: str
    success: bool
    session_id: str | None
    attempts: int
    repair_rounds: int
    turns: int
    tool_calls: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    termination: str | None = None
    failure_reason: str | None = None
    model: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        for name in ("proposal_id", "parent_id", "operator"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.session_id is not None and (
            not isinstance(self.session_id, str)
            or not self.session_id.strip()
        ):
            raise ValueError("session_id must be non-empty when present")
        integer_fields = (
            self.attempts,
            self.repair_rounds,
            self.turns,
            self.tool_calls,
            self.prompt_tokens,
            self.completion_tokens,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in integer_fields
        ):
            raise ValueError("summary accounting integers must be nonnegative")
        if self.repair_rounds != max(0, self.attempts - 1):
            raise ValueError("repair_rounds must equal attempts minus one")
        numeric_fields = (self.cost_usd, self.elapsed_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in numeric_fields
        ):
            raise ValueError("summary numeric values must be nonnegative")
        if self.success == (self.failure_reason is not None):
            raise ValueError(
                "failure_reason must be present exactly for failed summaries"
            )
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str)
            or not self.failure_reason.strip()
        ):
            raise ValueError("failure_reason must be non-empty when present")
        if self.termination is not None and (
            not isinstance(self.termination, str)
            or not self.termination.strip()
        ):
            raise ValueError("termination must be non-empty when present")
        if not isinstance(self.model, str):
            raise TypeError("model must be text")


@runtime_checkable
class ManagedEventSink(EventSink, Protocol):
    """Event sink whose lifecycle is owned by AgentSessionProposer."""

    trace_path: str | None

    def record_preflight(self, record: PreflightTraceRecord) -> None: ...

    def finalize(
        self,
        summary: ProposalTraceSummary,
        final_patch: str | None,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class EventSinkFactory(Protocol):
    """Open one managed sink for one proposal attempt series."""

    def open(
        self,
        *,
        proposal_id: str,
        parent_id: str,
        operator: str,
    ) -> ManagedEventSink:
        ...


@dataclass(frozen=True)
class AgentSessionLimits:
    """Hard limits available to one agent backend invocation."""

    max_turns: int = 12
    max_tool_calls: int = 40
    timeout_s: float = 300.0
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 1
        ):
            raise ValueError("max_turns must be at least 1")
        if (
            isinstance(self.max_tool_calls, bool)
            or not isinstance(self.max_tool_calls, int)
            or self.max_tool_calls < 0
        ):
            raise ValueError("max_tool_calls must be nonnegative")

        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be nonnegative and finite")


@dataclass(frozen=True)
class AgentSessionRequest:
    """One initial or resumed agent invocation over the same workspace."""

    system: str
    user: str
    parent: Candidate
    operator: str
    workdir: Path
    limits: AgentSessionLimits
    preflight: ProposalPreflight
    event_sink: EventSink | None = None
    session_id: str | None = None
    feedback: tuple[PreflightIssue, ...] = ()


@dataclass(frozen=True)
class AgentSessionResult:
    """Normalized result returned by every agent backend."""

    termination: AgentTermination
    session_id: str | None = None
    final_message: str = ""
    model: str | None = ""

    cost_usd: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0

    events: tuple[AgentEvent, ...] = ()

    @property
    def completed(self) -> bool:
        return self.termination is AgentTermination.COMPLETED

    def __post_init__(self) -> None:
        if not isinstance(self.termination, AgentTermination):
            raise TypeError("termination must be AgentTermination")
        if self.session_id is not None and (
            not isinstance(self.session_id, str)
            or not self.session_id.strip()
        ):
            raise ValueError("session_id must be non-empty when present")
        if not isinstance(self.final_message, str):
            raise TypeError("final_message must be text")
        if self.model is not None and not isinstance(self.model, str):
            raise TypeError("model must be text when present")

        for name in (
            "turns",
            "tool_calls",
            "prompt_tokens",
            "completion_tokens",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be a nonnegative integer"
                )
        for name in ("cost_usd", "elapsed_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be nonnegative and finite"
                )
        if not isinstance(self.events, tuple) or not all(
            isinstance(event, AgentEvent) for event in self.events
        ):
            raise TypeError("events must be a tuple of AgentEvent")


@runtime_checkable
class AgentBackend(Protocol):
    """Run one complete model/tool loop for an initial or resumed session."""

    def run(
        self,
        request: AgentSessionRequest,
    ) -> AgentSessionResult:
        ...

    def release(self, session_id: str) -> None:
        ...
