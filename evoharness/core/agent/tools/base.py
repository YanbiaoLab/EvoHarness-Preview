"""Provider-neutral agent tool execution contracts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from ...llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from ...population import Candidate
from ...preflight import PreflightContext, ProposalPreflight


_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


class AgentToolError(ValueError):
    """Expected tool failure with a stable model-facing error code."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not _ERROR_CODE_PATTERN.fullmatch(code):
            raise ValueError(
                "agent tool error code must use lowercase letters, digits, and hyphens"
            )
        if not isinstance(message, str) or not message.strip():
            raise ValueError("agent tool error message must be non-empty")
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AgentToolContext:
    """Harness state exposed to tools during one proposal session."""

    workdir: Path
    parent: Candidate
    operator: str
    preflight: ProposalPreflight
    remaining_timeout_s: float
    # Per-session record of what a read tool has already surfaced:
    # display path -> (mtime_ns, offset, limit). Owned by the session so an
    # unchanged file is never dumped into the conversation twice; mutating
    # tools deliberately do NOT populate it (a post-edit hit would point the
    # model at pre-edit content).
    read_state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        workdir = Path(self.workdir).resolve()
        if not workdir.is_dir():
            raise ValueError("agent tool workdir must be an existing directory")
        if (
            isinstance(self.remaining_timeout_s, bool)
            or not isinstance(self.remaining_timeout_s, (int, float))
            or not math.isfinite(self.remaining_timeout_s)
            or self.remaining_timeout_s <= 0
        ):
            raise ValueError("remaining_timeout_s must be positive and finite")
        object.__setattr__(self, "workdir", workdir)

    @property
    def preflight_context(self) -> PreflightContext:
        return PreflightContext(
            parent=self.parent,
            operator=self.operator,
            workdir=self.workdir,
        )


@runtime_checkable
class AgentTool(Protocol):
    """One tool callable by the agent runtime."""

    definition: LLMToolDefinition

    def is_concurrency_safe(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> bool:
        ...

    def invoke(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> LLMToolResult:
        ...


# How long a tool's result stays worth resending. Every uncompacted result
# is re-sent on every later turn, so a file dump read once is paid for many
# times, while a failure digest earns its keep for the whole session.
RETENTION_EPHEMERAL = "ephemeral"  # superseded once acted on (reads, greps)
RETENTION_DURABLE = "durable"      # diagnostic context worth carrying


def make_tool_result(
    call_id: str,
    payload: dict[str, object],
    *,
    is_error: bool = False,
) -> LLMToolResult:
    """Serialize one tool payload at the model-facing boundary."""

    return LLMToolResult(
        call_id=call_id,
        content=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        is_error=is_error,
    )


def make_tool_error(
    call_id: str,
    code: str,
    message: str,
) -> LLMToolResult:
    return make_tool_result(
        call_id,
        {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
        is_error=True,
    )


def truncate_tool_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Bound tool output while retaining both diagnostic context and tail."""

    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    if len(text) <= max_chars:
        return text, False

    marker = "\n...<truncated>...\n"
    remaining = max_chars - len(marker)
    head_chars = (remaining * 2) // 3
    tail_chars = remaining - head_chars
    return text[:head_chars] + marker + text[-tail_chars:], True


class AgentToolRegistry:
    """Validated name-to-tool dispatch table."""

    def __init__(self, tools: Iterable[AgentTool]):
        self.tools = tuple(tools)

        if not all(
            isinstance(tool, AgentTool)
            and isinstance(tool.definition, LLMToolDefinition)
            for tool in self.tools
        ):
            raise TypeError(
                "every agent tool must implement the AgentTool protocol"
            )

        names = tuple(tool.definition.name for tool in self.tools)
        if len(names) != len(set(names)):
            raise ValueError("agent tool names must be unique")

        self._by_name = {
            tool.definition.name: tool
            for tool in self.tools
        }

    @property
    def definitions(self) -> tuple[LLMToolDefinition, ...]:
        return tuple(tool.definition for tool in self.tools)

    def retention_of(self, name: str) -> str:
        """Retention class of a tool's results. Unknown or undeclared tools
        are treated as durable: dropping context must never be the default
        for something the runtime does not understand."""
        tool = self._by_name.get(name)
        if tool is None:
            return RETENTION_DURABLE
        value = getattr(tool, "retention", RETENTION_DURABLE)
        return (
            RETENTION_EPHEMERAL
            if value == RETENTION_EPHEMERAL
            else RETENTION_DURABLE
        )

    def is_concurrency_safe(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> bool:
        """Classify a call; unknown or broken classifiers fail closed."""

        tool = self._by_name.get(call.name)
        if tool is None:
            return False

        try:
            return tool.is_concurrency_safe(call, ctx) is True
        except Exception:
            return False

    def invoke(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> LLMToolResult:
        tool = self._by_name.get(call.name)
        if tool is None:
            return make_tool_error(
                call.call_id,
                "unknown-tool",
                f"unknown tool: {call.name}",
            )

        try:
            result = tool.invoke(call, ctx)
            if not isinstance(result, LLMToolResult):
                raise TypeError(
                    "tool invoke() must return LLMToolResult"
                )
            if result.call_id != call.call_id:
                raise ValueError(
                    "tool result call_id must match the request"
                )
            return result
        except AgentToolError as exc:
            return make_tool_error(call.call_id, exc.code, str(exc))
        except Exception as exc:
            return make_tool_error(
                call.call_id,
                "tool-error",
                f"{type(exc).__name__}: {exc}",
            )
