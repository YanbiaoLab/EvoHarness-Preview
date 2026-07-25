"""Agent tool: read another candidate's source on demand.

Reference programs used to be rendered into every mutation prompt in full.
That is the one part of the prompt whose size tracks the program being
evolved rather than a budget — three full copies per proposal, resent on
every turn of the session — so it is the first thing to breach a context
window as a project grows. Progressive disclosure instead: the prompt
carries an inventory, and the agent expands only what it decides to read.
"""

from __future__ import annotations

from ...llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from .base import (
    RETENTION_EPHEMERAL,
    AgentToolContext,
    AgentToolError,
    make_tool_result,
    truncate_tool_text,
)


class InspectCandidateTool:
    """Read the files of any evaluated candidate by id."""

    # Another program's source expires on use, exactly like a file read.
    retention = RETENTION_EPHEMERAL

    definition = LLMToolDefinition(
        name="inspect_candidate",
        description=(
            "Read the source of a previously evaluated candidate program. "
            "candidate_id must be copied verbatim from an 'id=...' shown in "
            "the reference-program list of your prompt — it is an opaque "
            "identifier, not a command. Pass path=null to see that "
            "candidate's files, then pass a path to read one. If your prompt "
            "listed no reference programs, there is nothing to inspect."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string"},
                # Optional args are nullable + required (strict-schema rule).
                "path": {"type": ["string", "null"]},
            },
            "required": ["candidate_id", "path"],
            "additionalProperties": False,
        },
    )

    def __init__(self, store, *, max_output_chars: int = 20_000):
        if max_output_chars < 32:
            raise ValueError("max_output_chars must be at least 32")
        self.store = store
        self.max_output_chars = max_output_chars

    def is_concurrency_safe(
        self, call: LLMToolCall, ctx: AgentToolContext
    ) -> bool:
        return True  # read-only over immutable, already-graded candidates

    def invoke(
        self, call: LLMToolCall, ctx: AgentToolContext
    ) -> LLMToolResult:
        args = call.arguments or {}
        candidate_id = args.get("candidate_id")
        if not candidate_id or not isinstance(candidate_id, str):
            raise AgentToolError(
                "invalid-arguments", "candidate_id must be a non-empty string"
            )
        candidate = self.store.get(candidate_id)
        if candidate is None:
            # Name real ids: the first live session called this with
            # candidate_id="list", reading the description's "list that
            # candidate's files" as a command word.
            raise AgentToolError(
                "unknown-candidate",
                f"no candidate with id {candidate_id!r}. Ids are opaque and "
                "must be copied from an 'id=...' in your prompt's "
                "reference-program list.",
            )
        try:
            texts = candidate.workspace.texts()
        except Exception as exc:  # noqa: BLE001 — surfaced as a tool error
            raise AgentToolError(
                "unreadable-candidate",
                f"candidate {candidate_id} has no readable workspace",
            ) from exc

        path = args.get("path")
        if not path:
            return make_tool_result(
                call.call_id,
                {
                    "ok": True,
                    "candidate_id": candidate_id,
                    "fitness": candidate.fitness,
                    "change_title": candidate.change_title,
                    "files": {
                        name: len(text.splitlines())
                        for name, text in sorted(texts.items())
                    },
                },
            )
        if path not in texts:
            raise AgentToolError(
                "unknown-path",
                f"candidate {candidate_id} has no file {path}; "
                f"available: {', '.join(sorted(texts))}",
            )
        content, truncated = truncate_tool_text(
            texts[path], self.max_output_chars
        )
        return make_tool_result(
            call.call_id,
            {
                "ok": True,
                "candidate_id": candidate_id,
                "path": path,
                "content": content,
                "truncated": truncated,
            },
        )


__all__ = ["InspectCandidateTool"]
