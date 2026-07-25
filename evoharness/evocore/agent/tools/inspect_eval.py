"""Agent tool: progressive, sanitized inspection of the parent's eval trace.

Domain-neutral. Backed by any ArtifactStore + TraceSanitizer, so the optimizer
reads the parent's per-item evaluation on demand (summary -> item -> search)
instead of getting it dumped into the prompt. Every payload passes through the
sanitizer at a fixed audience, so secrets never reach the model.
"""

from __future__ import annotations

from ...artifacts import ArtifactRef, ArtifactStore
from ...llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from ...sanitize import TraceSanitizer
from .base import (
    AgentToolContext,
    AgentToolError,
    make_tool_result,
    truncate_tool_text,
)


class InspectParentEvalTool:
    definition = LLMToolDefinition(
        name="inspect_parent_eval",
        description=(
            "Inspect the parent program's per-problem evaluation results to "
            "guide your improvement. PREFER action='failed_digest': it "
            "returns a compact critique of EVERY failed problem in one call "
            "— much cheaper than reading items one by one. Use "
            "action='item' with an item_id only when you need one problem's "
            "full output. action='summary' lists pass/fail ids; "
            "action='search' returns item ids whose trace text matches a "
            "query."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "summary",
                        "failed_digest",
                        "list_failed",
                        "item",
                        "search",
                    ],
                },
                # Optional args are nullable + required (strict-schema rule).
                "item_id": {"type": ["string", "null"]},
                "query": {"type": ["string", "null"]},
            },
            "required": ["action", "item_id", "query"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        store: ArtifactStore,
        sanitizer: TraceSanitizer,
        *,
        audience: str = "optimizer",
        # Every item result stays in the conversation and is resent on each
        # subsequent turn, so a generous per-field cap is paid many times
        # over (live run: 2.1M prompt tokens across 9 sessions). Keep the
        # full-item view useful but bounded; use failed_digest for breadth.
        max_field_chars: int = 2_000,
        digest_field_chars: int = 300,
    ):
        if max_field_chars < 32:
            raise ValueError("max_field_chars must be at least 32")
        if digest_field_chars < 32:
            raise ValueError("digest_field_chars must be at least 32")
        self.store = store
        self.sanitizer = sanitizer
        self.audience = audience
        self.max_field_chars = max_field_chars
        self.digest_field_chars = digest_field_chars

    def is_concurrency_safe(
        self, call: LLMToolCall, ctx: AgentToolContext
    ) -> bool:
        return True  # read-only over an immutable parent trace

    def _bound(self, item: dict, max_chars: int | None = None) -> dict:
        limit = self.max_field_chars if max_chars is None else max_chars
        out: dict[str, object] = {}
        for key, value in item.items():
            if isinstance(value, str) and len(value) > limit:
                value, _ = truncate_tool_text(value, limit)
            out[key] = value
        return out

    def invoke(
        self, call: LLMToolCall, ctx: AgentToolContext
    ) -> LLMToolResult:
        report = ctx.parent.report
        ref = report.artifacts_ref if report is not None else None
        if not ref:
            raise AgentToolError(
                "no-parent-trace",
                "the parent program has no evaluation trace to inspect",
            )
        view = self.store.open(ArtifactRef.decode(ref))
        args = call.arguments or {}
        action = args.get("action")

        if action == "summary":
            payload = {
                "ok": True,
                "summary": self.sanitizer.sanitize_summary(
                    view.summary(), self.audience
                ),
                "failed_items": view.item_ids(failed_only=True),
                "all_items": view.item_ids(),
            }
        elif action == "failed_digest":
            # One-call overview of every failure: the per-item fields are
            # truncated hard so ten failures cost less than two full items.
            # This exists because turn-limited agents burned their whole
            # session reading items one per turn (smoke run postmortem).
            digests = [
                self._bound(
                    self.sanitizer.sanitize_item(view.item(item_id), self.audience),
                    self.digest_field_chars,
                )
                for item_id in view.item_ids(failed_only=True)
            ]
            payload = {
                "ok": True,
                "failed_count": len(digests),
                "failed_items": digests,
            }
        elif action == "list_failed":
            payload = {"ok": True, "failed_items": view.item_ids(failed_only=True)}
        elif action == "item":
            item_id = args.get("item_id")
            if not item_id:
                raise AgentToolError(
                    "invalid-arguments", "action='item' requires item_id"
                )
            try:
                raw = view.item(item_id)
            except (FileNotFoundError, ValueError) as exc:
                raise AgentToolError(
                    "unknown-item", f"no such item: {item_id}"
                ) from exc
            payload = {
                "ok": True,
                "item": self._bound(
                    self.sanitizer.sanitize_item(raw, self.audience)
                ),
            }
        elif action == "search":
            query = args.get("query")
            if not query:
                raise AgentToolError(
                    "invalid-arguments", "action='search' requires query"
                )
            payload = {"ok": True, "matches": view.search(query)}
        else:
            raise AgentToolError(
                "invalid-arguments", f"unknown action: {action!r}"
            )

        return make_tool_result(call.call_id, payload)


__all__ = ["InspectParentEvalTool"]
