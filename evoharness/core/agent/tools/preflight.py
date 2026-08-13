"""Agent-facing adapter for the authoritative proposal preflight."""

from __future__ import annotations

from ...llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from .base import (
    AgentToolContext,
    AgentToolError,
    make_tool_result,
)


class RunPreflightTool:
    """Run the same preflight used by the final proposal admission gate."""

    definition = LLMToolDefinition(
        name="run_preflight",
        description=(
            "Check the current workspace with the authoritative proposal "
            "preflight. A passing result is diagnostic only; the harness "
            "will run the check again before accepting the proposal."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, max_issue_output_chars: int = 8_000):
        if max_issue_output_chars < 32:
            raise ValueError("max_issue_output_chars must be at least 32")
        self.max_issue_output_chars = max_issue_output_chars

    def is_concurrency_safe(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> bool:
        return False

    def invoke(
        self,
        call: LLMToolCall,
        ctx: AgentToolContext,
    ) -> LLMToolResult:
        if call.arguments:
            raise AgentToolError(
                "invalid-arguments",
                "run_preflight does not accept arguments",
            )

        # Deferred: agent.feedback imports this package for its text
        # helpers, so a module-level import here would close an import
        # cycle (feedback -> tools -> preflight -> feedback).
        from ..feedback import preflight_issue_to_payload

        outcome = ctx.preflight.check(ctx.preflight_context)
        issues = [
            preflight_issue_to_payload(
                issue,
                max_output_chars=self.max_issue_output_chars,
            )
            for issue in outcome.report.issues
        ]

        return make_tool_result(
            call.call_id,
            {
                "ok": outcome.ok,
                "issues": issues,
                "summary": {
                    "issue_count": len(issues),
                    "failed_stage": outcome.report.failed_stage,
                    "repairable": outcome.report.repairable,
                    "elapsed_s": outcome.report.elapsed_s,
                },
            },
        )
