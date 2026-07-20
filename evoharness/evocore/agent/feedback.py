"""Agent-facing rendering for provider-neutral preflight diagnostics."""

from __future__ import annotations

import json

from ..preflight import PreflightIssue
from .tools import truncate_tool_text


def preflight_issue_to_payload(
    issue: PreflightIssue,
    *,
    max_output_chars: int = 8_000,
) -> dict[str, object]:
    """Render one pure diagnostic IR value at an agent-facing boundary."""

    if max_output_chars < 32:
        raise ValueError("max_output_chars must be at least 32")

    stdout, stdout_truncated = truncate_tool_text(
        issue.stdout,
        max_output_chars,
    )
    stderr, stderr_truncated = truncate_tool_text(
        issue.stderr,
        max_output_chars,
    )
    return {
        "validator": issue.validator,
        "code": issue.code,
        "message": issue.message,
        "repairable": issue.repairable,
        "path": issue.path,
        "line": issue.line,
        "column": issue.column,
        "command": (
            list(issue.command) if issue.command is not None else None
        ),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def render_preflight_feedback(
    issues: tuple[PreflightIssue, ...],
    *,
    max_output_chars: int = 8_000,
) -> str:
    """Render repair feedback without mutating its diagnostic IR."""

    if not issues:
        raise ValueError("preflight feedback requires at least one issue")

    return json.dumps(
        {
            "type": "preflight_feedback",
            "instruction": (
                "Repair the reported issues in the existing workspace."
            ),
            "issues": [
                preflight_issue_to_payload(
                    issue,
                    max_output_chars=max_output_chars,
                )
                for issue in issues
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
