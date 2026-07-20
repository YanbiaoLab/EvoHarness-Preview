"""Controlled diagnostic command tool."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..llm import LLMToolCall, LLMToolDefinition, LLMToolResult
from .tools import (
    AgentToolContext,
    AgentToolError,
    make_tool_result,
    truncate_tool_text,
)


@runtime_checkable
class RunnerResult(Protocol):
    """Result shape required from an injected command runner."""

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float


@runtime_checkable
class Runner(Protocol):
    """Execute one argv-only command under an external safety policy."""

    def run(
        self,
        cmd: list[str],
        workdir: Path,
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> RunnerResult:
        ...


class RunTool:
    """Run a bounded diagnostic command inside the candidate workspace."""

    definition = LLMToolDefinition(
        name="run",
        description=(
            "Run one diagnostic command in the candidate workspace. "
            "Provide argv directly; shell syntax and pipelines are not supported."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 128,
                },
                "timeout_s": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["argv", "timeout_s"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        runner: Runner,
        *,
        runner_timeout_cap_s: float = 60.0,
        max_output_chars: int = 40_000,
        max_argv_items: int = 128,
        max_argument_chars: int = 16_384,
    ):
        if not isinstance(runner, Runner):
            raise TypeError("runner must implement the Runner protocol")
        if runner_timeout_cap_s <= 0:
            raise ValueError("runner_timeout_cap_s must be positive")
        if max_output_chars < 32:
            raise ValueError("max_output_chars must be at least 32")
        if max_argv_items < 1 or max_argument_chars < 1:
            raise ValueError("argv limits must be positive")

        self.runner = runner
        self.runner_timeout_cap_s = float(runner_timeout_cap_s)
        self.max_output_chars = max_output_chars
        self.max_argv_items = max_argv_items
        self.max_argument_chars = max_argument_chars

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
        if set(call.arguments) != {"argv", "timeout_s"}:
            raise AgentToolError(
                "invalid-arguments",
                "run requires exactly argv and timeout_s",
            )

        argv = self._validate_argv(call.arguments["argv"])
        requested_timeout_s = self._validate_timeout(
            call.arguments["timeout_s"]
        )
        timeout_s = min(
            requested_timeout_s,
            ctx.remaining_timeout_s,
            self.runner_timeout_cap_s,
        )

        try:
            result = self.runner.run(
                argv,
                workdir=ctx.workdir,
                timeout_s=timeout_s,
                env=self._minimal_env(),
            )
        except FileNotFoundError as exc:
            raise AgentToolError(
                "command-not-found",
                f"command not found: {argv[0]}",
            ) from exc
        except OSError as exc:
            raise AgentToolError(
                "run-failed",
                f"failed to start command: {exc}",
            ) from exc

        self._validate_result(result)
        stdout, stdout_truncated = truncate_tool_text(
            result.stdout,
            self.max_output_chars,
        )
        stderr, stderr_truncated = truncate_tool_text(
            result.stderr,
            self.max_output_chars,
        )

        return make_tool_result(
            call.call_id,
            {
                "ok": result.return_code == 0 and not result.timed_out,
                "argv": argv,
                "return_code": result.return_code,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "timed_out": result.timed_out,
                "elapsed_s": result.elapsed_s,
                "timeout_s": timeout_s,
            },
        )

    def _validate_argv(self, value: object) -> list[str]:
        if not isinstance(value, list) or not value:
            raise AgentToolError(
                "invalid-arguments",
                "argv must be a non-empty string array",
            )
        if len(value) > self.max_argv_items:
            raise AgentToolError(
                "invalid-arguments",
                f"argv exceeds {self.max_argv_items} items",
            )
        if not all(
            isinstance(item, str)
            and item
            and "\x00" not in item
            for item in value
        ):
            raise AgentToolError(
                "invalid-arguments",
                "argv items must be non-empty strings without NUL bytes",
            )
        if sum(len(item) for item in value) > self.max_argument_chars:
            raise AgentToolError(
                "invalid-arguments",
                f"argv exceeds {self.max_argument_chars} characters",
            )
        return list(value)

    @staticmethod
    def _validate_timeout(value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise AgentToolError(
                "invalid-arguments",
                "timeout_s must be a positive finite number",
            )
        return float(value)

    @staticmethod
    def _validate_result(result: RunnerResult) -> None:
        if not isinstance(result, RunnerResult):
            raise TypeError("runner returned an invalid result")
        if (
            isinstance(result.return_code, bool)
            or not isinstance(result.return_code, int)
        ):
            raise TypeError("runner return_code must be an integer")
        if not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
            raise TypeError("runner stdout and stderr must be strings")
        if not isinstance(result.timed_out, bool):
            raise TypeError("runner timed_out must be a boolean")
        if (
            isinstance(result.elapsed_s, bool)
            or not isinstance(result.elapsed_s, (int, float))
        ):
            raise TypeError("runner elapsed_s must be numeric")
        if not math.isfinite(result.elapsed_s) or result.elapsed_s < 0:
            raise ValueError("runner elapsed_s must be finite and non-negative")

    @staticmethod
    def _minimal_env() -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for name in ("LANG", "LC_ALL", "TZ"):
            value = os.environ.get(name)
            if value:
                env[name] = value
        return env
