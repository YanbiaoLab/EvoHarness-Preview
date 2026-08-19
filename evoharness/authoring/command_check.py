"""A preflight check declared as a command line rather than as Python.

A task directory has to be writable by whoever owns the domain, and by models
that are not strong enough to implement a `PreflightValidator` correctly. One
data-driven validator covers what almost every task actually wants — "does it
import", "does it compile", "do the smoke tests pass", "does the proof
checker accept it" — because every one of those is already a command that
exits non-zero when it fails.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from evoharness.core.preflight import (
    PreflightContext,
    PreflightIssue,
    PreflightResult,
)

#: Long enough for a compiler, short enough that a hung check cannot stall a
#: generation. Overridable per check.
DEFAULT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class CommandCheck:
    """Run `argv` in the candidate's directory; non-zero exit is a failure."""

    name: str
    argv: tuple[str, ...]
    timeout_s: float = DEFAULT_TIMEOUT_S
    #: Whether a failure should be handed back to the proposer as repair
    #: feedback. False means the candidate is rejected outright — right for a
    #: check whose failure says the genome is unusable rather than fixable.
    repairable: bool = True

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                list(self.argv),
                cwd=ctx.workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return self._failed(
                started,
                code="timeout",
                message=f"{self.name} exceeded {self.timeout_s:g}s",
            )
        except OSError as exc:
            # The command itself is missing or unrunnable. That is a broken
            # task declaration, not a broken candidate, so it must not be
            # handed to the model as something to repair.
            return self._failed(
                started,
                code="unrunnable",
                message=f"{self.name} could not start: {exc}",
                repairable=False,
            )
        if completed.returncode == 0:
            return PreflightResult(
                stage=self.name, elapsed_s=time.monotonic() - started
            )
        return self._failed(
            started,
            code=f"exit-{completed.returncode}",
            message=f"{self.name} failed with exit code {completed.returncode}",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _failed(
        self,
        started: float,
        *,
        code: str,
        message: str,
        stdout: str = "",
        stderr: str = "",
        repairable: bool | None = None,
    ) -> PreflightResult:
        return PreflightResult(
            stage=self.name,
            elapsed_s=time.monotonic() - started,
            issues=(
                PreflightIssue(
                    validator=self.name,
                    code=code,
                    message=message,
                    repairable=(
                        self.repairable if repairable is None else repairable
                    ),
                    command=self.argv,
                    stdout=stdout,
                    stderr=stderr,
                ),
            ),
        )

    def component_config(self) -> dict:
        """Explicit configuration so the check enters the frozen task spec."""

        return {
            "argv": list(self.argv),
            "timeout_s": self.timeout_s,
            "repairable": self.repairable,
        }
