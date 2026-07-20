# EvoHarness original guardrail. Upstream (shinka/launch/local.py) runs
# evaluation subprocesses bare; this sandbox adds resource limits, a minimal
# environment and process-group timeout kill. Network isolation is
# best-effort at this layer (proxy black-holing); the AntiHackScanner is the
# complementary static layer for raw-socket attempts.
"""Subprocess sandbox for candidate evaluation."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.return_code == 0 and not self.timed_out


class Sandbox:
    def __init__(
        self,
        cpu_time_s: int | None = None,
        memory_mb: int | None = None,
        allow_network: bool = False,
        extra_env: dict[str, str] | None = None,
    ):
        self.cpu_time_s = cpu_time_s
        self.memory_mb = memory_mb
        self.allow_network = allow_network
        self.extra_env = extra_env or {}

    def _build_env(self, env: dict[str, str] | None, workdir: Path) -> dict:
        base = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if not self.allow_network:
            # Black-hole proxies: blocks HTTP(S) clients honoring proxy vars.
            for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                base[var] = "http://127.0.0.1:9"
            base["NO_PROXY"] = ""
        base.update(self.extra_env)
        if env:
            base.update(env)
        return base

    def _preexec(self):  # pragma: no cover - runs in the child process
        os.setsid()
        try:
            import resource

            if self.cpu_time_s is not None:
                resource.setrlimit(
                    resource.RLIMIT_CPU, (self.cpu_time_s, self.cpu_time_s)
                )
            if self.memory_mb is not None:
                limit = self.memory_mb * 1024 * 1024
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                except (ValueError, OSError):
                    pass  # RLIMIT_AS unreliable on some platforms (macOS)
        except Exception:
            pass

    def run(
        self,
        cmd: list[str],
        workdir: Path,
        timeout_s: float,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=self._build_env(env, workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=self._preexec,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return SandboxResult(
            return_code=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=timed_out,
            elapsed_s=time.monotonic() - start,
        )
