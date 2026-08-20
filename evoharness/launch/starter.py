"""Start a run that outlives whoever started it.

`api.run` and the experiment driver both block until the search finishes,
which is right for a shell and impossible for a tool call: a run takes hours
and a tool call has to answer in seconds. This starts the same command in a
detached process and returns as soon as it is confirmed alive.

Durability stays where it already was — the run directory and its checkpoint.
This module adds no state of its own beyond a record of how the run was
invoked, written into the run directory so the directory stays
self-describing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evoharness.readout import run_status

JOB_FILE = "job.json"
LOG_FILE = "run.log"

DEFAULT_HANDSHAKE_S = 30.0


LOG_TAIL_BYTES = 4000


class StartError(RuntimeError):
    """The run could not be started, or died before it identified itself."""

@dataclass(frozen=True)
class JobSpec:
    """How this run was invoked, recorded beside its results.

    A detached run has no caller left to ask. Without this the only record of
    what produced a directory is somebody's shell history, and a resume has to
    reconstruct the command by hand — which is exactly how a resume ends up
    running a different configuration than the run it claims to continue.
    """

    argv: tuple[str, ...]
    cwd: str
    created_at: float
    env: dict[str, str] = field(default_factory=dict)
    #: The settings the process was launched WITH, when the caller supplied
    #: them. `argv` records how it was launched and `launch` records what was
    #: asked for; keeping them apart is what lets `python -m evoharness.launch`
    #: read the second without parsing the first.
    launch: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {**asdict(self), "argv": list(self.argv)}


    @classmethod
    def read(cls, run_dir: Path) -> "JobSpec | None":
        path = Path(run_dir) / JOB_FILE
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            argv=tuple(data["argv"]),
            cwd=data["cwd"],
            created_at=float(data["created_at"]),
            env=dict(data.get("env", {})),
            launch=dict(data.get("launch", {})),
        )

    def write(self, run_dir: Path) -> None:
        (Path(run_dir) / JOB_FILE).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class StartedRun:
    """A run confirmed to be alive."""

    run_dir: Path
    pid: int
    log_path: Path
    #: Whether the child wrote its manifest before this call returned. False
    #: means it is still starting up, NOT that anything is wrong — but a
    #: caller that needs the frozen identity has to poll for it.
    identified: bool

    def to_json(self) -> dict:
        # Paths spelled out rather than `asdict`: this crosses a tool boundary
        # as JSON, and a Path is not serializable — `asdict` would keep them
        # and every caller would fail at `json.dumps`, not here.
        return {
            "run_dir": str(self.run_dir),
            "pid": self.pid,
            "log_path": str(self.log_path),
            "identified": self.identified,
        }


def _log_tail(path: Path) -> str:
    if not path.exists():
        return "(no output)"
    data = path.read_bytes()
    return data[-LOG_TAIL_BYTES:].decode("utf-8", errors="replace").strip()



def _already_running(run_dir: Path) -> bool:
    """Whether a previous start is still in charge of this directory."""

    if not (run_dir / "manifest.json").exists():
        return False
    return not run_status(run_dir).finished

def start_run(
    run_dir: Path | str,
    argv: list[str] | tuple[str, ...],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    launch: dict | None = None,
    handshake_s: float = DEFAULT_HANDSHAKE_S,
    force: bool = False,
    now=time.time,
) -> StartedRun:
    """Spawn `argv` as a detached run and return once it is confirmed alive.

    :param launch: the settings the command was built from, recorded so the
        run directory describes itself. `python -m evoharness.launch` reads
        exactly this, which is what makes a resume the same command as the
        original start instead of a hand-reconstructed one.
    :param force: start even though the directory holds an unfinished run.
        Two live processes writing one checkpoint corrupt it, so this exists
        for the case where the previous process is known to be gone.
    """

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not force and _already_running(run_dir):
        raise StartError(
            f"{run_dir} holds a run that has not finished; two processes "
            "writing one checkpoint corrupt it. Pass force=True only if the "
            "previous process is known to be gone."
        )

    JobSpec(
        argv=tuple(str(part) for part in argv),
        cwd=str(cwd or Path.cwd()),
        created_at=now(),
        env=dict(env or {}),
        launch=dict(launch or {}),
    ).write(run_dir)

    log_path = run_dir / LOG_FILE
    child_env = {**os.environ, **(env or {})}

    # Append rather than truncate: a resume writes into the same directory,
    # and losing the previous attempt's traceback is losing the reason the
    # resume was needed.
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(part) for part in argv],
            cwd=str(cwd or Path.cwd()),
            env=child_env,
            # A detached process that inherits stdin blocks forever the first
            # time anything reads from it.
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Its own session, so the parent exiting, the terminal closing, or
            # a SIGHUP does not take the run with it. Without this the failure
            # looks like a run that silently stopped, with nothing written
            # down about why.
            start_new_session=True,
        )

    manifest = run_dir / "manifest.json"
    deadline = now() + handshake_s
    while now() < deadline:
        if manifest.exists():
            return StartedRun(run_dir, process.pid, log_path, identified=True)
        if process.poll() is not None:
            # Died before identifying itself. Returning a run directory here
            # would report a configuration error as a successful start, and
            # the caller would poll a run that never existed.
            raise StartError(
                f"run exited with code {process.returncode} before writing "
                f"its manifest:\n{_log_tail(log_path)}"
            )
        time.sleep(0.05)

    if process.poll() is not None:
        raise StartError(
            f"run exited with code {process.returncode} before writing "
            f"its manifest:\n{_log_tail(log_path)}"
        )
    # Alive but slow. Saying so beats both killing it and pretending the
    # identity is on disk.
    return StartedRun(run_dir, process.pid, log_path, identified=False)