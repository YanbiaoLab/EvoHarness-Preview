"""What a run is doing right now, and how much to trust that answer.

The single source of truth is the run directory `SearchLoop` writes. Nothing
here writes to it: reading a run must never be able to change one, which is
also why the population store is opened through `open_readonly` — the writable
constructor runs schema DDL and would manufacture an empty database from a
mistyped path, then report it as an empty run.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evoharness.core import PopulationStore

#: A run whose checkpoint has not moved for this long is reported as `stalled`
#: rather than `running`. The number is deliberately generous: a slow
#: generation on a large task can take minutes, and calling a live run dead is
#: worse than answering "unknown" a little late.
STALL_AFTER_S = 20 * 60

#: What `SearchLoop` writes into a mid-flight checkpoint's `stopped_reason`.
#: It reads like a stop reason and is the opposite of one, so it is named here
#: rather than compared inline: taking it at face value reports every live run
#: as finished, and a caller polling for completion stops waiting at the first
#: checkpoint.
RUNNING_SENTINEL = "running"


@dataclass(frozen=True)
class RunStatus:
    """The cheap answer: is it moving, how far has it got, how good is it."""

    name: str
    #: `running`, `stalled`, or the loop's own `stopped_reason` once it
    #: finished. `stalled` is not a state the loop ever writes — it is this
    #: module's verdict about a run that claims to be running while nothing
    #: on disk has changed.
    state: str
    generation: int
    target_generations: int | None
    best_fitness: float | None
    evaluations: int
    #: Seconds since the checkpoint last moved; None when there is no
    #: checkpoint yet.
    heartbeat_age_s: float | None
    stopped_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.stopped_reason is not None

    def to_json(self) -> dict:
        return {**asdict(self), "finished": self.finished}


@dataclass(frozen=True)
class RunDirectory:
    """Lazy readers over one run directory."""

    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_dir():
            raise ReadoutError(f"not a run directory: {self.path}")

    def json(self, name: str) -> dict:
        path = self.path / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A half-written checkpoint is normal while a run is mid-flight.
            # Reporting "no checkpoint" is honest; raising would make every
            # status poll a coin flip.
            return {}

    def jsonl(self, name: str) -> list[dict]:
        path = self.path / name
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Only ever the final line, and only while it is being
                # written. Skipping it beats failing the whole read.
                continue
        return rows

    def mtime(self, name: str) -> float | None:
        path = self.path / name
        return path.stat().st_mtime if path.exists() else None


class ReadoutError(RuntimeError):
    """The directory cannot be read as a run."""


def _best_fitness(run: RunDirectory, report: dict) -> float | None:
    recorded = report.get("best_fitness")
    if recorded is not None:
        return recorded
    db = run.path / "run.db"
    if not db.exists():
        return None
    store = PopulationStore.open_readonly(db)
    try:
        best = store.best()
    finally:
        store.close()
    return best.fitness if best else None


def run_status(run_dir: Path | str, *, now=time.time) -> RunStatus:
    """Read the run's state without opening the population unless needed."""

    run = RunDirectory(Path(run_dir))
    checkpoint = run.json("checkpoint.json")
    manifest = run.json("manifest.json")
    # The manifest's report is written once, at the end. The checkpoint's is
    # rewritten every generation and carries the in-flight sentinel, so it can
    # only be trusted while the finalized one is absent.
    report = manifest.get("report") or checkpoint.get("run_report") or {}

    stopped = report.get("stopped_reason")
    if stopped == RUNNING_SENTINEL:
        stopped = None
    heartbeat = run.mtime("checkpoint.json")
    age = None if heartbeat is None else max(0.0, now() - heartbeat)

    if stopped:
        state = stopped
    elif age is not None and age > STALL_AFTER_S:
        # The loop cannot write "my process was killed", so a run that claims
        # to be running while nothing moves is indistinguishable from a live
        # one unless somebody says so. A caller polling for completion would
        # otherwise wait forever.
        state = "stalled"
    else:
        state = "running"

    return RunStatus(
        name=run.path.name,
        state=state,
        generation=int(checkpoint.get("generation", 0) or 0),
        target_generations=(manifest.get("search") or {}).get("num_generations"),
        best_fitness=_best_fitness(run, report),
        evaluations=int(report.get("evaluations", 0) or 0),
        heartbeat_age_s=None if age is None else round(age, 1),
        stopped_reason=stopped,
    )


def list_runs(root: Path | str, *, now=time.time) -> list[RunStatus]:
    """Every run directory under `root`, oldest checkpoint last."""

    root = Path(root)
    found = [
        run_status(db.parent, now=now) for db in sorted(root.glob("*/run.db"))
    ]
    return sorted(found, key=lambda status: status.name)
