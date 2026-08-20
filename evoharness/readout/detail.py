"""What happened in a run: its identity, its population, its trajectory.

Split from `status` because the cost is different by an order of magnitude —
a status poll should not open the population store and read every candidate,
and a caller that wants the trajectory should not have to.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from evoharness.core import MetricLog, PopulationStore

from .status import ReadoutError, RunDirectory, RunStatus, run_status


@dataclass(frozen=True)
class CandidateRow:
    """One candidate as the outside world sees it."""

    id: str
    generation: int
    parent_id: str | None
    island: int
    operator: str
    fitness: float | None
    passed: bool
    title: str
    summary: str
    model: str
    in_archive: bool
    fault: str | None
    #: The dsh session that produced it, when an agent backend ran. Its absence
    #: on an agentic run is what `scripts/audit.py` looks for: a candidate with
    #: no session reference cannot be traced back to what the agent actually
    #: did.
    session_id: str | None

    def to_json(self) -> dict:
        return asdict(self)


def _row(candidate) -> CandidateRow:
    metadata = candidate.metadata or {}
    return CandidateRow(
        id=candidate.id,
        generation=candidate.generation,
        parent_id=candidate.parent_id,
        island=candidate.island_idx,
        operator=candidate.operator,
        fitness=candidate.fitness if candidate.report else None,
        passed=candidate.passed,
        title=candidate.change_title or "",
        summary=candidate.change_summary or "",
        model=candidate.model_name or "",
        in_archive=bool(candidate.in_archive),
        fault=candidate.report.fault if candidate.report else None,
        session_id=metadata.get("session_id") or metadata.get("agent_session_id"),
    )


def _candidates(run: RunDirectory) -> list[CandidateRow]:
    db = run.path / "run.db"
    if not db.exists():
        raise ReadoutError(f"{run.path} has no run.db")
    store = PopulationStore.open_readonly(db)
    try:
        return [_row(candidate) for candidate in store.all_candidates()]
    finally:
        store.close()


def population(run_dir: Path | str) -> list[CandidateRow]:
    """Every candidate the run produced, in store order."""

    return _candidates(RunDirectory(Path(run_dir)))


def candidate_detail(run_dir: Path | str, candidate_id: str) -> dict | None:
    """One candidate, with its program text and evaluation report."""

    run = RunDirectory(Path(run_dir))
    db = run.path / "run.db"
    if not db.exists():
        raise ReadoutError(f"{run.path} has no run.db")
    store = PopulationStore.open_readonly(db)
    try:
        candidate = store.get(candidate_id)
        if candidate is None:
            return None
        row = _row(candidate).to_json()
        # The program text, not the serialized genome: a multi-file workspace
        # serializes to a JSON blob nobody can read.
        row["code"] = candidate.workspace.main_text()
        row["report"] = (
            candidate.report.to_json() if candidate.report else None
        )
        row["metadata"] = candidate.metadata or {}
        return row
    finally:
        store.close()


@dataclass(frozen=True)
class Generation:
    """What one generation changed."""

    generation: int
    candidates: tuple[CandidateRow, ...]
    best_fitness: float | None
    #: Best fitness across every generation up to and including this one, so a
    #: reader can see whether the run is still improving without recomputing
    #: the running maximum itself.
    best_so_far: float | None

    def to_json(self) -> dict:
        return {
            "generation": self.generation,
            "candidates": [row.to_json() for row in self.candidates],
            "best_fitness": self.best_fitness,
            "best_so_far": self.best_so_far,
        }


def trajectory(run_dir: Path | str) -> list[Generation]:
    """The run generation by generation.

    This is the view an author revising a task needs: not "which candidate
    won" but "what did each generation try, and did anything move". A flat
    candidate list answers the first question and hides the second.
    """

    rows = _candidates(RunDirectory(Path(run_dir)))
    by_generation: dict[int, list[CandidateRow]] = {}
    for row in rows:
        by_generation.setdefault(row.generation, []).append(row)

    out: list[Generation] = []
    running_best: float | None = None
    for generation in sorted(by_generation):
        members = tuple(by_generation[generation])
        scored = [row.fitness for row in members if row.fitness is not None]
        best = max(scored) if scored else None
        if best is not None:
            running_best = best if running_best is None else max(running_best, best)
        out.append(
            Generation(
                generation=generation,
                candidates=members,
                best_fitness=best,
                best_so_far=running_best,
            )
        )
    return out


def run_detail(run_dir: Path | str) -> dict:
    """Identity, status and population in one payload."""

    run = RunDirectory(Path(run_dir))
    manifest = run.json("manifest.json")
    metrics = MetricLog(run.path / "metrics.jsonl")
    status: RunStatus = run_status(run.path)

    return {
        "status": status.to_json(),
        # The frozen identity: spec hashes, models, which backend ran, which
        # preflight checks were active. A result read without this is a number
        # with no experiment attached to it.
        "identity": {
            "task": manifest.get("task"),
            "recipe": manifest.get("recipe"),
            "spec_hashes": manifest.get("spec_hashes", {}),
            "models": manifest.get("models", []),
            "proposal": manifest.get("proposal", {}),
            "live": manifest.get("live"),
        },
        "candidates": [row.to_json() for row in _candidates(run)],
        "series": {
            "best_fitness": metrics.series("sys/best_fitness"),
            "fitness": metrics.series("sys/fitness"),
        },
    }
