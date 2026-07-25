# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/database/dbase.py (Program dataclass, ProgramDatabase,
#           SQLite schema, archive update), shinka/database/islands.py
#           (island assignment, elitist migration)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Behavior-aligned port with independent naming (docs/naming_map.md).
# Intentional deviations (recorded in porting notes):
#   - "crowding" archive update strategy not ported (only "fitness").
#   - Dynamic island spawning (stagnation detection) not ported.
#   - EvalReport is a typed contract instead of loose metrics dicts.
"""Candidate/EvalReport data model and the SQLite-backed PopulationStore."""

from __future__ import annotations

from functools import cached_property
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PopulationConfig
from .workspace import Workspace, load_workspace

_REQUIRED_REPORT_KEYS = ("fitness", "passed")


@dataclass
class EvalReport:
    """Typed evaluation report (independent design, replaces upstream's
    metrics.json/correct.json key-name convention)."""

    fitness: float
    passed: bool
    fault: str | None = None
    visible_metrics: dict = field(default_factory=dict)
    hidden_metrics: dict = field(default_factory=dict)
    notes: str = ""
    structured_feedback: dict | None = None
    artifacts_ref: str | None = None
    stdout_log: str = ""
    stderr_log: str = ""
    stage_reached: int = 3
    execution_time: float = 0.0
    eval_cost_usd: float = 0.0

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "fitness": self.fitness,
            "passed": self.passed,
            "fault": self.fault,
            "visible_metrics": self.visible_metrics,
            "hidden_metrics": self.hidden_metrics,
            "notes": self.notes,
            "structured_feedback": self.structured_feedback,
            "artifacts_ref": self.artifacts_ref,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "stage_reached": self.stage_reached,
            "execution_time": self.execution_time,
            "eval_cost_usd": self.eval_cost_usd,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EvalReport":
        missing = [k for k in _REQUIRED_REPORT_KEYS if k not in d]
        if missing:
            raise ValueError(f"EvalReport missing required keys: {missing}")
        fitness = float(d["fitness"])
        if np.isnan(fitness) or np.isinf(fitness):
            # Upstream marks NaN/inf combined_score as incorrect; we reject the
            # value into a failed report instead of storing it.
            return cls(
                fitness=0.0,
                passed=False,
                fault=f"non-finite fitness: {d['fitness']}",
            )
        return cls(
            fitness=fitness,
            passed=bool(d["passed"]),
            fault=d.get("fault"),
            visible_metrics=d.get("visible_metrics", {}),
            hidden_metrics=d.get("hidden_metrics", {}),
            notes=d.get("notes", ""),
            structured_feedback=d.get("structured_feedback"),
            artifacts_ref=d.get("artifacts_ref"),
            stdout_log=d.get("stdout_log", ""),
            stderr_log=d.get("stderr_log", ""),
            stage_reached=int(d.get("stage_reached", 3)),
            execution_time=float(d.get("execution_time", 0.0)),
            eval_cost_usd=float(d.get("eval_cost_usd", 0.0)),
        )

    def render_for_prompt(self) -> str:
        lines = [f"Fitness: {self.fitness:.6g}"]
        for k, v in self.visible_metrics.items():
            lines.append(f"{k}: {v}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


@dataclass
class Candidate:
    """One program in the population."""

    id: str
    code: str
    generation: int
    parent_id: str | None
    island_idx: int
    operator: str  # seed|revise|rewrite|recombine|repair
    workspace_kind: str = "file"
    change_title: str = ""
    change_summary: str = ""
    model_name: str = ""
    inspiration_ids: list[str] = field(default_factory=list)
    report: EvalReport | None = None
    embedding: list[float] | None = None
    behavior_signature: str | None = None
    behavior_duplicate: bool = False
    children_count: int = 0
    in_archive: bool = False
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    @property
    def fitness(self) -> float:
        return self.report.fitness if self.report else 0.0

    @property
    def passed(self) -> bool:
        return bool(self.report and self.report.passed)
    
    @cached_property
    def workspace(self) -> Workspace:
        return load_workspace(self.workspace_kind, self.code)
    



@dataclass
class IslandView:
    """Read-only snapshot of one island used by selection."""

    island_idx: int
    candidates: list[Candidate]  # all evaluated candidates in the island

    @property
    def passed_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.passed]

    @property
    def archive_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.in_archive and c.passed]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    generation INTEGER NOT NULL,
    parent_id TEXT,
    island_idx INTEGER NOT NULL,
    operator TEXT NOT NULL,
    change_title TEXT,
    change_summary TEXT,
    model_name TEXT,
    inspiration_ids TEXT,
    report TEXT,
    embedding TEXT,
    behavior_signature TEXT,
    behavior_duplicate INTEGER DEFAULT 0,
    children_count INTEGER DEFAULT 0,
    in_archive INTEGER DEFAULT 0,
    metadata TEXT,
    timestamp REAL,
    workspace_kind TEXT NOT NULL DEFAULT 'file'
);
CREATE INDEX IF NOT EXISTS idx_cand_generation ON candidates(generation);
CREATE INDEX IF NOT EXISTS idx_cand_parent ON candidates(parent_id);
CREATE INDEX IF NOT EXISTS idx_cand_island ON candidates(island_idx);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class _Rows(list):
    """Cursor-shaped view over rows already fetched under the lock."""

    def fetchall(self):
        return list(self)

    def fetchone(self):
        return self[0] if self else None


class _LockedConnection:
    """Serializes one sqlite connection so worker threads may read it.

    Rows are materialised while the lock is held: handing back a live
    cursor would let one thread's fetch interleave with another thread's
    execute on the same connection, which is exactly what sqlite's
    same-thread check exists to prevent.
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, *args, **kwargs) -> _Rows:
        with self._lock:
            return _Rows(self._conn.execute(*args, **kwargs).fetchall())

    def executescript(self, *args, **kwargs) -> None:
        with self._lock:
            self._conn.executescript(*args, **kwargs)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class PopulationStore:
    """SQLite-backed population with islands, elite archive and migration."""

    def __init__(self, cfg: PopulationConfig, path: Path | str = ":memory:"):
        self.cfg = cfg
        # Agent tools read candidates from the runtime's worker threads, and
        # sqlite refuses a connection outside its creating thread. One
        # connection guarded by one lock keeps the store single-writer while
        # letting those reads through; the evolution loop itself still
        # mutates only from the main thread.
        self._lock = threading.RLock()
        self._conn = _LockedConnection(
            sqlite3.connect(str(path), check_same_thread=False), self._lock
        )
        self._conn.executescript(_SCHEMA)
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(candidates)")}
        if "workspace_kind" not in cols:
            self._conn.execute(
                "ALTER TABLE candidates ADD COLUMN "
                "workspace_kind TEXT NOT NULL DEFAULT 'file'"
            )
        self._conn.commit()

    # -- (de)serialization ---------------------------------------------------

    @staticmethod
    def _to_row(c: Candidate) -> tuple:
        return (
            c.id,
            c.code,
            c.generation,
            c.parent_id,
            c.island_idx,
            c.operator,
            c.change_title,
            c.change_summary,
            c.model_name,
            json.dumps(c.inspiration_ids),
            json.dumps(c.report.to_json()) if c.report else None,
            json.dumps(c.embedding) if c.embedding is not None else None,
            c.behavior_signature,
            int(c.behavior_duplicate),
            c.children_count,
            int(c.in_archive),
            json.dumps(c.metadata),
            c.timestamp or time.time(),
            c.workspace_kind,
        )

    @staticmethod
    def _from_row(row: tuple) -> Candidate:
        (
            cid, code, generation, parent_id, island_idx, operator, change_title,
            change_summary, model_name, inspiration_ids, report, embedding,
            behavior_signature, behavior_duplicate, children_count, in_archive,
            metadata, timestamp, workspace_kind,
        ) = row
        return Candidate(
            id=cid,
            code=code,
            generation=generation,
            parent_id=parent_id,
            island_idx=island_idx,
            operator=operator,
            change_title=change_title or "",
            change_summary=change_summary or "",
            model_name=model_name or "",
            inspiration_ids=json.loads(inspiration_ids) if inspiration_ids else [],
            report=EvalReport.from_json(json.loads(report)) if report else None,
            embedding=json.loads(embedding) if embedding else None,
            behavior_signature=behavior_signature,
            behavior_duplicate=bool(behavior_duplicate),
            children_count=children_count,
            in_archive=bool(in_archive),
            metadata=json.loads(metadata) if metadata else {},
            timestamp=timestamp,
            workspace_kind=workspace_kind or "file",
        )

    _COLS = (
        "id, code, generation, parent_id, island_idx, operator, change_title, "
        "change_summary, model_name, inspiration_ids, report, embedding, "
        "behavior_signature, behavior_duplicate, children_count, in_archive, "
        "metadata, timestamp, workspace_kind"
    )

    # -- insertion & island assignment ----------------------------------------

    def insert(self, cand: Candidate) -> None:
        """Insert a candidate; assigns an island if island_idx < 0 and maintains
        the parent's children_count (upstream behavior)."""
        if cand.island_idx < 0:
            cand.island_idx = self._assign_island(cand)
        self._conn.execute(
            f"INSERT INTO candidates ({self._COLS}) VALUES "
            f"({','.join('?' * 19)})",
            self._to_row(cand),
        )
        if cand.parent_id:
            self._conn.execute(
                "UPDATE candidates SET children_count = children_count + 1 "
                "WHERE id = ?",
                (cand.parent_id,),
            )
        self._conn.commit()

    def _assign_island(self, cand: Candidate) -> int:
        # Upstream default assignment: inherit the parent's island; otherwise a
        # passed candidate goes to the first island lacking one; else random.
        if cand.parent_id:
            parent = self.get(cand.parent_id)
            if parent is not None:
                return parent.island_idx
        if cand.passed:
            for idx in range(self.cfg.num_islands):
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM candidates WHERE island_idx = ? AND "
                    "json_extract(report, '$.passed') = 1",
                    (idx,),
                ).fetchone()
                if row[0] == 0:
                    return idx
        return int(np.random.randint(self.cfg.num_islands))

    def seed_all_islands(self, seed: Candidate) -> list[Candidate]:
        """Copy the evaluated seed candidate into every island (upstream
        CopyInitialProgramIslandStrategy). Returns all inserted copies."""
        inserted = []
        for idx in range(self.cfg.num_islands):
            copy = Candidate(
                id=seed.id if idx == seed.island_idx else Candidate.new_id(),
                code=seed.code,
                generation=0,
                parent_id=None,
                island_idx=idx,
                operator="seed",
                change_title=seed.change_title,
                change_summary=seed.change_summary,
                report=seed.report,
                embedding=seed.embedding,
                behavior_signature=seed.behavior_signature,
                metadata={**seed.metadata, "seed_copy_of": seed.id}
                if idx != seed.island_idx
                else dict(seed.metadata),
                workspace_kind=seed.workspace_kind,
            )
            self.insert(copy)
            inserted.append(copy)
        return inserted

    # -- queries -------------------------------------------------------------

    def get(self, cand_id: str) -> Candidate | None:
        row = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE id = ?", (cand_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def all_candidates(self) -> list[Candidate]:
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates ORDER BY generation, timestamp"
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def island_view(self, island_idx: int) -> IslandView:
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE island_idx = ? "
            "AND report IS NOT NULL ORDER BY generation, timestamp",
            (island_idx,),
        ).fetchall()
        return IslandView(island_idx=island_idx, candidates=[self._from_row(r) for r in rows])

    def best(self) -> Candidate | None:
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE "
            "json_extract(report, '$.passed') = 1"
        ).fetchall()
        cands = [self._from_row(r) for r in rows]
        return max(cands, key=lambda c: c.fitness) if cands else None

    def latest_failed(self) -> Candidate | None:
        """Most recent failed candidate not yet targeted by a repair."""
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE report IS NOT NULL AND "
            "json_extract(report, '$.passed') = 0 AND "
            "json_extract(metadata, '$.repair_attempted') IS NULL "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return self._from_row(rows) if rows else None

    def mark_repair_attempted(self, cand_id: str) -> None:
        cand = self.get(cand_id)
        if cand is None:
            return
        cand.metadata["repair_attempted"] = True
        self._conn.execute(
            "UPDATE candidates SET metadata = ? WHERE id = ?",
            (json.dumps(cand.metadata), cand_id),
        )
        self._conn.commit()

    def all_signatures(self, island_idx: int | None = None) -> list[str]:
        if island_idx is None:
            rows = self._conn.execute(
                "SELECT behavior_signature FROM candidates "
                "WHERE behavior_signature IS NOT NULL"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT behavior_signature FROM candidates "
                "WHERE behavior_signature IS NOT NULL AND island_idx = ?",
                (island_idx,),
            ).fetchall()
        return [r[0] for r in rows]

    def update_candidate_flags(self, cand: Candidate) -> None:
        """Persist mutable flags observers may set before/after insertion."""
        self._conn.execute(
            "UPDATE candidates SET behavior_signature = ?, behavior_duplicate = ?,"
            " embedding = ?, metadata = ? WHERE id = ?",
            (
                cand.behavior_signature,
                int(cand.behavior_duplicate),
                json.dumps(cand.embedding) if cand.embedding is not None else None,
                json.dumps(cand.metadata),
                cand.id,
            ),
        )
        self._conn.commit()

    # -- archive maintenance ([parity]: "fitness" strategy == keep top-N) ----

    def refresh_archive(self) -> None:
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE "
            "json_extract(report, '$.passed') = 1 AND behavior_duplicate = 0"
        ).fetchall()
        cands = [self._from_row(r) for r in rows]
        cands.sort(key=lambda c: c.fitness, reverse=True)
        keep = {c.id for c in cands[: self.cfg.archive_size]}
        self._conn.execute("UPDATE candidates SET in_archive = 0")
        if keep:
            marks = ",".join("?" * len(keep))
            self._conn.execute(
                f"UPDATE candidates SET in_archive = 1 WHERE id IN ({marks})",
                tuple(keep),
            )
        self._conn.commit()

    # -- migration ([parity]: elitist random migration) -----------------------

    def maybe_migrate(
        self, generation: int, rng: np.random.Generator | None = None
    ) -> int:
        """Every migration_interval generations move migration_rate of each
        island to a random other island. Excludes generation-0 seeds, failed
        candidates and (with island_elitism) each island's best. Returns the
        number of migrated candidates."""
        cfg = self.cfg
        if cfg.migration_rate <= 0 or cfg.num_islands < 2:
            return 0
        if generation == 0 or generation % cfg.migration_interval != 0:
            return 0
        rng = rng or np.random.default_rng()
        # Select all migrants from a snapshot first, then apply, so a migrant
        # cannot be re-selected from its target island within the same event.
        moves: list[tuple[Candidate, int, int]] = []
        for idx in range(cfg.num_islands):
            view = self.island_view(idx)
            eligible = [
                c for c in view.passed_candidates if c.generation > 0
            ]
            if cfg.island_elitism and view.passed_candidates:
                best = max(view.passed_candidates, key=lambda c: c.fitness)
                eligible = [c for c in eligible if c.id != best.id]
            if not eligible:
                continue
            n_migrate = max(1, int(len(view.candidates) * cfg.migration_rate))
            n_migrate = min(n_migrate, len(eligible))
            chosen = rng.choice(len(eligible), size=n_migrate, replace=False)
            for ci in np.atleast_1d(chosen):
                cand = eligible[int(ci)]
                targets = [d for d in range(cfg.num_islands) if d != idx]
                moves.append((cand, idx, int(rng.choice(targets))))
        for cand, source, target in moves:
            history = cand.metadata.setdefault("migration_history", [])
            history.append(
                {"generation": generation, "from": source, "to": target}
            )
            self._conn.execute(
                "UPDATE candidates SET island_idx = ?, metadata = ? WHERE id = ?",
                (target, json.dumps(cand.metadata), cand.id),
            )
        self._conn.commit()
        return len(moves)

    def close(self) -> None:
        self._conn.close()
