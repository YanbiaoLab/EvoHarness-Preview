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
import math
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
# Core stays policy-free: it understands ONE generic flag. A quarantined
# candidate is recorded but excluded from selection, archive and repair;
# WHY it is quarantined lives outside Core (evaluation layer sets the flag,
# typed, single writer). Orthogonal to `passed`: an ordinary failed
# candidate is not quarantined — it stays repairable as before.
_NOT_QUARANTINED_SQL = (
    "COALESCE(json_extract(metadata, '$.quarantined'), 0) != 1"
)
_SELECTABLE_SQL = (
    f"json_extract(report, '$.passed') = 1 AND {_NOT_QUARANTINED_SQL}"
)
_ARCHIVE_ELIGIBLE_SQL = _SELECTABLE_SQL
_REPAIRABLE_SQL = (
    f"json_extract(report, '$.passed') = 0 AND {_NOT_QUARANTINED_SQL}"
)


def _wire_number(value: object, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"EvalReport {name} must be numeric")
    number = float(value)
    if non_negative and number < 0:
        raise ValueError(f"EvalReport {name} must be non-negative")
    return number


def _wire_int(value: object, name: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"EvalReport {name} must be an integer")
    if non_negative and value < 0:
        raise ValueError(f"EvalReport {name} must be non-negative")
    return value


def _wire_mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"EvalReport {name} must be a JSON object")
    return value


def _wire_string(value: object, name: str, *, optional: bool = False):
    if optional and value is None:
        return None
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise TypeError(f"EvalReport {name} must be a string{suffix}")
    return value


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
    # Precision of `fitness`, reported by the task (see serve.Grade). 0
    # means "not reported": every LCB below then equals the point estimate,
    # so a domain that stays silent keeps today's behaviour exactly.
    n_units: int = 0
    # Units whose measurements are safe to use as evidence. ``None`` means
    # the legacy wire did not report trust separately; a declared
    # MeasurementSpec must not silently treat that as complete coverage.
    trustworthy_units: int | None = None
    sem: float = 0.0
    # 结构化故障词表(见 evaluation/faults.py),None = 服务未上报(legacy)。
    # Wire schema 仍为 v1:可选键,缺省向后兼容。
    fault_kind: str | None = None

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
            "n_units": self.n_units,
            "trustworthy_units": self.trustworthy_units,
            "sem": self.sem,
            "fault_kind": self.fault_kind,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EvalReport":
        if not isinstance(d, dict):
            raise TypeError("EvalReport payload must be a JSON object")
        if "schema_version" in d and d["schema_version"] != 1:
            raise ValueError(
                f"unsupported EvalReport schema: {d['schema_version']!r}"
            )
        missing = [k for k in _REQUIRED_REPORT_KEYS if k not in d]
        if missing:
            raise ValueError(f"EvalReport missing required keys: {missing}")
        if isinstance(d["passed"], bool) is False:
            raise TypeError("EvalReport passed must be bool")
        fitness = _wire_number(d["fitness"], "fitness")
        fault_kind = d.get("fault_kind")
        if fault_kind is not None and (
            not isinstance(fault_kind, str) or not fault_kind.strip()
        ):
            raise TypeError("EvalReport fault_kind must be a string or None")
        if d["passed"] and fault_kind is not None:
            raise ValueError("passed report cannot carry fault_kind")
        if np.isnan(fitness) or np.isinf(fitness):
            # Upstream marks NaN/inf combined_score as incorrect; we reject the
            # value into a failed report instead of storing it.
            return cls(
                fitness=0.0,
                passed=False,
                fault=f"non-finite fitness: {d['fitness']}",
            )
        visible_metrics = _wire_mapping(
            d.get("visible_metrics", {}), "visible_metrics"
        )
        hidden_metrics = _wire_mapping(
            d.get("hidden_metrics", {}), "hidden_metrics"
        )
        structured_feedback = d.get("structured_feedback")
        if structured_feedback is not None:
            structured_feedback = _wire_mapping(
                structured_feedback, "structured_feedback"
            )
        trustworthy_units = d.get("trustworthy_units")
        if trustworthy_units is not None:
            trustworthy_units = _wire_int(
                trustworthy_units,
                "trustworthy_units",
                non_negative=True,
            )
        return cls(
            fitness=fitness,
            passed=d["passed"],
            fault=_wire_string(d.get("fault"), "fault", optional=True),
            visible_metrics=visible_metrics,
            hidden_metrics=hidden_metrics,
            notes=_wire_string(d.get("notes", ""), "notes"),
            structured_feedback=structured_feedback,
            artifacts_ref=_wire_string(
                d.get("artifacts_ref"), "artifacts_ref", optional=True
            ),
            stdout_log=_wire_string(d.get("stdout_log", ""), "stdout_log"),
            stderr_log=_wire_string(d.get("stderr_log", ""), "stderr_log"),
            stage_reached=_wire_int(
                d.get("stage_reached", 3), "stage_reached", non_negative=True
            ),
            execution_time=_wire_number(
                d.get("execution_time", 0.0),
                "execution_time",
                non_negative=True,
            ),
            eval_cost_usd=_wire_number(
                d.get("eval_cost_usd", 0.0),
                "eval_cost_usd",
                non_negative=True,
            ),
            n_units=_wire_int(
                d.get("n_units", 0), "n_units", non_negative=True
            ),
            trustworthy_units=trustworthy_units,
            sem=_wire_number(d.get("sem", 0.0), "sem", non_negative=True),
            fault_kind=fault_kind,
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
    def quarantined(self) -> bool:
        # Typed check: only the evidence producer writes this flag (bool
        # True); anything else means "not quarantined by us".
        return self.metadata.get("quarantined") is True

    @property
    def passed(self) -> bool:
        return (
            bool(self.report and self.report.passed)
            and not self.quarantined
        )

    @property
    def archive_eligible(self) -> bool:
        return self.passed

    def fitness_lcb(self, z: float = 1.0) -> float:
        """Pessimistic fitness, for COMMITMENT only.

        Picking the argmax of several noisy estimates scores the noise along
        with the program: the winner is partly whoever got lucky. Measured
        on this project's own IMO runs — final selection takes the max of 3
        candidates on 12 items (binomial sem ~0.14) and the chosen program
        lost 0.206 points between validation and test.

        Parent selection must NOT use this. Exploration wants optimism;
        discounting a high-variance candidate is how a search stops taking
        the risks that produce breakthroughs. Pessimism belongs only where
        we stop searching and commit.

        Domains that report no `sem` get their point estimate back unchanged.
        """
        if self.report is None:
            return 0.0
        return self.report.fitness - z * self.report.sem
    
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

    Storage blips are survived, not re-raised: run modmul_r15 died 17
    hours of state into `attempt to write a readonly database` when the
    box's overlay storage hiccuped for a moment (the same volume once
    returned EIO mid-benchmark-read). The filesystem was writable again
    by the time anyone looked. A transient fault through a stale fd needs
    a RECONNECT, not just a retry — the old descriptor can stay pinned to
    the read-only view after the volume recovers.
    """

    _TRANSIENT = ("readonly database", "disk i/o error")
    _BACKOFF_S = (0.5, 2.0, 8.0)

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        path: str = ":memory:",
        readonly: bool = False,
    ):
        self._conn = conn
        self._lock = lock
        self._path = path
        self._readonly = readonly

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
        # A read-only connection must come back read-only. The plain
        # reconnect would silently upgrade a viewer to writable -- the
        # exact breach the read-only mode exists to prevent.
        if self._readonly:
            self._conn = sqlite3.connect(
                f"file:{self._path}?mode=ro", uri=True,
                check_same_thread=False,
            )
        else:
            self._conn = sqlite3.connect(
                self._path, check_same_thread=False
            )

    def _is_transient(self, exc: sqlite3.OperationalError) -> bool:
        text = str(exc).lower()
        if self._path == ":memory:":
            return False
        if self._readonly and "readonly database" in text:
            # On a read-only connection this is the contract working, not
            # a storage blip: retrying would stall a viewer for the whole
            # backoff schedule and then fail anyway.
            return False
        return any(t in text for t in self._TRANSIENT)

    def execute(self, *args, **kwargs) -> _Rows:
        with self._lock:
            for pause in self._BACKOFF_S:
                try:
                    return _Rows(
                        self._conn.execute(*args, **kwargs).fetchall()
                    )
                except sqlite3.OperationalError as exc:
                    if not self._is_transient(exc):
                        raise
                    time.sleep(pause)
                    self._reconnect()
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

    def __init__(
        self,
        cfg: PopulationConfig,
        path: Path | str = ":memory:",
        readonly: bool = False,
    ):
        self.cfg = cfg
        self.readonly = readonly
        # Agent tools read candidates from the runtime's worker threads, and
        # sqlite refuses a connection outside its creating thread. One
        # connection guarded by one lock keeps the store single-writer while
        # letting those reads through; the evolution loop itself still
        # mutates only from the main thread.
        self._lock = threading.RLock()
        if readonly:
            # mode=ro refuses to create the file, runs no DDL, and makes
            # sqlite itself reject any write this class fails to guard.
            self._conn = _LockedConnection(
                sqlite3.connect(
                    f"file:{path}?mode=ro", uri=True,
                    check_same_thread=False,
                ),
                self._lock, path=str(path), readonly=True,
            )
            cols = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(candidates)")
            }
            if "workspace_kind" not in cols:
                raise RuntimeError(
                    f"{path} predates the current schema; open it once with "
                    "its own writer (resuming the run migrates it) before "
                    "viewing it read-only"
                )
            return
        self._conn = _LockedConnection(
            sqlite3.connect(str(path), check_same_thread=False), self._lock,
            path=str(path),
        )
        self._conn.executescript(_SCHEMA)
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(candidates)")}
        if "workspace_kind" not in cols:
            self._conn.execute(
                "ALTER TABLE candidates ADD COLUMN "
                "workspace_kind TEXT NOT NULL DEFAULT 'file'"
            )
        self._conn.commit()

    @classmethod
    def open_readonly(
        cls, path: Path | str, cfg: PopulationConfig | None = None
    ) -> "PopulationStore":
        """The viewer entry point.

        Every display surface used to instantiate the WRITABLE store, whose
        constructor runs schema DDL and creates the file when missing -- so
        merely LOOKING at a run could write to it, and a typo'd path
        manufactured an empty database where the viewer then reported an
        empty run instead of an error.
        """
        return cls(cfg or PopulationConfig(), path, readonly=True)

    def _assert_writable(self) -> None:
        if self.readonly:
            raise PermissionError(
                "this store was opened read-only (a viewer surface); "
                "mutations belong to the run's own writer"
            )

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

    def note_attempt(self, parent_id: str) -> None:
        """Charge a parent for an attempt at PLAN time.

        children_count is what stops the selector grinding on one parent —
        its weight carries a 1/(1+children_count) factor. Counting on insert
        made that feedback close once per generation, which was fine while a
        generation held one proposal. With eval_batch_size 16 every proposal
        in a batch is planned before any is graded, so all sixteen saw a
        count of zero: run modmul_r7 drew all fifteen offspring from one
        parent, and that parent finished the generation with
        children_count=15 that had never influenced anything.

        Charging at plan time also reads better than it did: what should
        discourage a parent is attempts spent on it, not children that
        happened to survive.
        """
        self._assert_writable()
        self._conn.execute(
            "UPDATE candidates SET children_count = children_count + 1 "
            "WHERE id = ?",
            (parent_id,),
        )
        self._conn.commit()

    def insert(self, cand: Candidate) -> None:
        """Insert a candidate; assigns an island if island_idx < 0. The
        parent's children_count is charged at plan time, by note_attempt."""
        self._assert_writable()
        if cand.island_idx < 0:
            cand.island_idx = self._assign_island(cand)
        self._conn.execute(
            f"INSERT INTO candidates ({self._COLS}) VALUES "
            f"({','.join('?' * 19)})",
            self._to_row(cand),
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
                    f"({_SELECTABLE_SQL})",
                    (idx,),
                ).fetchone()
                if row[0] == 0:
                    return idx
        return int(np.random.randint(self.cfg.num_islands))

    def seed_all_islands(
        self, seed: Candidate, islands: list[int] | None = None
    ) -> list[Candidate]:
        """Copy the evaluated seed candidate into every island (upstream
        CopyInitialProgramIslandStrategy). Returns all inserted copies.

        `islands` narrows the copy to a subset. Copying everywhere erases
        heterogeneous seeding whenever the primary outscores the natives,
        so SearchLoop uses it only to backfill islands that would otherwise
        have no parent."""
        self._assert_writable()
        inserted = []
        for idx in (
            range(self.cfg.num_islands) if islands is None else islands
        ):
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
            f"({_SELECTABLE_SQL})"
        ).fetchall()
        cands = [self._from_row(r) for r in rows]
        return max(cands, key=lambda c: c.fitness) if cands else None

    def latest_failed(self) -> Candidate | None:
        """Most recent failed candidate not yet targeted by a repair."""
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE report IS NOT NULL AND "
            f"({_REPAIRABLE_SQL}) AND "
            "json_extract(metadata, '$.repair_attempted') IS NULL "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return self._from_row(rows) if rows else None

    def mark_repair_attempted(self, cand_id: str) -> None:
        self._assert_writable()
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
        self._assert_writable()
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

    def _metric(self, cand: Candidate, key: str) -> float | None:
        """Numeric visible metric, or None when absent or non-numeric."""
        if not key or cand.report is None:
            return None
        value = cand.report.visible_metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def _feature(self, cand: Candidate) -> float | None:
        return self._metric(cand, self.cfg.archive_feature_metric)

    def _quality(self, cand: Candidate) -> float:
        value = self._metric(cand, self.cfg.archive_feature_quality)
        return cand.fitness if value is None else value

    def _bucket_elites(self, cands: list[Candidate]) -> list[Candidate]:
        """Best candidate per feature bucket, best bucket first.

        Reserve fewer slots than there are occupied buckets and the low-value
        buckets fall off the end -- which are exactly the ones worth keeping
        when the leaders are pinned against the ceiling. Size the reserve to
        the number of buckets the axis can hold, not to a fraction of the
        archive.
        """
        width = self.cfg.archive_feature_bucket
        if not self.cfg.archive_feature_metric or width <= 0:
            return []
        best: dict[int, Candidate] = {}
        for cand in cands:
            value = self._feature(cand)
            if value is None:
                continue
            idx = math.floor(value / width)
            held = best.get(idx)
            if held is None or (self._quality(cand), cand.id) > (
                self._quality(held), held.id
            ):
                best[idx] = cand
        return sorted(
            best.values(), key=lambda c: (self._quality(c), c.id), reverse=True
        )

    def refresh_archive(self) -> None:
        self._assert_writable()
        rows = self._conn.execute(
            f"SELECT {self._COLS} FROM candidates WHERE "
            f"({_ARCHIVE_ELIGIBLE_SQL}) "
            "AND behavior_duplicate = 0"
        ).fetchall()
        cands = [self._from_row(r) for r in rows]
        # Equal fitness, cheaper on the declared feature axis wins. A refactor
        # that frees resource without moving the score is progress, and a
        # pure-fitness sort cannot say so.
        cands.sort(
            key=lambda c: (
                c.fitness,
                -(self._feature(c) if self._feature(c) is not None
                  else math.inf),
                c.id,
            ),
            reverse=True,
        )
        size = self.cfg.archive_size
        keep: dict[str, None] = {}
        if self.cfg.archive_update_strategy == "feature_buckets":
            reserve = max(0, min(self.cfg.archive_feature_reserve, size))
            for cand in self._bucket_elites(cands)[:reserve]:
                keep[cand.id] = None
        for cand in cands:
            if len(keep) >= size:
                break
            keep[cand.id] = None
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
        self._assert_writable()
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
