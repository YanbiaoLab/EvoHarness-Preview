"""SQLite persistence for the proof graph: memoization, atomicity, leases.

Three primary design properties are managed here:

**Memoization is enforced by a UNIQUE constraint.** `goals.identity` is unique,
so two goals with identical mathematical identities map to a single node in the
DAG. This deduplication prevents duplicate search efforts across subgoals.

**Decompositions are transactional.** Creating subgoals, checking acyclicity,
and writing decomposition edges occur in a single atomic transaction, preventing
partially created decompositions.

**Leases are separated from proof status.** `Goal.status` reflects semantic state
(open, proved, exhausted), while transient scheduling state (lease holder and
expiry) is maintained in separate columns.

Storage uses SQLite with WAL mode to provide atomic multi-row updates and crash
consistency across runs.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from .graph import (
    Attempt,
    Decomposition,
    DecompositionStatus,
    Goal,
    GoalStatus,
    Outcome,
    check_acyclic,
    decomposition_status_from,
    goal_status_from,
)
from .sketch import Sketch

_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id                  TEXT PRIMARY KEY,
    identity            TEXT NOT NULL UNIQUE,
    statement           TEXT NOT NULL,
    status              TEXT NOT NULL,
    exhausted_at_budget REAL,
    exhausted_at_solver TEXT,
    lease_owner         TEXT,
    lease_expires_at    REAL,
    created_at          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decompositions (
    id              TEXT PRIMARY KEY,
    goal_id         TEXT NOT NULL REFERENCES goals(id),
    status          TEXT NOT NULL,
    rejected_reason TEXT NOT NULL DEFAULT '',
    sketch_json     TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decomposition_subgoals (
    decomposition_id TEXT NOT NULL REFERENCES decompositions(id),
    ordinal          INTEGER NOT NULL,
    subgoal_id       TEXT NOT NULL REFERENCES goals(id),
    PRIMARY KEY (decomposition_id, ordinal)
);
CREATE TABLE IF NOT EXISTS attempts (
    id           TEXT PRIMARY KEY,
    goal_id      TEXT NOT NULL REFERENCES goals(id),
    outcome      TEXT NOT NULL,
    proof_text   TEXT,
    run_dir      TEXT,
    evidence_ref TEXT,
    cost         REAL NOT NULL DEFAULT 0,
    note         TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS roots (
    goal_id TEXT PRIMARY KEY REFERENCES goals(id),
    label   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decomp_goal ON decompositions(goal_id);
CREATE INDEX IF NOT EXISTS ix_edge_subgoal ON decomposition_subgoals(subgoal_id);
CREATE INDEX IF NOT EXISTS ix_attempt_goal ON attempts(goal_id);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _goal(row: sqlite3.Row) -> Goal:
    return Goal(
        id=row["id"],
        identity=row["identity"],
        statement=row["statement"],
        status=GoalStatus(row["status"]),
        exhausted_at_budget=row["exhausted_at_budget"],
        exhausted_at_solver=row["exhausted_at_solver"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
    )


class ProofGraphStore:
    """The graph's one source of truth. Readable views are derived, never kept."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        # Durability matters more than throughput here: the recovery test kills
        # the process mid-run and expects the graph to be intact.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- goals ----------------------------------------------------------------

    def upsert_goal(self, identity: str, statement: str) -> Goal:
        """The memoization point: one identity is one node, always.

        A caller that re-derives the same goal -- from another branch, or after
        a resume -- gets the existing node back with whatever progress it has
        already accumulated, rather than a fresh one that has to be reproved.
        """

        with self._conn:
            self._conn.execute(
                "INSERT INTO goals (id, identity, statement, status, created_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(identity) DO NOTHING",
                (
                    _new_id("goal"),
                    identity,
                    statement,
                    GoalStatus.OPEN.value,
                    time.time(),
                ),
            )
        found = self.goal_by_identity(identity)
        assert found is not None  # the insert above guarantees a row
        return found

    def goal(self, goal_id: str) -> Goal:
        row = self._conn.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such goal: {goal_id}")
        return _goal(row)

    def goal_by_identity(self, identity: str) -> Goal | None:
        row = self._conn.execute(
            "SELECT * FROM goals WHERE identity = ?", (identity,)
        ).fetchone()
        return _goal(row) if row else None

    def open_goals(self) -> list[Goal]:
        rows = self._conn.execute(
            "SELECT * FROM goals WHERE status = ? ORDER BY created_at",
            (GoalStatus.OPEN.value,),
        ).fetchall()
        return [_goal(row) for row in rows]

    def add_root(self, goal_id: str, label: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO roots (goal_id, label) VALUES (?, ?)",
                (goal_id, label),
            )

    # -- graph shape ----------------------------------------------------------

    def parents_of(self, goal_id: str) -> list[str]:
        """Goals that own a decomposition having this goal as a subgoal."""

        rows = self._conn.execute(
            "SELECT DISTINCT d.goal_id FROM decomposition_subgoals e"
            " JOIN decompositions d ON d.id = e.decomposition_id"
            " WHERE e.subgoal_id = ?",
            (goal_id,),
        ).fetchall()
        return [row["goal_id"] for row in rows]

    def ancestor_identities(self, goal_id: str) -> set[str]:
        """Every identity reachable upward. The acyclicity check needs all of
        them: the degenerate decomposition restates the GRANDparent, not the
        parent, so stopping one level up would let it through.
        """

        seen: set[str] = set()
        frontier = [goal_id]
        while frontier:
            current = frontier.pop()
            for parent_id in self.parents_of(current):
                parent = self.goal(parent_id)
                if parent.identity in seen:
                    continue
                seen.add(parent.identity)
                frontier.append(parent_id)
        return seen

    # -- decompositions -------------------------------------------------------

    def add_decomposition(
        self,
        goal_id: str,
        subgoals: Sequence[tuple[str, str]],
        *,
        sketch: Sketch | None = None,
        status: DecompositionStatus = DecompositionStatus.PROPOSED,
    ) -> Decomposition:
        """Create or reuse the subgoals, check acyclicity, write the edges.

        `subgoals` is a sequence of (identity, statement). All of it happens in
        one transaction: a decomposition that landed without its edges would be
        an AND node the controller can never close, and nothing would say so.
        """

        if not subgoals:
            raise ValueError(
                "a decomposition with no subgoals would complete vacuously"
            )
        parent = self.goal(goal_id)
        ancestors = self.ancestor_identities(goal_id) | {parent.identity}
        check_acyclic(ancestors, [identity for identity, _ in subgoals])

        decomposition_id = _new_id("dec")
        now = time.time()
        with self._conn:
            subgoal_ids: list[str] = []
            for identity, statement in subgoals:
                self._conn.execute(
                    "INSERT INTO goals (id, identity, statement, status,"
                    " created_at) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(identity) DO NOTHING",
                    (
                        _new_id("goal"),
                        identity,
                        statement,
                        GoalStatus.OPEN.value,
                        now,
                    ),
                )
                row = self._conn.execute(
                    "SELECT id FROM goals WHERE identity = ?", (identity,)
                ).fetchone()
                subgoal_ids.append(row["id"])
            self._conn.execute(
                "INSERT INTO decompositions (id, goal_id, status, sketch_json,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    decomposition_id,
                    goal_id,
                    status.value,
                    json.dumps(sketch.to_json(), ensure_ascii=False)
                    if sketch else "",
                    now,
                ),
            )
            self._conn.executemany(
                "INSERT INTO decomposition_subgoals (decomposition_id, ordinal,"
                " subgoal_id) VALUES (?, ?, ?)",
                [
                    (decomposition_id, ordinal, subgoal_id)
                    for ordinal, subgoal_id in enumerate(subgoal_ids)
                ],
            )
        return self.decomposition(decomposition_id)

    def decomposition(self, decomposition_id: str) -> Decomposition:
        row = self._conn.execute(
            "SELECT * FROM decompositions WHERE id = ?", (decomposition_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such decomposition: {decomposition_id}")
        edges = self._conn.execute(
            "SELECT subgoal_id FROM decomposition_subgoals"
            " WHERE decomposition_id = ? ORDER BY ordinal",
            (decomposition_id,),
        ).fetchall()
        return Decomposition(
            id=row["id"],
            goal_id=row["goal_id"],
            subgoal_ids=tuple(edge["subgoal_id"] for edge in edges),
            status=DecompositionStatus(row["status"]),
            rejected_reason=row["rejected_reason"],
        )

    def decompositions_of(self, goal_id: str) -> list[Decomposition]:
        rows = self._conn.execute(
            "SELECT id FROM decompositions WHERE goal_id = ? ORDER BY created_at",
            (goal_id,),
        ).fetchall()
        return [self.decomposition(row["id"]) for row in rows]

    def sketch_of(self, decomposition_id: str) -> Sketch | None:
        """The Lean artifact behind a decomposition, or None if it never had one.

        Deliberately separate from `decomposition()`: `graph.py` stays free of
        Lean, and a caller that only needs the shape of the DAG should not have
        to deserialize a proof term to get it.
        """

        row = self._conn.execute(
            "SELECT sketch_json FROM decompositions WHERE id = ?",
            (decomposition_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"no such decomposition: {decomposition_id}")
        raw = row["sketch_json"]
        return Sketch.from_json(json.loads(raw)) if raw else None

    def set_decomposition_status(
        self,
        decomposition_id: str,
        status: DecompositionStatus,
        *,
        reason: str = "",
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE decompositions SET status = ?, rejected_reason = ?"
                " WHERE id = ?",
                (status.value, reason, decomposition_id),
            )

    # -- attempts -------------------------------------------------------------

    def record_attempt(
        self,
        goal_id: str,
        outcome: Outcome,
        *,
        proof_text: str | None = None,
        run_dir: str | None = None,
        evidence_ref: str | None = None,
        cost: float = 0.0,
        note: str = "",
    ) -> Attempt:
        attempt_id = _new_id("att")
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO attempts (id, goal_id, outcome, proof_text,"
                " run_dir, evidence_ref, cost, note, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attempt_id,
                    goal_id,
                    outcome.value,
                    proof_text,
                    run_dir,
                    evidence_ref,
                    cost,
                    note,
                    now,
                ),
            )
        return Attempt(
            id=attempt_id,
            goal_id=goal_id,
            outcome=outcome,
            proof_text=proof_text,
            run_dir=run_dir,
            evidence_ref=evidence_ref,
            cost=cost,
            note=note,
            created_at=now,
        )

    def total_cost(self) -> float:
        """Everything spent on this graph, derived rather than kept.

        A separate "spent so far" counter would be a second source of truth,
        and after a resume the two would disagree with nothing to say which is
        right. The attempts table already records every charge, so the ledger
        is a SUM over it -- correct after a crash for free.
        """

        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost), 0.0) AS total FROM attempts"
        ).fetchone()
        return float(row["total"])

    def attempts_of(self, goal_id: str) -> list[Attempt]:
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE goal_id = ? ORDER BY created_at",
            (goal_id,),
        ).fetchall()
        return [
            Attempt(
                id=row["id"],
                goal_id=row["goal_id"],
                outcome=Outcome(row["outcome"]),
                proof_text=row["proof_text"],
                run_dir=row["run_dir"],
                evidence_ref=row["evidence_ref"],
                cost=row["cost"],
                note=row["note"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- propagation ----------------------------------------------------------

    def propagate(
        self,
        goal_id: str,
        *,
        max_capability_attempts: int,
        budget_spent: float = 0.0,
        solver_level: str = "",
    ) -> None:
        """Recompute this goal and everything above it.

        A work queue rather than recursion, because this is a DAG: one goal can
        be a subgoal of several decompositions, and a shared lemma closing must
        reach every parent waiting on it.

        `budget_spent` and `solver_level` are recorded whenever a goal turns
        EXHAUSTED. Without them a later run with more budget has no basis to
        reopen the goal, and the search would route around it forever.
        """

        pending = [goal_id]
        while pending:
            current_id = pending.pop()
            current = self.goal(current_id)
            decompositions = self.decompositions_of(current_id)

            advanced: list[DecompositionStatus] = []
            for decomposition in decompositions:
                subgoal_statuses = [
                    self.goal(subgoal_id).status
                    for subgoal_id in decomposition.subgoal_ids
                ]
                new_status = decomposition_status_from(
                    current=decomposition.status,
                    subgoal_statuses=subgoal_statuses,
                )
                if new_status is not decomposition.status:
                    self.set_decomposition_status(decomposition.id, new_status)
                advanced.append(new_status)

            status = goal_status_from(
                attempt_outcomes=[a.outcome for a in self.attempts_of(current_id)],
                decomposition_statuses=advanced,
                max_capability_attempts=max_capability_attempts,
            )
            if status is current.status:
                continue

            with self._conn:
                if status is GoalStatus.EXHAUSTED:
                    self._conn.execute(
                        "UPDATE goals SET status = ?, exhausted_at_budget = ?,"
                        " exhausted_at_solver = ? WHERE id = ?",
                        (status.value, budget_spent, solver_level, current_id),
                    )
                else:
                    # Reopening clears the exhaustion context: it described a
                    # verdict that no longer holds.
                    self._conn.execute(
                        "UPDATE goals SET status = ?, exhausted_at_budget = NULL,"
                        " exhausted_at_solver = NULL WHERE id = ?",
                        (status.value, current_id),
                    )
            pending.extend(self.parents_of(current_id))

    # -- leases ---------------------------------------------------------------

    def claim(
        self, goal_id: str, owner: str, *, ttl_s: float, now: float | None = None
    ) -> bool:
        """Take the lease if it is free or expired. A conditional UPDATE, so two
        workers racing for the same goal cannot both win.
        """

        now = time.time() if now is None else now
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE goals SET lease_owner = ?, lease_expires_at = ?"
                " WHERE id = ? AND (lease_owner IS NULL"
                "                   OR lease_expires_at IS NULL"
                "                   OR lease_expires_at <= ?)",
                (owner, now + ttl_s, goal_id, now),
            )
        return cursor.rowcount == 1

    def release(self, goal_id: str, owner: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE goals SET lease_owner = NULL, lease_expires_at = NULL"
                " WHERE id = ? AND lease_owner = ?",
                (goal_id, owner),
            )


__all__ = ["ProofGraphStore"]
