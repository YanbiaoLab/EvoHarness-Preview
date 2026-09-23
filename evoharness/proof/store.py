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

**A graph has one scope, fixed when it is created.** Memoization hands back a
node with its accumulated status, so a graph whose premises changed between two
commands would let a PROVED earned under one set stand under another. The
`graph_scope` row records the premises; opening under different ones is
refused. See `scope.py`.

Storage uses SQLite with WAL mode to provide atomic multi-row updates and crash
consistency across runs.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

from .graph import (
    Attempt,
    Certification,
    Decomposition,
    DecompositionStatus,
    Goal,
    GoalStatus,
    GraphError,
    Outcome,
    check_acyclic,
    decomposition_status_from,
    goal_status_from,
)
from .envelope import SourceEnvelope
from .scope import GraphScope, ScopeError, ScopeMismatch, ScopeMissing
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
    lease_run_dir       TEXT,
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
CREATE TABLE IF NOT EXISTS certifications (
    id          TEXT PRIMARY KEY,
    goal_id     TEXT NOT NULL REFERENCES goals(id),
    ok          INTEGER NOT NULL,
    axioms_json TEXT NOT NULL DEFAULT '[]',
    reason      TEXT NOT NULL DEFAULT '',
    text_sha256 TEXT NOT NULL DEFAULT '',
    -- Nullable on purpose, and NOT defaulted to '': a row from before this
    -- column existed has an unknown route, while '' says there was none to
    -- pick. Defaulting would erase that difference on every old graph.
    decomposition_id TEXT,
    -- The trust level the axiom report put the finished proof at, read
    -- through the graph's AxiomPolicy. Nullable for the same reason as the
    -- route: NULL is "recorded before trust was kept", '' is "nothing
    -- compiled, so nothing to classify".
    trust       TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS roots (
    goal_id TEXT PRIMARY KEY REFERENCES goals(id),
    label   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decomp_goal ON decompositions(goal_id);
CREATE INDEX IF NOT EXISTS ix_edge_subgoal ON decomposition_subgoals(subgoal_id);
CREATE INDEX IF NOT EXISTS ix_attempt_goal ON attempts(goal_id);
CREATE INDEX IF NOT EXISTS ix_cert_goal ON certifications(goal_id);
-- Candidate files split into what the verifier checks (envelope.py). Written
-- once, before the first verification that uses it, and read back rather than
-- re-split: verify, assembly and publish must see byte-identical input.
CREATE TABLE IF NOT EXISTS envelopes (
    envelope_hash    TEXT PRIMARY KEY,
    envelope_json    TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    -- Where the original file lives, when known. The file itself stays there:
    -- The verifier keeps only its hash, in the publication's provenance.
    candidate_path   TEXT,
    created_at       REAL NOT NULL
);
-- One row or none. None means either a brand-new graph (the first open writes
-- it) or a graph built before scopes existed (opening it needs --adopt-scope).
CREATE TABLE IF NOT EXISTS graph_scope (
    id                      INTEGER PRIMARY KEY CHECK (id = 1),
    scope_version           INTEGER NOT NULL,
    environment             TEXT NOT NULL,
    base_id                 TEXT NOT NULL DEFAULT '',
    minimum_trust           TEXT NOT NULL,
    axiom_policy_version    INTEGER NOT NULL,
    identity_hasher         TEXT NOT NULL,
    preamble_sha256         TEXT NOT NULL,
    envelope_schema_version INTEGER NOT NULL,
    goalkey_schema_version  INTEGER NOT NULL,
    created_at              REAL NOT NULL
);
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
        lease_run_dir=row["lease_run_dir"],
    )


class ProofGraphStore:
    """The graph's one source of truth. Readable views are derived, never kept."""

    def __init__(
        self,
        path: Path | str,
        scope: GraphScope | None = None,
        *,
        adopt_scope: bool = False,
    ):
        """Open (or create) the graph at `path`.

        `scope` is what the caller will work under. Given, it is enforced: a
        new graph records it, a graph with a scope must match it field by
        field, and a graph with goals but no scope is refused unless
        `adopt_scope` says to record this one. `None` enforces nothing and is
        for callers that own the whole graph's lifetime themselves -- tests,
        and scripts that build a graph and throw it away. The CLI always
        passes one.
        """

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        # Durability matters more than throughput here: the recovery test kills
        # the process mid-run and expects the graph to be intact.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._add_missing_columns()
        self._conn.commit()
        try:
            self.scope = self._settle_scope(scope, adopt_scope)
        except ScopeError:
            self._conn.close()
            raise

    @staticmethod
    def read_scope(path: Path | str) -> GraphScope | None:
        """The scope recorded in the graph at `path`, without opening it for work.

        Callers use it to inherit the settings a command did not restate.
        A missing file or a graph without a scope reads as None.
        """

        path = Path(path)
        if not path.is_file():
            return None
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_scope'"
            ).fetchone()
            if table is None:
                return None
            row = conn.execute("SELECT * FROM graph_scope WHERE id = 1").fetchone()
            return GraphScope.from_row(row) if row else None
        finally:
            conn.close()

    def _settle_scope(
        self, wanted: GraphScope | None, adopt: bool
    ) -> GraphScope | None:
        row = self._conn.execute(
            "SELECT * FROM graph_scope WHERE id = 1"
        ).fetchone()
        stored = GraphScope.from_row(row) if row else None
        if wanted is None:
            if adopt:
                raise ScopeError("--adopt-scope needs a scope to adopt")
            return stored
        if stored is not None:
            if adopt:
                raise ScopeError(
                    "--adopt-scope only applies to a graph with no recorded "
                    "scope; this one has one"
                )
            diffs = stored.diff(wanted)
            if diffs:
                raise ScopeMismatch(diffs)
            return stored
        goals = self._conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
        if goals and not adopt:
            raise ScopeMissing(
                f"this graph has {goals} goal(s) but no recorded scope: it was "
                "built before scopes existed, under premises nobody wrote down. "
                "If you know it was built under the settings of this command, "
                "record them with --adopt-scope; if it may have mixed hashers or "
                "preambles, start over with --force-new-graph."
            )
        row = {**wanted.to_row(), "created_at": time.time()}
        with self._conn:
            self._conn.execute(
                "INSERT INTO graph_scope (id, " + ", ".join(row) + ") VALUES (1, "
                + ", ".join("?" for _ in row) + ")",
                tuple(row.values()),
            )
        return wanted

    def _add_missing_columns(self) -> None:
        """Bring an older graph's tables up to the current shape.

        `CREATE TABLE IF NOT EXISTS` adds tables but never columns, so a graph
        created before a column existed opens without it and fails on the
        first read -- and these graphs outlive the sessions that made them,
        which is the whole point of writing them to disk.
        """

        for table, column, decl in (
            ("certifications", "decomposition_id", "TEXT"),
            ("certifications", "trust", "TEXT"),
            ("goals", "lease_run_dir", "TEXT"),
        ):
            existing = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {decl}"
                )

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

    def reachable_from(self, goal_id: str) -> set[str]:
        """This goal and every goal below it, through its decompositions.

        `solve(root)` has to be scoped to this set. Without it `open_goals()`
        answers with everything open in the WORKSPACE, so attacking one subgoal
        spends budget on unrelated goals -- including other problems, since one
        workspace is meant to hold several. Observed: attacking one lemma
        quietly proved its sibling and charged the first lemma's budget for it.
        """

        seen = {goal_id}
        frontier = [goal_id]
        while frontier:
            current = frontier.pop()
            for decomposition in self.decompositions_of(current):
                for subgoal_id in decomposition.subgoal_ids:
                    if subgoal_id not in seen:
                        seen.add(subgoal_id)
                        frontier.append(subgoal_id)
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

    def decompositions_containing(self, goal_id: str) -> list[Decomposition]:
        """Every route that has this goal as one of its subgoals.

        Used to answer "if I prove this, does anything upstream close?" -- a
        subgoal whose routes are all unaccepted can be proved perfectly well
        and still shut nothing, because only an ACCEPTED decomposition may
        complete.
        """

        rows = self._conn.execute(
            "SELECT DISTINCT decomposition_id FROM decomposition_subgoals"
            " WHERE subgoal_id = ?",
            (goal_id,),
        ).fetchall()
        return [self.decomposition(row["decomposition_id"]) for row in rows]

    def proposed_decompositions(self) -> list[Decomposition]:
        """Decompositions nobody ever reached a verdict on.

        A sketch check that could not run leaves one of these: not rejected,
        because a dead toolchain is no verdict, but not accepted either. They
        have to be findable, because PROPOSED counts as a live route -- so an
        unchecked one both blocks its parent from being attacked directly AND
        never gets checked, which wedges the goal shut for good.
        """

        rows = self._conn.execute(
            "SELECT id FROM decompositions WHERE status = ? ORDER BY created_at",
            (DecompositionStatus.PROPOSED.value,),
        ).fetchall()
        return [self.decomposition(row["id"]) for row in rows]

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

    # -- certifications -------------------------------------------------------

    def record_certification(
        self,
        goal_id: str,
        *,
        ok: bool,
        axioms: Iterable[str] = (),
        reason: str = "",
        text_sha256: str = "",
        decomposition_id: str | None = "",
        trust: str | None = "",
    ) -> Certification:
        """Record one compile of this goal's assembled proof.

        Appended, never replaced: the graph can change under a certification
        (a lemma re-proved, a decomposition added), and the history of what
        was compiled when is what lets a reader tell a stale pass from a
        current one.
        """

        cert_id = _new_id("cert")
        now = time.time()
        axiom_set = frozenset(axioms)
        with self._conn:
            self._conn.execute(
                "INSERT INTO certifications (id, goal_id, ok, axioms_json,"
                " reason, text_sha256, decomposition_id, trust, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cert_id,
                    goal_id,
                    int(ok),
                    json.dumps(sorted(axiom_set)),
                    reason,
                    text_sha256,
                    decomposition_id,
                    trust,
                    now,
                ),
            )
        return Certification(
            id=cert_id,
            goal_id=goal_id,
            ok=ok,
            axioms=axiom_set,
            reason=reason,
            text_sha256=text_sha256,
            decomposition_id=decomposition_id,
            trust=trust,
            created_at=now,
        )

    def certifications_of(self, goal_id: str) -> list[Certification]:
        rows = self._conn.execute(
            "SELECT * FROM certifications WHERE goal_id = ? ORDER BY created_at",
            (goal_id,),
        ).fetchall()
        return [
            Certification(
                id=row["id"],
                goal_id=row["goal_id"],
                ok=bool(row["ok"]),
                axioms=frozenset(json.loads(row["axioms_json"])),
                reason=row["reason"],
                text_sha256=row["text_sha256"],
                decomposition_id=row["decomposition_id"],
                trust=row["trust"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def latest_certification(self, goal_id: str) -> Certification | None:
        """The most recent compile of the assembled proof, passed or not."""

        certs = self.certifications_of(goal_id)
        return certs[-1] if certs else None

    # -- envelopes ------------------------------------------------------------

    def record_envelope(
        self, envelope: SourceEnvelope, *, candidate_path: str | None = None
    ) -> str:
        """Persist the exact envelope a verification will use. Returns its hash.

        Idempotent by hash. A second write of the same hash must carry the same
        text; anything else means two different envelopes collided, and keeping
        either silently would bind a certificate to input nobody checked.
        """

        text = envelope.to_json()
        digest = envelope.envelope_hash
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO envelopes (envelope_hash, envelope_json,"
                " candidate_sha256, candidate_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (digest, text, envelope.candidate_file_sha256, candidate_path, time.time()),
            )
        stored = self._conn.execute(
            "SELECT envelope_json FROM envelopes WHERE envelope_hash = ?", (digest,)
        ).fetchone()["envelope_json"]
        if stored != text:
            raise GraphError(f"envelope {digest} is already stored with different content")
        return digest

    def envelope(self, envelope_hash: str) -> SourceEnvelope:
        """The stored envelope, read back and re-hashed -- never re-split."""

        row = self._conn.execute(
            "SELECT envelope_json FROM envelopes WHERE envelope_hash = ?", (envelope_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no such envelope: {envelope_hash}")
        envelope = SourceEnvelope.from_dict(json.loads(row["envelope_json"]))
        if envelope.envelope_hash != envelope_hash:
            raise GraphError(f"envelope {envelope_hash} no longer hashes to its key")
        return envelope

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
            # The working directory is cleared with the same UPDATE that takes
            # the lease: a new holder inherits nothing, and a stale pointer
            # from the previous one can never be read as this one's.
            cursor = self._conn.execute(
                "UPDATE goals SET lease_owner = ?, lease_expires_at = ?,"
                " lease_run_dir = NULL"
                " WHERE id = ? AND (lease_owner IS NULL"
                "                   OR lease_expires_at IS NULL"
                "                   OR lease_expires_at <= ?)",
                (owner, now + ttl_s, goal_id, now),
            )
        return cursor.rowcount == 1

    def stale_leases(self, now: float | None = None) -> list[Goal]:
        """Goals whose lease has run out with nobody having released it.

        A worker that died mid-attempt left one of these behind. The attempt
        itself was never recorded -- the process stopped before it could be --
        so without this scan the fact that anything was tried at all is simply
        gone, and the run directory it left is an orphan nothing points at.
        """

        now = time.time() if now is None else now
        rows = self._conn.execute(
            "SELECT * FROM goals WHERE lease_owner IS NOT NULL"
            " AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?"
            " ORDER BY created_at",
            (now,),
        ).fetchall()
        return [_goal(row) for row in rows]

    def note_attempt_dir(self, goal_id: str, run_dir: str) -> None:
        """Record where the lease holder is about to work.

        Called when the directory is claimed and before anything is written
        into it, because the case this exists for is the worker dying with the
        attempt unrecorded. Written after the fact it would be missing exactly
        when it is needed.
        """

        with self._conn:
            self._conn.execute(
                "UPDATE goals SET lease_run_dir = ? WHERE id = ?",
                (run_dir, goal_id),
            )

    def force_release(self, goal_id: str) -> None:
        """Clear a lease regardless of who holds it. Recovery only."""

        with self._conn:
            self._conn.execute(
                "UPDATE goals SET lease_owner = NULL, lease_expires_at = NULL,"
                " lease_run_dir = NULL WHERE id = ?",
                (goal_id,),
            )

    def release(self, goal_id: str, owner: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE goals SET lease_owner = NULL, lease_expires_at = NULL,"
                " lease_run_dir = NULL WHERE id = ? AND lease_owner = ?",
                (goal_id, owner),
            )


__all__ = ["ProofGraphStore"]
