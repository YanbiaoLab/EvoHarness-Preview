import json
import sqlite3
from pathlib import Path

from evoharness.contracts import canonical_json
from evoharness.core.checkpoint import (
    append_jsonl as _append_jsonl,
    atomic_write_json,
    read_jsonl as _read_jsonl,
)

from .models import (
    ExperimentOutcome,
    ExperimentSpec,
    Hypothesis,
    ProtocolChangeProposal,
    ResearchDecision,
    ResearchGoal,
)


class ResearchStoreError(RuntimeError):
    """Append-only violation or corrupted research record."""


class ResearchStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.ledger_path = self.root / "research.sqlite3"
        self._initialize_ledger()

    def _initialize_ledger(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.ledger_path, timeout=30.0) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_requests (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    experiment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_decision_requests_experiment
                    ON decision_requests(experiment_id);

                CREATE TABLE IF NOT EXISTS research_decisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE,
                    experiment_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_research_decisions_experiment
                    ON research_decisions(experiment_id);
                """
            )

    # -- write-once specs --------------------------------------------------

    def _put_once(self, path: Path, payload: dict, identity: str) -> None:
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

        if path.exists():
            # 比较规范化 JSON 而非 Python 对象:内存里的 tuple 落盘再读回
            # 是 list,直接 == 会把幂等重放误判成内容冲突。
            existing = json.dumps(
                json.loads(path.read_text()), sort_keys=True
            )
            incoming = json.dumps(
                json.loads(json.dumps(payload)), sort_keys=True
            )
            if existing == incoming:
                return                      # 幂等重放
            raise ResearchStoreError(
                f"{identity} already exists with different content; "
                "research specs are append-only — create a new id"
            )
        atomic_write_json(path, payload)

    def put_goal(self, goal: ResearchGoal) -> None:
        self._put_once(
            self.root / "goals" / f"{goal.goal_id}.json",
            {"schema_version": 1, "hash": goal.hash, **goal.__dict__},
            f"goal {goal.goal_id}",
        )

    def put_hypothesis(self, hypothesis: Hypothesis) -> None:
        self._put_once(
            self.root / "hypotheses" / f"{hypothesis.hypothesis_id}.json",
            {
                "schema_version": 1,
                "hash": hypothesis.hash,
                **hypothesis.__dict__,
            },
            f"hypothesis {hypothesis.hypothesis_id}",
        )

    def put_experiment(self, spec: ExperimentSpec) -> None:
        self._put_once(
            self._experiment_dir(spec.experiment_id) / "spec.json",
            spec.to_json(),
            f"experiment {spec.experiment_id}",
        )

    def load_experiment(self, experiment_id: str) -> ExperimentSpec:
        path = self._experiment_dir(experiment_id) / "spec.json"
        if not path.exists():
            raise ResearchStoreError(f"experiment {experiment_id} not found")
        return ExperimentSpec.from_json(json.loads(path.read_text()))

    # -- append-only records ------------------------------------------------

    def append_outcome(self, outcome: ExperimentOutcome) -> None:
        self._require_experiment(outcome.experiment_id)
        _append_jsonl(
            self._experiment_dir(outcome.experiment_id) / "outcomes.jsonl",
            outcome.to_json(),
        )

    def append_decision(self, decision: ResearchDecision) -> None:
        self._require_experiment(decision.experiment_id)
        if decision.request_id:
            raise ResearchStoreError(
                "decisions tied to a request must be committed through "
                "InboxStore.answer() so authorization and one-answer "
                "transaction rules cannot be bypassed"
            )
        encoded = canonical_json(decision.to_json())
        with sqlite3.connect(self.ledger_path, timeout=30.0) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO research_decisions(
                    request_id, experiment_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    None,
                    decision.experiment_id,
                    encoded,
                    decision.created_at,
                ),
            )

    def append_assessment(
        self, experiment_id: str, assessment: "ClaimAssessment"
    ) -> None:
        """Record one verdict so a later assessor can be caught contradicting it.

        Conflict detection is the whole point: two assessors that disagree
        about one claim are only visible if the earlier verdict survived.
        """
        self._require_experiment(experiment_id)
        _append_jsonl(
            self._experiment_dir(experiment_id) / "assessments.jsonl",
            assessment.to_json(),
        )

    def assessments(self, experiment_id: str) -> list["ClaimAssessment"]:
        from .assessment import ClaimAssessment

        return [
            ClaimAssessment.from_json(d)
            for d in _read_jsonl(
                self._experiment_dir(experiment_id) / "assessments.jsonl"
            )
        ]

    def append_routed_event(self, event) -> None:
        """The routing audit trail, including events executed without a human.

        Cards land in the ledger on their own; auto-executed events would
        otherwise leave no trace at all, which is exactly the silence an
        activity audit has to be able to detect.
        """
        self._require_experiment(event.experiment_id)
        _append_jsonl(
            self._experiment_dir(event.experiment_id) / "routing.jsonl",
            event.to_json(),
        )

    def routed_events(self, experiment_id: str) -> list[dict]:
        return _read_jsonl(
            self._experiment_dir(experiment_id) / "routing.jsonl"
        )

    def append_proposal(self, proposal: ProtocolChangeProposal) -> None:
        self._require_experiment(proposal.experiment_id)
        _append_jsonl(
            self._experiment_dir(proposal.experiment_id) / "proposals.jsonl",
            proposal.to_json(),
        )

    def outcomes(self, experiment_id: str) -> list[ExperimentOutcome]:
        return [
            ExperimentOutcome.from_json(d)
            for d in _read_jsonl(
                self._experiment_dir(experiment_id) / "outcomes.jsonl"
            )
        ]

    def decisions(self, experiment_id: str) -> list[dict]:
        # Keep old JSONL readable while all new decisions use the transactional
        # ledger shared with InboxStore.
        legacy = _read_jsonl(
            self._experiment_dir(experiment_id) / "decisions.jsonl"
        )
        with sqlite3.connect(self.ledger_path, timeout=30.0) as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM research_decisions
                WHERE experiment_id = ? ORDER BY sequence
                """,
                (experiment_id,),
            ).fetchall()
        return legacy + [json.loads(row[0]) for row in rows]

    # -- helpers -------------------------------------------------------------

    def _experiment_dir(self, experiment_id: str) -> Path:
        return self.root / "experiments" / experiment_id

    def _require_experiment(self, experiment_id: str) -> None:
        if not (self._experiment_dir(experiment_id) / "spec.json").exists():
            raise ResearchStoreError(
                f"experiment {experiment_id} has no frozen spec; "
                "records may only attach to a frozen experiment"
            )
