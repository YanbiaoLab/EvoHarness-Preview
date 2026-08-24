import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoreScorecard:
    run_id: str
    stopped_reason: str
    generations_completed: int
    evaluations: int
    infra_drops: int
    preflight_failures: int
    best_fitness: float | None
    total_eval_cost_usd: float
    evidence_count: int

    @property
    def infra_rate(self) -> float:
        total = self.evaluations + self.infra_drops
        return self.infra_drops / total if total else 0.0

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            **asdict(self),
            "infra_rate": self.infra_rate,
        }

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "CoreScorecard":
        run_dir = Path(run_dir)
        manifest_path = run_dir / "manifest.json"
        checkpoint_path = run_dir / "checkpoint.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {}
        # A finalized manifest is the authority for terminal state.  A
        # checkpoint is only a recovery point and commonly still says
        # ``running`` after a clean return or circuit-breaker stop.
        if manifest.get("status") == "completed" and isinstance(
            manifest.get("report"), dict
        ):
            report = manifest["report"]
        elif checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            report = checkpoint.get("run_report", {})
        else:
            raise FileNotFoundError(
                f"{run_dir} has neither a finalized manifest nor checkpoint"
            )
        history = report.get("history", [])
        evidence_path = run_dir / "evidence.jsonl"
        evidence_count = (
            len(evidence_path.read_text().splitlines())
            if evidence_path.exists() else 0
        )

        return cls(
            run_id=run_dir.name,
            stopped_reason=report.get("stopped_reason", "unknown"),
            generations_completed=int(report.get("generations_completed", 0)),
            evaluations=int(report.get("evaluations", 0)),
            infra_drops=sum(
                1 for h in history if h.get("status") == "infra_error"
            ),
            preflight_failures=sum(
                1 for h in history if h.get("status") == "preflight_failed"
            ),
            best_fitness=report.get("best_fitness"),
            total_eval_cost_usd=float(report.get("total_eval_cost", 0.0)),
            evidence_count=evidence_count,
        )

@dataclass(frozen=True)
class ResearchScorecard:
    experiments: int
    outcomes: int
    decisions: int
    pending_requests: int
    total_recorded_eval_cost_usd: float
    decisions_per_100_eval_usd: float | None

    discovery_latency_s: None = None
    discovery_latency_note: str = (
        "requires retrospective full validation (not built yet)"
    )
    cost_scope: str = "evaluation cost recorded in ExperimentOutcome only"

    def to_json(self) -> dict:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def build(
        cls, research_root: Path, *, pending_requests: int
    ) -> "ResearchScorecard":
        research_root = Path(research_root)
        root = research_root / "experiments"
        experiments = outcomes = decisions = 0
        total_cost = 0.0
        if root.exists():
            for exp_dir in sorted(root.iterdir()):
                if not (exp_dir / "spec.json").exists():
                    continue
                experiments += 1
                for line in _lines(exp_dir / "outcomes.jsonl"):
                    outcomes += 1
                    total_cost += float(json.loads(line).get("eval_cost_usd", 0.0))
                decisions += _decision_count(research_root, exp_dir.name)
        # 在 if 之外返回:research 目录不存在时给空记分卡,不给 None。
        per_100 = (
            decisions / (total_cost / 100.0) if total_cost > 0 else None
        )
        return cls(
            experiments=experiments,
            outcomes=outcomes,
            decisions=decisions,
            pending_requests=pending_requests,
            total_recorded_eval_cost_usd=total_cost,
            decisions_per_100_eval_usd=per_100,
        )

def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [x for x in path.read_text().splitlines() if x.strip()]


def _decision_count(research_root: Path, experiment_id: str) -> int:
    count = len(
        _lines(research_root / "experiments" / experiment_id / "decisions.jsonl")
    )
    ledger = research_root / "research.sqlite3"
    if not ledger.exists():
        return count
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*) FROM research_decisions WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return count
    finally:
        connection.close()
    return count + int(row[0])
