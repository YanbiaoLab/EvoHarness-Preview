import json
import os
from pathlib import Path

from evoharness.core.checkpoint import atomic_write_json

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


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())



def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


class ResearchStore:
    def __init__(self, root: Path):
        self.root = Path(root)

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
        _append_jsonl(
            self._experiment_dir(decision.experiment_id) / "decisions.jsonl",
            decision.to_json(),
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
        return _read_jsonl(
            self._experiment_dir(experiment_id) / "decisions.jsonl"
        )

    # -- helpers -------------------------------------------------------------

    def _experiment_dir(self, experiment_id: str) -> Path:
        return self.root / "experiments" / experiment_id

    def _require_experiment(self, experiment_id: str) -> None:
        if not (self._experiment_dir(experiment_id) / "spec.json").exists():
            raise ResearchStoreError(
                f"experiment {experiment_id} has no frozen spec; "
                "records may only attach to a frozen experiment"
            )