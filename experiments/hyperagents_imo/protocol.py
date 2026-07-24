"""Frozen comparison contract for HyperAgents on the EvoHarness IMO task.

The original HyperAgents driver owns a different candidate representation and
requires Docker.  This module prevents accidentally calling that driver with a
different dataset or a larger search budget and treating the result as a fair
comparison with ``results/imo_trace_run_v2``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RUN = PROJECT_ROOT / "results" / "imo_trace_run_v2"
EVO_SPEC = PROJECT_ROOT / "experiments" / "imo_proof" / "benchmark.v1.json"
EVO_DATASET = PROJECT_ROOT / "experiments" / "imo_proof" / "assets" / "proofbench.csv"
HYPER_ROOT = PROJECT_ROOT / "third_party" / "HyperAgents"
HYPER_DATASET = HYPER_ROOT / "domains" / "imo" / "proofbench.csv"


@dataclass(frozen=True)
class ComparisonContract:
    """Fields which must remain fixed for a score comparison."""

    benchmark_id: str
    dataset_sha256: str
    train_ids: tuple[str, ...]
    optimizer_model: str
    solver_model: str
    grader_model: str
    optimizer_max_output_tokens: int
    solver_max_output_tokens: int
    grader_max_output_tokens: int
    candidate_budget: int
    reference_run: str

    @classmethod
    def load(cls) -> "ComparisonContract":
        spec = json.loads(EVO_SPEC.read_text(encoding="utf-8"))
        trace_candidates = _load_trace_candidates()
        return cls(
            benchmark_id=spec["benchmark_id"],
            dataset_sha256=spec["dataset"]["sha256"],
            train_ids=tuple(spec["splits"]["train"]),
            optimizer_model=spec["optimizer"]["name"],
            solver_model=spec["solver"]["name"],
            grader_model=spec["grader"]["model"]["name"],
            optimizer_max_output_tokens=spec["optimizer"]["max_output_tokens"],
            solver_max_output_tokens=spec["solver"]["max_output_tokens"],
            grader_max_output_tokens=spec["grader"]["model"]["max_output_tokens"],
            # ``imo_trace_run_v2`` contains seed + four evaluated children.
            candidate_budget=len(trace_candidates),
            reference_run=str(BASELINE_RUN.relative_to(PROJECT_ROOT)),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_trace_candidates() -> list[dict]:
    """Read only the stable, user-visible lineage fields from the baseline."""
    import sqlite3

    database = BASELINE_RUN / "run.db"
    if not database.is_file():
        raise FileNotFoundError(f"missing reference database: {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, generation, island_idx, parent_id, operator "
            "FROM candidates ORDER BY generation, timestamp"
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "id": candidate_id,
            "generation": generation,
            "island_idx": island_idx,
            "parent_id": parent_id,
            "operator": operator,
        }
        for candidate_id, generation, island_idx, parent_id, operator in rows
    ]


def verify_contract(contract: ComparisonContract | None = None) -> dict[str, object]:
    """Return machine-readable admission checks without invoking a model."""
    contract = contract or ComparisonContract.load()
    checks: dict[str, bool] = {
        "reference_run_exists": (BASELINE_RUN / "run.db").is_file(),
        "hyperagents_checkout_exists": (HYPER_ROOT / "generate_loop.py").is_file(),
        "datasets_exist": EVO_DATASET.is_file() and HYPER_DATASET.is_file(),
        "datasets_byte_identical": (
            EVO_DATASET.is_file()
            and HYPER_DATASET.is_file()
            and EVO_DATASET.read_bytes() == HYPER_DATASET.read_bytes()
        ),
        "dataset_matches_frozen_hash": (
            EVO_DATASET.is_file() and _sha256(EVO_DATASET) == contract.dataset_sha256
        ),
        "train_ids_exist": False,
        "reference_budget_is_seed_plus_four_children": contract.candidate_budget == 5,
    }
    if checks["datasets_exist"]:
        with EVO_DATASET.open(newline="", encoding="utf-8") as handle:
            ids = {row["Problem ID"] for row in csv.DictReader(handle)}
        checks["train_ids_exist"] = set(contract.train_ids).issubset(ids)

    return {
        "contract": asdict(contract),
        "checks": checks,
        "ok": all(checks.values()),
        "not_a_live_run": True,
        "next_live_requirement": (
            "Docker runtime plus a HyperAgents-to-EvoHarness candidate adapter; "
            "do not score a native HyperAgents run as directly comparable before that adapter exists."
        ),
    }


__all__ = ["ComparisonContract", "verify_contract"]
