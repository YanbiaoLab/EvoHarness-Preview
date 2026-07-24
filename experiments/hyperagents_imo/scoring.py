"""Score a HyperAgents proof and an EvoHarness proof with one frozen grader."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evoharness.evocore import LLMClient

from experiments.imo_proof.evaluation.engine import (
    LLMProofGrader,
    ProofDataset,
    load_frozen_grader_prompt,
)
from experiments.imo_proof.protocol import BenchmarkSpec, default_spec_path
from experiments.imo_proof.transport import SpecOpenAITransport


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_hyperagents_proof(path: Path, problem_id: str) -> str:
    predictions = load_hyperagents_predictions(path)
    try:
        return predictions[problem_id]
    except KeyError as exc:
        raise ValueError(f"missing HyperAgents proof for {problem_id}") from exc


def load_hyperagents_predictions(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    predictions: dict[str, str] = {}
    for row in rows:
        problem_id = row.get("Problem ID", "").strip()
        if not problem_id:
            raise ValueError("HyperAgents prediction row has an empty Problem ID")
        if problem_id in predictions:
            raise ValueError(f"duplicate HyperAgents prediction for {problem_id}")
        proof = (row.get("prediction") or row.get("Response") or "").strip()
        if not proof:
            raise ValueError(f"HyperAgents proof is empty for {problem_id}")
        predictions[problem_id] = proof
    return predictions


def load_evoharness_proof(path: Path, problem_id: str) -> str:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    observed_id = value.get("item_id") or value.get("problem_id")
    if observed_id != problem_id:
        raise ValueError(
            f"EvoHarness artifact is for {observed_id!r}, expected {problem_id!r}"
        )
    proof = str(value.get("proof", "")).strip()
    if not proof:
        raise ValueError(f"EvoHarness proof is empty for {problem_id}")
    return proof


def _grader(spec: BenchmarkSpec) -> LLMProofGrader:
    api_base = os.environ.get("EVOHARNESS_API_BASE")
    api_key = os.environ.get("EVOHARNESS_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError(
            "live scoring requires EVOHARNESS_API_BASE and EVOHARNESS_API_KEY"
        )
    model = spec.grader.model
    client = LLMClient(
        model.temperature,
        model.max_output_tokens,
        transport=SpecOpenAITransport(
            api_base,
            api_key,
            model.enable_thinking,
            model.input_cost_per_million,
            model.output_cost_per_million,
        ),
    )
    return LLMProofGrader(
        client,
        spec,
        load_frozen_grader_prompt(PROJECT_ROOT, spec),
    )


def _outcome_dict(proof: str, outcome: Any, spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "proof_sha256": _sha256_text(proof),
        "proof_characters": len(proof),
        "label": outcome.label,
        "points": spec.scoring.points[outcome.label],
        "max_points": spec.scoring.max_points,
        "grader_usage": {
            "calls": outcome.usage.calls,
            "prompt_tokens": outcome.usage.prompt_tokens,
            "completion_tokens": outcome.usage.completion_tokens,
            "cost_usd": outcome.usage.cost_usd,
        },
        "grader_response": outcome.raw_response,
    }


def score_pair(
    *,
    problem_id: str,
    hyperagents_csv: Path,
    evoharness_item: Path,
) -> dict[str, Any]:
    spec = BenchmarkSpec.load(default_spec_path())
    spec.verify_workspace(PROJECT_ROOT)
    dataset = ProofDataset.load(PROJECT_ROOT, spec)
    record = dataset.select((problem_id,))[0]
    hyper_proof = load_hyperagents_proof(hyperagents_csv, problem_id)
    evo_proof = load_evoharness_proof(evoharness_item, problem_id)
    grader = _grader(spec)

    # The grader is stateless and receives the same frozen prompt. Method names
    # are deliberately not included in either grading request.
    hyper_outcome = grader.grade(record, hyper_proof)
    evo_outcome = grader.grade(record, evo_proof)
    hyper_result = _outcome_dict(hyper_proof, hyper_outcome, spec)
    evo_result = _outcome_dict(evo_proof, evo_outcome, spec)
    return {
        "comparison_type": "paired-proof-frozen-grader",
        "scope": "one_problem",
        "problem_id": problem_id,
        "benchmark_id": spec.benchmark_id,
        "benchmark_fingerprint": spec.fingerprint,
        "grader_model": spec.grader.model.name,
        "grader_temperature": spec.grader.model.temperature,
        "grader_prompt_sha256": spec.grader.prompt_sha256,
        "results": {
            "hyperagents": hyper_result,
            "evoharness": evo_result,
        },
        "point_delta_hyperagents_minus_evoharness": (
            hyper_result["points"] - evo_result["points"]
        ),
        "limitations": [
            "This is a paired one-problem evaluator comparison, not a full benchmark.",
            "The native HyperAgents solver run is not the HyperAgents outer evolution loop.",
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _split_summary(
    problem_ids: tuple[str, ...],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = [results[problem_id] for problem_id in problem_ids]
    earned = sum(item["points"] for item in selected)
    maximum = sum(item["max_points"] for item in selected)
    correct = sum(item["points"] == item["max_points"] for item in selected)
    labels: dict[str, int] = {}
    for item in selected:
        labels[item["label"]] = labels.get(item["label"], 0) + 1
    return {
        "problem_count": len(selected),
        "points": earned,
        "max_points": maximum,
        "points_percentage": earned / maximum if maximum else 0.0,
        "correct_count": correct,
        "correct_percentage": correct / len(selected) if selected else 0.0,
        "label_counts": dict(sorted(labels.items())),
    }


def score_benchmark(
    *,
    hyperagents_csv: Path,
    output_dir: Path,
    max_workers: int = 3,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    spec = BenchmarkSpec.load(default_spec_path())
    spec.verify_workspace(PROJECT_ROOT)
    dataset = ProofDataset.load(PROJECT_ROOT, spec)
    predictions = load_hyperagents_predictions(hyperagents_csv)
    expected_ids = set(
        spec.splits.train + spec.splits.validation + spec.splits.test
    )
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        extra = sorted(set(predictions) - expected_ids)
        raise ValueError(
            f"prediction IDs do not match benchmark; missing={missing}, extra={extra}"
        )

    output_dir = Path(output_dir)
    item_dir = output_dir / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for problem_id in sorted(expected_ids):
        item_path = item_dir / f"{problem_id}.json"
        if item_path.is_file():
            value = json.loads(item_path.read_text(encoding="utf-8"))
            expected_hash = _sha256_text(predictions[problem_id])
            if value.get("proof_sha256") != expected_hash:
                raise ValueError(
                    f"saved score proof hash mismatch for {problem_id}"
                )
            results[problem_id] = value
        else:
            pending.append(problem_id)

    grader = _grader(spec)

    def grade_one(problem_id: str) -> tuple[str, dict[str, Any]]:
        record = dataset.select((problem_id,))[0]
        outcome = grader.grade(record, predictions[problem_id])
        return problem_id, _outcome_dict(predictions[problem_id], outcome, spec)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(grade_one, problem_id): problem_id for problem_id in pending}
        for future in as_completed(futures):
            problem_id, value = future.result()
            _write_json(item_dir / f"{problem_id}.json", value)
            results[problem_id] = value
            print(
                f"graded {problem_id}: {value['points']}/{value['max_points']}",
                flush=True,
            )

    split_summaries = {
        split: _split_summary(spec.splits.ids(split), results)
        for split in ("train", "validation", "test")
    }
    all_ids = spec.splits.train + spec.splits.validation + spec.splits.test
    summary = {
        "comparison_type": "hyperagents-full-frozen-grader",
        "benchmark_id": spec.benchmark_id,
        "benchmark_fingerprint": spec.fingerprint,
        "dataset_sha256": spec.dataset.sha256,
        "grader_model": spec.grader.model.name,
        "grader_temperature": spec.grader.model.temperature,
        "grader_prompt_sha256": spec.grader.prompt_sha256,
        "prediction_csv_sha256": hashlib.sha256(
            Path(hyperagents_csv).read_bytes()
        ).hexdigest(),
        "splits": split_summaries,
        "overall": _split_summary(all_ids, results),
        "items": {
            problem_id: {
                key: results[problem_id][key]
                for key in (
                    "label",
                    "points",
                    "max_points",
                    "proof_sha256",
                    "proof_characters",
                    "grader_usage",
                )
            }
            for problem_id in sorted(results)
        },
        "generation_protocol": {
            "system": "HyperAgents native IMO TaskAgent",
            "solver_model": spec.solver.name,
            "solver_temperature": spec.solver.temperature,
            "solver_max_output_tokens_per_call": spec.solver.max_output_tokens,
            "budget_warning": (
                "Native HyperAgents uses its own iterative solve/verify stopping "
                "rule and is not constrained by BenchmarkSpec.solver_budget."
            ),
        },
    }
    _write_json(output_dir / "benchmark_result.json", summary)
    return summary


__all__ = [
    "load_evoharness_proof",
    "load_hyperagents_proof",
    "load_hyperagents_predictions",
    "score_benchmark",
    "score_pair",
]
