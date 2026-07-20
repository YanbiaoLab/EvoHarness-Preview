"""Compare two IMO Proof reports under one frozen evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def compare_reports(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, object]:
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    if baseline["total"] != candidate["total"]:
        raise ValueError("reports must cover the same number of problems")

    baseline_score = float(baseline["points_percentage"])
    candidate_score = float(candidate["points_percentage"])
    return {
        "baseline_report": str(baseline_path.resolve()),
        "candidate_report": str(candidate_path.resolve()),
        "num_samples": int(baseline["total"]),
        "baseline_points_percentage": baseline_score,
        "candidate_points_percentage": candidate_score,
        "absolute_improvement": candidate_score - baseline_score,
        "candidate_wins": candidate_score > baseline_score,
    }


def compare_runs(
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, object]:
    """Compare scores only after proving protocol fields are identical."""

    baseline_manifest = _load_manifest(baseline_dir / "manifest.json")
    candidate_manifest = _load_manifest(candidate_dir / "manifest.json")
    baseline_protocol = _protocol(baseline_manifest)
    candidate_protocol = _protocol(candidate_manifest)
    mismatches = {
        key: (baseline_protocol[key], candidate_protocol[key])
        for key in baseline_protocol
        if baseline_protocol[key] != candidate_protocol[key]
    }
    if mismatches:
        details = ", ".join(sorted(mismatches))
        raise ValueError(f"evaluation protocol mismatch: {details}")

    result = compare_reports(
        baseline_dir / "report.json",
        candidate_dir / "report.json",
    )
    result["protocol_verified"] = True
    result["protocol"] = baseline_protocol
    result["baseline_algorithm"] = _algorithm_name(baseline_manifest)
    result["candidate_algorithm"] = _algorithm_name(candidate_manifest)
    result["paired_analysis"] = _paired_analysis(
        baseline_dir / "gradings" / "predictions.csv",
        candidate_dir / "gradings" / "predictions.csv",
    )
    baseline_calls = _model_calls(baseline_dir)
    candidate_calls = _model_calls(candidate_dir)
    result["model_calls"] = {
        "baseline": baseline_calls,
        "candidate": candidate_calls,
        "candidate_to_baseline_ratio": (
            candidate_calls["total"] / baseline_calls["total"]
        ),
    }
    return result


def _load_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    required = {"total", "points_percentage"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(f"invalid IMO Proof report: {path}")
    return value


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"invalid experiment manifest: {path}")
    return value


def _protocol(manifest: dict[str, object]) -> dict[str, object]:
    dataset = manifest.get("dataset")
    grader = manifest.get("grader")
    model = manifest.get("model", manifest.get("solver"))
    if not all(isinstance(item, dict) for item in (dataset, grader, model)):
        raise ValueError("manifest lacks dataset, model/solver, or grader")
    assert isinstance(dataset, dict)
    assert isinstance(grader, dict)
    assert isinstance(model, dict)
    return {
        "dataset_sha256": dataset.get("sha256"),
        "dataset_selection": dataset.get("selection"),
        "num_samples": dataset.get("num_samples"),
        "model": model.get("name", model.get("model")),
        "enable_thinking": model.get("enable_thinking"),
        "temperature": model.get("temperature"),
        "max_tokens": model.get("max_tokens"),
        "grader_model": grader.get("model"),
        "grader_source_sha256": grader.get("source_sha256"),
    }


def _algorithm_name(manifest: dict[str, object]) -> str:
    algorithm = manifest.get("algorithm")
    if isinstance(algorithm, dict) and isinstance(algorithm.get("name"), str):
        return algorithm["name"]
    solver = manifest.get("solver")
    if isinstance(solver, dict) and isinstance(solver.get("agent"), str):
        return solver["agent"]
    return "unknown"


def _paired_analysis(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    points = {"incorrect": 0, "partial": 1, "almost": 6, "correct": 7}

    def load(path: Path) -> dict[str, int]:
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle)
            values = {}
            for row in rows:
                problem_id = row.get("Problem ID")
                label = row.get("prediction", "").strip().lower()
                if not problem_id or label not in points:
                    raise ValueError(f"invalid grader prediction row: {path}")
                values[problem_id] = points[label]
            return values

    baseline = load(baseline_path)
    candidate = load(candidate_path)
    if baseline.keys() != candidate.keys():
        raise ValueError("grader predictions cover different problem IDs")
    deltas = [candidate[key] - baseline[key] for key in sorted(baseline)]
    if not deltas:
        raise ValueError("grader predictions cannot be empty")

    rng = random.Random(seed)
    n = len(deltas)
    bootstrap = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / (7 * n)
        for _ in range(bootstrap_samples)
    )
    lower = bootstrap[int(0.025 * (bootstrap_samples - 1))]
    upper = bootstrap[int(0.975 * (bootstrap_samples - 1))]
    return {
        "candidate_problem_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "candidate_problem_losses": sum(delta < 0 for delta in deltas),
        "mean_normalized_point_improvement": sum(deltas) / (7 * n),
        "paired_bootstrap_95_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def _model_calls(run_dir: Path) -> dict[str, object]:
    counts = {}
    for path in sorted((run_dir / "agent_evals").glob("*.md")):
        problem_id = path.stem.removeprefix("chat_history_")
        counts[problem_id] = sum(
            line.startswith("Input:")
            for line in path.read_text(errors="replace").splitlines()
        )
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError(f"invalid solver transcripts: {run_dir}")
    values = list(counts.values())
    return {
        "total": sum(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    comparison = compare_runs(args.baseline, args.candidate)
    rendered = json.dumps(comparison, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if comparison["candidate_wins"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
