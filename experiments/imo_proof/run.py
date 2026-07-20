"""Run EvoProof on HyperAgents' frozen IMO Proof evaluation pipeline.

Required environment variables are the same ones used by the vendored
HyperAgents model adapter, normally ``OPENAI_API_KEY`` and
``OPENAI_API_BASE`` for an OpenAI-compatible provider. Secrets are never
written to the experiment manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYPER_ROOT = PROJECT_ROOT / "third_party" / "HyperAgents"
AGENT_PATH = Path(__file__).resolve().with_name("agent.py")
HYPER_AGENT_PATH = HYPER_ROOT / "baselines" / "imo_proof" / "agent.py"
DATASET_PATH = HYPER_ROOT / "domains" / "imo" / "proofbench.csv"
GRADER_LABELS = {"incorrect", "partial", "almost", "correct"}


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observed_model_calls(output_dir: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    for path in sorted((output_dir / "agent_evals").glob("*.md")):
        problem_id = path.stem.removeprefix("chat_history_")
        counts[problem_id] = sum(
            line.startswith("Input:")
            for line in path.read_text(errors="replace").splitlines()
        )
    values = list(counts.values())
    return {
        "per_problem": counts,
        "total": sum(values),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "mean": sum(values) / len(values) if values else 0.0,
    }


def _validate_artifacts(
    output_dir: Path,
    *,
    expected_samples: int,
    min_calls: int,
    max_calls: int,
) -> dict[str, object]:
    """Reject incomplete, mismatched, or budget-drifting experiment runs."""

    def load(path: Path, *, labels: set[str] | None = None):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_samples:
            raise RuntimeError(
                f"{path} contains {len(rows)} rows; "
                f"expected {expected_samples}"
            )
        ids = [row.get("Problem ID", "").strip() for row in rows]
        if any(not problem_id for problem_id in ids):
            raise RuntimeError(f"{path} contains an empty Problem ID")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{path} contains duplicate Problem IDs")
        predictions = [row.get("prediction", "").strip() for row in rows]
        if any(not prediction for prediction in predictions):
            raise RuntimeError(f"{path} contains an empty prediction")
        if labels is not None and any(
            prediction.lower() not in labels for prediction in predictions
        ):
            raise RuntimeError(f"{path} contains an invalid grader label")
        return ids

    solver_ids = load(output_dir / "predictions.csv")
    grader_ids = load(
        output_dir / "gradings" / "predictions.csv",
        labels=GRADER_LABELS,
    )
    if set(solver_ids) != set(grader_ids):
        raise RuntimeError("solver and grader cover different Problem IDs")

    calls = _observed_model_calls(output_dir)
    per_problem = calls["per_problem"]
    assert isinstance(per_problem, dict)
    if set(per_problem) != set(solver_ids):
        raise RuntimeError("solver transcripts cover different Problem IDs")
    invalid_counts = {
        problem_id: count
        for problem_id, count in per_problem.items()
        if not min_calls <= count <= max_calls
    }
    if invalid_counts:
        raise RuntimeError(
            "observed solver call counts exceeded declared bounds: "
            f"{invalid_counts}"
        )

    grader_transcripts = {
        path.stem.removeprefix("chat_history_")
        for path in (output_dir / "gradings" / "agent_evals").glob("*.md")
    }
    if grader_transcripts != set(solver_ids):
        raise RuntimeError("grader transcripts cover different Problem IDs")
    return calls


def _clear_invalid_grader_predictions(path: Path) -> int:
    """Make malformed prior grader outputs eligible for harness resume."""

    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "prediction" not in fieldnames:
        raise RuntimeError(f"invalid grader checkpoint schema: {path}")

    cleared = 0
    for row in rows:
        if row.get("prediction", "").strip().lower() not in GRADER_LABELS:
            row["prediction"] = ""
            cleared += 1
    if not cleared:
        return 0

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return cleared


def _ensure_runtime_dependencies() -> None:
    """Compose the already-installed benchmark and project dependencies."""

    if importlib.util.find_spec("litellm") is None:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        project_site = PROJECT_ROOT / ".venv" / "lib" / version / "site-packages"
        if project_site.is_dir():
            sys.path.insert(0, str(project_site))

    missing = [
        name
        for name in ("litellm", "pandas", "hydra")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"missing benchmark runtime dependencies: {names}; "
            "run with the repository's base Python environment"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("evoproof_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--agent",
        choices=("evoproof", "hyperagents"),
        default="evoproof",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="zero-based dataset row at which the evaluation slice starts",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--model",
        default=os.environ.get("IMO_PROOF_MODEL", "openai/qwen3.7-plus"),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout-s", type=float, default=60.0)
    parser.add_argument("--retry-max-time-s", type=float, default=120.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HYPER_ROOT / "outputs",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)

    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")

    _ensure_runtime_dependencies()

    os.environ["IMO_PROOF_MODEL"] = args.model
    os.environ["IMO_PROOF_GRADER_MODEL"] = args.model
    os.environ["LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["LLM_REQUEST_TIMEOUT_S"] = str(args.request_timeout_s)
    os.environ["LLM_RETRY_MAX_TIME_S"] = str(args.retry_max_time_s)
    os.environ["LLM_ENABLE_THINKING"] = str(args.enable_thinking).lower()

    for path in (HYPER_ROOT, HYPER_ROOT / "proofgrader_repo"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Import only after freezing environment-backed model configuration.
    from domains.harness import harness
    from domains.imo.proof_eval import report_proof_grading

    output_root = args.output_root.resolve()
    agent_path = (
        AGENT_PATH if args.agent == "evoproof" else HYPER_AGENT_PATH
    )
    with _working_directory(HYPER_ROOT):
        output_dir = Path(
            harness(
                agent_path=str(agent_path),
                output_dir=str(output_root),
                run_id=args.run_id,
                domain="imo_proof",
                start_index=args.start_index,
                num_samples=args.num_samples,
                save_interval=1,
                num_workers=args.num_workers,
            )
        ).resolve()
        invalid_grader_predictions = _clear_invalid_grader_predictions(
            output_dir / "gradings" / "predictions.csv"
        )
        grading_dir = Path(
            harness(
                domain="imo_proof_grading",
                agent_path="proofgrader.task_agent",
                proofs_dname=str(output_dir),
                output_dir=str(output_dir),
                run_id="gradings",
                save_interval=1,
                num_workers=args.num_workers,
            )
        ).resolve()
        _, grading_report = report_proof_grading(
            dname=str(grading_dir)
        )
        shutil.move(grading_report, output_dir / "report.json")

    if args.agent == "evoproof":
        algorithm = {
            "name": "EvoProof",
            "stages": [
                "candidate_a",
                "candidate_b",
                "candidate_c",
                "cross_review",
                "synthesis",
                "initial_double_audit",
                "adaptive_repair_1",
                "fresh_double_audit",
                "adaptive_repair_2",
                "final_adversarial_audit",
                "conditional_final_rewrite",
            ],
            "min_model_calls_per_problem": 7,
            "max_model_calls_per_problem": 13,
            "source_sha256": _sha256(
                AGENT_PATH.with_name("strategy.py")
            ),
        }
    else:
        algorithm = {
            "name": "HyperAgents IMO specialized baseline",
            "max_iterations": 30,
            "required_correct_passes": 5,
            "max_error_passes": 10,
            "min_model_calls_per_problem": 12,
            "max_model_calls_per_problem": 91,
            "source_sha256": _sha256(HYPER_AGENT_PATH),
        }

    report = json.loads((output_dir / "report.json").read_text())
    observed_calls = _validate_artifacts(
        output_dir,
        expected_samples=args.num_samples,
        min_calls=algorithm["min_model_calls_per_problem"],
        max_calls=algorithm["max_model_calls_per_problem"],
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": args.run_id,
        "framework": (
            "EvoHarness" if args.agent == "evoproof" else "HyperAgents"
        ),
        "algorithm": algorithm,
        "observed_solver_model_calls": observed_calls,
        "regraded_invalid_predictions": invalid_grader_predictions,
        "dataset": {
            "path": str(DATASET_PATH.relative_to(HYPER_ROOT)),
            "sha256": _sha256(DATASET_PATH),
            "selection": (
                f"rows {args.start_index}.."
                f"{args.start_index + args.num_samples - 1}"
            ),
            "num_samples": args.num_samples,
        },
        "model": {
            "name": args.model,
            "enable_thinking": args.enable_thinking,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "request_timeout_s": args.request_timeout_s,
            "retry_max_time_s": args.retry_max_time_s,
            "num_workers": args.num_workers,
        },
        "grader": {
            "source": "baselines/imo_grading/proofautograder.py",
            "source_sha256": _sha256(
                HYPER_ROOT
                / "baselines"
                / "imo_grading"
                / "proofautograder.py"
            ),
            "executed_source_sha256": _sha256(
                HYPER_ROOT
                / "proofgrader_repo"
                / "proofgrader"
                / "task_agent.py"
            ),
            "model": args.model,
            "enable_thinking": args.enable_thinking,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        },
        "artifacts": {
            "predictions": "predictions.csv",
            "solver_transcripts": "agent_evals/",
            "grader_predictions": "gradings/predictions.csv",
            "grader_transcripts": "gradings/agent_evals/",
            "report": "report.json",
        },
        "results": report,
        "secrets_persisted": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"{algorithm['name']} report: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
