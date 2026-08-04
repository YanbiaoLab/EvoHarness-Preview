"""Grade a candidate coding harness on the Polyglot Python exercises.

The candidate is a small program that turns an exercise into source files,
with a model handle and a test-running tool. Fitness is the fraction of
exercises whose real test suite passes on a run the candidate did not perform
itself. The harness runs in a subprocess (see runner.py) so that a candidate
that hangs, leaks, or crashes the interpreter costs one evaluation and not
the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from evoharness.evoserve import Grade, GradeContext, InfraError

_RUNNER = Path(__file__).resolve().parent / "runner.py"

# Above this share of exercises lost to the harness's own errors (rather than
# to wrong answers) the pass rate stops describing the program: it is
# measuring how often the harness crashed. Same threshold and same reasoning
# as the IMO task's unscored-item guard.
_HARNESS_FAULT_RATIO = 0.5


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def make_grade_func(*, split: str = "train"):
    """Bind a split; the loop grades on train, the report on holdout."""

    def grade_workspace(candidate_dir: Path, ctx: GradeContext) -> Grade:
        report_path = Path(ctx.workdir) / f"polyglot_{split}.json"
        command = [
            sys.executable,
            str(_RUNNER),
            "--candidate-dir", str(candidate_dir),
            "--output", str(report_path),
            "--split", split,
            "--train-size", str(_env_int("POLYGLOT_TRAIN_SIZE", 12)),
            "--model", os.environ.get("POLYGLOT_SOLVER_MODEL", "gpt-5.1"),
            "--max-llm-calls", str(_env_int("POLYGLOT_MAX_LLM_CALLS", 6)),
            "--max-test-runs", str(_env_int("POLYGLOT_MAX_TEST_RUNS", 3)),
            "--test-timeout-s", str(_env_float("POLYGLOT_TEST_TIMEOUT_S", 60.0)),
            "--workers", str(_env_int("POLYGLOT_WORKERS", 4)),
        ]
        timeout_s = _env_float("POLYGLOT_TIMEOUT_S", 5400.0)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except subprocess.TimeoutExpired:
            return Grade(
                fitness=0.0,
                passed=False,
                fault=f"harness exceeded {timeout_s:.0f}s over the whole split",
                visible_metrics={"timeout": True, "structural_doa": False},
                structured_feedback={
                    "schema_version": 1,
                    "summary": "the harness did not finish the exercise set in time",
                },
                stage_reached=1,
            )
        except OSError as exc:
            raise InfraError(f"could not launch the polyglot runner: {exc}") from exc

        if completed.returncode == 3:
            # The runner refuses to start without model credentials. That is
            # the machine's problem, not the candidate's.
            raise InfraError("polyglot runner: model endpoint is not configured")

        if not report_path.is_file():
            tail = ((completed.stdout or "") + (completed.stderr or ""))[-1500:]
            return Grade(
                fitness=0.0,
                passed=False,
                fault=f"harness produced no report (rc={completed.returncode})",
                visible_metrics={"structural_doa": True},
                structured_feedback={
                    "schema_version": 1,
                    "summary": "the harness could not be run at all",
                    "error_excerpt": tail,
                },
                stderr_log=tail,
                stage_reached=1,
            )

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InfraError(f"unreadable polyglot report: {exc}") from exc

        items = report.get("items", [])
        total = len(items)
        if total == 0:
            raise InfraError("polyglot report contains no exercises")

        passed_items = [item for item in items if item.get("passed")]
        harness_faults = [
            item for item in items
            if str(item.get("error_category", "")).startswith("harness-")
        ]
        starved = len(harness_faults) > _HARNESS_FAULT_RATIO * total
        rate = len(passed_items) / total

        categories = Counter(
            str(item.get("error_category") or "passed") for item in items
        )
        feedback_items = [
            {
                "item_id": item.get("item_id"),
                "passed": bool(item.get("passed")),
                "error_category": item.get("error_category", ""),
                "llm_calls": item.get("llm_calls", 0),
                "test_runs": item.get("test_runs", 0),
            }
            for item in items
        ]

        return Grade(
            fitness=rate,
            passed=not starved,
            fault=(
                f"{len(harness_faults)}/{total} exercises failed inside the "
                "harness itself; this pass rate does not measure the program"
                if starved
                else None
            ),
            visible_metrics={
                "pass_rate": rate,
                "passed": len(passed_items),
                "total": total,
                "harness_faults": len(harness_faults),
                "llm_calls": report.get("llm_calls", 0),
                "test_runs": report.get("test_runs", 0),
                "structural_doa": False,
            },
            structured_feedback={
                "schema_version": 1,
                "summary": f"{len(passed_items)}/{total} exercises pass their tests",
                "items": feedback_items,
                "failure_categories": dict(categories),
            },
            execution_time=float(report.get("elapsed_s", 0.0)),
            eval_cost_usd=float(report.get("cost_usd", 0.0)),
            n_units=total,
            # Bernoulli standard error over the exercise set — the only
            # variance estimate a single pass over a fixed corpus supports.
            sem=(rate * (1.0 - rate) / total) ** 0.5,
        )

    return grade_workspace


grade_workspace = make_grade_func(split="train")

__all__ = ["grade_workspace", "make_grade_func"]
