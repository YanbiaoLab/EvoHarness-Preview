"""ShinkaEvolve evaluator for modmul — a thin adapter, not a second grader.

Everything that decides a score already exists in `experiments/modmul/grade.py`:
ASHA promotion rungs, per-tier scoring, the official inference wall-clock
semantics, and the weight-perturbation compliance gate. This file only
translates between ShinkaEvolve's file contract and that grader.

    Shinka runs:  python evaluate.py --program_path <evolved.py> --results_dir <dir>
    and reads:    <dir>/metrics.json   (metrics["combined_score"] is fitness)
                  <dir>/correct.json   ({"correct": bool, "error": str|None})

`metrics["public"]` is fed back into the mutation prompt; `metrics["private"]`
is not. Per-tier inference seconds go in `public` deliberately — see the score
discussion below, the entire point of this run is invisible without them.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evoharness.evoserve import GradeContext  # noqa: E402
from experiments.modmul.grade import (  # noqa: E402
    SCORED_TIERS,
    SECONDS_PER_PROBLEM,
    grade_workspace,
)

# Operand width per scored tier (modchallenge/config.py TIERS). Inference cost
# is dominated by the Horner step count, which is operand_bits / RADIX_BITS, so
# these ratios are how an unreached tier's cost is projected from a reached one.
OPERAND_BITS = {
    1: 32, 2: 48, 3: 64, 4: 96, 5: 128,
    6: 256, 7: 512, 8: 1024, 9: 2048, 10: 4096,
}

# Weight on the time penalty. Calibrated against the measured seed: it is
# 2.95x over budget, so the penalty is LAMBDA * log2(2.95) = 0.234 against an
# overall_accuracy of 0.892 — large enough to rank a within-budget candidate
# above it, small enough that gutting accuracy for speed still loses.
LAMBDA = 0.15


def project_seconds(seconds: dict[int, float], cases_per_tier: int) -> tuple[float, float, bool]:
    """Estimate what all ten scored tiers would cost, and what is allowed.

    Observed time alone cannot express the failure being optimised away. When
    a candidate blows the budget the harness stops issuing tiers, so its
    observed seconds land at roughly the budget no matter how far over it
    truly is — the measured seed shows 0% on tier 10 with no `infer_s_tier_10`
    metric at all, because tier 10 never ran. Projecting from the tiers that
    did complete is what turns "did not finish" back into a number with a
    gradient on it.

    Returns (projected_total_seconds, allowed_seconds, is_projected).
    """
    allowed = SECONDS_PER_PROBLEM * cases_per_tier * len(SCORED_TIERS)
    if not seconds:
        return float("inf"), allowed, True
    total = sum(seconds.values())
    last = max(seconds)
    projected = False
    for tier in SCORED_TIERS:
        if tier in seconds:
            continue
        # Cost scales with the Horner step count, i.e. with operand width.
        total += seconds[last] * (OPERAND_BITS[tier] / OPERAND_BITS[last])
        projected = True
    return total, allowed, projected


def build_metrics(grade, cases_per_tier: int) -> dict:
    merged = {**grade.visible_metrics, **grade.hidden_metrics}
    seconds = {
        tier: float(merged[f"infer_s_tier_{tier}"])
        for tier in SCORED_TIERS
        if f"infer_s_tier_{tier}" in merged
    }
    accuracies = {
        tier: float(merged.get(f"acc_tier_{tier}", 0.0)) for tier in SCORED_TIERS
    }
    overall = float(merged.get("overall_accuracy", 0.0))
    projected, allowed, is_projected = project_seconds(seconds, cases_per_tier)
    over = projected / allowed if allowed > 0 else float("inf")

    # Score = accuracy, minus a penalty only for being OVER the deadline.
    #
    # Two deliberate choices. First, the leaderboard key (h90 + overall)/11 is
    # NOT the fitness: a tier accuracy doubling moves it by 0.0018 while
    # crossing a 0.9 threshold moves it by 0.091, which is a flat plain with
    # occasional cliffs and close to the worst terrain an evolutionary search
    # can be given. overall_accuracy is continuous and climbs toward the same
    # place. Second, the penalty is one-sided: the deadline is a constraint,
    # not an objective, so beating it early buys nothing and a candidate has
    # no reason to trade accuracy for pointless speed. h90 is still computed
    # and reported — it just does not steer.
    penalty = LAMBDA * math.log2(max(1.0, over))
    combined = overall - penalty

    public = {
        "h90": int(merged.get("h90", 0)),
        "overall_accuracy": round(overall, 4),
        "params": int(merged.get("params", 0)),
        "rung_reached": merged.get("rung", ""),
        "train_seconds": merged.get("train_seconds", 0),
        # The whole run turns on these two lines. Tier 10 scores zero because
        # the wall clock runs out, not because the cell is wrong, and without
        # per-tier seconds in the prompt the search cannot see that.
        "inference_seconds_by_tier": {str(t): round(s, 2) for t, s in sorted(seconds.items())},
        "inference_seconds_projected_all_tiers": round(projected, 1),
        "inference_seconds_allowed": round(allowed, 1),
        # >1 means the ten tiers do not fit in the official allowance, and the
        # value is exactly the speedup still needed.
        "over_budget_factor": round(over, 2),
        "time_penalty": round(penalty, 4),
        "accuracy_by_tier": {str(t): round(a, 4) for t, a in accuracies.items()},
        "perturbation": merged.get("perturbation", "unknown"),
    }
    if grade.fault:
        public["fault"] = grade.fault[:300]

    return {
        "combined_score": float(combined),
        "public": public,
        "private": {
            "leaderboard_fitness": grade.fitness,
            "passed": bool(grade.passed),
            "stage_reached": grade.stage_reached,
            "time_projection_used": is_projected,
            "execution_time": grade.execution_time,
            "perturbation_original_acc": merged.get("perturbation_original_acc"),
            "perturbation_random_acc": merged.get("perturbation_random_acc"),
        },
    }


def main(program_path: str, results_dir: str) -> int:
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)

    # A candidate directory, never a repo path: train() is resumable by
    # contract, so it continues from whatever weights sit in the directory it
    # is handed and writes back there. grade_workspace now refuses paths under
    # seeds/ for exactly this reason.
    candidate = results / "candidate"
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir()
    shutil.copy(program_path, candidate / "model.py")

    error = None
    try:
        grade = grade_workspace(
            candidate,
            GradeContext(
                candidate_id=results.name,
                workdir=results / "work",
                operator="shinka",
                generation=0,
            ),
        )
        cases = _cases_per_tier()
        metrics = build_metrics(grade, cases)
        correct = bool(grade.passed)
        if not correct:
            error = grade.fault or "candidate did not pass"
    except Exception as exc:  # noqa: BLE001 — a crash is a candidate verdict
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        # A failed candidate still needs a finite score, or it cannot be
        # ranked against the ones that ran.
        metrics = {
            "combined_score": -1.0,
            "public": {"fault": error[:300]},
            "private": {"traceback": traceback.format_exc()[-2000:]},
        }
        correct = False

    (results / "metrics.json").write_text(json.dumps(metrics, indent=4))
    (results / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=4)
    )
    print(json.dumps(metrics["public"], indent=2, sort_keys=True))
    print(f"combined_score = {metrics['combined_score']:.4f}  correct = {correct}")
    return 0


def _cases_per_tier() -> int:
    """Cases per tier in the deepest rung — the denominator of the official
    time allowance. Read from the grader so QUICK and full runs agree."""
    from experiments.modmul.grade import _rungs

    return _rungs()[-1].cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="modmul evaluator for ShinkaEvolve")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    raise SystemExit(main(args.program_path, args.results_dir))
