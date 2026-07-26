"""Measure evaluation noise for the modmul domain.

Companion to measure_eval_noise.py, and a deliberate contrast: modmul's
candidate is a TRAINING RUN, so repeating it re-rolls initialisation and
data order. Where the IMO domain runs three temperature-0 models and may
be perfectly repeatable, this one has real run-to-run variance — and its
fitness carries a discontinuous term (h90 is the highest tier reaching
90%), so the same jitter is inert far from a threshold and worth 1/11 of
the whole score at one.

Grades the same seed workspace N times and reports the spread of fitness,
of overall accuracy, and of every per-tier accuracy. Usage:

    MODMUL_QUICK=1 python scripts/measure_modmul_noise.py [repeats]
"""

import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoharness.evoserve import GradeContext  # noqa: E402
from experiments.modmul.grade import grade_workspace  # noqa: E402
from experiments.modmul.task import PRIMARY_SEED, _SEEDS  # noqa: E402


def _tier_accuracies(grade) -> dict:
    merged = {**grade.visible_metrics, **grade.hidden_metrics}
    return {
        key: value
        for key, value in sorted(merged.items())
        if "acc_tier" in key or key in ("h90", "overall_accuracy")
    }


def main() -> int:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed_dir = _SEEDS / PRIMARY_SEED
    print(f"seed {PRIMARY_SEED} at {seed_dir}, {repeats} repeats\n")

    runs = []
    for index in range(repeats):
        with tempfile.TemporaryDirectory(prefix=f"modmul_noise_{index}_") as tmp:
            grade = grade_workspace(
                seed_dir,
                GradeContext(
                    candidate_id=f"noise-{index}",
                    workdir=Path(tmp),
                    operator="seed",
                    generation=0,
                ),
            )
        metrics = _tier_accuracies(grade)
        runs.append({"fitness": grade.fitness, "passed": grade.passed, **metrics})
        print(
            f"run {index}: fitness {grade.fitness:.4f} passed={grade.passed} "
            f"{json.dumps(metrics, sort_keys=True)}"
        )

    print()
    keys = sorted({k for r in runs for k in r if k != "passed"})
    print(f"{'metric':<22} {'min':>8} {'max':>8} {'spread':>8} {'sd':>8}")
    for key in keys:
        values = [float(r[key]) for r in runs if key in r]
        if len(values) < 2:
            continue
        sd = statistics.stdev(values)
        print(
            f"{key:<22} {min(values):>8.4f} {max(values):>8.4f} "
            f"{max(values) - min(values):>8.4f} {sd:>8.4f}"
        )

    fitness = [r["fitness"] for r in runs]
    spread = max(fitness) - min(fitness)
    print()
    if len(fitness) < 2:
        print("=> one run measures nothing; pass repeats >= 2.")
    elif spread < 1e-9:
        print("=> repeating a training run reproduced the score exactly; the "
              "noise lives elsewhere (data or tier sampling).")
    else:
        print(f"=> repeating a training run moved fitness by up to {spread:.4f}. "
              "Repeat-sampling has something real to average here, unlike a "
              "temperature-0 deterministic evaluation.")
    h90 = [r.get("h90") for r in runs if r.get("h90") is not None]
    if h90 and len(set(h90)) > 1:
        print(f"=> h90 itself moved ({sorted(set(h90))}): the discontinuous term "
              "is live, so a symmetric confidence interval on fitness would "
              "misdescribe this domain.")
    Path("results").mkdir(exist_ok=True)
    json.dump(runs, open("results/modmul_noise.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
