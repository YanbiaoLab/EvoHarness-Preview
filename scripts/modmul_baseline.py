"""Grade a modmul seed at FULL rungs and report every tier.

Nothing about attacking this domain can be planned from the QUICK numbers:
QUICK trains for 20 seconds and scores 5 cases per tier, so a tier's
accuracy can only ever be a multiple of 0.2 and the whole run finishes
before the model has learned anything. The recorded QUICK result (h90=1)
says almost nothing about where the seed actually stands.

This runs the real rungs (480s + 1320s + 3600s of cumulative training,
30/40/50 cases per tier) with promotion forced, so every tier gets scored
even when ASHA would have cut the candidate off early. Usage:

    python scripts/modmul_baseline.py [seed_name ...]
"""

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.pop("MODMUL_QUICK", None)          # full rungs, not the 20s probe
os.environ["MODMUL_FORCE_ALL_RUNGS"] = "1"    # score every tier, no early exit

from evoharness.serve import GradeContext  # noqa: E402
from experiments.modmul.grade import SCORED_TIERS, grade_workspace  # noqa: E402
from experiments.modmul.task import PRIMARY_SEED, _SEEDS  # noqa: E402


def main() -> int:
    names = sys.argv[1:] or [PRIMARY_SEED]
    root = Path("results/modmul_baseline")
    root.mkdir(parents=True, exist_ok=True)
    out = Path("results/modmul_baseline.json")
    recorded = json.loads(out.read_text()) if out.exists() else {}

    for name in names:
        started = time.monotonic()
        print(f"\n=== seed {name} (full rungs, this takes ~1.5h of training)")
        # Grade a COPY, and KEEP it. Two separate mistakes made in one day:
        #
        #  * training is resumable by contract — train() continues from
        #    whatever weights sit in model_dir and writes back there — so
        #    handing grade_workspace a repo path retrains the committed seed
        #    in place, which pushed limb_horner from 5,577 to 37,088 steps
        #    and silently stopped the measurement being a baseline;
        #  * the fix for that graded inside a TemporaryDirectory, which threw
        #    away the trained weights on exit. 85 minutes of training reached
        #    h90=9 and left nothing behind but a metrics dict — and those
        #    weights are the artifact a submission is built from.
        #
        # So: a fresh copy under a persistent directory, refusing to reuse an
        # existing one, since silently resuming is the first mistake again.
        candidate = root / name / "candidate"
        if candidate.exists():
            print(f"  {candidate} already exists — refusing to resume into it.")
            print("  Move or delete it to re-measure from the committed seed.")
            return 1
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_SEEDS / name, candidate)
        grade = grade_workspace(
            candidate,
            GradeContext(
                candidate_id=f"baseline-{name}",
                workdir=root / name / "work",
                operator="seed",
                generation=0,
            ),
        )
        print(f"  trained weights kept at {candidate}")
        metrics = {**grade.visible_metrics, **grade.hidden_metrics}
        accuracies = {
            tier: float(metrics.get(f"acc_tier_{tier}", 0.0))
            for tier in SCORED_TIERS
        }
        print(f"  fitness {grade.fitness:.4f}  passed={grade.passed}")
        if grade.fault:
            print(f"  fault: {grade.fault}")
        print(f"  h90 {metrics.get('h90')}  overall {metrics.get('overall_accuracy')}")
        for tier, value in accuracies.items():
            bar = "#" * int(value * 40)
            print(f"    tier {tier:>2}  {value:>6.1%}  {bar}")
        # The gap between the top scoring tier and the h90 frontier is what
        # a search actually has to cross. If every tier above h90 is at 0,
        # there is no partial credit to climb and the terrain is a wall,
        # not a slope — that changes the plan, not just the schedule.
        frontier = [t for t in SCORED_TIERS if 0.0 < accuracies[t] < 0.9]
        print(f"  tiers with partial credit (climbable): {frontier or 'NONE'}")
        recorded[name] = {
            "fitness": grade.fitness,
            "passed": grade.passed,
            "fault": grade.fault,
            "elapsed_s": round(time.monotonic() - started, 1),
            "metrics": {k: v for k, v in metrics.items()},
        }
        out.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {out} ({recorded[name]['elapsed_s']:.0f}s)")

    if len(recorded) > 1:
        fitness = [r["fitness"] for r in recorded.values()]
        print(f"\nacross {len(fitness)} seeds: fitness "
              f"{min(fitness):.4f}..{max(fitness):.4f} "
              f"(sd {statistics.stdev(fitness):.4f})" if len(fitness) > 1 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
