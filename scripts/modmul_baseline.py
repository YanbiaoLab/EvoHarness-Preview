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
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.pop("MODMUL_QUICK", None)          # full rungs, not the 20s probe
os.environ["MODMUL_FORCE_ALL_RUNGS"] = "1"    # score every tier, no early exit

from evoharness.evoserve import GradeContext  # noqa: E402
from experiments.modmul.grade import SCORED_TIERS, grade_workspace  # noqa: E402
from experiments.modmul.task import PRIMARY_SEED, _SEEDS  # noqa: E402


def main() -> int:
    names = sys.argv[1:] or [PRIMARY_SEED]
    out = Path("results/modmul_baseline.json")
    out.parent.mkdir(exist_ok=True)
    recorded = json.loads(out.read_text()) if out.exists() else {}

    for name in names:
        started = time.monotonic()
        print(f"\n=== seed {name} (full rungs, this takes ~1.5h of training)")
        with tempfile.TemporaryDirectory(prefix=f"modmul_base_{name}_") as tmp:
            # Grade a COPY. Training is resumable by contract — train() picks
            # up from whatever weights already sit in model_dir and writes
            # back there — so handing grade_workspace a repo path retrains
            # the committed seed in place. The first version of this script
            # did exactly that and pushed limb_horner from 5,577 steps to
            # 37,088, which both invalidated the measurement (it was no
            # longer a baseline) and moved the starting point of every
            # future run.
            candidate = Path(tmp) / "candidate"
            shutil.copytree(_SEEDS / name, candidate)
            grade = grade_workspace(
                candidate,
                GradeContext(
                    candidate_id=f"baseline-{name}",
                    workdir=Path(tmp) / "work",
                    operator="seed",
                    generation=0,
                ),
            )
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
