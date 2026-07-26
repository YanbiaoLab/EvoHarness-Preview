"""Measure how much of the fitness signal is evaluation noise.

Everything downstream — selection, the reflector's noise verdict, whether
any A/B is readable — rests on an unmeasured premise: how much does the
same candidate's score move when nothing about it changes? All three
models run at temperature 0, so the answer may be "not at all", in which
case repeat-sampling buys nothing and only item count can help.

Evaluates one candidate on the same split twice, then on a disjoint split,
and reports both variances. Usage:

    python scripts/measure_eval_noise.py results/e5s_r2 [candidate_id]
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoharness.evocore.workspace import load_workspace  # noqa: E402
from experiments.imo_proof.evaluation.engine import (  # noqa: E402
    make_live_evaluator,
)
from experiments.imo_proof.protocol import (  # noqa: E402
    BenchmarkSpec,
    default_spec_path,
)
from experiments.imo_proof.run import PROJECT_ROOT, _clients  # noqa: E402


def _pass_vector(evaluation) -> str:
    return "".join(
        "1" if p.label == "correct" else ("~" if p.label == "almost" else "0")
        for p in sorted(evaluation.problems, key=lambda x: x.problem_id)
    )


def main() -> int:
    run_dir = Path(sys.argv[1])
    con = sqlite3.connect(run_dir / "run.db")
    rows = list(
        con.execute(
            "SELECT id, code, workspace_kind, json_extract(report,'$.fitness') "
            "FROM candidates WHERE json_extract(report,'$.fitness') IS NOT NULL "
            "ORDER BY json_extract(report,'$.fitness') DESC"
        )
    )
    con.close()
    wanted = sys.argv[2] if len(sys.argv) > 2 else None
    row = next((r for r in rows if r[0] == wanted), rows[0])
    candidate_id, code, kind, recorded = row
    print(f"candidate {candidate_id}, recorded train fitness {recorded:.4f}")

    spec = BenchmarkSpec.load(default_spec_path())
    spec.verify_workspace(PROJECT_ROOT)
    _, solver, grader = _clients(spec)
    backend = make_live_evaluator(
        project_root=PROJECT_ROOT,
        spec=spec,
        solver_client=solver,
        grader_client=grader,
        max_workers=3,
    )
    workspace = load_workspace(kind or "file", code)

    results = {}
    partial = run_dir / "eval_noise_partial.json"
    for label, split in (("train#1", "train"), ("train#2", "train"),
                         ("validation", "validation")):
        # One transient provider failure used to discard every split already
        # paid for; the first attempt died 12 problems in, competing with a
        # concurrent experiment run for the same endpoint.
        evaluation = None
        for attempt in range(1, 4):
            try:
                with tempfile.TemporaryDirectory(prefix="noise_") as tmp:
                    root = workspace.materialize(Path(tmp))
                    with tempfile.TemporaryDirectory(prefix="noise_out_") as out:
                        evaluation = backend.evaluate_directory(
                            candidate_id=f"{candidate_id}-{label}",
                            candidate_root=root,
                            split=split,
                            output_dir=Path(out),
                        )
                break
            except Exception as exc:  # noqa: BLE001 — transient provider faults
                print(f"{label}: attempt {attempt}/3 failed: {exc}")
        if evaluation is None:
            print(f"{label}: giving up after 3 attempts")
            return 1
        results[label] = evaluation
        partial.write_text(json.dumps(
            {k: {"points_percentage": v.points_percentage,
                 "vector": _pass_vector(v)} for k, v in results.items()},
            indent=2))
        print(
            f"{label:<11} points {evaluation.points_percentage:.4f}  "
            f"correct {evaluation.correct_percentage:.4f}  "
            f"vector {_pass_vector(evaluation)}"
        )

    a, b = results["train#1"], results["train#2"]
    repeat_delta = abs(a.points_percentage - b.points_percentage)
    flips = sum(
        1
        for x, y in zip(
            sorted(a.problems, key=lambda p: p.problem_id),
            sorted(b.problems, key=lambda p: p.problem_id),
        )
        if x.label != y.label
    )
    split_delta = abs(a.points_percentage - results["validation"].points_percentage)

    print()
    print(f"REPEAT  same split twice : delta {repeat_delta:.4f}, {flips} items changed")
    print(f"SPLIT   train vs validation: delta {split_delta:.4f}")
    print()
    if repeat_delta < 1e-9 and flips == 0:
        print("=> evaluation is deterministic at temperature 0. Repeat-sampling "
              "cannot reduce noise; only item count can.")
    else:
        print(f"=> evaluation is stochastic: one repeat moved the score by "
              f"{repeat_delta:.4f}. Repeat-sampling has something to average.")
    print(f"=> item-sampling spread (train vs validation) is {split_delta:.4f}; "
          "this is the term more items would shrink.")
    json.dump(
        {
            label: {
                "points_percentage": e.points_percentage,
                "correct_percentage": e.correct_percentage,
                "vector": _pass_vector(e),
            }
            for label, e in results.items()
        },
        open(run_dir / "eval_noise.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
