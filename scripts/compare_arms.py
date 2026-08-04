"""Compare experiment arms that differ only by recipe.

    python scripts/compare_arms.py genesis_e0_s1 genesis_e3r_s1 genesis_e6p_s1

Reads each run's manifest, checkpoint and metric log and prints one row per
arm: what the seed scored, what the search reached, how many candidates were
actually evaluated, and what was spent getting there. The seed column matters
more than it looks — every arm starts from the same genome, so an arm whose
seed score differs is reporting evaluation noise, and differences smaller
than that spread are not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarise(run_dir: Path) -> dict:
    manifest = _read_json(run_dir / "manifest.json")
    checkpoint = _read_json(run_dir / "checkpoint.json")
    report = checkpoint.get("run_report", {})
    rows = _metrics(run_dir)

    fitnesses = [
        row["metrics"].get("sys/fitness")
        for row in rows
        if isinstance(row.get("metrics"), dict)
    ]
    fitnesses = [value for value in fitnesses if isinstance(value, (int, float))]
    seed_fitness = fitnesses[0] if fitnesses else None
    best = report.get("best_fitness")
    if best is None and fitnesses:
        best = max(fitnesses)

    improved_at = None
    if seed_fitness is not None:
        for index, value in enumerate(fitnesses):
            if value > seed_fitness:
                improved_at = index
                break

    return {
        "run": run_dir.name,
        "recipe": manifest.get("recipe", "?"),
        "task": manifest.get("task", "?"),
        "seed_fitness": seed_fitness,
        "best_fitness": best,
        "evaluated": len(fitnesses),
        "failed_proposals": report.get("proposals_failed"),
        "first_improvement": improved_at,
        "stopped": report.get("stopped_reason", "running"),
        "llm_cost": report.get("llm_cost"),
        "fitnesses": fitnesses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    summaries = [summarise(args.results_dir / name) for name in args.runs]

    header = (
        f"{'run':<28} {'recipe':<6} {'seed':>8} {'best':>8} {'gain':>8} "
        f"{'evals':>6} {'failed':>7} {'first+':>7}  status"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        seed = item["seed_fitness"]
        best = item["best_fitness"]
        gain = (best - seed) if (seed is not None and best is not None) else None
        print(
            f"{item['run']:<28} {item['recipe']:<6} "
            f"{seed if seed is None else round(seed, 4):>8} "
            f"{best if best is None else round(best, 4):>8} "
            f"{gain if gain is None else round(gain, 4):>8} "
            f"{item['evaluated']:>6} "
            f"{item['failed_proposals'] if item['failed_proposals'] is not None else '-':>7} "
            f"{item['first_improvement'] if item['first_improvement'] is not None else '-':>7}"
            f"  {item['stopped']}"
        )

    print("\nfitness trajectories (seed first):")
    for item in summaries:
        trail = ", ".join(f"{value:.4f}" for value in item["fitnesses"])
        print(f"  {item['run']:<28} {trail}")

    if args.json:
        args.json.write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
