"""Compare experiment arms that differ only by recipe.

    python scripts/compare_arms.py genesis_e0_s1 genesis_e3r_s1 ...
    python scripts/compare_arms.py --group-by-arm genesis_{e0,e3r,e6p}_s{1..5}

Per-run it reports the seed's score, the best reached, and three statistics
that keep working when every arm reaches the same ceiling:

* `to_thr`  — how many candidates until fitness first reached the threshold.
  Under a ceiling this is what the extensions claim to improve: not a higher
  summit, a faster climb.
* `auc`     — mean of the best-so-far curve over the children. Uses every
  candidate rather than only the maximum, so an arm that reaches 0.98 at
  candidate 1 scores above one that reaches it at candidate 5.
* `collapse`— children scoring below half the seed. Catastrophic proposals
  (a reward that trains a degenerate policy, a harness that crashes) cost a
  generation, and an arm that produces more of them is worse in a way the
  maximum never shows.

With --group-by-arm the runs are pooled by arm across search seeds and each
statistic is reported as mean ± standard error over seeds. That is the only
form in which arms can actually be ranked: one run per arm is one sample.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

_RUN_NAME = re.compile(r"^(?P<domain>[a-z]+)_(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)$")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _fitnesses(run_dir: Path) -> list[float]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    values = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The seed is logged once per island; count a candidate once.
        key = row.get("candidate_id")
        if key in seen:
            continue
        seen.add(key)
        value = (row.get("metrics") or {}).get("sys/fitness")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarise(run_dir: Path, threshold: float) -> dict:
    manifest = _read_json(run_dir / "manifest.json")
    checkpoint = _read_json(run_dir / "checkpoint.json")
    report = checkpoint.get("run_report", {})
    fitnesses = _fitnesses(run_dir)

    seed_fitness = fitnesses[0] if fitnesses else None
    children = fitnesses[1:]

    best_so_far, running = [], seed_fitness if seed_fitness is not None else 0.0
    for value in children:
        running = max(running, value)
        best_so_far.append(running)

    to_threshold = None
    for index, value in enumerate(children, start=1):
        if value >= threshold:
            to_threshold = index
            break

    collapse = (
        sum(1 for value in children if seed_fitness and value < 0.5 * seed_fitness)
        if seed_fitness
        else 0
    )

    name = run_dir.name
    match = _RUN_NAME.match(name)
    return {
        "run": name,
        "arm": match.group("arm") if match else manifest.get("recipe", "?"),
        "search_seed": int(match.group("seed")) if match else None,
        "recipe": manifest.get("recipe", "?"),
        "seed_fitness": seed_fitness,
        "best_fitness": max(fitnesses) if fitnesses else None,
        "gain": (max(children) - seed_fitness)
        if (children and seed_fitness is not None)
        else None,
        "to_threshold": to_threshold,
        "auc": statistics.fmean(best_so_far) if best_so_far else None,
        "collapse": collapse,
        "evaluated": len(fitnesses),
        "children": len(children),
        "stopped": report.get("stopped_reason", "running"),
        "fitnesses": fitnesses,
    }


def _cell(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _mean_sem(values: list[float]) -> tuple[float | None, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return mean, 0.0
    return mean, statistics.stdev(clean) / (len(clean) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--group-by-arm", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    summaries = [
        summarise(args.results_dir / name, args.threshold)
        for name in args.runs
        if (args.results_dir / name).is_dir()
    ]
    if not summaries:
        print("no runs found")
        return 1

    header = (
        f"{'run':<24} {'arm':<5} {'seed':>7} {'best':>7} {'gain':>7} "
        f"{'to_thr':>7} {'auc':>7} {'clps':>5} {'n':>4}  status"
    )
    print(header)
    print("-" * len(header))
    for item in sorted(summaries, key=lambda s: (s["arm"], s["search_seed"] or 0)):
        print(
            f"{item['run']:<24} {item['arm']:<5} "
            f"{_cell(item['seed_fitness']):>7} {_cell(item['best_fitness']):>7} "
            f"{_cell(item['gain']):>7} {_cell(item['to_threshold']):>7} "
            f"{_cell(item['auc']):>7} {_cell(item['collapse']):>5} "
            f"{item['children']:>4}  {item['stopped']}"
        )

    if args.group_by_arm:
        arms: dict[str, list[dict]] = {}
        for item in summaries:
            arms.setdefault(item["arm"], []).append(item)
        print(f"\nper arm, mean ± sem over {len(summaries) // max(len(arms), 1)} "
              f"search seeds (threshold {args.threshold}):")
        head = (
            f"  {'arm':<5} {'best':>16} {'auc':>16} {'to_thr':>14} "
            f"{'collapse':>10} {'seeds':>6}"
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for arm, items in sorted(arms.items()):
            best_m, best_e = _mean_sem([i["best_fitness"] for i in items])
            auc_m, auc_e = _mean_sem([i["auc"] for i in items])
            # A run that never reached the threshold is not a small number;
            # it is a censored observation, and averaging it as if it were
            # would flatter the arm. Report reached/total beside the mean.
            reached = [i["to_threshold"] for i in items if i["to_threshold"]]
            thr_m, thr_e = _mean_sem([float(v) for v in reached])
            clps_m, _ = _mean_sem([float(i["collapse"]) for i in items])
            thr_text = (
                f"{_cell(thr_m, 1)}±{_cell(thr_e, 1)} ({len(reached)}/{len(items)})"
                if reached
                else f"- (0/{len(items)})"
            )
            print(
                f"  {arm:<5} "
                f"{_cell(best_m)}±{_cell(best_e, 4):<8} "
                f"{_cell(auc_m)}±{_cell(auc_e, 4):<8} "
                f"{thr_text:>14} {_cell(clps_m, 2):>10} {len(items):>6}"
            )

    print("\nfitness trajectories (seed first):")
    for item in sorted(summaries, key=lambda s: (s["arm"], s["search_seed"] or 0)):
        trail = ", ".join(f"{value:.4f}" for value in item["fitnesses"])
        print(f"  {item['run']:<24} {trail}")

    if args.json:
        args.json.write_text(
            json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
