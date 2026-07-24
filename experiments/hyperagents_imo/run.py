"""Admission gate for the comparable HyperAgents IMO experiment.

Usage:
    python -m experiments.hyperagents_imo.run verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import verify_contract
from .scoring import score_benchmark, score_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    score = subparsers.add_parser(
        "score-pair",
        help="grade one HyperAgents proof and one EvoHarness proof identically",
    )
    score.add_argument("--problem-id", required=True)
    score.add_argument("--hyperagents-csv", type=Path, required=True)
    score.add_argument("--evoharness-item", type=Path, required=True)
    score.add_argument("--output", type=Path)
    benchmark = subparsers.add_parser(
        "score-benchmark",
        help="grade all 60 HyperAgents proofs with the frozen benchmark grader",
    )
    benchmark.add_argument("--hyperagents-csv", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_contract()
    elif args.command == "score-pair":
        result = score_pair(
            problem_id=args.problem_id,
            hyperagents_csv=args.hyperagents_csv,
            evoharness_item=args.evoharness_item,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        result = score_benchmark(
            hyperagents_csv=args.hyperagents_csv,
            output_dir=args.output_dir,
            max_workers=args.workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.command != "verify" or result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
