"""Run one evolved HyperAgents TaskAgent on one problem.

This process receives only the problem statement. Reference solutions and
grading guidelines remain in the privileged outer evaluator.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--problem-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    args = parser.parse_args(argv)

    workspace = args.workspace.resolve()
    value = json.loads(args.problem_file.read_text(encoding="utf-8"))
    problem_id = str(value["problem_id"])
    problem = str(value["problem"])
    if set(value) != {"problem_id", "problem"}:
        raise ValueError("candidate input must contain only problem_id and problem")

    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    llm_module = importlib.import_module("agent.llm")
    original_get_response = llm_module.get_response_from_llm
    usage = {"calls": 0}

    def budgeted_get_response(*call_args, **call_kwargs):
        if usage["calls"] >= args.max_calls:
            raise RuntimeError(
                f"solver call budget exceeded ({args.max_calls} calls)"
            )
        usage["calls"] += 1
        return original_get_response(*call_args, **call_kwargs)

    llm_module.get_response_from_llm = budgeted_get_response
    module = importlib.import_module("task_agent")
    agent = module.TaskAgent(
        model=args.model,
        chat_history_file=str(args.log),
    )
    result = agent.forward({"domain": "imo_proof", "problem": problem})
    prediction = result[0] if isinstance(result, tuple) else result
    if prediction is None:
        raise ValueError("TaskAgent returned no proof")
    proof = str(prediction).strip()
    if not proof:
        raise ValueError("TaskAgent returned an empty proof")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "problem_id": problem_id,
                "proof": proof,
                "solver_calls": usage["calls"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
