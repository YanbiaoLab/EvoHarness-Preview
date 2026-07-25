"""Command entrypoint for the independent IMO proof experiment."""

from __future__ import annotations
import dotenv
import argparse
import json
import os
from pathlib import Path

from evoharness.evocore import (
    LLMClient,
    LLMMessage,
    LLMToolChoice,
    LLMToolChoiceMode,
    LLMToolDefinition,
    LLMToolResult,
)

from .evaluation.engine import make_live_evaluator
from .evaluation.contract import EvaluationRouter
from .evaluation.service import HttpEvaluationBackend
from .evolution import run_experiment
from .protocol import BenchmarkSpec, default_spec_path
from .transport import SpecOpenAITransport

dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clients(spec: BenchmarkSpec):
    api_base = os.environ.get("EVOHARNESS_API_BASE")
    api_key = os.environ.get("EVOHARNESS_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError(
            "live runs require EVOHARNESS_API_BASE and EVOHARNESS_API_KEY"
        )

    def make_client(model):
        return LLMClient(
            model.temperature,
            model.max_output_tokens,
            transport=SpecOpenAITransport(
                api_base,
                api_key,
                model.enable_thinking,
                model.input_cost_per_million,
                model.output_cost_per_million,
            ),
        )

    return (
        make_client(spec.optimizer),
        make_client(spec.solver),
        make_client(spec.grader.model),
    )


def _probe_live(spec: BenchmarkSpec) -> dict[str, object]:
    optimizer, solver, grader = _clients(spec)
    tool = LLMToolDefinition(
        name="protocol_probe",
        description="Return the supplied value through a protocol probe.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    first_messages = (
        LLMMessage("user", "Call protocol_probe with value PROBE_OK."),
    )
    first = optimizer.query_messages(
        first_messages,
        spec.optimizer.name,
        tools=(tool,),
        tool_choice=LLMToolChoice(LLMToolChoiceMode.REQUIRED),
        parallel_tool_calls=False,
    )
    if len(first.tool_calls) != 1 or first.tool_calls[0].name != tool.name:
        raise RuntimeError("optimizer provider did not return the required tool call")
    followup = optimizer.query_messages(
        (
            *first_messages,
            LLMMessage(
                "assistant",
                content=first.text,
                tool_calls=first.tool_calls,
            ),
            LLMMessage(
                "tool",
                tool_results=(
                    LLMToolResult(first.tool_calls[0].call_id, "PROBE_OK"),
                ),
            ),
        ),
        spec.optimizer.name,
        tools=(tool,),
        tool_choice=LLMToolChoice(LLMToolChoiceMode.AUTO),
        parallel_tool_calls=False,
    )
    solver_response = solver.query(
        "Return exactly PROBE_OK.",
        "PROBE_OK",
        spec.solver.name,
    )
    grader_response = grader.query(
        "Return exactly PROBE_OK.",
        "PROBE_OK",
        spec.grader.model.name,
    )
    responses = (first, followup, solver_response, grader_response)
    return {
        "tool_call_roundtrip": bool(followup.text.strip()),
        "optimizer_model": followup.model,
        "solver_model": solver_response.model,
        "grader_model": grader_response.model,
        "observed_calls": len(responses),
        "observed_tokens": sum(
            response.prompt_tokens + response.completion_tokens
            for response in responses
        ),
        "observed_cost_usd": sum(response.cost for response in responses),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=default_spec_path())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("probe-live")
    run = sub.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="parallel per-problem evaluations in the live evaluator",
    )
    run.add_argument("--train-eval-url")
    run.add_argument("--validation-eval-url")
    run.add_argument("--test-eval-url")
    run.add_argument(
        "--experience-mode",
        default="lessons+scratchpad",
        choices=[
            "off",
            "global",
            "retrieval",
            "retrieval+rejected",
            "lessons",
            "lessons+scratchpad",
        ],
        help="experience-injection arm (design §5): E3r=retrieval, "
        "E4a=retrieval+rejected, E5r=lessons, E5s=lessons+scratchpad",
    )
    run.add_argument(
        "--lesson-directive",
        action="store_true",
        help="promote the parent's own lesson into an explicit mutation "
        "directive (probabilistic; requires a lessons arm to have effect)",
    )
    run.add_argument(
        "--reflect-batch-size",
        type=int,
        default=4,
        help="graded mutations per reflection batch (small runs need small "
        "batches so lessons arrive early enough to matter)",
    )
    run.add_argument(
        "--operator-bandit",
        action="store_true",
        help="adaptive operator scheduling: floor+softmax over decayed "
        "per-operator fitness deltas instead of the static probabilities",
    )
    args = parser.parse_args(argv)

    spec = BenchmarkSpec.load(args.spec)
    spec.verify_workspace(PROJECT_ROOT)
    if args.command == "verify":
        print(
            json.dumps(
                {
                    "experiment_id": spec.benchmark_id,
                    "fingerprint": spec.fingerprint,
                    "verified": True,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "probe-live":
        print(json.dumps(_probe_live(spec), indent=2))
        return 0

    optimizer, solver, grader = _clients(spec)
    eval_urls = {
        "train": args.train_eval_url,
        "validation": args.validation_eval_url,
        "test": args.test_eval_url,
    }
    if any(eval_urls.values()) and not all(eval_urls.values()):
        parser.error(
            "remote evaluation requires train, validation, and test URLs"
        )
    if all(eval_urls.values()):
        eval_token = os.environ.get("EVOHARNESS_EVAL_TOKEN", "")
        evaluation_backend = EvaluationRouter(
            {
                split: HttpEvaluationBackend(
                    url,
                    spec,
                    profile=split,
                    auth_token=eval_token,
                )
                for split, url in eval_urls.items()
            }
        )
    else:
        evaluation_backend = make_live_evaluator(
            project_root=PROJECT_ROOT,
            spec=spec,
            solver_client=solver,
            grader_client=grader,
            max_workers=args.concurrency,
        )
    summary = run_experiment(
        spec=spec,
        evaluation_backend=evaluation_backend,
        optimizer_client=optimizer,
        run_dir=args.run_dir,
        evolution_seed=args.seed,
        experience_mode=args.experience_mode,
        lesson_directive=args.lesson_directive,
        operator_bandit=args.operator_bandit,
        reflect_batch_size=args.reflect_batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
