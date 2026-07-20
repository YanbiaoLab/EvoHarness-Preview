"""Optional paid S8 smoke against an OpenAI-compatible model service.

The command first verifies a real two-request tool-call round trip, then runs
one agentic generation on the GitWorkspace-backed ``s8_multifile`` task.
Nothing in the default test suite invokes this command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import recipes
from evoharness.evocore import (
    LLMClient,
    LLMMessage,
    LLMToolChoice,
    LLMToolChoiceMode,
    LLMToolDefinition,
    LLMToolResult,
    PopulationConfig,
    ProposalConfig,
    SearchConfig,
    atomic_write_json,
    make_openai_compat_transport,
)
from evoharness.evoguard import BudgetMeter
from evoharness.evoplus.config import PlusConfig
from recipes.common import RecipeContext
from tasks import get_task


def _with_token_pricing(
    transport,
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
):
    for name, value in (
        ("input_cost_per_million", input_cost_per_million),
        ("output_cost_per_million", output_cost_per_million),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be nonnegative and finite")

    def priced_transport(**kwargs):
        response = transport(**kwargs)
        cost = (
            response.prompt_tokens * input_cost_per_million
            + response.completion_tokens * output_cost_per_million
        ) / 1_000_000
        return replace(response, cost=cost)

    return priced_transport


def _protocol_probe(
    client: LLMClient,
    *,
    model: str,
    timeout_s: float,
) -> dict[str, object]:
    definition = LLMToolDefinition(
        name="record_probe",
        description="Record the fixed S8 protocol probe token.",
        input_schema={
            "type": "object",
            "properties": {
                "token": {"type": "string", "enum": ["s8"]},
            },
            "required": ["token"],
            "additionalProperties": False,
        },
    )
    initial = (
        LLMMessage("system", "Follow tool instructions exactly."),
        LLMMessage("user", "Call record_probe with token s8."),
    )
    first = client.query_messages(
        initial,
        model,
        tools=(definition,),
        tool_choice=LLMToolChoice(
            LLMToolChoiceMode.SPECIFIC,
            "record_probe",
        ),
        parallel_tool_calls=False,
        timeout_s=timeout_s,
    )
    if len(first.tool_calls) != 1:
        raise RuntimeError("provider did not return exactly one probe tool call")
    call = first.tool_calls[0]
    if call.name != "record_probe" or call.arguments != {"token": "s8"}:
        raise RuntimeError("provider did not preserve the forced probe schema")

    history = (
        *initial,
        LLMMessage(
            "assistant",
            first.text,
            tool_calls=first.tool_calls,
        ),
        LLMMessage(
            "tool",
            tool_results=(
                LLMToolResult(call.call_id, '{"recorded":true}'),
            ),
        ),
    )
    second = client.query_messages(
        history,
        model,
        tools=(definition,),
        tool_choice=LLMToolChoice(LLMToolChoiceMode.NONE),
        parallel_tool_calls=False,
        timeout_s=timeout_s,
    )
    if not second.text.strip() or second.tool_calls:
        raise RuntimeError("provider did not complete after the probe tool result")

    return {
        "model": second.model or first.model or model,
        "call_id": call.call_id,
        "arguments": dict(call.arguments),
        "followup_text_present": bool(second.text.strip()),
        "prompt_tokens": first.prompt_tokens + second.prompt_tokens,
        "completion_tokens": (
            first.completion_tokens + second.completion_tokens
        ),
        "cost_usd": first.cost + second.cost,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("EVOHARNESS_API_BASE"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EVOHARNESS_API_KEY"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EVOHARNESS_MODEL"),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=os.environ.get("EVOHARNESS_INPUT_COST_PER_MILLION"),
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=os.environ.get("EVOHARNESS_OUTPUT_COST_PER_MILLION"),
    )
    args = parser.parse_args(argv)
    if not args.api_base or not args.api_key or not args.model:
        parser.error(
            "provide --api-base/--api-key/--model or set "
            "EVOHARNESS_API_BASE, EVOHARNESS_API_KEY and EVOHARNESS_MODEL"
        )
    if (
        args.input_cost_per_million is None
        or args.output_cost_per_million is None
    ):
        parser.error(
            "provide explicit input/output token pricing with "
            "--input-cost-per-million and --output-cost-per-million "
            "(use 0 for a free local model)"
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    budget = BudgetMeter(
        args.budget_usd,
        state_path=args.run_dir / "budget.json",
    )
    transport = _with_token_pricing(
        make_openai_compat_transport(
            args.api_base,
            args.api_key,
            timeout_s=args.timeout_s,
        ),
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    client = LLMClient(
        temperature=0.0,
        max_tokens=2_048,
        transport=transport,
    )

    protocol = _protocol_probe(
        client,
        model=args.model,
        timeout_s=args.timeout_s,
    )
    budget.charge(float(protocol["cost_usd"]))
    atomic_write_json(args.run_dir / "protocol_probe.json", protocol)
    if budget.should_stop():
        raise RuntimeError("budget exhausted by protocol probe")

    task = get_task("s8_multifile")
    run_dir = args.run_dir / "multifile"
    ctx = RecipeContext(
        search=SearchConfig(
            num_generations=1,
            operators=["rewrite"],
            operator_probs=[1.0],
            llm_models=[args.model],
            task_sys_msg=task.task_sys_msg,
        ),
        population=PopulationConfig(num_islands=1),
        plus=PlusConfig(),
        proposal=ProposalConfig(
            mode="agentic",
            model=args.model,
            max_turns=12,
            max_tool_calls=30,
            timeout_s=args.timeout_s,
            max_cost_usd=max(0.0, args.budget_usd - budget.spent_usd),
            max_repair_rounds=2,
        ),
        grader=task.grader,
        llm=client,
        run_dir=run_dir,
        budget=budget,
        preflight_validators=task.preflight_validators,
        runner=task.runner,
    )
    loop = recipes.get_recipe("e0").build(ctx)
    report = loop.run(
        task.initial_code,
        initial_workspace=task.initial_workspace,
    )
    generated = [
        candidate
        for candidate in loop.store.all_candidates()
        if candidate.generation == 1
    ]
    if not generated or report.best_fitness != 1.0:
        raise RuntimeError("live agent did not produce a valid multi-file child")
    child = generated[-1]
    if child.workspace_kind != "git":
        raise RuntimeError("live child did not preserve GitWorkspace lineage")
    changed_files = sorted(child.workspace.texts())
    required = {"math_ops.py", "metadata.py"}
    parent_texts = task.initial_workspace.texts()
    actually_changed = sorted(
        path
        for path in set(parent_texts) | set(child.workspace.texts())
        if parent_texts.get(path) != child.workspace.texts().get(path)
    )
    if not required.issubset(actually_changed):
        raise RuntimeError(
            "live child passed without modifying both required project files"
        )

    output = {
        "schema_version": 1,
        "protocol_probe": protocol,
        "multifile": {
            "report": asdict(report),
            "proposal": ctx.extras["proposal_manifest"],
            "workspace_kind": child.workspace_kind,
            "workspace_files": changed_files,
            "changed_files": actually_changed,
            "trace_path": child.metadata.get("trace_path"),
        },
        "budget": {
            "hard_cap_usd": args.budget_usd,
            "spent_usd": budget.spent_usd,
            "input_cost_per_million": args.input_cost_per_million,
            "output_cost_per_million": args.output_cost_per_million,
        },
    }
    atomic_write_json(args.run_dir / "live_smoke.json", output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
