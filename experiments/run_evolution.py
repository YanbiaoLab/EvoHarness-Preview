# EvoHarness original: the thin experiment driver. Group selection is
# `--recipe` (a Python module in recipes/, verl algorithm-zoo style);
# hyperparameters come from an optional YAML + dot-path overrides.
"""Run one experiment group.

    python -m experiments.run_evolution --recipe e3r --task demo_counter \
        --run-dir results/e3r_s1 --set search.seed=1 search.num_generations=20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import recipes
from evoharness.evocore import LLMClient, make_openai_compat_transport
from evoharness.evoguard import BudgetMeter, write_manifest
from recipes.common import RecipeContext
from tasks import get_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True,
                        help=f"one of {sorted(recipes.REGISTRY)}")
    parser.add_argument("--task", default="demo_counter")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help=(
                            "hyperparameter YAML "
                            "({search, population, plus, proposal})"
                        ))
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument(
        "--brief", type=Path, default=None,
        help="frozen research brief (markdown); overrides the task's own",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="use a real LLM instead of the task's bundled fake transport; "
        "reads EVOHARNESS_API_BASE and EVOHARNESS_API_KEY from the "
        "environment (OpenAI-compatible endpoint)",
    )
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="section.key=value")
    args = parser.parse_args(argv)

    recipe = recipes.get_recipe(args.recipe)
    search, population, plus, proposal = recipes.load_experiment_config(
        args.config, args.overrides
    )
    task = get_task(args.task)
    if task.task_sys_msg and not search.task_sys_msg:
        search.task_sys_msg = task.task_sys_msg

    budget = None
    if args.budget_usd is not None:
        budget = BudgetMeter(
            args.budget_usd, state_path=args.run_dir / "budget.json"
        )

    brief = args.brief.read_text() if args.brief else task.research_brief
    brief_sha = hashlib.sha256(brief.encode()).hexdigest() if brief else None

    transport = task.transport
    if args.live:
        api_base = os.environ.get("EVOHARNESS_API_BASE")
        api_key = os.environ.get("EVOHARNESS_API_KEY")
        if not api_base or not api_key:
            parser.error(
                "--live requires EVOHARNESS_API_BASE and EVOHARNESS_API_KEY"
            )
        # Reasoning models spend minutes before the first token, and this
        # task's prompt is large (task brief + research brief + a
        # multi-file genome). The 120s default timed out every proposal on
        # MiniMax-M3, which the proposer breaker correctly reads as a dead
        # proposer — a healthy model would be mistaken for a broken one.
        transport = make_openai_compat_transport(
            api_base,
            api_key,
            # Measured on MiniMax-M3 with this task's ~12k-token prompt: a
            # complete proposal takes ~105s. Sized to 3x that, not to the
            # worst imaginable case — an over-long timeout does not make a
            # slow call succeed, it only lets a stalled socket hold the run,
            # which is how a previous run sat dead for 2h04m looking healthy.
            timeout_s=float(os.environ.get("EVOHARNESS_LLM_TIMEOUT_S", 300)),
        )

    # Only the run knows where lineage state can live, and only graders that
    # opted in have the attribute at all.
    if getattr(task.grader, "lineage_dir", "absent") is None:
        task.grader.lineage_dir = args.run_dir / "lineage"

    ctx = RecipeContext(
        search=search,
        population=population,
        plus=plus,
        grader=task.grader,
        llm=LLMClient(
            temperature=search.llm_temperature,
            max_tokens=search.llm_max_tokens,
            transport=transport,
        ),
        run_dir=args.run_dir,
        proposal=proposal,
        budget=budget,
        research_brief=brief or None,
        preflight_validators=tuple(task.preflight_validators),
        runner=task.runner,
    )
    loop = recipe.build(ctx)
    report = loop.run(
        task.initial_code,
        extra_seeds=task.extra_seeds or None,
        initial_workspace=task.initial_workspace,
    )

    write_manifest(
        args.run_dir / "manifest.json",
        recipe=recipe.NAME,
        recipe_description=recipe.DESCRIPTION,
        task=args.task,
        search=search,
        population=population,
        plus=plus,
        proposal=ctx.extras["proposal_manifest"],
        budget={
            "hard_cap_usd": args.budget_usd,
            "spent_usd": (
                budget.spent_usd
                if budget is not None
                else report.total_llm_cost + report.total_eval_cost
            ),
            "llm_cost_usd": report.total_llm_cost,
            "eval_cost_usd": report.total_eval_cost,
        },
        report=report,
        research_brief_sha256=brief_sha,
        metric_summary=loop.metric_log.summary() if loop.metric_log else {},
    )
    print(json.dumps({
        "recipe": recipe.NAME,
        "stopped_reason": report.stopped_reason,
        "generations": report.generations_completed,
        "evaluations": report.evaluations,
        "best_fitness": report.best_fitness,
        "llm_cost": round(report.total_llm_cost, 4),
        "eval_cost": round(report.total_eval_cost, 4),
        "run_dir": str(args.run_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
