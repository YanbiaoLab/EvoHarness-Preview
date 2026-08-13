"""Compile public specs into the existing Core configuration objects."""

from __future__ import annotations

from evoharness.contracts import (
    BasicSearchProfile,
    EvolutionSearchProfile,
    RunSpec,
    SearchProfile,
    TaskSpec,
)
from evoharness.core import PopulationConfig, ProposalConfig, SearchConfig


def compile_specs(
    task: TaskSpec,
    run: RunSpec,
    profile: SearchProfile,
) -> tuple[SearchConfig, PopulationConfig, ProposalConfig]:
    common = dict(
        task_sys_msg=task.domain_prompt,
        llm_models=list(run.models),
        llm_temperature=run.llm_temperature,
        llm_max_tokens=run.llm_max_tokens,
        seed=run.seed,
        eval_batch_size=run.evaluation_concurrency,
        proposal_concurrency=run.proposal_concurrency,
        max_consecutive_infra_failures=run.max_consecutive_infra_failures,
    )
    if isinstance(profile, BasicSearchProfile):
        search = SearchConfig(
            num_generations=profile.num_trajectories,
            operators=["rewrite"],
            operator_probs=[1.0],
            novelty_enabled=profile.novelty_enabled,
            repair_enabled=False,
            **common,
        )
        population = PopulationConfig(
            num_islands=1,
            parent_strategy="seed_only",
            migration_rate=0.0,
        )
    elif isinstance(profile, EvolutionSearchProfile):
        search = SearchConfig(
            num_generations=profile.num_generations,
            operators=list(profile.operators),
            operator_probs=list(profile.operator_probs),
            novelty_enabled=profile.novelty_enabled,
            repair_enabled=profile.repair_enabled,
            repair_probability=profile.repair_probability,
            **common,
        )
        population = PopulationConfig(
            num_islands=profile.num_islands,
            archive_size=profile.archive_size,
            migration_interval=profile.migration_interval,
            migration_rate=profile.migration_rate,
            parent_strategy=profile.parent_strategy,
        )
    else:
        raise TypeError(f"unknown search profile: {type(profile).__name__}")
    limits = run.proposal_limits
    proposal = ProposalConfig(
        mode=profile.proposal_mode,
        model=run.proposal_model,
        max_turns=limits.max_turns,
        max_tool_calls=limits.max_tool_calls,
        timeout_s=limits.timeout_s,
        max_cost_usd=limits.max_cost_usd,
        max_input_tokens=limits.max_input_tokens,
    )
    return search, population, proposal


def spec_hashes(
    task: TaskSpec,
    run: RunSpec,
    profile: SearchProfile,
) -> dict[str, str]:
    return {
        "task_hash": task.hash,
        "run_hash": run.hash,
        "search_hash": profile.hash,
    }
