# EvoHarness original (verl-inspired recipe pattern): the engine provides
# components, each experiment group is a thin assembly file. This module is
# the shared assembly core; recipes stay ~20 lines and their mutual diffs
# ARE the ablation definitions.
"""Shared recipe machinery: RecipeContext and assemble()."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from evoharness.core import (
    AgentSessionLimits,
    AgentSessionProposer,
    ConversationalAgentBackend,
    HybridProposalSelector,
    InspirationSelector,
    JsonlEventSinkFactory,
    LLMClient,
    MetricLog,
    NativeToolAgentBackend,
    PopulationConfig,
    PopulationStore,
    PreflightPipeline,
    ProposalConfig,
    ProposalLane,
    Proposer,
    ProposalPreflight,
    PromptBuilder,
    SearchConfig,
    SearchLoop,
    SingleShotProposer,
    StaticRouter,
    make_parent_selector,
)
from evoharness.core.agent import (
    AgentToolRegistry,
    InspectCandidateTool,
    Runner,
    TokenEstimator,
    make_default_agent_tools,
)
from evoharness.core.interfaces import BudgetLike, Grader
from evoharness.core.novelty import NoveltyGate, hashing_embedding
from evoharness.core.preflight import PreflightValidator
from evoharness.guard import Sandbox
from evoharness.evoplus.config import PlusConfig


@dataclass
class RecipeContext:
    search: SearchConfig
    population: PopulationConfig
    plus: PlusConfig
    grader: Grader
    llm: LLMClient
    run_dir: Path
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    budget: BudgetLike | None = None
    research_brief: str | None = None  # shared by ALL groups, not an ablation
    preflight_validators: tuple[PreflightValidator, ...] = ()
    runner: Runner | None = None
    token_estimator: TokenEstimator | None = None
    extra_agent_tools: tuple = ()  # task-injected agent tools (appended to defaults)
    extras: dict = field(default_factory=dict)  # recipes may stash handles here
    frozen_spec_hashes: dict[str, str] = field(default_factory=dict)


def _estimate_agent_tokens(messages, tools) -> int:
    """Conservative provider-neutral estimate used for context admission."""

    characters = 0
    for message in messages:
        characters += len(message.role) + len(message.content)
        for call in message.tool_calls:
            characters += len(call.call_id) + len(call.name)
            characters += len(
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
            )
        for result in message.tool_results:
            characters += len(result.call_id) + len(result.content)
    for tool in tools:
        characters += len(tool.name) + len(tool.description)
        characters += len(
            json.dumps(tool.input_schema, ensure_ascii=False, sort_keys=True)
        )
    return (characters + 3) // 4


def _resolve_agent_model(ctx: RecipeContext) -> str:
    if ctx.proposal.model is not None:
        return ctx.proposal.model
    if len(ctx.search.llm_models) != 1:
        raise ValueError(
            "agent proposal modes require proposal.model when "
            "search.llm_models contains more than one model"
        )
    return ctx.search.llm_models[0]


def _build_proposer(
    ctx: RecipeContext,
    model_router: StaticRouter,
    store=None,
) -> tuple[Proposer, Proposer | None]:
    mode = ctx.proposal.mode
    if mode == "single_shot":
        ctx.extras["proposal_manifest"] = {
            "mode": mode,
            "models": list(ctx.search.llm_models),
            "max_resamples": ctx.search.max_op_resamples,
            "tools": [],
            "transcript": False,
        }
        return SingleShotProposer(
            llm=ctx.llm,
            model_router=model_router,
            language=ctx.search.language,
            max_resamples=ctx.search.max_op_resamples,
        ), None

    model = _resolve_agent_model(ctx)
    runner = ctx.runner or Sandbox(allow_network=False)
    tools = (
        ()
        if mode == "conversational"
        else (
            *make_default_agent_tools(runner),
            # Reference programs reach the prompt as an inventory; this is
            # how the agent expands one it actually wants to read.
            *((InspectCandidateTool(store),) if store is not None else ()),
            *ctx.extra_agent_tools,
        )
    )
    registry = AgentToolRegistry(tools)
    backend = NativeToolAgentBackend(
        client=ctx.llm,
        model=model,
        registry=registry,
        max_input_tokens=ctx.proposal.max_input_tokens,
        token_estimator=ctx.token_estimator or _estimate_agent_tokens,
        max_parallel_tools=ctx.proposal.max_parallel_tools,
        recent_tool_results_to_keep=(
            ctx.proposal.recent_tool_results_to_keep
        ),
        compact_trigger_ratio=ctx.proposal.compact_trigger_ratio,
    )
    if mode == "conversational":
        backend = ConversationalAgentBackend(
            backend,
            language=ctx.search.language,
        )

    limits = AgentSessionLimits(
        max_turns=ctx.proposal.max_turns,
        max_tool_calls=(
            0
            if mode == "conversational"
            else ctx.proposal.max_tool_calls
        ),
        timeout_s=ctx.proposal.timeout_s,
        max_cost_usd=ctx.proposal.max_cost_usd,
    )
    preflight = ProposalPreflight(
        PreflightPipeline(ctx.preflight_validators)
    )
    ctx.extras["proposal_manifest"] = {
        "mode": mode,
        "model": model,
        "limits": dataclasses.asdict(limits),
        "max_repair_rounds": ctx.proposal.max_repair_rounds,
        "max_input_tokens": ctx.proposal.max_input_tokens,
        "max_parallel_tools": ctx.proposal.max_parallel_tools,
        "recent_tool_results_to_keep": (
            ctx.proposal.recent_tool_results_to_keep
        ),
        "compact_trigger_ratio": ctx.proposal.compact_trigger_ratio,
        "tools": [definition.name for definition in registry.definitions],
        "preflight_validators": [
            validator.name for validator in ctx.preflight_validators
        ],
        "transcript": True,
    }
    agent_proposer = AgentSessionProposer(
        backend=backend,
        preflight=preflight,
        limits=limits,
        event_sink_factory=JsonlEventSinkFactory(ctx.run_dir),
        max_repair_rounds=ctx.proposal.max_repair_rounds,
        work_root=ctx.run_dir / ".agent_work",
    )
    if mode == "hybrid":
        ctx.extras["proposal_manifest"].update(
            {
                "transcript_scope": "agentic_routes",
                "routing": {
                    "agent_probability": (
                        ctx.proposal.hybrid_agent_probability
                    ),
                    "stagnation_generations": (
                        ctx.proposal.hybrid_stagnation_generations
                    ),
                },
            }
        )
        return SingleShotProposer(
            llm=ctx.llm,
            model_router=model_router,
            language=ctx.search.language,
            max_resamples=ctx.search.max_op_resamples,
        ), agent_proposer
    return agent_proposer, None


# Which feedback layer a mounted plugin implies. Declared capability is
# derived from the assembly rather than asserted by hand, so it cannot drift
# from what was actually mounted; observed capability comes from the
# CapabilityLedger at runtime, and the manifest reports both side by side.
_CAPABILITY_MARKERS: dict[str, tuple[str, ...]] = {
    "structured_feedback": ("FeedbackContributor", "SignatureRecorder"),
    "behavior_signature": ("SignatureRecorder", "BehavioralNoveltyPolicy"),
    "experience": ("ExperienceContributor", "ExperienceStore"),
    "reflection": ("MutationReflector",),
}


def _declared_capabilities(mounted_class_names: set[str]) -> list[str]:
    declared = ["scalar"]     # every grader produces at least a fitness
    for capability, markers in _CAPABILITY_MARKERS.items():
        if any(marker in mounted_class_names for marker in markers):
            declared.append(capability)
    return sorted(declared)


def assemble(
    ctx: RecipeContext,
    contributors: list | None = None,
    observers: list | None = None,
    weight_policies: list | None = None,
    operator_selector: object | None = None,
    merge_planner: object | None = None,
    island_health: object | None = None,
    inspiration_policy: object | None = None,
) -> SearchLoop:
    """One canonical wiring; recipes differ only in the plugin lists.

    The research brief (if any) is injected here, ahead of all recipe
    contributors, so every experiment group shares identical external
    knowledge — it is task infrastructure, never an ablation variable."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    contributors = list(contributors or [])
    if ctx.research_brief:
        from evoharness.evoplus import StaticBriefContributor

        contributors.insert(0, StaticBriefContributor(ctx.research_brief))
    store = PopulationStore(ctx.population, ctx.run_dir / "run.db")

    # HITL channel (hitl_design.md): always mounted in every group, inert
    # until a human writes control/directives.json (via the console or by
    # hand). Human guidance is injected ahead of all machine sections.
    from evoharness.evoplus import (
        DirectiveBook,
        HumanDirectiveContributor,
        IslandBriefContributor,
        LineageVetoPolicy,
    )

    book = DirectiveBook(ctx.run_dir / "control" / "directives.json")
    contributors.insert(0, HumanDirectiveContributor(book))
    # Per-island directives, mounted the same way: control/island_briefs.json
    # is absent by default, in which case this is inert and no other task is
    # affected. When present, islands are separated by task as well as by genome,
    # so the fitness gradient cannot pull them all onto one slope.
    contributors.insert(
        1, IslandBriefContributor(ctx.run_dir / "control" / "island_briefs.json")
    )
    # Cost axis, mounted only when the population declares one. The archive
    # reserves slots per bucket along this axis; without the ledger the model
    # never learns those cheaper elites exist, so the diversity the archive
    # preserves stays unreachable from the prompt.
    if ctx.population.archive_feature_metric:
        from evoharness.evoplus import ResourceLedgerContributor

        contributors.append(
            ResourceLedgerContributor(
                ctx.population.archive_feature_metric,
                store=store,
                cap=ctx.population.archive_feature_cap,
                quality_metric=ctx.population.archive_feature_quality,
                unit=ctx.population.archive_feature_unit,
            )
        )
    weight_policies = list(weight_policies or [])
    weight_policies.append(LineageVetoPolicy(book, store))
    ctx.extras["directive_book"] = book

    # Observed-capability ledger: always mounted, like the HITL channel --
    # observability, not an ablation variable. Appended LAST below so every
    # recipe observer (signature recorders above all) has already run when
    # it looks at the candidate.
    from evoharness.evoplus import CapabilityLedger

    capability_ledger = CapabilityLedger()
    ctx.extras["capability_ledger"] = capability_ledger
    model_router = StaticRouter(ctx.search.llm_models)
    # Upstream's rejection sampling was never actually mounted here: with no
    # gate, candidates carried no embeddings, novelty_rejections was pinned
    # at 0, and the rejected_novelty experience channel could never fire.
    # The default embedder is provider-free (see hashing_embedding).
    novelty_gate = (
        NoveltyGate(
            hashing_embedding,
            threshold=ctx.search.similarity_threshold,
            mode=ctx.search.novelty_mode,
        )
        if ctx.search.novelty_enabled
        else None
    )
    proposer, hybrid_agent_proposer = _build_proposer(ctx, model_router, store)
    assembly_fingerprint = {
        "contributors": [
            f"{item.__class__.__module__}.{item.__class__.__qualname__}"
            for item in contributors
        ],
        "observers": [
            f"{item.__class__.__module__}.{item.__class__.__qualname__}"
            for item in (observers or [])
        ],
        "weight_policies": [
            f"{item.__class__.__module__}.{item.__class__.__qualname__}"
            for item in weight_policies
        ],
        # Both change which candidates exist, not merely how they are
        # ranked, so a run that used them is not comparable to one that did
        # not. Naming them in the fingerprint is what makes the two runs
        # distinguishable after the fact.
        "merge_planner": (
            f"{merge_planner.__class__.__module__}."
            f"{merge_planner.__class__.__qualname__}"
            if merge_planner is not None
            else None
        ),
        "island_health": (
            f"{island_health.__class__.__module__}."
            f"{island_health.__class__.__qualname__}"
            if island_health is not None
            else None
        ),
        "inspiration_policy": (
            f"{inspiration_policy.__class__.__module__}."
            f"{inspiration_policy.__class__.__qualname__}"
            if inspiration_policy is not None
            else None
        ),
    }
    ctx.extras["assembly_fingerprint"] = assembly_fingerprint
    mounted_names = {
        item.__class__.__qualname__
        for item in [*contributors, *(observers or []), *weight_policies]
    }
    ctx.extras["capabilities_declared"] = _declared_capabilities(mounted_names)
    prompt_builder = PromptBuilder(
        ctx.search.task_sys_msg,
        language=ctx.search.language,
        contributors=contributors,
        workspace_agent=ctx.proposal.mode == "agentic",
    )
    proposal_selector = None
    if hybrid_agent_proposer is not None:
        agent_prompt_builder = PromptBuilder(
            ctx.search.task_sys_msg,
            language=ctx.search.language,
            contributors=contributors,
            workspace_agent=True,
        )
        proposal_selector = HybridProposalSelector(
            single_shot=ProposalLane(
                "single_shot",
                prompt_builder,
                proposer,
            ),
            agentic=ProposalLane(
                "agentic",
                agent_prompt_builder,
                hybrid_agent_proposer,
            ),
            agent_probability=ctx.proposal.hybrid_agent_probability,
            stagnation_generations=(
                ctx.proposal.hybrid_stagnation_generations
            ),
        )
    return SearchLoop(
        cfg=ctx.search,
        pop_cfg=ctx.population,
        store=store,
        grader=ctx.grader,
        llm=ctx.llm,
        prompt_builder=prompt_builder,
        parent_selector=make_parent_selector(ctx.population, weight_policies),
        inspiration_selector=InspirationSelector(
            ctx.population, policy=inspiration_policy
        ),
        model_router=model_router,
        novelty_gate=novelty_gate,
        observers=[*(observers or []), capability_ledger],
        budget=ctx.budget,
        workdir=ctx.run_dir,
        metric_log=MetricLog(ctx.run_dir / "metrics.jsonl"),
        proposer=proposer,
        proposal_selector=proposal_selector,
        checkpoint_configs=(
            ctx.plus,
            ctx.proposal,
            assembly_fingerprint,
            ctx.frozen_spec_hashes,
        ),
        operator_selector=operator_selector,
        merge_planner=merge_planner,
        island_health=island_health,
        preflight_pipeline=PreflightPipeline(ctx.preflight_validators),
    )
