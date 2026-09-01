"""Turn a `LaunchConfig` into a running search, and run it.

Lifted verbatim out of the experiment driver so that a detached start and a
resume can reach it: both need exactly what the command line was providing,
and neither has a parser. The steps and their order are unchanged — in
particular the manifest is still written BEFORE the loop starts, because a run
that dies mid-flight has to leave behind what it was.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import functools
import os
from dataclasses import dataclass
from typing import Any

import recipes
from evoharness import (
    ComponentSpec,
    EvolutionSearchProfile,
    ProposalLimits,
    RunSpec,
    spec_hashes,
)
from evoharness.core import (
    LLMClient,
    make_openai_compat_transport,
    make_openai_responses_transport,
)
from evoharness.evaluation import evidence_manifest, make_evidence_grader
from evoharness.guard import BudgetMeter, finalize_manifest, start_manifest
from recipes.common import RecipeContext
from tasks import get_task

from .config import LaunchConfig, LaunchConfigError


DEFAULT_LLM_API_BASE = (
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


@dataclass
class BuiltRun:
    """A search assembled but not yet started.

    `ctx` travels with the loop because assembly stashes things in
    `ctx.extras` — the capability ledger, the declared capabilities, the
    proposal manifest, the assembly fingerprint — that only the finalizing
    step reads. Returning the loop alone would drop them silently and the
    manifest would come out short.
    """

    loop: Any
    ctx: RecipeContext
    recipe: Any
    task: Any
    budget: BudgetMeter | None
    run_spec: RunSpec
    search_profile: EvolutionSearchProfile
    frozen_hashes: dict
    brief_sha: str | None
    task_label: str
    search: Any
    population: Any
    plus: Any


def configure_logging() -> None:
    """Root stays at WARNING so urllib3 and friends do not flood; only our own
    tree is verbose."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("evoharness").setLevel(
        os.environ.get("EVOHARNESS_LOG_LEVEL", "INFO").upper()
    )


def build(cfg: LaunchConfig) -> BuiltRun:
    """Assemble the search described by `cfg`. Writes nothing."""

    recipe = recipes.get_recipe(cfg.recipe)
    search, population, plus, proposal = recipes.load_experiment_config(
        cfg.config_path, cfg.overrides
    )
    if cfg.task_dir is not None:
        from evoharness.authoring import load_task_from_dir

        task = load_task_from_dir(cfg.task_dir)
        task_label = f"dir:{cfg.task_dir}"
    else:
        task = get_task(cfg.task)
        task_label = cfg.task
    if task.task_sys_msg and not search.task_sys_msg:
        search.task_sys_msg = task.task_sys_msg
    # Same handoff as the line above: the task supplies it, an explicit
    # setting overrides it. A task that knows what "solved" means should not
    # need the caller to remember, and a caller who disagrees can still say so.
    if search.stop_at_fitness is None and task.spec.criterion.solved_at is not None:
        search.stop_at_fitness = task.spec.criterion.solved_at

    budget = None
    if cfg.budget_usd is not None:
        budget = BudgetMeter(
            cfg.budget_usd, state_path=cfg.run_dir / "budget.json"
        )

    brief = cfg.brief.read_text() if cfg.brief else task.research_brief
    brief_sha = hashlib.sha256(brief.encode()).hexdigest() if brief else None

    transport = task.transport
    if cfg.live:
        api_base = os.environ.get("EVOHARNESS_API_BASE", DEFAULT_LLM_API_BASE)
        api_key = os.environ.get("ALIYUN_MAAS_API_KEY") or os.environ.get(
            "EVOHARNESS_API_KEY"
        )
        if not api_key:
            # Raised rather than routed through `parser.error`: the assembly
            # has to run where there is no parser — a detached start, a
            # resume — and reaching back into argparse is the one line that
            # would keep it here.
            raise LaunchConfigError(
                "--live requires ALIYUN_MAAS_API_KEY "
                "(or legacy EVOHARNESS_API_KEY)"
            )
        # Reasoning models spend minutes before the first token, and this
        # task's prompt is large (task brief + research brief + a
        # multi-file genome). The 120s default timed out every proposal on
        # MiniMax-M3, which the proposer breaker correctly reads as a dead
        # proposer — a healthy model would be mistaken for a broken one.
        # The protocol is explicit, never autodetected: a gateway that serves
        # only /responses rejects /chat/completions with the same 403 it uses
        # for an unauthorized token, so a silent fallback would run the whole
        # search on a protocol nobody chose and report nothing.
        responses = os.environ.get("EVOHARNESS_LLM_PROTOCOL", "chat") == "responses"
        make_transport = (
            make_openai_responses_transport if responses
            else make_openai_compat_transport
        )
        # Streaming is Responses-only and opt-in. It exists for delivery
        # robustness: a proxy that cuts the body mid-flight ends the stream
        # without its terminal event, which retries cleanly, instead of
        # yielding a JSON prefix that may or may not parse.
        if responses and os.environ.get("EVOHARNESS_LLM_STREAM") == "1":
            make_transport = functools.partial(
                make_openai_responses_transport, stream=True
            )
        transport = make_transport(
            api_base,
            api_key,
            # Sized to measured latency, remeasured after the prompt grew.
            # MiniMax-M3 on this task's ~12k-token prompt, concurrency 4:
            # median 242s, max 283s, no failures. At 300s the median sat 19%
            # under the cliff and calls were being truncated mid-flight — in
            # the concurrency probe every "failure" landed on exactly 300s,
            # so they were this timeout, not the provider.
            #
            # 600s was 2.5x the median, and that was the wrong anchor: the
            # right one is the measured MAX of 283s. At 600 a hung call cost
            # ten minutes, and retries nest, so run modmul_r6 spent thirty
            # minutes on a single proposal while the rest of its generation
            # waited. 400s clears the observed max by 41% and caps a hang at
            # two thirds of what it did.
            timeout_s=float(os.environ.get("EVOHARNESS_LLM_TIMEOUT_S", 400)),
        )

    # Built before RunSpec, not after: when a dsh runtime drives the proposals
    # it IS the proposer backend, and describing the transport instead would
    # leave the deployment out of every hash that decides whether a resume is
    # the same experiment. The pair is validated in LaunchConfig, which every
    # caller goes through — including the ones with no command line.
    agent_backend = None
    if cfg.dsh_config is not None:
        if proposal.mode == "single_shot":
            # The single-shot lane answers in one message and never opens a
            # session, so it returns before it looks at the backend. Accepting
            # the pair here would put a dsh runtime in the run's identity for
            # a run that proposed entirely in-process — the manifest would name
            # a runtime no candidate ever ran inside.
            raise LaunchConfigError(
                "proposal.mode=single_shot cannot use an agent runtime; "
                "pass --set proposal.mode=agentic (or conversational/hybrid) "
                "or drop --dsh-config"
            )

        from evoharness.core.agent import DshAgentBackend, DshRuntimeSpec

        agent_backend = DshAgentBackend(
            DshRuntimeSpec(
                config_path=cfg.dsh_config.resolve(),
                runtime_argv=(
                    "node", "--import", "tsx/esm",
                    str(cfg.dsh_runtime.resolve()),
                ),
                provider=cfg.dsh_provider,
                runtime_cwd=cfg.dsh_runtime.resolve().parents[3],
                session_root=cfg.run_dir / "dsh_sessions",
                run_dir=cfg.run_dir,
                # The tool `candidate.cordis.yml` mounts so a candidate can
                # read the reference programs its prompt lists. Declared here
                # because Python cannot see the runtime's tool table; the
                # prompt names exactly this or nothing.
                peer_fetch_tool="evo_inspect_candidate",
                model=proposal.model or search.llm_models[0],
            )
        )

    if agent_backend is not None:
        backend_spec = ComponentSpec.for_object(
            "proposer_backend",
            agent_backend,
            version="dsh-v1",
            config=agent_backend.spec.fingerprint(),
        )
    else:
        backend_spec = ComponentSpec.for_object(
            "proposer_backend",
            transport if transport is not None else type(None),
            version="live-v1" if cfg.live else "task-default-v1",
        )
    run_spec = RunSpec(
        models=tuple(search.llm_models),
        proposer_backend=backend_spec,
        output_dir=str(cfg.run_dir),
        seed=search.seed,
        budget_usd=cfg.budget_usd,
        llm_temperature=search.llm_temperature,
        llm_max_tokens=search.llm_max_tokens,
        proposal_model=proposal.model,
        proposal_concurrency=search.proposal_concurrency,
        evaluation_concurrency=search.eval_batch_size,
        max_consecutive_infra_failures=search.max_consecutive_infra_failures,
        proposal_limits=ProposalLimits(
            max_turns=proposal.max_turns,
            max_tool_calls=proposal.max_tool_calls,
            timeout_s=proposal.timeout_s,
            max_cost_usd=proposal.max_cost_usd,
            max_input_tokens=proposal.max_input_tokens,
        ),
        metadata_json=json.dumps(
            {"live": bool(cfg.live)}, ensure_ascii=False
        ),
    )
    search_profile = EvolutionSearchProfile(
        num_generations=search.num_generations,
        operators=tuple(search.operators),
        operator_probs=tuple(search.operator_probs),
        proposal_mode=proposal.mode,
        num_islands=population.num_islands,
        archive_size=population.archive_size,
        migration_interval=population.migration_interval,
        migration_rate=population.migration_rate,
        parent_strategy=population.parent_strategy,
        novelty_enabled=search.novelty_enabled,
        repair_enabled=search.repair_enabled,
        repair_probability=search.repair_probability,
        options_json=json.dumps(
            {
                "effective_search": dataclasses.asdict(search),
                "effective_population": dataclasses.asdict(population),
                "effective_plus": dataclasses.asdict(plus),
                "effective_proposal": dataclasses.asdict(proposal),
                "recipe": recipe.NAME,
            },
            ensure_ascii=False,
        ),
    )
    frozen_hashes = spec_hashes(task.spec, run_spec, search_profile)

    # Only the run knows where lineage state can live, and only graders that
    # opted in have the attribute at all.
    if getattr(task.grader, "lineage_dir", "absent") is None:
        task.grader.lineage_dir = cfg.run_dir / "lineage"
    runtime_grader = make_evidence_grader(
        task.grader,
        task_spec=task.spec,
        run_dir=cfg.run_dir,
    )

    ctx = RecipeContext(
        search=search,
        population=population,
        plus=plus,
        grader=runtime_grader,
        llm=LLMClient(
            temperature=search.llm_temperature,
            max_tokens=search.llm_max_tokens,
            transport=transport,
        ),
        run_dir=cfg.run_dir,
        proposal=proposal,
        budget=budget,
        research_brief=brief or None,
        preflight_validators=tuple(task.preflight_validators),
        runner=task.runner,
        frozen_spec_hashes=frozen_hashes,
        agent_backend=agent_backend,
        # 任务自带的活工具:要绑定任务自己的 ArtifactStore 与脱敏白名单,
        # 只能由任务构造。
        extra_agent_tools=tuple(getattr(task, "agent_tools", ())),
    )
    loop = recipe.build(ctx)

    return BuiltRun(
        loop=loop,
        ctx=ctx,
        recipe=recipe,
        task=task,
        budget=budget,
        run_spec=run_spec,
        search_profile=search_profile,
        frozen_hashes=frozen_hashes,
        brief_sha=brief_sha,
        task_label=task_label,
        search=search,
        population=population,
        plus=plus,
    )


def execute(built: BuiltRun, cfg: LaunchConfig, argv: list[str] | None = None) -> dict:
    """Run an assembled search and return its summary."""

    recipe = built.recipe
    ctx = built.ctx
    loop = built.loop
    task = built.task
    budget = built.budget
    search = built.search
    population = built.population
    plus = built.plus
    run_spec = built.run_spec
    search_profile = built.search_profile
    frozen_hashes = built.frozen_hashes
    brief_sha = built.brief_sha
    task_label = built.task_label

    manifest_path = cfg.run_dir / "manifest.json"
    start_manifest(
        manifest_path,
        recipe=recipe.NAME,
        recipe_description=recipe.DESCRIPTION,
        task=task_label,
        argv=list(argv) if argv is not None else None,
        overrides=list(cfg.overrides),
        search=search,
        population=population,
        plus=plus,
        proposal=ctx.extras["proposal_manifest"],
        assembly=ctx.extras["assembly_fingerprint"],
        capabilities_declared=ctx.extras["capabilities_declared"],
        models=list(search.llm_models),
        live=bool(cfg.live),
        budget_cap_usd=cfg.budget_usd,
        research_brief_sha256=brief_sha,
        spec_hashes=frozen_hashes,
        task_spec={
            "task_id": task.spec.task_id,
            "version": task.spec.version,
            "criterion_hash": task.spec.criterion.hash,
            "measurement_hash": task.spec.measurement.hash,
        },
        evidence=evidence_manifest(task.spec),
        run_spec=run_spec.to_payload(),
        search_profile=search_profile.to_payload(),
    )

    report = loop.run(
        task.initial_code,
        extra_seeds=task.extra_seeds or None,
        initial_workspace=task.initial_workspace,
    )

    finalize_manifest(
        manifest_path,
        budget={
            "hard_cap_usd": cfg.budget_usd,
            "spent_usd": (
                budget.spent_usd
                if budget is not None
                else report.total_llm_cost + report.total_eval_cost
            ),
            "llm_cost_usd": report.total_llm_cost,
            "eval_cost_usd": report.total_eval_cost,
        },
        report=report,
        metric_summary=loop.metric_log.summary() if loop.metric_log else {},
        # Declared next to observed, with the silent channels named: the
        # explicit-downgrade record. A task that returns bare scalars under
        # a full-stack recipe is degraded, and now the manifest says so.
        capabilities_observed=ctx.extras["capability_ledger"].summary(
            declared=ctx.extras["capabilities_declared"]
        ),
    )

    return {
        "recipe": recipe.NAME,
        "stopped_reason": report.stopped_reason,
        "generations": report.generations_completed,
        "evaluations": report.evaluations,
        "best_fitness": report.best_fitness,
        "llm_cost": round(report.total_llm_cost, 4),
        "eval_cost": round(report.total_eval_cost, 4),
        "run_dir": str(cfg.run_dir),
    }
