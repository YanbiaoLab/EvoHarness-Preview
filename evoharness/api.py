"""Public execution entry point for the three-contract API."""

from __future__ import annotations

import json
from pathlib import Path

from evoharness.contracts import RunSpec, SearchProfile
from evoharness.core import (
    AgentSessionLimits,
    AgentSessionProposer,
    ConversationalAgentBackend,
    InspirationSelector,
    JsonlEventSinkFactory,
    LLMClient,
    MetricLog,
    NativeToolAgentBackend,
    NoveltyGate,
    PopulationStore,
    PreflightPipeline,
    ProposalPreflight,
    PromptBuilder,
    SearchLoop,
    StaticRouter,
    make_parent_selector,
)
from evoharness.core.agent import AgentToolRegistry, make_default_agent_tools
from evoharness.core.novelty import hashing_embedding
from evoharness.evaluation import evidence_manifest, make_evidence_grader
from evoharness.guard import BudgetMeter, Sandbox, finalize_manifest, start_manifest
from evoharness.runtime.compiler import compile_specs, spec_hashes
from evoharness.runtime.task import ResolvedTask


def _estimate_tokens(messages, tools) -> int:
    characters = 0
    for message in messages:
        characters += len(message.role) + len(message.content)
        for call in message.tool_calls:
            characters += len(call.call_id) + len(call.name)
            characters += len(json.dumps(call.arguments, sort_keys=True))
        for result in message.tool_results:
            characters += len(result.call_id) + len(result.content)
    for tool in tools:
        characters += len(tool.name) + len(tool.description)
        characters += len(json.dumps(tool.input_schema, sort_keys=True))
    return (characters + 3) // 4


class _KnowledgeContributor:
    """TaskSpec.knowledge 进了任务身份,就必须真实到达 prompt——
    身份声明与实际注入不一致,是最隐蔽的实验条件失效。"""

    def __init__(self, brief: str):
        self._brief = brief

    def contribute(self, ctx) -> str | None:
        return self._brief or None


def run(
    task: ResolvedTask,
    run_spec: RunSpec,
    search_profile: SearchProfile,
    *,
    transport=None,
    experiment_ref: dict | None = None,
):
    """Execute one frozen task/run/search triple on the existing Core."""

    search, population, proposal = compile_specs(
        task.spec, run_spec, search_profile
    )
    run_dir = Path(run_spec.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_transport = transport or task.default_transport
    if effective_transport is None:
        raise ValueError(
            "no proposer transport resolved; pass transport= or provide a "
            "task default transport"
        )
    llm = LLMClient(
        temperature=run_spec.llm_temperature,
        max_tokens=run_spec.llm_max_tokens,
        transport=effective_transport,
    )
    router = StaticRouter(list(run_spec.models))
    prompt_builder = PromptBuilder(
        task.spec.domain_prompt,
        contributors=(
            [_KnowledgeContributor(task.spec.research_brief)]
            if task.spec.knowledge else []
        ),
        workspace_agent=proposal.mode in {"agentic", "conversational"},
    )
    proposer = None
    if proposal.mode in {"agentic", "conversational"}:
        if proposal.model is not None:
            model = proposal.model
        elif len(run_spec.models) == 1:
            model = run_spec.models[0]
        else:
            raise ValueError(
                "agent proposal mode requires proposal_model when multiple "
                "models are configured"
            )
        runner = task.runner or Sandbox(allow_network=False)
        registry = AgentToolRegistry(make_default_agent_tools(runner))
        backend = NativeToolAgentBackend(
            client=llm,
            model=model,
            registry=registry,
            max_input_tokens=proposal.max_input_tokens,
            token_estimator=_estimate_tokens,
            max_parallel_tools=proposal.max_parallel_tools,
            recent_tool_results_to_keep=proposal.recent_tool_results_to_keep,
            compact_trigger_ratio=proposal.compact_trigger_ratio,
        )
        if proposal.mode == "conversational":
            backend = ConversationalAgentBackend(
                backend, language=search.language
            )
        proposer = AgentSessionProposer(
            backend=backend,
            preflight=ProposalPreflight(
                PreflightPipeline(task.preflight_validators)
            ),
            limits=AgentSessionLimits(
                max_turns=proposal.max_turns,
                max_tool_calls=(
                    0 if proposal.mode == "conversational"
                    else proposal.max_tool_calls
                ),
                timeout_s=proposal.timeout_s,
                max_cost_usd=proposal.max_cost_usd,
            ),
            event_sink_factory=JsonlEventSinkFactory(run_dir),
            max_repair_rounds=proposal.max_repair_rounds,
            work_root=run_dir / ".agent_work",
        )
    elif proposal.mode == "hybrid":
        raise NotImplementedError(
            "hybrid routing requires an explicit SearchProfile policy"
        )

    budget = (
        BudgetMeter(
            run_spec.budget_usd,
            state_path=run_dir / "budget.json",
        )
        if run_spec.budget_usd is not None else None
    )
    frozen_hashes = spec_hashes(task.spec, run_spec, search_profile)
    if getattr(task.grader, "lineage_dir", "absent") is None:
        task.grader.lineage_dir = run_dir / "lineage"
    # 在 TaskSpec 构建之后才包:身份是内层 grader 的,包装不进 hash。
    evidence_grader = make_evidence_grader(
        task.grader,
        task_spec=task.spec,
        run_dir=run_dir,
    )
    store = PopulationStore(population, run_dir / "run.db")
    loop = SearchLoop(
        cfg=search,
        pop_cfg=population,
        store=store,
        grader=evidence_grader,
        llm=llm,
        prompt_builder=prompt_builder,
        parent_selector=make_parent_selector(population),
        inspiration_selector=InspirationSelector(population),
        model_router=router,
        novelty_gate=(
            NoveltyGate(
                hashing_embedding,
                threshold=search.similarity_threshold,
                mode=search.novelty_mode,
            )
            if search.novelty_enabled else None
        ),
        budget=budget,
        workdir=run_dir,
        metric_log=MetricLog(run_dir / "metrics.jsonl"),
        proposer=proposer,
        checkpoint_configs=(frozen_hashes,),
        preflight_pipeline=PreflightPipeline(task.preflight_validators),
    )
    manifest_path = run_dir / "manifest.json"
    start_manifest(
        manifest_path,
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
        # 只在实验上下文里写该 section:普通 run 的 manifest 不该带
        # 一个恒为 null 的键,冻结语义下它还会阻止未来 resume 补写。
        **({"experiment_ref": experiment_ref} if experiment_ref else {}),
    )
    try:
        report = loop.run(
            task.initial_code,
            extra_seeds=list(task.extra_seeds) or None,
            initial_workspace=task.initial_workspace,
        )
        finalize_manifest(
            manifest_path,
            report=report,
            metric_summary=loop.metric_log.summary(),
        )
        return report
    finally:
        store.close()
