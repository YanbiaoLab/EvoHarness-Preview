# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/core/async_runner.py (proposal->evaluate->insert flow,
#           novelty retry loop, repair-on-failure flow)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Intentional deviation (the project's largest, recorded in porting notes):
# the loop is a SYNCHRONOUS sequential port. Upstream's three concurrent
# asyncio tasks, logical slot pools and adaptive pipelining serve large-scale
# throughput and are not needed at this project's budget (~150 evals/run).
"""SearchLoop: the main evolution loop orchestrating all core components."""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_write_json,
    config_fingerprint,
    load_json,
)
from .config import PopulationConfig, SearchConfig
from .interfaces import (
    BudgetLike,
    Grader,
    IslandHealthPolicy,
    LoopObserver,
    MergePlanner,
    MutationContext,
    NullBudget,
    OperatorSelector,
    RejectionEvent,
)
from .llm import LLMBillingError, LLMClient
from .metrics import MetricLog
from .novelty import NoveltyGate, novelty_text
from .operators import PromptBuilder, sample_operator
from .population import Candidate, PopulationStore
from .preflight import (
    PreflightContext,
    PreflightIssue,
    PreflightPipeline,
    PreflightReport,
    PreflightResult,
    ProposalPreflight,
)
from .proposer import (
    HybridProposalSelector,
    Proposal,
    ProposalLane,
    Proposer,
    ProposeResult,
    SingleShotProposer,
)
from .remote import EvalInfraError
from .routing import ModelRouter
from .selection import InspirationDraw, InspirationSelector, ParentSelector
from .workspace import Workspace

logger = logging.getLogger(__name__)

# How far back a child records its ancestry. Deep enough to step over
# a run of failed generations, short enough not to bloat every row.
_ANCESTRY_DEPTH = 8


@dataclass(frozen=True)
class _ProposalPlan:
    """Everything chosen for one proposal before the LLM is called. Split out
    so the call can overlap with its siblings while the choices that consume
    self.rng stay strictly ordered."""

    lane: ProposalLane
    island: object
    parent: Candidate
    operator: str
    inspirations: InspirationDraw
    system: str
    user: str
    # Set only for a state merge, which needs no model call at all: the
    # genome is the base's, unchanged, and the mutation lives entirely in
    # which trained state the domain is told to inherit.
    merge: object | None = None


@dataclass
class RunReport:
    generations_completed: int = 0
    evaluations: int = 0
    proposals_failed: int = 0
    novelty_rejections: int = 0
    best_id: str | None = None
    best_fitness: float | None = None
    total_llm_cost: float = 0.0
    total_eval_cost: float = 0.0
    stopped_reason: str = "completed"
    history: list[dict] = field(default_factory=list)


class SearchLoop:
    def __init__(
        self,
        cfg: SearchConfig,
        pop_cfg: PopulationConfig,
        store: PopulationStore,
        grader: Grader,
        llm: LLMClient,
        prompt_builder: PromptBuilder,
        parent_selector: ParentSelector,
        inspiration_selector: InspirationSelector,
        model_router: ModelRouter,
        novelty_gate: NoveltyGate | None = None,
        observers: list[LoopObserver] | None = None,
        budget: BudgetLike | None = None,
        workdir: Path | None = None,
        metric_log: MetricLog | None = None,
        checkpoint_path: Path | None = None,
        proposer: Proposer | None = None,
        proposal_selector: HybridProposalSelector | None = None,
        checkpoint_configs: tuple[object, ...] = (),
        operator_selector: OperatorSelector | None = None,
        merge_planner: MergePlanner | None = None,
        island_health: IslandHealthPolicy | None = None,
        preflight_pipeline: PreflightPipeline | None = None,
    ):
        self.cfg = cfg
        self.pop_cfg = pop_cfg
        self.store = store
        self.grader = grader
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.parent_selector = parent_selector
        self.inspiration_selector = inspiration_selector
        self.model_router = model_router
        self.novelty_gate = novelty_gate
        self.observers = observers or []
        self.budget = budget or NullBudget()
        self.metric_log = metric_log
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="evoharness_"))
        self.checkpoint_path = checkpoint_path or self.workdir / "checkpoint.json"
        self.checkpoint_configs = tuple(checkpoint_configs)
        self.rng = np.random.default_rng(cfg.seed)
        self.proposer = proposer or SingleShotProposer(
            llm=llm,
            model_router=model_router,
            language=cfg.language,
            max_resamples=cfg.max_op_resamples,
        )
        self.proposal_selector = proposal_selector
        self.operator_selector = operator_selector
        self.merge_planner = merge_planner
        self.island_health = island_health
        # The loop is the final admission boundary. Agentic proposers may run
        # this same pipeline earlier to obtain repair feedback, but every
        # proposal mode is checked here before an expensive Grader call.
        self.preflight_pipeline = preflight_pipeline

    # -- checkpoint / resume -----------------------------------------------------

    def _stateful_components(self) -> dict[str, object]:
        """Duck-typed: any router/selector/observer/contributor exposing
        state()/set_state() is checkpointed, keyed by class name + position."""
        objs = [
            self.model_router,
            self.parent_selector,
            *self.observers,
            *self.prompt_builder.contributors,
            self.merge_planner,
            self.island_health,
        ]
        out: dict[str, object] = {}
        counts: dict[str, int] = {}
        # An island monitor is registered as an observer AND held directly,
        # so the same object arrives twice; keying it under two names would
        # restore it twice and shift every later index of its class.
        seen: set[int] = set()
        for obj in objs:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "state") and hasattr(obj, "set_state"):
                name = obj.__class__.__name__
                idx = counts.get(name, 0)
                counts[name] = idx + 1
                out[f"{name}:{idx}"] = obj
        return out

    def _save_checkpoint(self, generation: int, report: RunReport) -> None:
        atomic_write_json(
            self.checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_fingerprint": self._config_fingerprint(),
                "generation": generation,
                "run_report": dataclasses.asdict(report),
                "rng_state": self.rng.bit_generator.state,
                "components": {
                    key: obj.state()
                    for key, obj in self._stateful_components().items()
                },
            },
        )

    def _restore_checkpoint(self, ckpt: dict) -> tuple[RunReport, int]:
        if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {ckpt.get('schema_version')} not supported"
            )
        expected = self._config_fingerprint()
        if ckpt.get("config_fingerprint") != expected:
            raise ValueError(
                "checkpoint was written with a different configuration; "
                "resume with the original config or start a fresh run dir"
            )
        if self.store.count() == 0:
            raise ValueError(
                "checkpoint exists but the population store is empty — "
                "resume requires the file-backed run.db it was written with"
            )
        self.rng.bit_generator.state = ckpt["rng_state"]
        components = self._stateful_components()
        for key, state in ckpt.get("components", {}).items():
            if key in components:
                components[key].set_state(state)
        report = RunReport(**ckpt["run_report"])
        return report, int(ckpt["generation"]) + 1

    def _config_fingerprint(self) -> str:
        return config_fingerprint(
            self.cfg,
            self.pop_cfg,
            *self.checkpoint_configs,
        )

    # -- seeding ---------------------------------------------------------------

    @staticmethod
    def _as_seed_workspace(
        entry: str | Workspace,
        initial_workspace: Workspace | None,
    ) -> Workspace | None:
        """Normalize one extra seed to a Workspace (None = legacy raw string).

        Returning None for the single-file legacy path keeps old run.db rows
        byte-identical: `code` stays the raw program text."""
        if not isinstance(entry, str):
            return entry
        if initial_workspace is None or initial_workspace.kind == "file":
            return None
        return initial_workspace.with_main_text(entry)

    def _seed(
        self,
        initial_code: str,
        report: RunReport,
        extra_seeds: list[str | Workspace] | None = None,
        initial_workspace: Workspace | None = None,
    ) -> None:
        seed_code = (
            initial_code
            if initial_workspace is None
            else initial_workspace.serialize()
        )
        workspace_kind = (
            "file" if initial_workspace is None else initial_workspace.kind
        )
        seed = Candidate(
            id=Candidate.new_id(),
            code=seed_code,
            workspace_kind=workspace_kind,
            generation=0,
            parent_id=None,
            island_idx=0,
            operator="seed",
            change_title="initial program",
        )
        seed.report = self._grade(seed, report)
        for observer in self.observers:
            observer.on_candidate_graded(seed, self.store)
        self.store.insert(seed)
        self._log_metrics(0, seed, report)
        # Heterogeneous island seeding: each extra seed lands on its own
        # island. An extra seed may be a whole Workspace (multi-file genome)
        # or a bare main-file string; a string is lifted into the primary's
        # workspace so `workspace_kind` never disagrees with what `code`
        # actually holds.

        for i, entry in enumerate(extra_seeds or []):
            if isinstance(entry, Workspace):
                code = entry.serialize()
                workspace_kind = entry.kind
            else:
                variant_ws = self._as_seed_workspace(entry, initial_workspace)
                if variant_ws is None:
                    code, workspace_kind = entry, "file"
                else:
                    code = variant_ws.serialize()
                    workspace_kind = variant_ws.kind
            variant = Candidate(
                id=Candidate.new_id(),
                code=code,
                workspace_kind=workspace_kind,
                generation=0,
                parent_id=None,
                island_idx=(i + 1) % self.pop_cfg.num_islands,
                operator="seed",
                change_title=f"initial program variant {i + 1}",
            )
            variant.report = self._grade(variant, report)
            for observer in self.observers:
                observer.on_candidate_graded(variant, self.store)
            self.store.insert(variant)
            self._log_metrics(0, variant, report)
        # Backfill: an island with no usable parent cannot produce a proposal
        # at all, so _pick_island would skip it for the whole run. This covers
        # both "fewer extra seeds than islands" and "an extra seed failed to
        # grade" — the robustness the blanket copy was providing.
        barren = [
            idx
            for idx in range(self.pop_cfg.num_islands)
            # Never the seed's own island: its row is already inserted above,
            # and seed_all_islands reuses seed.id there, so asking for it back
            # is a primary-key collision. Reachable exactly when the primary
            # seed FAILED -- then island 0 has a row but no passed candidate,
            # so it looks barren. Run modmul_r9 died on this at generation 0
            # the first time the seed timed out.
            if idx != seed.island_idx
            and not self.store.island_view(idx).passed_candidates
        ]
        # Only a seed that PASSED is worth copying: a copy reuses the
        # original's report, so backfilling with a failed seed fills the
        # island with a row that can never be selected and calls it covered.
        if barren and seed.passed:
            self.store.seed_all_islands(seed, islands=barren)
        self.store.refresh_archive()

    def _grade(self, cand: Candidate, report: RunReport):
        gen_dir = self.workdir / f"gen_{cand.generation}_{cand.id}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        result = self.grader.grade(cand, gen_dir)
        self.budget.charge(result.eval_cost_usd)
        report.total_eval_cost += result.eval_cost_usd
        report.evaluations += 1
        return result

    def _log_metrics(
        self, generation: int, cand: Candidate, report: RunReport
    ) -> None:
        """Forward built-ins (sys/) and the candidate's evaluation metrics
        (eval/) to the metric log; free-form keys chart automatically in the
        display layer."""
        if self.metric_log is None or cand.report is None:
            return
        best = self.store.best()
        metrics: dict = {
            "sys": {
                "fitness": cand.report.fitness,
                "passed": cand.passed,
                "operator": cand.operator,
                "best_fitness": best.fitness if best else cand.report.fitness,
                "llm_cost_total": report.total_llm_cost,
                "eval_cost_total": report.total_eval_cost,
                "novelty_rejections_total": report.novelty_rejections,
            },
            "eval": {
                **cand.report.visible_metrics,
                **cand.report.hidden_metrics,
            },
        }
        self.metric_log.log(generation, metrics, candidate_id=cand.id)

    def _notify_rejected(self, event: RejectionEvent) -> None:
        """Duck-typed dispatch (same style as _stateful_components): only
        observers implementing RejectionObserver hear rejections."""
        for observer in self.observers:
            handler = getattr(observer, "on_proposal_rejected", None)
            if handler is not None:
                handler(event)

    def _run_admission_preflight(
        self,
        *,
        parent: Candidate,
        operator: str,
        child_workspace: Workspace,
        require_workspace_change: bool,
    ) -> tuple[Workspace | None, PreflightReport | None]:
        """Run the authoritative pre-evaluation gate on a child workspace.

        Ordinary proposals must be real workspace mutations, so they use
        ProposalPreflight's capture and no-change checks. A state merge has
        an intentionally unchanged genome; it still runs every domain
        validator, but skips only the workspace-change requirement.

        ``None`` means no pipeline was configured. This keeps direct Core
        construction backwards-compatible while all public/recipe assembly
        paths install the gate explicitly.
        """

        if self.preflight_pipeline is None:
            return child_workspace, None

        try:
            with tempfile.TemporaryDirectory(
                prefix="evoharness_preflight_"
            ) as temp_dir:
                workdir = child_workspace.materialize(Path(temp_dir))
                ctx = PreflightContext(
                    parent=parent,
                    operator=operator,
                    workdir=workdir,
                )
                if require_workspace_change:
                    checked = ProposalPreflight(
                        self.preflight_pipeline
                    ).check(ctx)
                    return checked.child_workspace, checked.report

                report = self.preflight_pipeline.run(ctx)
                return (child_workspace if report.ok else None), report
        except Exception as exc:
            # Workspace materialization is part of admission. It must fail
            # closed just like a validator exception, never crash the run or
            # get reclassified as an evaluation result.
            report = PreflightReport(
                results=(
                    PreflightResult(
                        stage="workspace",
                        issues=(
                            PreflightIssue(
                                validator="workspace",
                                code="materialize-error",
                                message=f"{type(exc).__name__}: {exc}",
                                repairable=False,
                            ),
                        ),
                    ),
                )
            )
            return None, report

    # -- proposal --------------------------------------------------------------

    def _pick_island(self, generation: int, slot: int = 0):
        """Round-robin over islands, skipping islands without any passed parent.

        Rotates per PROPOSAL, not per generation. Keyed on the generation
        alone, every proposal in a batch drew the same island: run modmul_r7
        put 17 of its 20 candidates on one island and took all 15 offspring
        from a single parent. Multiplying by the batch size keeps the
        per-generation offset the old behavior had while giving each slot
        its own island, and keeps rotation numbers from colliding across
        generations.

        Derived from (generation, slot) rather than kept in a counter so
        no new state has to survive a checkpoint for a resumed run to
        reproduce itself.
        """
        rotation = generation * max(1, self.cfg.eval_batch_size) + slot
        order = [
            (rotation + i) % self.pop_cfg.num_islands
            for i in range(self.pop_cfg.num_islands)
        ]
        for idx in order:
            view = self.store.island_view(idx)
            if view.passed_candidates:
                return view
        return None

    def _plan_proposal(
        self, generation: int, slot: int = 0
    ) -> "_ProposalPlan | None":
        """Pick island, parent, operator and build the prompts.

        Sequential by contract. Every draw here comes from self.rng, so
        running two of these at once would make a seeded run stop
        reproducing itself — which is why proposal concurrency overlaps only
        the network call, never this.
        """
        island = self._pick_island(generation, slot)
        if island is None:
            return None

        lane = self._proposal_lane(generation)

        # A state merge is offered first because it is the cheapest proposal
        # the loop can make -- no model call, no training -- and it competes
        # for the same evaluation slot as everything else. The planner does
        # its own probability roll, so this consumes one rng draw whether or
        # not it fires, keeping a seeded run's stream predictable.
        if self.merge_planner is not None:
            merge = self.merge_planner.plan(island, self.rng)
            if merge is not None:
                self.store.note_attempt(merge.base.id)
                logger.info(
                    "plan gen=%d slot=%d island=%d merge base=%s donors=%s",
                    generation, slot, island.island_idx,
                    merge.base.id[:8],
                    [d["id"][:8] for d in merge.state_donors()],
                )
                return _ProposalPlan(
                    lane=lane,
                    island=island,
                    parent=merge.base,
                    operator="merge",
                    inspirations=InspirationDraw(),
                    system="",
                    user="",
                    merge=merge,
                )

        failed = (
            self.store.latest_failed() if self.cfg.repair_enabled else None
        )
        if failed is not None and self.rng.random() >= self.cfg.repair_probability:
            failed = None    # throttled: fall through to a normal proposal
        if failed is not None:
            operator = "repair"
            parent = failed
            inspirations = InspirationDraw()
            self.store.mark_repair_attempted(failed.id)
            system, user = lane.prompt_builder.build_repair(parent)
        else:
            parent = self.parent_selector.sample(island, self.rng)
            if parent is None:
                return None
            # Charge the parent NOW, not when a child survives. The
            # 1/(1+children_count) factor is the only thing stopping the
            # selector grinding on one parent, and it cannot work if the
            # whole batch is planned before any of it is counted.
            self.store.note_attempt(parent.id)
            inspirations = self.inspiration_selector.sample(
                parent, self.store, self.rng
            )
            has_insp = bool(inspirations.archive or inspirations.top_k)
            if self.operator_selector is not None:
                operator = self.operator_selector.sample_operator(
                    has_insp, self.rng
                )
            else:
                operator = sample_operator(
                    self.cfg, has_inspirations=has_insp, rng=self.rng
                )
            ctx = MutationContext(
                parent=parent,
                archive_inspirations=inspirations.archive,
                top_k_inspirations=inspirations.top_k,
                inspiration_notes=inspirations.notes,
                operator=operator,
                generation=generation,
            )
            system, user = lane.prompt_builder.build(ctx)

        logger.info(
            "plan gen=%d slot=%d island=%d parent=%s children=%d op=%s",
            generation, slot, island.island_idx, parent.id[:8],
            parent.children_count, operator,
        )
        return _ProposalPlan(
            lane=lane,
            island=island,
            parent=parent,
            operator=operator,
            inspirations=inspirations,
            system=system,
            user=user,
        )

    def _absorb_proposal(
        self,
        generation: int,
        report: RunReport,
        plan: "_ProposalPlan",
        result,
    ) -> tuple[Candidate | None, bool]:
        """Turn one proposer result into a candidate.

        Returns (candidate, retry). `retry` is the novelty gate's resample
        signal — the caller should plan again. Everything here mutates
        budget, report, the store or observers, so it stays on the main
        thread even when the calls that produced `result` overlapped.
        """
        lane, island = plan.lane, plan.island
        parent, operator = plan.parent, plan.operator
        inspirations = plan.inspirations
        self.budget.charge(result.llm_cost)
        report.total_llm_cost += result.llm_cost
        if result.proposal is None:
            report.proposals_failed += 1
            report.history.append(
                {
                    "generation": generation,
                    "status": "proposal_failed",
                    "parent_id": parent.id,
                    "operator": operator,
                    **(
                        {"proposal_mode": lane.name}
                        if self.proposal_selector is not None
                        else {}
                    ),
                    "failure_reason": result.failure_reason,
                    "trace_path": result.trace_path,
                    "attempts": result.attempts,
                    "llm_cost": result.llm_cost,
                }
            )
            self._notify_rejected(
                RejectionEvent(
                    kind="proposal_failed",
                    generation=generation,
                    operator=operator,
                    parent=parent,
                    failure_reason=result.failure_reason or "",
                )
            )
            return None, False
        proposal = result.proposal
        if self.proposal_selector is not None:
            proposal.metadata.setdefault("proposal_mode", lane.name)

        # Build the child genome BEFORE judging it. A single-file proposer
        # returns code without a workspace, and the gate used to fall back to
        # the PARENT's workspace — which novelty_text renders in full,
        # discarding the proposed code entirely. Every single-file proposal
        # was therefore judged as a copy of its parent and rejected outright
        # in identity mode: a single-file domain could not produce offspring
        # at all with the gate on. Multi-file domains ship a child workspace
        # and never took that path, which is why the run this project spends
        # its GPU budget on never showed it.
        child_ws = proposal.workspace or parent.workspace.with_main_text(
            proposal.code
        )

        child_ws, preflight_report = self._run_admission_preflight(
            parent=parent,
            operator=operator,
            child_workspace=child_ws,
            require_workspace_change=plan.merge is None,
        )
        if child_ws is None:
            assert preflight_report is not None
            issues = preflight_report.issues
            first_issue = issues[0] if issues else None
            failure_reason = (
                "preflight-failed"
                if first_issue is None
                else (
                    f"{first_issue.validator}:{first_issue.code}: "
                    f"{first_issue.message}"
                )
            )
            report.proposals_failed += 1
            report.history.append(
                {
                    "generation": generation,
                    "status": "preflight_failed",
                    "parent_id": parent.id,
                    "operator": operator,
                    "failure_reason": failure_reason,
                    "failed_stage": preflight_report.failed_stage,
                    "issues": [dataclasses.asdict(issue) for issue in issues],
                }
            )
            self._notify_rejected(
                RejectionEvent(
                    kind="preflight",
                    generation=generation,
                    operator=operator,
                    parent=parent,
                    proposal_code=proposal.code,
                    proposal_workspace=proposal.workspace,
                    change_title=proposal.title,
                    failure_reason=failure_reason,
                )
            )
            # A fresh proposal may pass even when this one did not. Reuse the
            # loop's existing bounded re-proposal mechanism.
            return None, True
        if preflight_report is not None:
            proposal.metadata["admission_preflight"] = {
                "ok": True,
                "stages": [
                    result.stage for result in preflight_report.results
                ],
                "elapsed_s": preflight_report.elapsed_s,
            }

        embedding = None
        # A state merge is a duplicate BY CONSTRUCTION: its genome is the
        # base's, byte for byte, and the mutation lives in the inherited
        # state instead. Judging it on program text would reject every merge
        # the planner ever produced.
        if self.novelty_gate is not None and plan.merge is None:
            # Judge the whole workspace: a mutation that only touches a
            # non-main file leaves main_text identical (see novelty_text).
            verdict = self.novelty_gate.check(
                novelty_text(child_ws, proposal.code),
                island,
            )
            if not verdict.accepted:
                report.novelty_rejections += 1
                self._notify_rejected(
                    RejectionEvent(
                        kind="novelty",
                        generation=generation,
                        operator=operator,
                        parent=parent,
                        proposal_code=proposal.code,
                        proposal_workspace=proposal.workspace,
                        change_title=proposal.title,
                        max_similarity=verdict.max_similarity,
                        most_similar_id=verdict.most_similar_id,
                    )
                )
                # re-sample a parent, upstream retry semantics
                return None, True
            embedding = verdict.embedding

        # A seed copy is a database row, not an evaluated candidate: it reuses
        # the original's report and is never handed to the grader, so nothing
        # is ever published under its id. A domain that carries per-candidate
        # state keyed by id -- trained weights above all -- therefore finds
        # nothing for it, and every child of a copy starts from scratch.
        #
        # That is what happened to run modmul_r7: 25 of its 26 offspring
        # reported "parent-published-no-weights", because islands 1 and 2 both
        # drew from copies. The fitness column of that entire run measures
        # cold starts, not mutations.
        #
        # parent_id keeps pointing at the copy so island bookkeeping and the
        # lineage tree stay honest; only the identity used to look up
        # inherited state is redirected.
        metadata = dict(proposal.metadata)
        origin = parent.metadata.get("seed_copy_of")
        if origin:
            metadata["lineage_parent_id"] = origin
        # Carry the ancestry so a domain that inherits per-candidate state can
        # walk past a parent that has none. Self-propagating: each child
        # prepends its own parent to what the parent was carrying, so no store
        # lookup is needed and a resumed run reconstructs nothing. Bounded,
        # because this rides in every row.
        lineage_id = origin or parent.id
        ancestry = [lineage_id, *parent.metadata.get("lineage_ancestors", [])]
        metadata["lineage_ancestors"] = ancestry[:_ANCESTRY_DEPTH]
        return Candidate(
            id=Candidate.new_id(),
            code=child_ws.serialize(),
            workspace_kind=child_ws.kind,
            generation=generation,
            parent_id=parent.id,
            island_idx=parent.island_idx,
            operator=operator,
            change_title=proposal.title,
            change_summary=proposal.summary,
            model_name=proposal.model,
            inspiration_ids=[
                c.id for c in inspirations.all
            ],
            embedding=embedding,
            metadata=metadata,
        ), False

    @staticmethod
    def _execute_plan(plan: "_ProposalPlan") -> ProposeResult:
        """Turn a plan into a proposal, by model call or by merge.

        A merge is assembled here rather than by a Proposer because there is
        nothing to generate: the child's genome IS the base's, and the only
        new information is which state the domain should inherit. Charging it
        an LLM cost of zero is not a formality — it is the whole economic
        argument for the operator.
        """
        if plan.merge is None:
            return plan.lane.proposer.propose(
                plan.operator, plan.parent, plan.system, plan.user
            )
        merge = plan.merge
        base = merge.base
        return ProposeResult(
            proposal=Proposal(
                code=base.workspace.main_text,
                title=merge.title(),
                summary=merge.summary(),
                model="(state-merge)",
                workspace=base.workspace,
                metadata={
                    "state_donors": merge.state_donors(),
                    **(merge.audit() if hasattr(merge, "audit") else {}),
                },
            ),
            llm_cost=0.0,
            attempts=1,
        )

    def _propose(
        self, generation: int, report: RunReport, slot: int = 0
    ) -> Candidate | None:
        for _attempt in range(self.cfg.max_novelty_attempts):
            plan = self._plan_proposal(generation, slot)
            if plan is None:
                return None
            # An empty balance does not clear, so neither novelty resampling
            # nor the next generation can help. Let it out to the main loop,
            # which stops the run and names the cause.
            result = self._execute_plan(plan)
            cand, retry = self._absorb_proposal(
                generation, report, plan, result
            )
            if cand is not None:
                return cand
            if not retry:
                return None
        return None

    def _propose_many(
        self, generation: int, report: RunReport, count: int
    ) -> list[Candidate]:
        """Produce up to `count` candidates, overlapping the LLM calls.

        Proposals used to be generated strictly one at a time. Once
        eval_batch_size went to 16 that became the dominant cost of a
        generation — sixteen sequential calls at ~105s each, while grading
        all sixteen concurrently took minutes when the fast path applied.
        The bottleneck had simply moved.

        Planning stays sequential so rng draws keep their order, the calls
        overlap because they are pure network waits, and absorption runs in
        planning order so history and observers see the same sequence a
        serial run would produce.
        """
        workers = max(1, min(self.cfg.proposal_concurrency, count))
        if workers == 1:
            found = [
                self._propose(generation, report, slot)
                for slot in range(count)
            ]
            return [c for c in found if c is not None]

        plans: list[_ProposalPlan] = []
        for slot in range(count):
            plan = self._plan_proposal(generation, slot)
            if plan is None:
                break
            plans.append(plan)
        if not plans:
            return []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._execute_plan, plan) for plan in plans
            ]
            results = [f.result() for f in futures]

        cands: list[Candidate] = []
        for plan, result in zip(plans, results):
            cand, retry = self._absorb_proposal(
                generation, report, plan, result
            )
            if cand is not None:
                cands.append(cand)
            elif retry:
                # Novelty resample: one sequential retry rather than a whole
                # second parallel round, which would need re-planning anyway.
                cand = self._propose(generation, report)
                if cand is not None:
                    cands.append(cand)
        return cands

    def _proposal_lane(self, generation: int) -> ProposalLane:
        if self.proposal_selector is not None:
            return self.proposal_selector.select(
                generation,
                self.store,
                self.rng,
            )
        return ProposalLane(
            name="default",
            prompt_builder=self.prompt_builder,
            proposer=self.proposer,
        )

    # -- batched generation (WS-2: minute-scale remote evaluation) ---------------

    def _run_batch_generation(
        self, generation: int, report: RunReport, infra_streak: int
    ) -> tuple[int, bool]:
        """Propose eval_batch_size candidates, grade them CONCURRENTLY, absorb
        sequentially. Threads touch ONLY grader.grade(); population, budget
        and router mutations stay on the main thread. Returns the updated
        infra_streak (same drop semantics as the serial path) and whether the
        generation produced any candidate at all."""
        cands = self._propose_many(
            generation, report, self.cfg.eval_batch_size
        )
        if not cands:
            report.history.append({"generation": generation, "status": "skipped"})
            return infra_streak, False

        def _grade_only(cand: Candidate):
            gen_dir = self.workdir / f"gen_{cand.generation}_{cand.id}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            return self.grader.grade(cand, gen_dir)

        with ThreadPoolExecutor(max_workers=len(cands)) as pool:
            futures = [pool.submit(_grade_only, c) for c in cands]
            outcomes = []
            for cand, fut in zip(cands, futures):
                try:
                    outcomes.append((cand, fut.result(), None))
                except EvalInfraError as exc:
                    outcomes.append((cand, None, exc))

        for cand, result, exc in outcomes:
            if exc is not None:
                logger.warning("generation %d dropped: %s", generation, exc)
                infra_streak += 1
                report.history.append({
                    "generation": generation,
                    "status": "infra_error",
                    "candidate_id": cand.id,
                    "error": str(exc),
                })
                continue
            infra_streak = 0
            self.budget.charge(result.eval_cost_usd)
            report.total_eval_cost += result.eval_cost_usd
            report.evaluations += 1
            cand.report = result
            parent = self.store.get(cand.parent_id) if cand.parent_id else None
            for observer in self.observers:
                observer.on_candidate_graded(cand, self.store)
            self.store.insert(cand)
            self._log_metrics(generation, cand, report)
            reward = 0.0
            if parent is not None and parent.report is not None:
                reward = cand.fitness - parent.fitness
            self.model_router.settle(cand.model_name, reward, 0.0)
            report.history.append({
                "generation": generation,
                "status": "ok" if cand.passed else "failed",
                "candidate_id": cand.id,
                "operator": cand.operator,
                "fitness": cand.fitness,
                "parent_id": cand.parent_id,
            })
        self.store.refresh_archive()
        self.store.maybe_migrate(generation, self.rng)
        self._revive_islands(generation, report)
        best = self.store.best()
        if best is not None:
            report.best_id, report.best_fitness = best.id, best.fitness
        return infra_streak, True

    def _revive_islands(self, generation: int, report: RunReport) -> None:
        """Give an island that has gone quiet something new to breed from.

        Runs after archive refresh and migration, the only point where the
        whole generation's results are visible. Every restart is written to
        history: an injection changes where later candidates came from, and
        a run that cannot show when it happened cannot be read afterwards.
        """
        if self.island_health is None:
            return
        for event in self.island_health.maybe_restart(
            self.store, generation, self.pop_cfg.num_islands
        ):
            logger.info(
                "island %d revived at gen %d: best %.4f vs global %.4f after "
                "%d quiet generations; seeded %s from island %d",
                event.island_idx, generation, event.island_best,
                event.global_best, event.stalled_for,
                event.donor_id[:8], event.donor_island,
            )
            report.history.append(
                {
                    "generation": generation,
                    "status": "island_revived",
                    "island_idx": event.island_idx,
                    "donor_island": event.donor_island,
                    "donor_id": event.donor_id,
                    "injected_id": event.injected_id,
                    "island_best": event.island_best,
                    "global_best": event.global_best,
                    "stalled_for": event.stalled_for,
                }
            )

    # -- main loop ---------------------------------------------------------------

    def run(
        self,
        initial_code: str,
        extra_seeds: list[str | Workspace] | None = None,
        initial_workspace: Workspace | None = None,
    ) -> RunReport:
        report = RunReport()
        start_generation = 1
        ckpt = load_json(self.checkpoint_path)
        if ckpt is not None:
            report, start_generation = self._restore_checkpoint(ckpt)
            logger.info("resuming from generation %d", start_generation)
        elif self.store.count() > 0:
            raise RuntimeError(
                "population store is non-empty but no checkpoint was found; "
                "refusing to guess — start a fresh run dir or restore "
                "checkpoint.json"
            )
        if self.store.count() == 0:
            # Checkpoints are written every generation; without this, a
            # mid-run checkpoint carries the dataclass default "completed"
            # and monitoring UIs lie about liveness.
            report.stopped_reason = "running"
            self._seed(
                initial_code,
                report,
                extra_seeds,
                initial_workspace,
            )
            self._save_checkpoint(0, report)
        elif report.stopped_reason == "completed" and ckpt is not None \
                and start_generation <= self.cfg.num_generations:
            report.stopped_reason = "running"

        infra_streak = 0  # consecutive EvalInfraError drops (circuit breaker)
        # A dead PROPOSER used to have no breaker at all. When the LLM
        # endpoint is unreachable every generation records "skipped", the
        # loop runs to the end, and stopped_reason becomes "completed" —
        # observed on the L40S, where a run reported 2 completed generations
        # while producing zero offspring. On a multi-day run an expired
        # credential would drain the whole schedule and still look fine.
        propose_streak = 0
        for generation in range(start_generation, self.cfg.num_generations + 1):
            if self.budget.should_stop():
                report.stopped_reason = "budget"
                break

            if self.cfg.eval_batch_size > 1:
                try:
                    infra_streak, produced = self._run_batch_generation(
                        generation, report, infra_streak
                    )
                except LLMBillingError as exc:
                    # Not "the proposer is dead": that reading sends someone to
                    # debug the proposer while the account simply needs topping
                    # up. Stop on the first one -- five more generations of
                    # retrying an empty balance produce nothing.
                    logger.error("stopping: %s", exc)
                    report.stopped_reason = "llm_billing"
                    break
                propose_streak = 0 if produced else propose_streak + 1
                report.generations_completed = generation
                self._save_checkpoint(generation, report)
                if infra_streak >= self.cfg.max_consecutive_infra_failures:
                    report.stopped_reason = "eval_infra"
                    break
                if propose_streak >= self.cfg.max_consecutive_infra_failures:
                    report.stopped_reason = "proposer_dead"
                    break
                continue

            try:
                cand = self._propose(generation, report)
            except LLMBillingError as exc:
                logger.error("stopping: %s", exc)
                report.stopped_reason = "llm_billing"
                break
            if cand is None:
                report.history.append(
                    {"generation": generation, "status": "skipped"}
                )
                propose_streak += 1
                report.generations_completed = generation
                self._save_checkpoint(generation, report)
                if propose_streak >= self.cfg.max_consecutive_infra_failures:
                    # Same reasoning as the eval breaker: a run that cannot
                    # produce offspring must stop and say so, not drain the
                    # generation schedule and report success.
                    report.stopped_reason = "proposer_dead"
                    break
                continue
            propose_streak = 0

            parent = self.store.get(cand.parent_id) if cand.parent_id else None
            try:
                cand.report = self._grade(cand, report)
            except EvalInfraError as e:
                # Protocol §5: no evaluation signal — the candidate must not
                # enter the population (seed grading intentionally propagates:
                # a run cannot start without a graded seed).
                logger.warning("generation %d dropped: %s", generation, e)
                infra_streak += 1
                report.history.append(
                    {
                        "generation": generation,
                        "status": "infra_error",
                        "candidate_id": cand.id,
                        "error": str(e),
                    }
                )
                report.generations_completed = generation
                self._save_checkpoint(generation, report)
                if infra_streak >= self.cfg.max_consecutive_infra_failures:
                    # Every dropped candidate already cost an LLM proposal;
                    # a dead eval service must stop the run, not drain it.
                    report.stopped_reason = "eval_infra"
                    break
                continue
            infra_streak = 0
            for observer in self.observers:
                observer.on_candidate_graded(cand, self.store)
            self.store.insert(cand)
            self.store.refresh_archive()
            self.store.maybe_migrate(generation, self.rng)
            self._revive_islands(generation, report)
            self._log_metrics(generation, cand, report)

            reward = 0.0
            if parent is not None and parent.report is not None:
                reward = cand.fitness - parent.fitness
            self.model_router.settle(cand.model_name, reward, 0.0)

            best = self.store.best()
            if best is not None:
                report.best_id, report.best_fitness = best.id, best.fitness
            report.history.append(
                {
                    "generation": generation,
                    "status": "ok" if cand.passed else "failed",
                    "candidate_id": cand.id,
                    "operator": cand.operator,
                    "fitness": cand.fitness,
                    "parent_id": cand.parent_id,
                }
            )
            report.generations_completed = generation
            self._save_checkpoint(generation, report)

        if report.stopped_reason == "running":   # natural end of the loop
            report.stopped_reason = "completed"
        best = self.store.best()
        if best is not None:
            report.best_id = best.id
            report.best_fitness = best.fitness
        return report
