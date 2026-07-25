# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/core/async_runner.py (proposal->evaluate->insert flow,
#           novelty retry loop, repair-on-failure flow)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Intentional deviation (the project's largest, recorded in porting notes):
# the loop is a SYNCHRONOUS sequential port. Upstream's three concurrent
# asyncio tasks, logical slot pools and adaptive pipelining serve large-scale
# throughput and are not needed at this project's budget (~150 evals/run).
"""SearchLoop: the main evolution loop orchestrating all evocore components."""

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
    LoopObserver,
    MutationContext,
    NullBudget,
    OperatorSelector,
    RejectionEvent,
)
from .llm import LLMClient
from .metrics import MetricLog
from .novelty import NoveltyGate, novelty_text
from .operators import PromptBuilder, sample_operator
from .population import Candidate, PopulationStore
from .proposer import (
    HybridProposalSelector,
    ProposalLane,
    Proposer,
    SingleShotProposer,
)
from .remote import EvalInfraError
from .routing import ModelRouter
from .selection import InspirationSelector, ParentSelector
from .workspace import Workspace

logger = logging.getLogger(__name__)


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

    # -- checkpoint / resume -----------------------------------------------------

    def _stateful_components(self) -> dict[str, object]:
        """Duck-typed: any router/selector/observer/contributor exposing
        state()/set_state() is checkpointed, keyed by class name + position."""
        objs = [
            self.model_router,
            self.parent_selector,
            *self.observers,
            *self.prompt_builder.contributors,
        ]
        out: dict[str, object] = {}
        counts: dict[str, int] = {}
        for obj in objs:
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
        self.store.seed_all_islands(seed)
        self._log_metrics(0, seed, report)
        # Heterogeneous island seeding: each extra seed lands on its own
        # island (alongside the primary copy — selection arbitrates). A failed
        # extra seed does not kill the run: every island still has the primary.
        # An extra seed may be a whole Workspace (multi-file genome) or a bare
        # main-file string; a string is lifted into the primary's workspace so
        # `workspace_kind` never disagrees with what `code` actually holds.
        for i, entry in enumerate(extra_seeds or []):
            variant_ws = self._as_seed_workspace(entry, initial_workspace)
            variant = Candidate(
                id=Candidate.new_id(),
                code=(
                    entry
                    if variant_ws is None
                    else variant_ws.serialize()
                ),
                workspace_kind=(
                    "file" if variant_ws is None else variant_ws.kind
                ),
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
                "passed": cand.report.passed,
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

    # -- proposal --------------------------------------------------------------

    def _pick_island(self, generation: int):
        """Round-robin over islands, skipping islands without any passed parent."""
        order = [
            (generation + i) % self.pop_cfg.num_islands
            for i in range(self.pop_cfg.num_islands)
        ]
        for idx in order:
            view = self.store.island_view(idx)
            if view.passed_candidates:
                return view
        return None

    def _propose(self, generation: int, report: RunReport) -> Candidate | None:
        for _attempt in range(self.cfg.max_novelty_attempts):
            island = self._pick_island(generation)
            if island is None:
                return None

            lane = self._proposal_lane(generation)

            failed = (
                self.store.latest_failed() if self.cfg.repair_enabled else None
            )
            if failed is not None and self.rng.random() >= self.cfg.repair_probability:
                failed = None    # throttled: fall through to a normal proposal
            if failed is not None:
                operator = "repair"
                parent = failed
                inspirations: tuple[list, list] = ([], [])
                self.store.mark_repair_attempted(failed.id)
                system, user = lane.prompt_builder.build_repair(parent)
            else:
                parent = self.parent_selector.sample(island, self.rng)
                if parent is None:
                    return None
                inspirations = self.inspiration_selector.sample(
                    parent, self.store, self.rng
                )
                has_insp = bool(inspirations[0] or inspirations[1])
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
                    archive_inspirations=inspirations[0],
                    top_k_inspirations=inspirations[1],
                    operator=operator,
                    generation=generation,
                )
                system, user = lane.prompt_builder.build(ctx)

            result = lane.proposer.propose(
                operator, parent, system, user
            )
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
                return None
            proposal = result.proposal
            if self.proposal_selector is not None:
                proposal.metadata.setdefault("proposal_mode", lane.name)

            embedding = None
            if self.novelty_gate is not None:
                # Judge the whole workspace: a mutation that only touches a
                # non-main file leaves main_text identical (see novelty_text).
                verdict = self.novelty_gate.check(
                    novelty_text(
                        proposal.workspace or parent.workspace, proposal.code
                    ),
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
                    continue  # re-sample a parent, upstream retry semantics
                embedding = verdict.embedding

            child_ws = proposal.workspace or parent.workspace.with_main_text(
                proposal.code
            )
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
                    c.id for c in inspirations[0] + inspirations[1]
                ],
                embedding=embedding,
                metadata=dict(proposal.metadata),
            )
        return None

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
    ) -> int:
        """Propose eval_batch_size candidates, grade them CONCURRENTLY, absorb
        sequentially. Threads touch ONLY grader.grade(); population, budget
        and router mutations stay on the main thread. Returns updated
        infra_streak (same drop semantics as the serial path)."""
        cands: list[Candidate] = []
        for _ in range(self.cfg.eval_batch_size):
            cand = self._propose(generation, report)
            if cand is not None:
                cands.append(cand)
        if not cands:
            report.history.append({"generation": generation, "status": "skipped"})
            return infra_streak

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
        best = self.store.best()
        if best is not None:
            report.best_id, report.best_fitness = best.id, best.fitness
        return infra_streak

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
        for generation in range(start_generation, self.cfg.num_generations + 1):
            if self.budget.should_stop():
                report.stopped_reason = "budget"
                break

            if self.cfg.eval_batch_size > 1:
                infra_streak = self._run_batch_generation(
                    generation, report, infra_streak
                )
                report.generations_completed = generation
                self._save_checkpoint(generation, report)
                if infra_streak >= self.cfg.max_consecutive_infra_failures:
                    report.stopped_reason = "eval_infra"
                    break
                continue

            cand = self._propose(generation, report)
            if cand is None:
                report.history.append(
                    {"generation": generation, "status": "skipped"}
                )
                report.generations_completed = generation
                self._save_checkpoint(generation, report)
                continue

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
