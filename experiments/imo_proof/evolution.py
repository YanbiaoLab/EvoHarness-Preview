"""EvoHarness evolution loop for the IMO proof experiment."""

from __future__ import annotations

import json
import tempfile
from time import monotonic
from pathlib import Path

from evoharness import ScorableTask
from evoharness.evocore import (
    Candidate,
    LLMClient,
    PopulationConfig,
    ProposalConfig,
    SearchConfig,
)
from evoharness.evocore.preflight import (
    PreflightContext,
    PreflightIssue,
    PreflightResult,
)
from evoharness.evoguard import BudgetMeter, Sandbox
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    LessonDirectiveContributor,
    MutationReflector,
    OperatorBandit,
    RegressionSoftPenalty,
    SignatureRecorder,
)
from evoharness.evoplus.config import PlusConfig
from recipes.common import RecipeContext, assemble

from .evaluation.contract import EvaluationBackend, EvaluationUnavailable
from .evaluation.engine import AdmissionGate
from .grade import make_grade_func
from .protocol import BenchmarkSpec
from .result import RunManifest, code_version
from .seed import seed_directory, seed_sha256
from evoharness.evocore.artifacts import FileArtifactStore
from evoharness.evocore.agent.tools import InspectParentEvalTool
from evoharness.evocore.sanitize import AllowlistSanitizer


TASK_SYSTEM_PROMPT = """Improve the IMO solver agent in this workspace.
The candidate is an agent program, not a proof for one particular problem.
Use the parent's aggregate evaluation feedback to improve generalization.
Call inspect_parent_eval (action='summary', then action='item') to see which
problems the parent failed and read the grader's critique before you edit.
Never access benchmark datasets, reference solutions, grading guidelines, or
network APIs directly. Keep solve(problem, llm) as the public entrypoint.
"""


# Egress policy for the parent-eval inspection tool. The reference solution is
# never stored in the trace, so this is defense-in-depth; grader_critique is
# allowed through to the optimizer (per project decision).
_IMO_TRACE_SANITIZER = AllowlistSanitizer(
    item_allow={
        "optimizer": frozenset({
            "item_id", "passed", "label", "points", "max_points",
            "proof", "failure", "grader_critique",
        }),
    },
    summary_allow={
        "optimizer": frozenset({"points_percentage", "correct_percentage"}),
    },
)


def _evaluate_candidate(
    backend: EvaluationBackend,
    candidate: Candidate,
    *,
    split: str,
    output_dir: Path,
    retries: int = 3,
    deadline_s: float = 3600.0,
):
    # Final validation/test evals aren't checkpointed; a single transient
    # provider timeout would otherwise discard a whole completed run. Retry
    # the eval so a flaky call doesn't cost us the run.
    #
    # The retry count alone is not a bound: each attempt starts a fresh
    # per-problem budget, so three retries of a 3000s budget let one stuck
    # connection hold the run for 2.5h — observed live at 2h04m, with the
    # process alive, silent, and indistinguishable from slow progress. Cap
    # the wall clock too.
    started = monotonic()
    last: EvaluationUnavailable | None = None
    for attempt in range(1, retries + 1):
        if attempt > 1 and monotonic() - started > deadline_s:
            print(
                f"[{split}] candidate {candidate.id} retry deadline "
                f"{deadline_s:.0f}s exhausted after {attempt - 1} attempts"
            )
            break
        with tempfile.TemporaryDirectory(prefix="imo_evo_candidate_") as temporary:
            candidate_root = candidate.workspace.materialize(Path(temporary))
            try:
                return backend.evaluate_directory(
                    candidate_id=candidate.id,
                    candidate_root=candidate_root,
                    split=split,
                    output_dir=output_dir,
                )
            except EvaluationUnavailable as exc:
                last = exc
                print(
                    f"[{split}] candidate {candidate.id} transient infra error "
                    f"(attempt {attempt}/{retries}): {exc}"
                )
    assert last is not None
    raise last


def _add_usage(total: dict[str, float | int], value: dict[str, object]) -> None:
    for name in ("calls", "prompt_tokens", "completion_tokens"):
        total[name] += int(value.get(name, 0))
    total["cost_usd"] += float(value.get("cost_usd", 0.0))


def _native_usage(run_dir: Path) -> dict[str, object]:
    optimizer: dict[str, float | int] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "tool_calls": 0,
    }
    for path in run_dir.glob("agent_sessions/*/summary.json"):
        value = json.loads(path.read_text())
        optimizer["calls"] += int(value.get("turns", 0))
        optimizer["tool_calls"] += int(value.get("tool_calls", 0))
        optimizer["prompt_tokens"] += int(value.get("prompt_tokens", 0))
        optimizer["completion_tokens"] += int(value.get("completion_tokens", 0))
        optimizer["cost_usd"] += float(value.get("cost_usd", 0.0))

    solver: dict[str, float | int] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
    grader = dict(solver)
    for path in run_dir.glob("gen_*/evaluation.json"):
        value = json.loads(path.read_text())
        _add_usage(solver, value.get("solver_usage", {}))
        _add_usage(grader, value.get("grader_usage", {}))
    total = {
        "calls": int(optimizer["calls"]) + int(solver["calls"]) + int(grader["calls"]),
        "prompt_tokens": (
            int(optimizer["prompt_tokens"])
            + int(solver["prompt_tokens"])
            + int(grader["prompt_tokens"])
        ),
        "completion_tokens": (
            int(optimizer["completion_tokens"])
            + int(solver["completion_tokens"])
            + int(grader["completion_tokens"])
        ),
        "cost_usd": (
            float(optimizer["cost_usd"])
            + float(solver["cost_usd"])
            + float(grader["cost_usd"])
        ),
    }
    return {"optimizer": optimizer, "train_solver": solver, "train_grader": grader, "total": total}


def _evolution_metrics(candidates) -> dict[str, object]:
    ordered = sorted(candidates, key=lambda candidate: (candidate.generation, candidate.id))
    by_id = {candidate.id: candidate for candidate in ordered}
    best = float("-inf")
    curve = []
    improved = 0
    comparable = 0
    structural_doa = 0
    for candidate in ordered:
        best = max(best, candidate.fitness)
        curve.append(
            {
                "generation": candidate.generation,
                "candidate_id": candidate.id,
                "fitness": candidate.fitness,
                "best_fitness": best,
            }
        )
        if candidate.report and candidate.report.visible_metrics.get("structural_doa"):
            structural_doa += 1
        parent = by_id.get(candidate.parent_id)
        if parent is not None:
            comparable += 1
            improved += candidate.fitness > parent.fitness
    offspring = max(0, len(ordered) - 1)
    return {
        "candidate_count": len(ordered),
        "offspring_count": offspring,
        "structural_doa_count": structural_doa,
        "structural_doa_rate": structural_doa / offspring if offspring else 0.0,
        "parent_child_pairs": comparable,
        "parent_to_child_improvement_rate": improved / comparable if comparable else 0.0,
        "best_fitness_curve": curve,
    }


class IMOAdmissionValidator:
    name = "imo-agent-admission"

    def __init__(self, gate: AdmissionGate):
        self.gate = gate

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        report = self.gate.check(ctx.workdir)
        return PreflightResult(
            self.name,
            tuple(
                PreflightIssue(
                    validator=self.name,
                    code=issue.code,
                    message=issue.message,
                    repairable=True,
                    path=issue.path,
                    line=issue.line,
                )
                for issue in report.issues
            ),
        )


def make_task(spec: BenchmarkSpec,
              backend: EvaluationBackend,
              *,
              artifact_store=None) -> ScorableTask:
    module_name = spec.candidate.entrypoint.split(":", 1)[0]
    main_file = module_name.replace(".", "/") + ".py"
    validator = IMOAdmissionValidator(AdmissionGate(spec.candidate))
    return ScorableTask.from_directory(
        seed_directory(),
        make_grade_func(backend, artifact_store=artifact_store),
        main_file=main_file,
        include_files=spec.candidate.mutable_files,
        task_sys_msg=TASK_SYSTEM_PROMPT,
        preflight_validators=(validator,),
        runner=Sandbox(allow_network=False),
    )


def run_experiment(
    *,
    spec: BenchmarkSpec,
    evaluation_backend: EvaluationBackend,
    optimizer_client: LLMClient,
    run_dir: Path,
    evolution_seed: int,
    experience_mode: str = "lessons+scratchpad",
    lesson_directive: bool = False,
    operator_bandit: bool = False,
    reflect_batch_size: int = 4,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty: {run_dir}")
    if optimizer_client.temperature != spec.optimizer.temperature:
        raise ValueError("optimizer client temperature does not match BenchmarkSpec")
    if optimizer_client.max_tokens != spec.optimizer.max_output_tokens:
        raise ValueError("optimizer client max_tokens does not match BenchmarkSpec")
    artifact_store = FileArtifactStore(run_dir / "artifacts")
    task = make_task(spec, evaluation_backend, artifact_store=artifact_store)
    inspect_tool = InspectParentEvalTool(
        artifact_store, _IMO_TRACE_SANITIZER, audience="optimizer"
    )
    budget = None
    if spec.optimizer_budget.max_total_cost_usd is not None:
        budget = BudgetMeter(
            spec.optimizer_budget.max_total_cost_usd,
            state_path=run_dir / "budget.json",
        )
    search = SearchConfig(
        num_generations=max(0, spec.optimizer_budget.max_candidates - 1),
        task_sys_msg=task.task_sys_msg,
        language="python",
        llm_models=[spec.optimizer.name],
        llm_temperature=spec.optimizer.temperature,
        llm_max_tokens=spec.optimizer.max_output_tokens,
        seed=evolution_seed,
    )
    population = PopulationConfig(
        num_islands=2,
        # An archive that holds every candidate is not an archive: with
        # archive_size == max_candidates the elite filter kept regressions
        # too, so "archive inspiration" degenerated into "random ancestor".
        # Keep roughly the top third (floor 4 so small runs still have peers).
        archive_size=max(4, spec.optimizer_budget.max_candidates // 3),
    )
    proposal = ProposalConfig(
        mode="agentic",
        model=spec.optimizer.name,
        max_turns=spec.optimizer_budget.max_turns_per_candidate,
        max_tool_calls=spec.optimizer_budget.max_tool_calls_per_candidate,
        timeout_s=spec.optimizer_budget.timeout_s_per_candidate,
        max_cost_usd=None,
        max_repair_rounds=3,
    )
    ctx = RecipeContext(
        search=search,
        population=population,
        plus=PlusConfig(),
        grader=task.grader,
        llm=optimizer_client,
        run_dir=run_dir,
        proposal=proposal,
        budget=budget,
        preflight_validators=task.preflight_validators,
        runner=task.runner,
        extra_agent_tools=(inspect_tool,),
    )
    # Cross-generation experience memory (global across islands → gives
    # cross-island learning without migration). v2 three-layer stack
    # (todo/experience_buffer_design.md): L0 evidence buffer + L1 batched
    # LLM attribution + L2 bounded scratchpad. `experience_mode` selects the
    # experiment arm (design §5): E3r=retrieval, E4a=retrieval+rejected,
    # E5r=lessons, E5s=lessons+scratchpad. The reflector (and its LLM cost)
    # only exists on lessons arms so E3r/E4a stay v1-faithful baselines.
    experience_store = ExperienceStore(run_dir / "experience.jsonl")
    # C1/C2 behaviour tracking, always on: without a recorded signature the
    # archive's behaviour-duplicate filter is a no-op and two candidates
    # that score identically cannot be told apart from genuine duplicates.
    # SignatureRecorder MUST precede the policy that reads its output.
    behaviour_policy = BehavioralNoveltyPolicy(
        hamming_threshold=ctx.plus.hamming_threshold,
        duplicate_penalty=ctx.plus.duplicate_penalty,
    )
    # Soft validation gate (backlog item 3): discount a parent whose recent
    # children keep regressing, without removing it from the population.
    regression_penalty = RegressionSoftPenalty()
    observers: list[object] = [
        SignatureRecorder(),
        behaviour_policy,
        regression_penalty,
        experience_store,
    ]
    weight_policies: list[object] = [behaviour_policy, regression_penalty]
    reflector = None
    if experience_mode.startswith("lessons"):
        # batch_size=4 (not the module default 8): with max_candidates=15
        # per spec, an 8-batch would produce its first lessons only after
        # candidate ~9, leaving too few proposals to benefit.
        reflector = MutationReflector(
            experience_store,
            llm=optimizer_client,
            model=spec.optimizer.name,
            budget=budget,
            batch_size=reflect_batch_size,
        )
        # Order is load-bearing: the store must record the graded entry
        # before the reflector counts pending work.
        observers.append(reflector)
    experience_contributor = ExperienceContributor(
        experience_store,
        mode=experience_mode,
        llm=optimizer_client,
        model=spec.optimizer.name,
        reflector=reflector,
    )
    contributors: list[object] = [experience_contributor]
    if lesson_directive:
        # Backlog item 6: promote the parent's own lesson from passive
        # context to the mutation's explicit directive (probabilistic).
        contributors.insert(0, LessonDirectiveContributor(experience_store))
    selector = None
    if operator_bandit:
        # Backlog item 2: soft adaptive operator scheduling. Registered as
        # an observer too so it settles rewards and rides the checkpoint.
        selector = OperatorBandit(list(search.operators))
        observers.append(selector)
    loop = assemble(
        ctx,
        contributors=contributors,
        observers=observers,
        weight_policies=weight_policies,
        operator_selector=selector,
    )
    manifest = RunManifest(
        schema_version=1,
        run_id=run_dir.name,
        experiment_id=spec.benchmark_id,
        protocol_fingerprint=spec.fingerprint,
        evolution_seed=evolution_seed,
        seed_sha256=seed_sha256(spec.candidate),
        code_version=code_version(),
        actual={
            "proposal": ctx.extras["proposal_manifest"],
            "experience_mode": experience_mode,
            "lesson_directive": lesson_directive,
            "operator_bandit": operator_bandit,
            "reflect_batch_size": reflect_batch_size,
        },
    )
    manifest.write(run_dir / "experiment_manifest.json")
    report = loop.run(
        task.initial_code,
        initial_workspace=task.initial_workspace,
    )

    archive = sorted(
        (
            candidate
            for candidate in loop.store.all_candidates()
            if candidate.passed
        ),
        key=lambda candidate: candidate.fitness,
        reverse=True,
    )[:3]
    validation = []
    for candidate in archive:
        result = _evaluate_candidate(
            evaluation_backend,
            candidate,
            split="validation",
            output_dir=run_dir / "validation" / candidate.id,
        )
        validation.append((candidate, result))
    selected = max(
        validation,
        key=lambda item: (
            item[1].points_percentage,
            -item[1].solver_usage.prompt_tokens
            - item[1].solver_usage.completion_tokens,
            -item[0].generation,
        ),
    )[0]
    try:
        test_result = _evaluate_candidate(
            evaluation_backend,
            selected,
            split="test",
            output_dir=run_dir / "test",
        )
        test_status = "ok"
    except EvaluationUnavailable as exc:
        # Evolution + validation already succeeded; don't discard the run over
        # a transient final-eval failure. Persist the summary with test=None.
        test_result = None
        test_status = f"unavailable: {exc}"
        print(f"[test] giving up after retries, saving partial summary: {exc}")
    all_candidates = tuple(loop.store.all_candidates())
    summary = {
        "experiment": spec.benchmark_id,
        "protocol_fingerprint": spec.fingerprint,
        "selected_candidate_id": selected.id,
        "selected_generation": selected.generation,
        "train_fitness": selected.fitness,
        "test_status": test_status,
        "test_points_percentage": (
            test_result.points_percentage if test_result else None
        ),
        "test_correct_percentage": (
            test_result.correct_percentage if test_result else None
        ),
        "native_usage": _native_usage(run_dir),
        "evolution_metrics": _evolution_metrics(all_candidates),
        "run_report": report.__dict__,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
