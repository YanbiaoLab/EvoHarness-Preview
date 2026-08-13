import time
from pathlib import Path

from evoharness.api import run as api_run
from evoharness.contracts import RunSpec, SearchProfile
from evoharness.runtime.compiler import compile_specs, spec_hashes
from evoharness.runtime.task import ResolvedTask

from .models import ExperimentOutcome, ExperimentSpec
from .store import ResearchStore


class ExperimentRefMismatch(RuntimeError):
    """构建出的三元组与冻结 refs 不符——实验不允许在漂移的契约上执行。"""



def verify_refs(experiment: ExperimentSpec,
                task: ResolvedTask,
                run_spec: RunSpec,
                profile: SearchProfile
) -> None:
    actual = spec_hashes(task.spec, run_spec, profile)

    frozen = {
        "task_hash": experiment.task_ref,
        "run_hash": experiment.run_ref,
        "search_hash": experiment.search_ref,
    }

    diffs = [
        f"{key}: frozen {frozen[key]} != built{actual[key]}"
        for key in frozen
        if frozen[key] != actual[key]
    ]

    if diffs:
        raise ExperimentRefMismatch(
            f"experiment {experiment.experiment_id}: " + "; ".join(diffs)
        )


def run_experiment(
    store: ResearchStore,
    experiment: ExperimentSpec,
    *,
    task: ResolvedTask,
    run_spec: RunSpec,
    profile: SearchProfile,
    transport=None,
    now=time.time,
) -> ExperimentOutcome:
    # 冻结先于执行:已存在且内容一致则幂等,内容不同在 store 层报错。
    store.put_experiment(experiment)
    verify_refs(experiment, task, run_spec, profile)

    search, _population, _proposal = compile_specs(
        task.spec, run_spec, profile
    )
    report = api_run(
        task,
        run_spec,
        profile,
        transport=transport,
        experiment_ref={
            "experiment_id": experiment.experiment_id,
            "hypothesis_id": experiment.hypothesis_id,
            "goal_id": experiment.goal_id,
            "spec_hash": experiment.hash,
        },

    )

    run_dir = Path(run_spec.output_dir)
    evidence_path = run_dir / "evidence.jsonl"
    evidence_count = (
        len(evidence_path.read_text().splitlines())
        if evidence_path.exists() else 0
    )
    infra_drops = sum(
        1 for h in report.history if h.get("status") == "infra_error"
    )
    outcome = ExperimentOutcome(
        experiment_id=experiment.experiment_id,
        spec_hash=experiment.hash,
        run_id=run_dir.name,
        stopped_reason=report.stopped_reason,
        generations_planned=search.num_generations,
        generations_completed=report.generations_completed,
        evaluations=report.evaluations,
        infra_drops=infra_drops,
        evidence_count=evidence_count,
        best_fitness=report.best_fitness,
        eval_cost_usd=report.total_eval_cost,
        created_at=now(),
    )
    store.append_outcome(outcome)
    return outcome