"""Thin CandidateEvaluation -> EvoHarness Grade adapter."""

from __future__ import annotations

from pathlib import Path

from evoharness.evocore.artifacts import ArtifactStore
from evoharness import WorkspaceGradeFn
from evoharness.evoserve import Grade, GradeContext, InfraError

from .evaluation.contract import (
    CandidateEvaluation,
    EvaluationBackend,
    EvaluationProtocolError,
    EvaluationUnavailable,
)

def _error_category(item) -> str:
    if item.failure:
        return "exec:" + item.failure.split(":", 1)[0].strip()
    return item.label      # incorrect | partial | almost | correct


def _to_trace_blob(evaluation: CandidateEvaluation) -> dict:
    """冷 trace:完整逐题产物(证明 + grader 批语)。批语脱敏留到出口。"""
    items = [
        {
            "item_id": it.problem_id,
            "passed": it.points == it.max_points,
            "label": it.label,
            "points": it.points,
            "max_points": it.max_points,
            "proof": it.proof,                       # 候选自产,安全
            "grader_critique": it.grader_critique,   # 冷层保留,出口再脱敏
            "failure": it.failure or "",
        }
        for it in evaluation.problems
    ]
    return {
        "summary": {
            "points_percentage": evaluation.points_percentage,
            "correct_percentage": evaluation.correct_percentage,
        },
        "items": items,
    }

def to_grade(evaluation: CandidateEvaluation) -> Grade:
    """Map the task-owned result into the framework grading contract."""

    if not evaluation.admitted:
        return Grade(
            fitness=0.0,
            passed=False,
            fault="admission-rejected",
            visible_metrics={"structural_doa": True},
            structured_feedback={
                "admission_issues": list(evaluation.admission_issues)
            },
        )


    total_usage = evaluation.solver_usage + evaluation.grader_usage

    items = [
        {
            "item_id": item.problem_id,
            "passed": item.points == item.max_points,
            "predicted": item.label,
            "expected": "correct",
            "error_category": _error_category(item),
        }
        for item in evaluation.problems
    ]

    n_correct = sum(1 for it in evaluation.problems if it.points == it.max_points)

    n = len(evaluation.problems)

    return Grade(
        fitness=evaluation.points_percentage,
        passed=True,
        visible_metrics={
            "points_percentage": evaluation.points_percentage,
            "correct_percentage": evaluation.correct_percentage,
            "solver_calls": evaluation.solver_usage.calls,
            "solver_tokens": (
                    evaluation.solver_usage.prompt_tokens
                    + evaluation.solver_usage.completion_tokens
            ),
            "grader_calls": evaluation.grader_usage.calls,
            "structural_doa": False,
        },
        structured_feedback={
            "schema_version": 1,
            "items": items,
            "summary": f"{n_correct}/{n} problems fully correct",
        },
        execution_time=sum(item.elapsed_s for item in evaluation.problems),
        eval_cost_usd=total_usage.cost_usd,
        n_units=n,
        sem=evaluation.points_sem,
    )


def make_grade_func(
    backend: EvaluationBackend,
    *,
    split: str = "train",
    artifact_store: ArtifactStore | None = None,
) -> WorkspaceGradeFn:
    """Bind an evaluation backend without importing its implementation."""

    def grade_func(candidate_dir: Path, ctx: GradeContext) -> Grade:
        try:
            evaluation = backend.evaluate_directory(
                candidate_id=ctx.candidate_id,
                candidate_root=candidate_dir,
                split=split,
                output_dir=ctx.workdir,
            )
        except EvaluationUnavailable as exc:
            raise InfraError(str(exc)) from exc
        except EvaluationProtocolError as exc:
            raise InfraError(f"evaluation protocol error: {exc}") from exc
        grade = to_grade(evaluation)

        if artifact_store is not None and evaluation.admitted:
            ref = artifact_store.put(ctx.candidate_id, _to_trace_blob(evaluation))
            grade.artifacts_ref = ref.encode()
        return grade

    return grade_func


__all__ = ["make_grade_func", "to_grade"]
