"""Thin CandidateEvaluation -> EvoHarness Grade adapter."""

from __future__ import annotations

from collections import Counter
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

# Above this share of unscored problems the fitness stops describing the
# program. 1/4 is deliberately loose: one or two transient faults are part
# of life, half the set being cut off is not.
_UNSCORED_FAULT_RATIO = 0.25


def _error_category(item) -> str:
    """Keep the REASON, not just the exception class.

    This used to be `failure.split(":", 1)[0]`, which collapsed every
    budget exhaustion, timeout and transport fault into one bucket named
    "CandidateExecutionError" — and this string is the ONLY description of
    a failure that reaches the mutation prompt or the reflector. Run
    e5s_r2: a candidate that raised max_revisions from 3 to 5 had 9 of 12
    problems cut off at the per-problem call limit, and the reflector,
    seeing only the class name, wrote it up as "over-editing valid
    solutions" and advised capping revision loops at 3. That invented
    advice was then injected into later prompts.
    """
    if item.failure:
        kind, _, detail = item.failure.partition(":")
        return "exec:" + (detail.strip() or kind.strip())[:60]
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

    # A problem carrying a `failure` produced no answer at all. Scoring it
    # zero is right — nothing was delivered — but counting it as an ordinary
    # wrong answer is not: 4 of 15 candidates in run e5s_r2 scored exactly
    # 0.25 this way and were indistinguishable from a program that genuinely
    # solved 3 of 12. Nothing anywhere recorded the difference.
    unscored = [item for item in evaluation.problems if item.failure]
    reasons = Counter(_error_category(item) for item in unscored)
    starved = len(unscored) > _UNSCORED_FAULT_RATIO * n

    return Grade(
        fitness=evaluation.points_percentage,
        # Not a verdict on the program: we failed to obtain one. Marking it
        # unpassed keeps it out of the archive and out of parent selection,
        # which is what an unmeasurable candidate deserves.
        passed=not starved,
        fault=(
            f"unscored: {len(unscored)}/{n} problems produced no answer "
            f"({reasons.most_common(1)[0][0]}); this fitness does not "
            "measure the program"
            if starved
            else None
        ),
        visible_metrics={
            "unscored_items": len(unscored),
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
