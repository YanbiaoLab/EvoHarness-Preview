from dataclasses import dataclass, field, fields
import math
from pathlib import Path
from typing import Callable, Union

WIRE_SCHEMA_VERSION = 1


class InfraError(Exception):
    """Raise inside grade_fn when a DEPENDENCY failed (sandbox service down,
    network error) — no verdict on the candidate; the job becomes infra_error
    and the framework will retry. Candidate-caused failures must NOT use this:
    return Grade(passed=False) instead."""


@dataclass(frozen=True)
class GradeContext:
    """Read-only per-job context handed to grade_fn (protocol 'hints')."""
    candidate_id: str
    workdir: Path 
    operator: str | None = None
    generation: int | None = None


@dataclass
class Grade:
    """Full-control return type."""
    fitness: float
    passed: bool = True
    fault: str | None = None
    visible_metrics: dict = field(default_factory=dict)
    hidden_metrics: dict = field(default_factory=dict)
    notes: str = ""
    structured_feedback: dict | None = None   # None = 任务没提供(协议红线探测依据)
    artifacts_ref: str | None = None          # 指向 ArtifactStore 里的冷 trace(层②)
    stdout_log: str = ""
    stderr_log: str = ""
    stage_reached: int = 3
    execution_time: float = 0.0
    eval_cost_usd: float = 0.0


GradeValue = Union["Grade", dict, float, int]
GradeFn = Callable[[str, GradeContext], GradeValue]
_GRADE_FIELDS = {f.name for f in fields(Grade)}

def coerce_grade(raw: GradeValue) -> dict:
    """Normalize any legal grade_fn return into a full report dict"""
    if isinstance(raw, Grade):
        grade = raw
    elif isinstance(raw, bool):                    # bool is int; reject early
        raise TypeError("grade_fn returned bool; return float or dict")
    elif isinstance(raw, (int, float)):
        grade = Grade(fitness=float(raw))
    elif isinstance(raw, dict):
        unknown = set(raw) - _GRADE_FIELDS

        if unknown:
            raise ValueError(f"Unknown fields in grade dict: {unknown}")
        if "fitness" not in raw:
            raise ValueError(f"Missing required field 'fitness' in grade dict: {raw}")
        grade = Grade(**raw)

    else:
        raise TypeError(f"Invalid grade value: {type(raw).__name__}")
    
    if not math.isfinite(grade.fitness):
        raise ValueError(f"fitness must be finite, got {grade.fitness!r}")

    return {
        "schema_version": WIRE_SCHEMA_VERSION,
        "fitness": float(grade.fitness),
        "passed": bool(grade.passed),
        "fault": grade.fault,
        "visible_metrics": dict(grade.visible_metrics),
        "hidden_metrics": dict(grade.hidden_metrics),
        "notes": grade.notes,
        "structured_feedback": grade.structured_feedback,
        "artifacts_ref": grade.artifacts_ref,
        "stdout_log": grade.stdout_log,
        "stderr_log": grade.stderr_log,
        "stage_reached": int(grade.stage_reached),
        "execution_time": float(grade.execution_time),
        "eval_cost_usd": float(grade.eval_cost_usd),
    }