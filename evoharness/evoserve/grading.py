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
    # Lineage. Some domains produce expensive per-candidate state that the
    # genome cannot carry — trained weights above all — and re-deriving it
    # from scratch every generation throws away the run's whole compute
    # budget. The framework supplies a persistent directory and the parent's
    # id; deciding what may be inherited, and when, is the domain's call,
    # because only the domain knows when two genomes are compatible.
    parent_id: str | None = None
    lineage_dir: Path | None = None
    # The parent, then ITS parent, and so on, nearest first. A candidate that
    # failed evaluation publishes nothing, so its children find no state under
    # its id and start from scratch however much the lineage had accumulated.
    # Run modmul_r9 lost 323,546 training steps in three generations that way:
    # one candidate faulted, its repair child cold-started, and the cold start
    # outscored the fully-trained seed because the seed was scoring 0 for an
    # unrelated reason. Domains that inherit should walk this until they find
    # something, rather than treating the immediate parent as the only source.
    ancestor_ids: tuple[str, ...] = ()
    # A state merge, planned by the framework and executed by the domain.
    # Each entry is {"id": candidate id, "weight": float}, weights summing to
    # one, the base first. The framework picks the ids -- it can see from the
    # per-item pass vectors which candidates fail DIFFERENT items -- but it
    # cannot know what the carried state is, so combining is the domain's job.
    # Empty for every ordinary candidate; a domain that carries no state, or
    # cannot blend it, ignores this and inherits as usual.
    #
    # Two hazards, both learned the expensive way. A merge child's genome is
    # byte-identical to the base's, so (a) any content-addressed cache keyed
    # on genome text will hand back the BASE's state unless the merge is
    # applied after inheritance, and (b) publishing merged state into that
    # cache poisons every future candidate sharing the recipe.
    state_donors: tuple[dict, ...] = ()


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
    # How precise this fitness is. Only the task can know: a temperature-0
    # evaluation has zero repeat variance but real item-sampling variance,
    # while a training run has both. Left at 0 the harness treats the score
    # as exact, which is the current behaviour.
    n_units: int = 0                          # scored items/cases, 0 = not reported
    sem: float = 0.0                          # standard error of fitness


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
        "n_units": int(grade.n_units),
        "sem": float(grade.sem),
    }