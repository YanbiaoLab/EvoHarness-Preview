"""Pure evaluation contracts shared by local, worker, and HTTP execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable


class EvaluationUnavailable(RuntimeError):
    """Evaluation infrastructure failed without judging the candidate."""


class EvaluationProtocolError(RuntimeError):
    """An evaluation backend violated the frozen experiment protocol."""


@dataclass(frozen=True)
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.calls, self.prompt_tokens, self.completion_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("usage counters must be nonnegative integers")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError("usage cost must be nonnegative and finite")

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        if not isinstance(other, ModelUsage):
            return NotImplemented
        return ModelUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True)
class ProofResult:
    proof: str
    status: str = "completed"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.proof, str) or not self.proof.strip():
            raise ValueError("proof must be non-empty")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be non-empty")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")


@runtime_checkable
class SolverLLM(Protocol):
    def complete(self, stage: str, prompt: str) -> str: ...


@runtime_checkable
class Solver(Protocol):
    def __call__(self, problem: str, llm: SolverLLM) -> object: ...


def normalize_proof_result(value: object) -> ProofResult:
    if isinstance(value, ProofResult):
        return value
    if isinstance(value, str):
        return ProofResult(value)
    if isinstance(value, Mapping):
        return ProofResult(
            proof=value.get("proof"),  # type: ignore[arg-type]
            status=value.get("status", "completed"),  # type: ignore[arg-type]
            metadata=value.get("metadata", {}),  # type: ignore[arg-type]
        )
    proof = getattr(value, "proof", None)
    if proof is not None:
        return ProofResult(
            proof=proof,
            status=getattr(value, "status", "completed"),
            metadata=getattr(value, "metadata", {}),
        )
    raise TypeError("solver must return proof text, a mapping, or ProofResult")


@dataclass(frozen=True)
class ProblemResult:
    problem_id: str
    label: str
    points: int
    max_points: int
    proof: str
    solver_usage: ModelUsage
    grader_usage: ModelUsage
    elapsed_s: float
    failure: str | None = None
    grader_critique: str = ""

    def __post_init__(self) -> None:
        if not self.problem_id or not self.label:
            raise ValueError("problem_id and label must be non-empty")
        if self.points < 0 or self.max_points <= 0 or self.points > self.max_points:
            raise ValueError("problem points are outside the scoring range")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be nonnegative and finite")
        if self.failure is not None and not self.failure.strip():
            raise ValueError("failure must be non-empty when present")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProblemResult":
        raw_solver_usage = value.get("solver_usage")
        raw_grader_usage = value.get("grader_usage")
        if not isinstance(raw_solver_usage, Mapping):
            raise ValueError("solver_usage must be an object")
        if not isinstance(raw_grader_usage, Mapping):
            raise ValueError("grader_usage must be an object")
        return cls(
            problem_id=str(value["problem_id"]),
            label=str(value["label"]),
            points=int(value["points"]),
            max_points=int(value["max_points"]),
            proof=str(value.get("proof", "")),
            solver_usage=ModelUsage(**raw_solver_usage),  # type: ignore[arg-type]
            grader_usage=ModelUsage(**raw_grader_usage),  # type: ignore[arg-type]
            elapsed_s=float(value["elapsed_s"]),
            failure=value.get("failure"),  # type: ignore[arg-type]
            grader_critique=value.get("grader_critique", ""),
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """The sole task-domain and wire result for one candidate evaluation."""

    candidate_id: str
    split: str
    admitted: bool
    problems: tuple[ProblemResult, ...] = ()
    admission_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("invalid evaluation split")
        if self.admitted == bool(self.admission_issues):
            raise ValueError("admission issues must be present exactly when rejected")
        ids = [result.problem_id for result in self.problems]
        if len(ids) != len(set(ids)):
            raise ValueError("problem results must have unique IDs")

    @property
    def points_percentage(self) -> float:
        maximum = sum(item.max_points for item in self.problems)
        return sum(item.points for item in self.problems) / maximum if maximum else 0.0

    @property
    def points_sem(self) -> float:
        """Standard error of `points_percentage`.

        The score is an average over problems, so its precision is bounded
        by how much the problems disagree — and at 12 problems that bound is
        loose (~0.14 when half the items are solved). Repeat evaluation
        cannot shrink this: all three models run at temperature 0, so the
        same program on the same problems returns the same answer. Only more
        problems can.

        Uses the per-problem point fraction, which is exactly the mean
        behind `points_percentage` when every problem has the same
        max_points (true for IMO's 7-point scale) and a close approximation
        otherwise.
        """
        n = len(self.problems)
        if n < 2:
            return 0.0
        scores = [item.points / item.max_points for item in self.problems]
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
        return math.sqrt(variance / n)

    @property
    def correct_percentage(self) -> float:
        if not self.problems:
            return 0.0
        return sum(item.points == item.max_points for item in self.problems) / len(self.problems)

    @property
    def solver_usage(self) -> ModelUsage:
        return sum((item.solver_usage for item in self.problems), ModelUsage())

    @property
    def grader_usage(self) -> ModelUsage:
        return sum((item.grader_usage for item in self.problems), ModelUsage())

    def to_dict(self, *, include_proofs: bool = True) -> dict[str, object]:
        value = asdict(self)
        if not include_proofs:
            for problem in value["problems"]:
                problem["proof"] = ""
        value.update(
            {
                "points_percentage": self.points_percentage,
                "correct_percentage": self.correct_percentage,
                "solver_usage": asdict(self.solver_usage),
                "grader_usage": asdict(self.grader_usage),
            }
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CandidateEvaluation":
        raw_problems = value.get("problems", ())
        if not isinstance(raw_problems, (list, tuple)):
            raise ValueError("evaluation problems must be a list")
        raw_issues = value.get("admission_issues", ())
        if not isinstance(raw_issues, (list, tuple)):
            raise ValueError("admission_issues must be a list")
        admitted = value.get("admitted")
        if not isinstance(admitted, bool):
            raise ValueError("admitted must be a boolean")
        return cls(
            candidate_id=str(value["candidate_id"]),
            split=str(value["split"]),
            admitted=admitted,
            problems=tuple(ProblemResult.from_dict(item) for item in raw_problems),
            admission_issues=tuple(str(item) for item in raw_issues),
        )

    def write(self, path: Path, *, include_proofs: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.to_dict(include_proofs=include_proofs),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> "CandidateEvaluation":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("candidate evaluation must be a JSON object")
        return cls.from_dict(value)


@runtime_checkable
class EvaluationBackend(Protocol):
    def evaluate_directory(
        self,
        *,
        candidate_id: str,
        candidate_root: Path,
        split: str,
        output_dir: Path | None = None,
    ) -> CandidateEvaluation: ...


class EvaluationRouter:
    """Route train/validation/test to separately secured backends."""

    def __init__(self, backends: Mapping[str, EvaluationBackend]):
        expected = {"train", "validation", "test"}
        if set(backends) != expected:
            raise ValueError(
                "evaluation router requires train, validation, and test backends"
            )
        self.backends = dict(backends)

    def evaluate_directory(
        self,
        *,
        candidate_id: str,
        candidate_root: Path,
        split: str,
        output_dir: Path | None = None,
    ) -> CandidateEvaluation:
        try:
            backend = self.backends[split]
        except KeyError as exc:
            raise ValueError(f"unknown evaluation split {split!r}") from exc
        return backend.evaluate_directory(
            candidate_id=candidate_id,
            candidate_root=candidate_root,
            split=split,
            output_dir=output_dir,
        )


__all__ = [
    "CandidateEvaluation",
    "EvaluationBackend",
    "EvaluationProtocolError",
    "EvaluationRouter",
    "EvaluationUnavailable",
    "ModelUsage",
    "ProblemResult",
    "ProofResult",
    "Solver",
    "SolverLLM",
    "normalize_proof_result",
]
