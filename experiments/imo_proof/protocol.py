"""Frozen protocol for the IMO proof experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _positive_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    sha256: str
    id_column: str = "Problem ID"
    problem_column: str = "Problem"
    expected_rows: int = 60

    def __post_init__(self) -> None:
        for name in ("path", "sha256", "id_column", "problem_column"):
            _nonempty(name, getattr(self, name))
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("dataset sha256 must be a lowercase hex digest")
        _positive_int("expected_rows", self.expected_rows)


@dataclass(frozen=True)
class SplitSpec:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (self.train, self.validation, self.test)
        if any(not group for group in groups):
            raise ValueError("train, validation, and test splits must be non-empty")
        if any(not isinstance(item, str) or not item.strip() for group in groups for item in group):
            raise ValueError("split IDs must be non-empty strings")
        all_ids = [item for group in groups for item in group]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("split IDs must be unique and disjoint")

    def ids(self, name: str) -> tuple[str, ...]:
        if name not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split {name!r}")
        return getattr(self, name)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    temperature: float = 0.0
    max_output_tokens: int = 4096
    enable_thinking: bool = False
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None

    def __post_init__(self) -> None:
        _nonempty("model name", self.name)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be nonnegative and finite")
        _positive_int("max_output_tokens", self.max_output_tokens)
        if not isinstance(self.enable_thinking, bool):
            raise TypeError("enable_thinking must be bool")
        prices = (self.input_cost_per_million, self.output_cost_per_million)
        if (prices[0] is None) != (prices[1] is None):
            raise ValueError("model input and output prices must be set together")
        for price in prices:
            if price is not None and (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price < 0
            ):
                raise ValueError("model prices must be nonnegative and finite")


@dataclass(frozen=True)
class GraderSpec:
    model: ModelSpec
    prompt_path: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        _nonempty("grader prompt_path", self.prompt_path)
        if len(self.prompt_sha256) != 64:
            raise ValueError("grader prompt_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class SolverBudget:
    max_calls_per_problem: int = 8
    max_prompt_tokens_per_problem: int = 32_768
    max_completion_tokens_per_problem: int = 16_384
    timeout_s_per_problem: float = 300.0
    # Ceiling for ONE request, distinct from the per-problem budget above.
    # Handing the whole problem budget to a socket makes a half-dead
    # connection (ESTABLISHED, no bytes) hang for the full budget, once per
    # retry: a live run stalled 2h04m on a 3000s budget before anyone
    # noticed, because nothing distinguishes "this call is stuck" from
    # "this problem may legitimately take a while".
    request_timeout_s: float = 180.0
    max_cost_usd_per_problem: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_calls_per_problem",
            "max_prompt_tokens_per_problem",
            "max_completion_tokens_per_problem",
        ):
            _positive_int(name, getattr(self, name))
        _positive_number("timeout_s_per_problem", self.timeout_s_per_problem)
        _positive_number("request_timeout_s", self.request_timeout_s)
        if self.max_cost_usd_per_problem is not None:
            _positive_number("max_cost_usd_per_problem", self.max_cost_usd_per_problem)


@dataclass(frozen=True)
class OptimizerBudget:
    max_candidates: int = 5
    max_turns_per_candidate: int = 12
    max_tool_calls_per_candidate: int = 40
    timeout_s_per_candidate: float = 300.0
    max_total_cost_usd: float | None = None

    def __post_init__(self) -> None:
        _positive_int("max_candidates", self.max_candidates)
        _positive_int("max_turns_per_candidate", self.max_turns_per_candidate)
        _positive_int("max_tool_calls_per_candidate", self.max_tool_calls_per_candidate, minimum=0)
        _positive_number("timeout_s_per_candidate", self.timeout_s_per_candidate)
        if self.max_total_cost_usd is not None:
            _positive_number("max_total_cost_usd", self.max_total_cost_usd)


@dataclass(frozen=True)
class CandidateSpec:
    entrypoint: str
    mutable_files: tuple[str, ...]
    max_file_bytes: int = 262_144

    def __post_init__(self) -> None:
        _nonempty("candidate entrypoint", self.entrypoint)
        if ":" not in self.entrypoint:
            raise ValueError("candidate entrypoint must be module:function")
        if not self.mutable_files or len(self.mutable_files) != len(set(self.mutable_files)):
            raise ValueError("mutable_files must be non-empty and unique")
        for value in self.mutable_files:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or value != path.as_posix():
                raise ValueError(f"unsafe mutable file path: {value!r}")
        _positive_int("max_file_bytes", self.max_file_bytes)


@dataclass(frozen=True)
class ScoringSpec:
    label_points: tuple[tuple[str, int], ...] = (
        ("incorrect", 0),
        ("partial", 1),
        ("almost", 6),
        ("correct", 7),
    )

    def __post_init__(self) -> None:
        names = [name for name, _ in self.label_points]
        if not names or len(names) != len(set(names)):
            raise ValueError("scoring labels must be non-empty and unique")
        if any(not name or isinstance(points, bool) or not isinstance(points, int) or points < 0 for name, points in self.label_points):
            raise ValueError("scoring entries must be nonnegative integer points")
        if max(points for _, points in self.label_points) <= 0:
            raise ValueError("scoring requires a positive maximum")

    @property
    def points(self) -> dict[str, int]:
        return dict(self.label_points)

    @property
    def max_points(self) -> int:
        return max(self.points.values())


@dataclass(frozen=True)
class BenchmarkSpec:
    schema_version: int
    benchmark_id: str
    dataset: DatasetSpec
    splits: SplitSpec
    solver: ModelSpec
    optimizer: ModelSpec
    grader: GraderSpec
    solver_budget: SolverBudget
    optimizer_budget: OptimizerBudget
    candidate: CandidateSpec
    scoring: ScoringSpec

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported benchmark schema {self.schema_version}")
        _nonempty("benchmark_id", self.benchmark_id)
        total = len(self.splits.train) + len(self.splits.validation) + len(self.splits.test)
        if total != self.dataset.expected_rows:
            raise ValueError("split size must equal dataset expected_rows")
        if (
            self.solver_budget.max_cost_usd_per_problem is not None
            and self.solver.input_cost_per_million is None
        ):
            raise ValueError("solver cost cap requires frozen solver pricing")
        if self.optimizer_budget.max_total_cost_usd is not None:
            models = (self.optimizer, self.solver, self.grader.model)
            if any(model.input_cost_per_million is None for model in models):
                raise ValueError("total cost cap requires frozen pricing for all models")

    def to_dict(self) -> dict[str, Any]:
        value = _json_ready(asdict(self))
        value["scoring"]["label_points"] = {
            name: points for name, points in self.scoring.label_points
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkSpec":
        try:
            scoring_value = dict(value["scoring"])
            raw_points = scoring_value["label_points"]
            if not isinstance(raw_points, Mapping):
                raise TypeError("label_points must be an object")
            scoring = ScoringSpec(tuple((str(k), int(v)) for k, v in raw_points.items()))
            split_value = value["splits"]
            return cls(
                schema_version=int(value["schema_version"]),
                benchmark_id=str(value["benchmark_id"]),
                dataset=DatasetSpec(**value["dataset"]),
                splits=SplitSpec(
                    train=tuple(split_value["train"]),
                    validation=tuple(split_value["validation"]),
                    test=tuple(split_value["test"]),
                ),
                solver=ModelSpec(**value["solver"]),
                optimizer=ModelSpec(**value["optimizer"]),
                grader=GraderSpec(
                    model=ModelSpec(**value["grader"]["model"]),
                    prompt_path=value["grader"]["prompt_path"],
                    prompt_sha256=value["grader"]["prompt_sha256"],
                ),
                solver_budget=SolverBudget(**value["solver_budget"]),
                optimizer_budget=OptimizerBudget(**value["optimizer_budget"]),
                candidate=CandidateSpec(
                    entrypoint=value["candidate"]["entrypoint"],
                    mutable_files=tuple(value["candidate"]["mutable_files"]),
                    max_file_bytes=int(value["candidate"].get("max_file_bytes", 262_144)),
                ),
                scoring=scoring,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark spec: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> "BenchmarkSpec":
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("benchmark spec must be a JSON object")
        return cls.from_dict(value)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify_workspace(self, root: Path) -> None:
        root = Path(root).resolve()
        dataset_path = root / self.dataset.path
        grader_path = root / self.grader.prompt_path
        for path, expected, label in (
            (dataset_path, self.dataset.sha256, "dataset"),
            (grader_path, self.grader.prompt_sha256, "grader prompt"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path}")
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(f"{label} SHA-256 mismatch")

        with dataset_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != self.dataset.expected_rows:
            raise ValueError("dataset row count mismatch")
        ids = [row.get(self.dataset.id_column, "").strip() for row in rows]
        if any(not problem_id for problem_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("dataset IDs must be non-empty and unique")
        frozen = set(self.splits.train + self.splits.validation + self.splits.test)
        if frozen != set(ids):
            raise ValueError("frozen splits do not cover exactly the dataset IDs")


def default_spec_path() -> Path:
    return Path(__file__).with_name("benchmark.v1.json")


def load_default_spec() -> BenchmarkSpec:
    return BenchmarkSpec.load(default_spec_path())
