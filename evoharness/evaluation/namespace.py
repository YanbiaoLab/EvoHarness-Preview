"""Score identities: values are comparable only inside one namespace."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evoharness.contracts import TaskSpec


class NamespaceMismatch(TypeError):
    """Two scores live in different measurement realities."""


@dataclass(frozen=True)
class ScoreNamespace:
    criterion_hash: str
    measurement_hash: str
    evaluator_hash: str
    universe_hash: str

    def __post_init__(self) -> None:
        for spec_field in fields(self):
            value = getattr(self, spec_field.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{spec_field.name} must be non-empty")

    def require_comparable(self, other: "ScoreNamespace") -> None:
        if not isinstance(other, ScoreNamespace):
            raise TypeError("other must be ScoreNamespace")
        if self == other:
            return
        diffs = ", ".join(
            f"{field.name}: {getattr(self, field.name)!r} != "
            f"{getattr(other, field.name)!r}"
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        )
        raise NamespaceMismatch(
            "scores from different namespaces cannot be compared "
            f"({diffs})"
        )

    def to_json(self) -> dict[str, str]:
        return {
            "criterion_hash": self.criterion_hash,
            "measurement_hash": self.measurement_hash,
            "evaluator_hash": self.evaluator_hash,
            "universe_hash": self.universe_hash,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ScoreNamespace":
        if not isinstance(payload, dict):
            raise TypeError("score namespace must be a JSON object")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "score namespace fields mismatch: expected "
                f"{sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**payload)

    @classmethod
    def from_task(
        cls,
        task: "TaskSpec",
        *,
        evaluator_hash: str | None = None,
    ) -> "ScoreNamespace":
        return cls(
            criterion_hash=task.criterion.hash,
            measurement_hash=task.measurement.hash,
            evaluator_hash=evaluator_hash or task.grader.hash,
            universe_hash=task.measurement.universe_hash,
        )


__all__ = ["NamespaceMismatch", "ScoreNamespace"]
