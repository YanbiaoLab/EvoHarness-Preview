"""SearchProfile: independent Basic trajectories or Evolution."""

from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass
from typing import TypeAlias, Union

from .fingerprint import canonical_object_json, spec_hash


_PROPOSAL_MODES = {"single_shot", "conversational", "agentic", "hybrid"}


@dataclass(frozen=True)
class BasicSearchProfile:
    """Independent Agent trajectories, each starting from the seed."""

    num_trajectories: int = 1
    proposal_mode: str = "agentic"
    novelty_enabled: bool = False
    options_json: str = "{}"

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_trajectories, bool)
            or not isinstance(self.num_trajectories, int)
            or self.num_trajectories < 1
        ):
            raise ValueError("num_trajectories must be at least 1")
        if self.proposal_mode not in _PROPOSAL_MODES:
            raise ValueError(f"invalid proposal_mode: {self.proposal_mode}")
        object.__setattr__(
            self,
            "options_json",
            canonical_object_json(json.loads(self.options_json)),
        )

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["options"] = json.loads(payload.pop("options_json"))
        return {"kind": "basic", **payload}

    @property
    def hash(self) -> str:
        return spec_hash(self.to_payload())


@dataclass(frozen=True)
class EvolutionSearchProfile:
    num_generations: int = 150
    operators: tuple[str, ...] = ("revise", "rewrite", "recombine")
    operator_probs: tuple[float, ...] = (0.6, 0.3, 0.1)
    proposal_mode: str = "single_shot"
    num_islands: int = 2
    archive_size: int = 40
    migration_interval: int = 10
    migration_rate: float = 0.0
    parent_strategy: str = "weighted"
    novelty_enabled: bool = True
    repair_enabled: bool = True
    repair_probability: float = 1.0
    options_json: str = "{}"

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_generations, bool)
            or not isinstance(self.num_generations, int)
            or self.num_generations < 1
        ):
            raise ValueError("num_generations must be at least 1")
        if self.proposal_mode not in _PROPOSAL_MODES:
            raise ValueError(f"invalid proposal_mode: {self.proposal_mode}")
        if not self.operators or len(self.operators) != len(self.operator_probs):
            raise ValueError("operators and operator_probs must be aligned")
        if any(not item.strip() for item in self.operators):
            raise ValueError("operator names must be non-empty")
        if any(
            isinstance(prob, bool)
            or not isinstance(prob, (int, float))
            or not math.isfinite(prob)
            or prob < 0
            for prob in self.operator_probs
        ) or not math.isclose(sum(self.operator_probs), 1.0, abs_tol=1e-9):
            raise ValueError("operator_probs must be non-negative and sum to 1")
        for name in ("num_islands", "archive_size", "migration_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 <= self.migration_rate <= 1:
            raise ValueError("migration_rate must be between 0 and 1")
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("repair_probability must be between 0 and 1")
        object.__setattr__(
            self,
            "options_json",
            canonical_object_json(json.loads(self.options_json)),
        )

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["options"] = json.loads(payload.pop("options_json"))
        return {"kind": "evolution", **payload}

    @property
    def hash(self) -> str:
        return spec_hash(self.to_payload())


SearchProfile: TypeAlias = Union[BasicSearchProfile, EvolutionSearchProfile]
