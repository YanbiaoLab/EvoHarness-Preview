"""RunSpec: frozen runtime resources and reproducibility controls."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .component import ComponentSpec
from .fingerprint import canonical_object_json, spec_hash


@dataclass(frozen=True)
class ProposalLimits:
    max_turns: int = 48
    max_tool_calls: int = 120
    timeout_s: float = 5400.0
    max_cost_usd: float | None = None
    max_input_tokens: int = 131_072

    def __post_init__(self) -> None:
        for name, minimum in (
            ("max_turns", 1),
            ("max_tool_calls", 0),
            ("max_input_tokens", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValueError("max_cost_usd must be non-negative and finite")


@dataclass(frozen=True)
class RunSpec:
    models: tuple[str, ...]
    proposer_backend: ComponentSpec
    output_dir: str
    seed: int = 0
    budget_usd: float | None = None
    llm_temperature: float = 0.75
    llm_max_tokens: int = 4096
    proposal_model: str | None = None
    proposal_concurrency: int = 1
    evaluation_concurrency: int = 1
    max_consecutive_infra_failures: int = 5
    proposal_limits: ProposalLimits = field(default_factory=ProposalLimits)
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.models or any(
            not isinstance(model, str) or not model.strip()
            for model in self.models
        ):
            raise ValueError("models must contain non-empty names")
        if len(self.models) != len(set(self.models)):
            raise ValueError("models must be unique")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        self.proposer_backend.require_role("proposer_backend")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.budget_usd is not None and (
            isinstance(self.budget_usd, bool)
            or not isinstance(self.budget_usd, (int, float))
            or not math.isfinite(self.budget_usd)
            or self.budget_usd < 0
        ):
            raise ValueError("budget_usd must be non-negative and finite")
        if not math.isfinite(self.llm_temperature) or self.llm_temperature < 0:
            raise ValueError("llm_temperature must be non-negative and finite")
        for name, minimum in (
            ("llm_max_tokens", 1),
            ("proposal_concurrency", 1),
            ("evaluation_concurrency", 1),
            ("max_consecutive_infra_failures", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json must be valid JSON") from exc
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode a JSON object")
        object.__setattr__(self, "metadata_json", canonical_object_json(metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "models": list(self.models),
            "proposer_backend": self.proposer_backend.to_payload(),
            "output_dir": self.output_dir,
            "seed": self.seed,
            "budget_usd": self.budget_usd,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "proposal_model": self.proposal_model,
            "proposal_concurrency": self.proposal_concurrency,
            "evaluation_concurrency": self.evaluation_concurrency,
            "max_consecutive_infra_failures": self.max_consecutive_infra_failures,
            "proposal_limits": {
                "max_turns": self.proposal_limits.max_turns,
                "max_tool_calls": self.proposal_limits.max_tool_calls,
                "timeout_s": self.proposal_limits.timeout_s,
                "max_cost_usd": self.proposal_limits.max_cost_usd,
                "max_input_tokens": self.proposal_limits.max_input_tokens,
            },
            "metadata": json.loads(self.metadata_json),
        }

    @property
    def hash(self) -> str:
        payload = self.to_payload()
        # 物理输出位置是部署细节,不是实验身份:同一冻结实验换目录重跑
        # 必须得到同一 run_hash,否则 I-3 的确定性编译会被路径打散。
        payload.pop("output_dir")
        return spec_hash({"kind": "run", **payload})
