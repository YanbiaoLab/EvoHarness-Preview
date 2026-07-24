"""Run-level manifest for the IMO proof experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    experiment_id: str
    protocol_fingerprint: str
    evolution_seed: int
    seed_sha256: str
    actual: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (
            self.run_id,
            self.experiment_id,
            self.protocol_fingerprint,
            self.seed_sha256,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("manifest string fields must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema")
        if isinstance(self.evolution_seed, bool) or not isinstance(
            self.evolution_seed,
            int,
        ):
            raise ValueError("evolution_seed must be an integer")

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("run manifest must be an object")
        return cls(**value)


__all__ = ["RunManifest"]
