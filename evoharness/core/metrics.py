# EvoHarness original: wandb-style free-form metric logging. Anything logged
# here is auto-discovered by the display layer (evoviz/evoweb) and rendered
# as one panel per key, grouped by "namespace/" prefix — no hardcoded views.
"""Append-only metric log (JSON lines) with nested-dict flattening.

Conventions:
- step = generation.
- Keys use "namespace/name"; nested dicts are flattened with "/".
- Reserved namespaces: "sys/" (SearchLoop built-ins), "eval/" (auto-forwarded
  from EvalReport visible/hidden metrics). Anything else is user-defined —
  graders, observers and contributors may log freely.
- Values may be int/float/bool/str; the display layer charts numeric keys
  and tabulates the rest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def flatten(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a/b": 1} (wandb-style nested logging)."""
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{full}/"))
        else:
            flat[full] = value
    return flat


@dataclass
class MetricPoint:
    step: int
    key: str
    value: Any
    candidate_id: str | None = None
    ts: float = 0.0


class MetricLog:
    """Append-only log. path=None keeps points in memory only (tests)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.points: list[MetricPoint] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for key, value in rec["metrics"].items():
                    self.points.append(
                        MetricPoint(
                            step=rec["step"],
                            key=key,
                            value=value,
                            candidate_id=rec.get("candidate_id"),
                            ts=rec.get("ts", 0.0),
                        )
                    )

    def log(
        self,
        step: int,
        metrics: dict[str, Any],
        candidate_id: str | None = None,
    ) -> None:
        flat = flatten(metrics)
        if not flat:
            return
        ts = time.time()
        for key, value in flat.items():
            self.points.append(MetricPoint(step, key, value, candidate_id, ts))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "ts": ts,
                            "candidate_id": candidate_id,
                            "metrics": flat,
                        }
                    )
                    + "\n"
                )

    def keys(self) -> list[str]:
        return sorted({p.key for p in self.points})

    def series(self, key: str) -> list[tuple[int, Any]]:
        return [(p.step, p.value) for p in self.points if p.key == key]

    def summary(self) -> dict[str, Any]:
        """Last logged value per key (wandb-style run summary)."""
        out: dict[str, Any] = {}
        for p in self.points:
            out[p.key] = p.value
        return out
