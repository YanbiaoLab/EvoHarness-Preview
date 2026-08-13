# EvoHarness original guardrail (no upstream counterpart; upstream tracks
# costs but has no hard cap / circuit breaker).
"""Global API budget metering with a hard cap and persistent state."""

from __future__ import annotations

import json
import time
from pathlib import Path


class BudgetExhausted(RuntimeError):
    """Raised only when strict=True; the default contract is should_stop()."""


class BudgetMeter:
    """Satisfies core.interfaces.BudgetLike.

    charge() accumulates; the SearchLoop polls should_stop() each generation
    and terminates gracefully. strict=True additionally raises on the charge
    that crosses the cap (for callers that must never overshoot).
    """

    def __init__(
        self,
        hard_cap_usd: float,
        state_path: Path | str | None = None,
        strict: bool = False,
    ):
        if hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be positive")
        self.hard_cap_usd = hard_cap_usd
        self.state_path = Path(state_path) if state_path else None
        self.strict = strict
        self.spent_usd = 0.0
        self.n_charges = 0
        if self.state_path and self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            self.spent_usd = float(state["spent_usd"])
            self.n_charges = int(state.get("n_charges", 0))

    def charge(self, usd: float) -> None:
        if usd < 0:
            raise ValueError("cannot charge a negative amount")
        self.spent_usd += usd
        self.n_charges += 1
        self._persist()
        if self.strict and self.spent_usd >= self.hard_cap_usd:
            raise BudgetExhausted(
                f"budget exhausted: ${self.spent_usd:.4f} >= "
                f"${self.hard_cap_usd:.4f}"
            )

    def remaining(self) -> float:
        return max(0.0, self.hard_cap_usd - self.spent_usd)

    def should_stop(self) -> bool:
        return self.spent_usd >= self.hard_cap_usd

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "spent_usd": self.spent_usd,
                    "hard_cap_usd": self.hard_cap_usd,
                    "n_charges": self.n_charges,
                    "updated_at": time.time(),
                }
            )
        )
