# EvoHarness original research extension (framework-evo backlog item 2:
# adaptive operator scheduling, ShinkaEvolve-inspired). Deliberate deviation
# from a plain UCB (routing.BanditRouter): operators use a probability-FLOOR
# softmax over decayed rewards, so a lagging operator is throttled but never
# starved — the platform-breaking "restructure" moves in the IMO runs came
# from operators that a greedy bandit would have suppressed.
"""Adaptive operator scheduling: soft bandit over mutation operators."""

from __future__ import annotations

import numpy as np

from evoharness.core.population import Candidate, PopulationStore


class OperatorBandit:
    """Implements OperatorSelector + LoopObserver.

    Reward is the child-parent fitness delta, folded into a per-operator
    exponential moving average by on_candidate_graded (register it as an
    observer). sample_operator draws from floor + softmax(ema/temperature):
    recently effective operators gain share, every operator keeps at least
    `floor` probability. State survives checkpoints (duck-typed)."""

    def __init__(
        self,
        operators: list[str],
        floor: float = 0.1,
        temperature: float = 0.1,
        decay: float = 0.8,
    ):
        if not operators:
            raise ValueError("OperatorBandit needs at least one operator")
        if not 0.0 <= floor * len(operators) < 1.0:
            raise ValueError("floor * len(operators) must be in [0, 1)")
        self.operators = list(operators)
        self.floor = floor
        self.temperature = temperature
        self.decay = decay
        self.ema = {op: 0.0 for op in self.operators}
        self.n = {op: 0 for op in self.operators}

    # -- OperatorSelector ------------------------------------------------------

    def probabilities(self, has_inspirations: bool) -> tuple[list[str], list[float]]:
        ops = [
            op for op in self.operators
            if has_inspirations or op != "recombine"
        ]
        scores = np.array([self.ema[op] for op in ops]) / self.temperature
        weights = np.exp(scores - scores.max())
        soft = weights / weights.sum()
        floor = min(self.floor, 1.0 / len(ops))
        probs = floor + (1.0 - floor * len(ops)) * soft
        probs = probs / probs.sum()
        return ops, [float(p) for p in probs]

    def sample_operator(self, has_inspirations: bool, rng) -> str:
        ops, probs = self.probabilities(has_inspirations)
        return str(ops[int(rng.choice(len(ops), p=np.array(probs)))])

    # -- LoopObserver ----------------------------------------------------------

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        if cand.operator not in self.ema:
            return
        if not cand.parent_id or cand.report is None:
            return
        parent = store.get(cand.parent_id)
        if parent is None or parent.report is None:
            return
        reward = cand.report.fitness - parent.report.fitness
        op = cand.operator
        self.ema[op] = self.decay * self.ema[op] + (1.0 - self.decay) * reward
        self.n[op] += 1

    # -- checkpoint ------------------------------------------------------------

    def state(self) -> dict:
        return {"ema": dict(self.ema), "n": dict(self.n)}

    def set_state(self, state: dict) -> None:
        """Aligned-by-name restore, tolerating operator list changes."""
        for field in ("ema", "n"):
            saved = state.get(field, {})
            target = getattr(self, field)
            for op in self.operators:
                if op in saved:
                    target[op] = saved[op]
