# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/llm/prioritization.py (multi-arm interface, reward =
#           fitness improvement over the parent)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Intentional deviation (porting note): upstream's AsymmetricUCB (epsilon-
# greedy + baseline shifting + adaptive scaling + cost awareness) is NOT
# ported; BanditRouter below is a plain UCB1 that keeps the same interface,
# and StaticRouter is the project default. A faithful AsymmetricUCB port is
# an optional later work item.
"""Model routing: which LLM serves the next mutation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class ModelRouter(ABC):
    @abstractmethod
    def pick(self) -> str: ...

    @abstractmethod
    def settle(self, model: str, reward: float, cost: float) -> None: ...


class StaticRouter(ModelRouter):
    """Fixed sampling probabilities (uniform if not given)."""

    def __init__(
        self,
        models: list[str],
        probs: list[float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        if not models:
            raise ValueError("StaticRouter needs at least one model")
        self.models = list(models)
        if probs is None:
            probs = [1.0 / len(models)] * len(models)
        total = sum(probs)
        self.probs = [p / total for p in probs]
        self.rng = rng or np.random.default_rng()

    def pick(self) -> str:
        return self.models[int(self.rng.choice(len(self.models), p=self.probs))]

    def settle(self, model: str, reward: float, cost: float) -> None:
        pass


class BanditRouter(ModelRouter):
    """UCB1 over models; reward is the fitness improvement over the parent."""

    def __init__(
        self,
        models: list[str],
        exploration_coef: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        if not models:
            raise ValueError("BanditRouter needs at least one model")
        self.models = list(models)
        self.c = exploration_coef
        self.n = {m: 0 for m in models}
        self.total_reward = {m: 0.0 for m in models}
        self.total_cost = {m: 0.0 for m in models}
        self.rng = rng or np.random.default_rng()

    def pick(self) -> str:
        untried = [m for m in self.models if self.n[m] == 0]
        if untried:
            return str(self.rng.choice(untried))
        t = sum(self.n.values())
        scores = {
            m: self.total_reward[m] / self.n[m]
            + self.c * math.sqrt(2.0 * math.log(t) / self.n[m])
            for m in self.models
        }
        best = max(scores.values())
        winners = [m for m, s in scores.items() if s == best]
        return str(self.rng.choice(winners))

    def settle(self, model: str, reward: float, cost: float) -> None:
        if model not in self.n:
            return
        self.n[model] += 1
        self.total_reward[model] += max(reward, 0.0)
        self.total_cost[model] += cost

    def state(self) -> dict:
        return {
            "n": dict(self.n),
            "total_reward": dict(self.total_reward),
            "total_cost": dict(self.total_cost),
        }

    def set_state(self, state: dict) -> None:
        """Restore counts for known arms; arms added since the checkpoint
        keep zero counts (aligned-by-name, tolerating arm list changes)."""
        for field in ("n", "total_reward", "total_cost"):
            saved = state.get(field, {})
            target = getattr(self, field)
            for model in self.models:
                if model in saved:
                    target[model] = saved[model]
