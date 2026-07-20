# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/database/parents.py (five sampling strategies, exact
#           weight formulas), shinka/database/inspirations.py (archive +
#           top-k inspiration sampling)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Behavior-aligned port; the multiplicative SamplingWeightPolicy hook is an
# EvoHarness extension (weights default to 1.0, preserving upstream behavior).
"""Parent selection strategies and inspiration sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import PopulationConfig
from .interfaces import SamplingWeightPolicy
from .population import Candidate, IslandView, PopulationStore


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (upstream parents.py)."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


class ParentSelector(ABC):
    def __init__(
        self,
        cfg: PopulationConfig,
        weight_policies: list[SamplingWeightPolicy] | None = None,
    ):
        self.cfg = cfg
        self.weight_policies = weight_policies or []

    def _policy_multiplier(self, cand: Candidate) -> float:
        m = 1.0
        for p in self.weight_policies:
            m *= float(p.weight_multiplier(cand))
        return m

    @staticmethod
    def _pool(island: IslandView) -> list[Candidate]:
        """Sampling pool: island archive members, falling back to all passed
        candidates (upstream fallback chain, simplified)."""
        pool = island.archive_candidates
        return pool if pool else island.passed_candidates

    @abstractmethod
    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        ...


class WeightedSelector(ParentSelector):
    """Upstream default: performance sigmoid x novelty discount.

    w_i = sigmoid(lambda * (f_i - median) / MAD) * 1 / (1 + children_count)
    """

    def probabilities(self, pool: list[Candidate]) -> np.ndarray:
        scores = np.array([c.fitness for c in pool], dtype=float)
        alpha0 = float(np.median(scores))
        mad = float(np.median(np.abs(scores - alpha0)))
        scale = max(mad, 1e-6)
        s = stable_sigmoid(self.cfg.weighted_lambda * (scores - alpha0) / scale)
        h = np.array([1.0 / (1.0 + c.children_count) for c in pool])
        m = np.array([self._policy_multiplier(c) for c in pool])
        w = s * h * m
        total = w.sum()
        if total <= 0:
            return np.full(len(pool), 1.0 / len(pool))
        return w / total

    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        pool = self._pool(island)
        if not pool:
            return None
        probs = self.probabilities(pool)
        return pool[int(rng.choice(len(pool), p=probs))]


class PowerLawSelector(ParentSelector):
    """P(rank i) proportional to (i+1)^(-alpha), ranked by fitness desc."""

    def probabilities(self, pool: list[Candidate]) -> np.ndarray:
        order = np.argsort([-c.fitness for c in pool], kind="stable")
        raw = np.zeros(len(pool))
        for rank, idx in enumerate(order):
            raw[idx] = (rank + 1) ** (-self.cfg.power_alpha)
        m = np.array([self._policy_multiplier(c) for c in pool])
        raw = raw * m
        return raw / raw.sum()

    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        pool = self._pool(island)
        if not pool:
            return None
        probs = self.probabilities(pool)
        return pool[int(rng.choice(len(pool), p=probs))]


class BeamSelector(ParentSelector):
    """Stick with the current best parent until it has beam_width children,
    then move to the new best (stateful, per selector instance)."""

    def __init__(self, cfg, weight_policies=None):
        super().__init__(cfg, weight_policies)
        self._current_id: str | None = None

    def state(self) -> dict:
        return {"current_id": self._current_id}

    def set_state(self, state: dict) -> None:
        self._current_id = state.get("current_id")

    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        pool = self._pool(island)
        if not pool:
            return None
        by_id = {c.id: c for c in pool}
        current = by_id.get(self._current_id) if self._current_id else None
        if current is not None and current.children_count < self.cfg.beam_width:
            return current
        best = max(pool, key=lambda c: c.fitness)
        self._current_id = best.id
        return best


class SeedOnlySelector(ParentSelector):
    """Always the generation-0 seed (baseline strategy)."""

    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        seeds = [c for c in island.passed_candidates if c.generation == 0]
        return seeds[0] if seeds else None


class LatestSelector(ParentSelector):
    """Always the most recent passed candidate (baseline strategy)."""

    def sample(self, island: IslandView, rng: np.random.Generator) -> Candidate | None:
        pool = island.passed_candidates
        return max(pool, key=lambda c: (c.generation, c.timestamp)) if pool else None


_STRATEGIES = {
    "weighted": WeightedSelector,
    "power_law": PowerLawSelector,
    "beam": BeamSelector,
    "seed_only": SeedOnlySelector,
    "latest": LatestSelector,
}


def make_parent_selector(
    cfg: PopulationConfig,
    weight_policies: list[SamplingWeightPolicy] | None = None,
) -> ParentSelector:
    try:
        return _STRATEGIES[cfg.parent_strategy](cfg, weight_policies)
    except KeyError:
        raise ValueError(
            f"unknown parent_strategy {cfg.parent_strategy!r}; "
            f"expected one of {sorted(_STRATEGIES)}"
        ) from None


class InspirationSelector:
    """Two-route inspiration sampling for the mutation prompt.

    Archive route: best + elites (elite_selection_ratio) + random archive
    members. Top-k route: best remaining by fitness, excluding the parent and
    already chosen archive inspirations. Island separation on by default.
    """

    def __init__(self, cfg: PopulationConfig):
        self.cfg = cfg

    def sample(
        self,
        parent: Candidate,
        store: PopulationStore,
        rng: np.random.Generator,
    ) -> tuple[list[Candidate], list[Candidate]]:
        cfg = self.cfg
        if cfg.enforce_island_separation:
            view = store.island_view(parent.island_idx)
            scope = view.passed_candidates
            archive_scope = view.archive_candidates
        else:
            scope = [c for c in store.all_candidates() if c.passed]
            archive_scope = [c for c in scope if c.in_archive]

        chosen: list[Candidate] = []
        chosen_ids = {parent.id}

        n = cfg.num_archive_inspirations
        if n > 0 and archive_scope:
            best = max(archive_scope, key=lambda c: c.fitness)
            if best.id not in chosen_ids:
                chosen.append(best)
                chosen_ids.add(best.id)
            num_elites = max(0, int(n * cfg.elite_selection_ratio))
            elites = sorted(archive_scope, key=lambda c: -c.fitness)
            for c in elites:
                if len(chosen) >= min(n, 1 + num_elites):
                    break
                if c.id not in chosen_ids:
                    chosen.append(c)
                    chosen_ids.add(c.id)
            remaining = [c for c in archive_scope if c.id not in chosen_ids]
            while len(chosen) < n and remaining:
                pick = remaining.pop(int(rng.integers(len(remaining))))
                chosen.append(pick)
                chosen_ids.add(pick.id)

        top_k: list[Candidate] = []
        k = cfg.num_top_k_inspirations
        if k > 0:
            ranked = sorted(scope, key=lambda c: -c.fitness)
            for c in ranked:
                if len(top_k) >= k:
                    break
                if c.id not in chosen_ids:
                    top_k.append(c)
                    chosen_ids.add(c.id)

        return chosen, top_k
