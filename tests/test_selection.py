import numpy as np
import pytest

from conftest import make_candidate
from evoharness.evocore import (
    BeamSelector,
    IslandView,
    InspirationSelector,
    LatestSelector,
    PopulationConfig,
    PopulationStore,
    PowerLawSelector,
    SeedOnlySelector,
    WeightedSelector,
    make_parent_selector,
)
from evoharness.evocore.selection import stable_sigmoid


def view(cands):
    return IslandView(island_idx=0, candidates=cands)


def test_weighted_formula_matches_upstream_definition():
    cfg = PopulationConfig(weighted_lambda=10.0)
    pool = [
        make_candidate("a", 1.0, children=0),
        make_candidate("b", 2.0, children=0),
        make_candidate("c", 3.0, children=0),
    ]
    probs = WeightedSelector(cfg).probabilities(pool)
    scores = np.array([1.0, 2.0, 3.0])
    alpha0 = np.median(scores)
    mad = np.median(np.abs(scores - alpha0))
    s = stable_sigmoid(10.0 * (scores - alpha0) / max(mad, 1e-6))
    w = s * 1.0
    expected = w / w.sum()
    assert np.allclose(probs, expected)
    assert probs[2] > probs[1] > probs[0]


def test_weighted_children_discount():
    cfg = PopulationConfig()
    fresh = make_candidate("a", 2.0, children=0)
    exhausted = make_candidate("b", 2.0, children=9)
    probs = WeightedSelector(cfg).probabilities([fresh, exhausted])
    assert probs[0] == pytest.approx(probs[1] * 10.0)


def test_weighted_empirical_distribution_fixed_seed():
    cfg = PopulationConfig()
    pool = [make_candidate(c, f) for c, f in [("a", 1.0), ("b", 2.0), ("c", 3.0)]]
    sel = WeightedSelector(cfg)
    probs = sel.probabilities(pool)
    rng = np.random.default_rng(42)
    counts = {"a": 0, "b": 0, "c": 0}
    n = 4000
    for _ in range(n):
        counts[sel.sample(view(pool), rng).id] += 1
    for i, cid in enumerate(["a", "b", "c"]):
        assert counts[cid] / n == pytest.approx(probs[i], abs=0.03)


def test_power_law_probabilities():
    cfg = PopulationConfig(power_alpha=1.0)
    pool = [make_candidate(c, f) for c, f in [("low", 1.0), ("hi", 3.0), ("mid", 2.0)]]
    probs = PowerLawSelector(cfg).probabilities(pool)
    # ranks: hi=1, mid=2, low=3 -> raw 1, 1/2, 1/3
    raw = np.array([1 / 3, 1.0, 1 / 2])
    assert np.allclose(probs, raw / raw.sum())


def test_beam_sticks_until_width_then_switches():
    cfg = PopulationConfig(beam_width=2)
    a = make_candidate("a", 3.0, children=0)
    b = make_candidate("b", 5.0, children=0)
    sel = BeamSelector(cfg)
    rng = np.random.default_rng(0)
    assert sel.sample(view([a, b]), rng).id == "b"  # best picked first
    b.children_count = 1
    assert sel.sample(view([a, b]), rng).id == "b"  # still under width
    b.children_count = 2
    a2 = make_candidate("a2", 9.0)
    assert sel.sample(view([a, b, a2]), rng).id == "a2"  # switched to new best


def test_baseline_selectors():
    cfg = PopulationConfig()
    seed = make_candidate("seed", 1.0, generation=0)
    late = make_candidate("late", 0.5, generation=7)
    rng = np.random.default_rng(0)
    assert SeedOnlySelector(cfg).sample(view([seed, late]), rng).id == "seed"
    assert LatestSelector(cfg).sample(view([seed, late]), rng).id == "late"


def test_factory_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        make_parent_selector(PopulationConfig(parent_strategy="nope"))


def test_weight_policy_multiplier_applies():
    class Penalize:
        def weight_multiplier(self, cand):
            return 0.0 if cand.id == "b" else 1.0

    cfg = PopulationConfig()
    pool = [make_candidate("a", 2.0), make_candidate("b", 2.0)]
    probs = WeightedSelector(cfg, [Penalize()]).probabilities(pool)
    assert probs[1] == 0.0 and probs[0] == pytest.approx(1.0)


def test_inspiration_selector_excludes_parent_and_dedups():
    cfg = PopulationConfig(
        num_archive_inspirations=2, num_top_k_inspirations=2
    )
    store = PopulationStore(cfg)
    cands = [make_candidate(f"c{i}", float(i), island=0) for i in range(6)]
    for c in cands:
        store.insert(c)
    parent = cands[5]  # the best
    rng = np.random.default_rng(1)
    archive, top_k = InspirationSelector(cfg).sample(parent, store, rng)
    ids = [c.id for c in archive] + [c.id for c in top_k]
    assert parent.id not in ids
    assert len(ids) == len(set(ids))
    assert len(archive) == 2 and len(top_k) == 2
