import numpy as np

from conftest import make_candidate
from evoharness.core import (
    BanditRouter,
    IslandView,
    NoveltyGate,
    StaticRouter,
    cosine_similarity,
)
from evoharness.core.novelty import parse_judge_verdict


def test_cosine_similarity():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert cosine_similarity(np.zeros(2), np.ones(2)) == 0.0


def test_parse_judge_verdict():
    assert parse_judge_verdict("NOVEL — different algorithm")
    assert parse_judge_verdict("**NOVEL** because ...")
    assert not parse_judge_verdict("NOT NOVEL: same logic")
    assert not parse_judge_verdict("hard to say")


def _island_with_embedding(vec):
    existing = make_candidate("e", 1.0, embedding=list(vec))
    return IslandView(island_idx=0, candidates=[existing])


# These three cover the upstream-parity "similarity" path explicitly; the
# default mode is identity (see NoveltyGate for why fuzzy needs a semantic
# embedder to be meaningful).
def test_gate_accepts_below_threshold():
    gate = NoveltyGate(
        embed_fn=lambda code: [0.0, 1.0], threshold=0.9, mode="similarity"
    )
    verdict = gate.check("new code", _island_with_embedding([1.0, 0.0]))
    assert verdict.accepted and verdict.max_similarity == 0.0
    assert verdict.embedding == [0.0, 1.0]


def test_gate_rejects_identical_without_judge():
    gate = NoveltyGate(
        embed_fn=lambda code: [1.0, 0.0], threshold=0.99, mode="similarity"
    )
    verdict = gate.check("dup", _island_with_embedding([1.0, 0.0]))
    assert not verdict.accepted and verdict.most_similar_id == "e"


def test_gate_judge_overrides():
    def fake_judge(old, new):
        # Contract guard: the gate must hand the judge TEXT, not a Workspace
        # object (a repr slipping into the judge prompt degrades silently).
        assert isinstance(old, str) and isinstance(new, str)
        return "NOVEL: different in substance"

    gate = NoveltyGate(
        embed_fn=lambda code: [1.0, 0.0],
        threshold=0.99,
        judge_fn=fake_judge,
        mode="similarity",
    )
    verdict = gate.check("dup", _island_with_embedding([1.0, 0.0]))
    assert verdict.accepted and verdict.judged


def test_static_router_respects_probs():
    rng = np.random.default_rng(0)
    router = StaticRouter(["a", "b"], probs=[0.9, 0.1], rng=rng)
    picks = [router.pick() for _ in range(1000)]
    assert 0.85 < picks.count("a") / 1000 < 0.95


def test_bandit_router_prefers_rewarding_arm():
    rng = np.random.default_rng(0)
    router = BanditRouter(["good", "bad"], exploration_coef=0.1, rng=rng)
    for _ in range(30):
        m = router.pick()
        router.settle(m, reward=1.0 if m == "good" else 0.0, cost=0.0)
    picks = [router.pick() for _ in range(20)]
    assert picks.count("good") > picks.count("bad")
