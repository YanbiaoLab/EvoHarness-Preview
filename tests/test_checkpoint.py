"""Checkpoint format + resume flow (run.db is the primary state,
checkpoint.json carries loop counters, RNG and component states)."""

import pytest

from evoharness.evocore import (
    BanditRouter,
    BeamSelector,
    InspirationSelector,
    LLMClient,
    PopulationConfig,
    PopulationStore,
    PromptBuilder,
    SearchConfig,
    SearchLoop,
    StaticRouter,
    atomic_write_json,
    config_fingerprint,
    load_json,
    make_parent_selector,
)
from test_loop import INITIAL, MockGrader, make_rewrite_transport


def test_atomic_json_and_fingerprint(tmp_path):
    path = tmp_path / "sub" / "x.json"
    atomic_write_json(path, {"a": 1})
    assert load_json(path) == {"a": 1}
    assert load_json(tmp_path / "missing.json") is None

    a = config_fingerprint(SearchConfig(), PopulationConfig())
    b = config_fingerprint(SearchConfig(), PopulationConfig())
    c = config_fingerprint(SearchConfig(num_generations=99), PopulationConfig())
    assert a == b and a != c


def test_config_fingerprint_accepts_effective_assembly_mapping():
    base = config_fingerprint(
        SearchConfig(),
        {"contributors": ["recipes.e0.Baseline"]},
    )
    changed = config_fingerprint(
        SearchConfig(),
        {"contributors": ["recipes.e1.Reflection"]},
    )

    assert base != changed


def test_component_state_roundtrip():
    router = BanditRouter(["m1", "m2"])
    router.settle("m1", reward=1.0, cost=0.1)
    router.settle("m2", reward=0.0, cost=0.2)
    fresh = BanditRouter(["m1", "m2", "m3"])  # arm added since checkpoint
    fresh.set_state(router.state())
    assert fresh.n == {"m1": 1, "m2": 1, "m3": 0}
    assert fresh.total_reward["m1"] == 1.0

    beam = BeamSelector(PopulationConfig())
    beam._current_id = "abc"
    fresh_beam = BeamSelector(PopulationConfig())
    fresh_beam.set_state(beam.state())
    assert fresh_beam._current_id == "abc"


def _make_loop(tmp_path, num_generations, budget=None, seed=7):
    cfg = SearchConfig(
        num_generations=num_generations,
        operators=["rewrite"],
        operator_probs=[1.0],
        seed=seed,
    )
    pop_cfg = PopulationConfig(num_islands=2)
    store = PopulationStore(pop_cfg, tmp_path / "run.db")
    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=MockGrader(),
        llm=LLMClient(transport=make_rewrite_transport(), sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize increments"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        budget=budget,
        workdir=tmp_path,
    )
    return loop, store


class CapBudget:
    def __init__(self, cap):
        self.cap, self.spent = cap, 0.0

    def charge(self, usd):
        self.spent += usd

    def should_stop(self):
        return self.spent >= self.cap


def test_resume_after_budget_stop(tmp_path):
    loop1, store1 = _make_loop(tmp_path, 20, budget=CapBudget(0.02))
    report1 = loop1.run(INITIAL)
    assert report1.stopped_reason == "budget"
    interrupted_at = report1.generations_completed
    assert 0 < interrupted_at < 20
    store1.close()

    ckpt = load_json(tmp_path / "checkpoint.json")
    assert ckpt["generation"] == interrupted_at
    assert ckpt["schema_version"] == 1

    # fresh process: same config, same run dir, unmetered budget
    loop2, store2 = _make_loop(tmp_path, 20)
    report2 = loop2.run(INITIAL)
    assert report2.stopped_reason == "completed"
    assert report2.generations_completed == 20
    # counters continued rather than restarting
    assert report2.evaluations == 21  # seed + 20 total across both runs
    assert len(report2.history) == 20
    assert [h["generation"] for h in report2.history] == list(range(1, 21))
    # store holds one lineage, not a re-seeded second one
    seeds = [
        c for c in store2.all_candidates()
        if c.generation == 0 and "seed_copy_of" not in c.metadata
    ]
    assert len(seeds) == 1
    assert load_json(tmp_path / "checkpoint.json")["generation"] == 20


def test_resume_refuses_config_change(tmp_path):
    loop1, _ = _make_loop(tmp_path, 5)
    loop1.run(INITIAL)
    loop2, _ = _make_loop(tmp_path, 5, seed=99)  # different config
    with pytest.raises(ValueError, match="different configuration"):
        loop2.run(INITIAL)


def test_refuses_store_without_checkpoint(tmp_path):
    from conftest import make_candidate

    pop_cfg = PopulationConfig(num_islands=2)
    store = PopulationStore(pop_cfg, tmp_path / "run.db")
    store.insert(make_candidate("orphan", 1.0))
    loop, _ = _make_loop(tmp_path, 5)
    loop.store = store
    with pytest.raises(RuntimeError, match="no checkpoint"):
        loop.run(INITIAL)


def test_refuses_checkpoint_without_store(tmp_path):
    loop1, store1 = _make_loop(tmp_path, 3)
    loop1.run(INITIAL)
    store1.close()
    (tmp_path / "run.db").unlink()  # lost the primary state
    loop2, _ = _make_loop(tmp_path, 3)
    with pytest.raises(ValueError, match="store is empty"):
        loop2.run(INITIAL)


def test_completed_run_resumes_to_noop(tmp_path):
    loop1, _ = _make_loop(tmp_path, 4)
    report1 = loop1.run(INITIAL)
    assert report1.generations_completed == 4
    loop2, _ = _make_loop(tmp_path, 4)
    report2 = loop2.run(INITIAL)
    assert report2.evaluations == report1.evaluations  # nothing re-run
    assert report2.best_fitness == report1.best_fitness
