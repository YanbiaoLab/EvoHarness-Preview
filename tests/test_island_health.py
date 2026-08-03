"""Island revival: detecting a lineage that stopped contributing, and the
deliberately soft intervention that follows."""

import pytest

from evoharness.evocore.config import PopulationConfig
from evoharness.evocore.population import Candidate, EvalReport, PopulationStore
from evoharness.evoplus.islands import IslandHealthMonitor


def _cand(cid, fitness, island, generation, passed=True):
    return Candidate(
        id=cid,
        code=f"# {cid}",
        generation=generation,
        parent_id=None,
        island_idx=island,
        operator="revise",
        report=EvalReport(fitness=fitness, passed=passed),
    )


def _store(num_islands=3):
    return PopulationStore(PopulationConfig(num_islands=num_islands))


def _observe(monitor, store, cand):
    monitor.on_candidate_graded(cand, store)
    store.insert(cand)


# -- detection ------------------------------------------------------------


def test_a_quiet_underperforming_island_is_flagged():
    monitor = IslandHealthMonitor(patience=5, relative_floor=0.75)
    store = _store()
    _observe(monitor, store, _cand("strong", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("weak", 0.2, 1, generation=1))

    health = {h.island_idx: h for h in monitor.survey(generation=10, num_islands=3)}
    assert not health[0].stalled          # holds the record
    assert health[1].stalled              # quiet for 9, and far behind


def test_holding_the_record_is_proof_of_life():
    """The leading island is never restarted, however long it has been quiet.

    A search that has converged is not a search that has broken, and paving
    over the best lineage would lose the run's best program.
    """
    monitor = IslandHealthMonitor(patience=2)
    store = _store()
    _observe(monitor, store, _cand("best", 1.0, 0, generation=1))
    health = {h.island_idx: h for h in monitor.survey(generation=50, num_islands=3)}
    assert not health[0].stalled


def test_recent_improvement_clears_the_clock():
    monitor = IslandHealthMonitor(patience=5, relative_floor=0.9)
    store = _store()
    _observe(monitor, store, _cand("strong", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("weak", 0.2, 1, generation=1))
    _observe(monitor, store, _cand("weak2", 0.3, 1, generation=9))

    health = {h.island_idx: h for h in monitor.survey(generation=10, num_islands=3)}
    assert not health[1].stalled          # improved one generation ago


def test_a_close_second_is_not_stalled():
    """Behind is not dead. Only CLEAR underperformance plus a long silence."""
    monitor = IslandHealthMonitor(patience=3, relative_floor=0.75)
    store = _store()
    _observe(monitor, store, _cand("a", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("b", 0.9, 1, generation=1))

    health = {h.island_idx: h for h in monitor.survey(generation=20, num_islands=3)}
    assert not health[1].stalled


def test_failures_do_not_count_as_signs_of_life():
    monitor = IslandHealthMonitor(patience=3)
    store = _store()
    _observe(monitor, store, _cand("a", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("b", 0.1, 1, generation=1))
    _observe(monitor, store, _cand("junk", 9.9, 1, generation=9, passed=False))

    health = {h.island_idx: h for h in monitor.survey(generation=10, num_islands=3)}
    assert health[1].stalled
    assert monitor.best[1] == pytest.approx(0.1)


def test_nothing_fires_before_the_minimum_generation():
    monitor = IslandHealthMonitor(patience=1, min_generation=10)
    store = _store()
    _observe(monitor, store, _cand("a", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("b", 0.1, 1, generation=1))
    assert not any(h.stalled for h in monitor.survey(5, num_islands=3))


# -- intervention ---------------------------------------------------------


def test_restart_injects_the_best_foreign_candidate_and_deletes_nothing():
    monitor = IslandHealthMonitor(patience=3, relative_floor=0.75)
    store = _store()
    _observe(monitor, store, _cand("champion", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("native", 0.2, 1, generation=1))
    _observe(monitor, store, _cand("healthy", 0.95, 2, generation=1))

    events = monitor.maybe_restart(store, generation=10, num_islands=3)
    assert [e.island_idx for e in events] == [1]
    event = events[0]
    assert event.island_idx == 1 and event.donor_id == "champion"

    island = store.island_view(1)
    ids = {c.id for c in island.candidates}
    assert "native" in ids, "a restart must never remove what was there"
    assert event.injected_id in ids
    injected = store.get(event.injected_id)
    assert injected.code == "# champion"
    # The redirect a domain needs to find the donor's carried state.
    assert injected.metadata["seed_copy_of"] == "champion"
    assert injected.metadata["island_reseed_from"] == "champion"


def test_an_island_that_never_produced_anything_is_supplied_too():
    """An island whose every candidate failed holds nothing to breed from.

    It is the most under-supplied island there is, and the loop's round-robin
    keeps handing it slots regardless, so it gets a migrant on the same terms
    as one that merely went quiet.
    """
    monitor = IslandHealthMonitor(patience=3, relative_floor=0.75)
    store = _store()
    _observe(monitor, store, _cand("champion", 1.0, 0, generation=1))

    events = monitor.maybe_restart(store, generation=10, num_islands=2)
    assert [e.island_idx for e in events] == [1]
    assert store.island_view(1).passed_candidates


def test_a_restart_resets_the_clock_rather_than_firing_every_generation():
    monitor = IslandHealthMonitor(patience=3, relative_floor=0.75)
    store = _store()
    _observe(monitor, store, _cand("champion", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("native", 0.2, 1, generation=1))

    assert monitor.maybe_restart(store, generation=10, num_islands=3)
    assert monitor.maybe_restart(store, generation=11, num_islands=3) == []


def test_restarts_are_capped_per_island():
    monitor = IslandHealthMonitor(
        patience=2, relative_floor=0.75, max_restarts_per_island=1
    )
    store = _store()
    _observe(monitor, store, _cand("champion", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("native", 0.2, 1, generation=1))

    assert monitor.maybe_restart(store, generation=10, num_islands=3)
    assert monitor.maybe_restart(store, generation=20, num_islands=3) == []


def test_a_single_island_run_is_untouched():
    monitor = IslandHealthMonitor(patience=1)
    store = _store(num_islands=1)
    _observe(monitor, store, _cand("only", 0.1, 0, generation=1))
    assert monitor.maybe_restart(store, generation=99, num_islands=1) == []


def test_state_survives_a_checkpoint():
    monitor = IslandHealthMonitor(patience=3)
    store = _store()
    _observe(monitor, store, _cand("champion", 1.0, 0, generation=1))
    _observe(monitor, store, _cand("native", 0.2, 1, generation=1))
    monitor.maybe_restart(store, generation=10, num_islands=3)

    restored = IslandHealthMonitor(patience=3)
    restored.set_state(monitor.state())
    assert restored.best == monitor.best
    assert restored.improved_at == monitor.improved_at
    assert restored.restarts == monitor.restarts
    # And the cap is still remembered after a resume.
    assert restored.restarts[1] == 1


def test_the_loop_actually_revives_a_dead_island(tmp_path):
    """Liveness, not unit correctness: the monitor must fire from inside a
    real run, not only when a test calls it directly.

    This project has a standing lesson that a mechanism can pass every unit
    test while never being reached in a live run -- a day of auditing once
    turned up eight such defects behind a green suite.
    """
    from evoharness.evocore import (
        InspirationSelector,
        LLMClient,
        LLMResponse,
        PromptBuilder,
        SearchConfig,
        SearchLoop,
        StaticRouter,
        make_parent_selector,
    )

    class _LopsidedGrader:
        """Island 0 thrives; island 1 can never score. A dead lineage."""

        def grade(self, cand, workdir):
            alive = cand.island_idx == 0
            return EvalReport(
                fitness=(1.0 + 0.01 * cand.generation) if alive else 0.05,
                passed=True,
            )

    def transport(messages, model, **kw):
        transport.n += 1
        body = "x = 0\n" + "x += 1\n" * (transport.n + 1)
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        return LLMResponse(
            text=f"TITLE: t{transport.n}\nSUMMARY: s\n```python\n{code}```",
            model=model,
            cost=0.0,
        )

    transport.n = 0
    pop_cfg = PopulationConfig(num_islands=2)
    store = PopulationStore(pop_cfg)
    monitor = IslandHealthMonitor(
        patience=3, relative_floor=0.75, min_generation=2
    )
    loop = SearchLoop(
        cfg=SearchConfig(
            num_generations=12,
            operators=["rewrite"],
            operator_probs=[1.0],
            seed=1,
        ),
        pop_cfg=pop_cfg,
        store=store,
        grader=_LopsidedGrader(),
        llm=LLMClient(transport=transport, sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        observers=[monitor],
        workdir=tmp_path,
        island_health=monitor,
    )
    report = loop.run(
        "# EDIT-REGION-BEGIN\nx = 0\nx += 1\n# EDIT-REGION-END\nprint(x)\n"
    )

    revivals = [h for h in report.history if h["status"] == "island_revived"]
    assert revivals, "the monitor never fired inside a live run"
    assert all(event["island_idx"] == 1 for event in revivals)
    # An injection changes where later candidates came from, so the run has
    # to be able to say when it happened.
    assert revivals[0]["donor_island"] == 0
    assert store.get(revivals[0]["injected_id"]) is not None
    assert monitor.restarts[1] == len(revivals)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patience": 0},
        {"relative_floor": 1.5},
        {"min_generation": -1},
        {"max_restarts_per_island": -1},
    ],
)
def test_nonsense_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        IslandHealthMonitor(**kwargs)
