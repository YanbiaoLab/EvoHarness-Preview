"""Protocol §5 loop-side guarantees: candidates with no verdict are dropped
(never inserted, never counted as evaluations), a dead eval service trips the
circuit breaker instead of draining the run, and an ungradeable seed is fatal.
"""

import time
from pathlib import Path

import pytest

from evoharness.core import (
    EvalReport,
    InspirationSelector,
    LLMClient,
    LLMResponse,
    PopulationConfig,
    PopulationStore,
    PromptBuilder,
    SearchConfig,
    SearchLoop,
    StaticRouter,
    make_parent_selector,
)
from evoharness.core.remote import EvalInfraError

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


class FlakyGrader:
    """MockGrader semantics + an EvalInfraError schedule.

    fail_on: 1-based indices of CHILD gradings that raise (seed not counted).
    fail_seed: raise on the seed itself (must be fatal for the run).
    """

    def __init__(self, fail_on=(), fail_all_children=False, fail_seed=False):
        self.fail_on = set(fail_on)
        self.fail_all_children = fail_all_children
        self.fail_seed = fail_seed
        self.children_seen = 0

    def grade(self, cand, workdir: Path) -> EvalReport:
        if cand.operator == "seed":
            if self.fail_seed:
                raise EvalInfraError("eval service down at seeding")
        else:
            self.children_seen += 1
            if self.fail_all_children or self.children_seen in self.fail_on:
                raise EvalInfraError(f"eval service down (child #{self.children_seen})")
        return EvalReport(
            fitness=float(cand.code.count("x += 1")),
            passed=True,
            eval_cost_usd=0.001,
        )


def make_rewrite_transport():
    counter = {"n": 1}

    def transport(messages, model, **kw):
        counter["n"] += 1
        body = "x = 0\n" + "x += 1\n" * counter["n"]
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        text = (
            f"TITLE: add increment {counter['n']}\n"
            "SUMMARY: one more increment\n"
            f"```python\n{code}```"
        )
        return LLMResponse(text=text, model=model, cost=0.002)

    return transport


def build_loop(grader, tmp_path, generations=10, breaker=5, transport=None):
    cfg = SearchConfig(
        num_generations=generations,
        operators=["rewrite"],
        operator_probs=[1.0],
        seed=7,
        max_consecutive_infra_failures=breaker,
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=grader,
        llm=LLMClient(
            transport=transport or make_rewrite_transport(), sleep=lambda s: None
        ),
        prompt_builder=PromptBuilder("maximize increments"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        workdir=tmp_path,
    )
    return loop, store


def test_single_outage_is_dropped_and_run_continues(tmp_path):
    loop, store = build_loop(FlakyGrader(fail_on={3}), tmp_path)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "completed"
    assert report.generations_completed == 10
    dropped = [h for h in report.history if h["status"] == "infra_error"]
    assert len(dropped) == 1 and "child #3" in dropped[0]["error"]
    assert len([h for h in report.history if h["status"] == "ok"]) == 9
    assert store.count() == 1 + 9              # seed + 9 children; NO dropped one
    assert report.evaluations == 1 + 9         # failed grading is not an evaluation
    assert report.total_eval_cost == pytest.approx(10 * 0.001)


def test_dead_service_trips_circuit_breaker(tmp_path):
    loop, store = build_loop(
        FlakyGrader(fail_all_children=True), tmp_path, generations=20, breaker=3
    )
    report = loop.run(INITIAL)

    assert report.stopped_reason == "eval_infra"
    assert report.generations_completed == 3    # stopped, not drained to 20
    assert store.count() == 1                   # seed only
    assert report.evaluations == 1
    assert len([h for h in report.history if h["status"] == "infra_error"]) == 3


def test_streak_resets_on_success(tmp_path):
    """Two separate 2-long outages never trip a breaker of 3 — this pins the
    `infra_streak = 0` reset after every successful grading."""
    loop, store = build_loop(
        FlakyGrader(fail_on={2, 3, 6, 7}), tmp_path, generations=10, breaker=3
    )
    report = loop.run(INITIAL)

    assert report.stopped_reason == "completed"
    assert len([h for h in report.history if h["status"] == "infra_error"]) == 4
    assert store.count() == 1 + 6


def test_ungradeable_seed_is_fatal(tmp_path):
    loop, _ = build_loop(FlakyGrader(fail_seed=True), tmp_path)
    with pytest.raises(EvalInfraError, match="seeding"):
        loop.run(INITIAL)


def dead_proposer_transport(messages, model, **kw):
    """What an unreachable LLM endpoint looks like to the loop: a response
    the parser cannot turn into a proposal."""
    return LLMResponse(text="", model=model, cost=0.0)


def test_a_dead_proposer_stops_the_run_instead_of_draining_it(tmp_path):
    """A run that produces no offspring must say so.

    The eval service had a circuit breaker; the proposer had none. With the
    LLM unreachable every generation recorded "skipped", the loop ran to the
    end, and stopped_reason came out "completed" — observed live on the L40S,
    where a two-generation run reported success having produced only its
    seeds. Over a multi-day schedule an expired credential would burn the
    entire run and still look healthy in the summary.
    """
    loop, store = build_loop(
        FlakyGrader(), tmp_path, generations=30, breaker=5,
        transport=dead_proposer_transport,
    )

    report = loop.run(INITIAL)

    assert report.stopped_reason == "proposer_dead"
    # It must give up early rather than walking the whole schedule.
    assert report.generations_completed <= 6
    # Only the seed made it into the population.
    assert len([c for c in store.all_candidates() if c.operator != "seed"]) == 0


def test_a_healthy_proposer_never_trips_the_new_breaker(tmp_path):
    """The breaker must not fire on a run that is working — a guard that
    misfires is worse than the defect it guards against."""
    loop, store = build_loop(FlakyGrader(), tmp_path, generations=10, breaker=5)
    report = loop.run(INITIAL)

    assert report.stopped_reason == "completed"
    assert report.generations_completed == 10


def test_parallel_proposals_plan_in_the_same_order_as_sequential(tmp_path):
    """Overlapping the LLM calls must not change which parent and operator
    each slot draws.

    Planning consumes self.rng, so doing it concurrently would make a seeded
    run stop reproducing itself. Only the network wait overlaps: plans are
    made in order, calls run together, results are absorbed in planning
    order.

    What this does NOT promise: identical output when something inside the
    proposer or transport is itself order-dependent. This test's fake
    transport numbers its answers from a shared counter, and four concurrent
    calls consume it in whatever order they finish — so the resulting code
    differs even though every choice this loop makes is the same. A model
    router that rotates between models has the same property.
    """
    plans = []
    for concurrency in (1, 4):
        loop, store = build_loop(
            FlakyGrader(), tmp_path / f"c{concurrency}", generations=1
        )
        loop.cfg.eval_batch_size = 4
        loop.cfg.proposal_concurrency = concurrency
        loop.run(INITIAL)
        by_id = {c.id: c for c in store.all_candidates()}
        # Candidate ids are random UUIDs, so they cannot be compared across
        # two independent runs; compare the choices themselves.
        plans.append(sorted(
            (c.operator, by_id[c.parent_id].code)
            for c in store.all_candidates()
            if c.operator != "seed" and c.parent_id in by_id
        ))

    assert plans[0], "the run produced no offspring to compare"
    assert plans[0] == plans[1]


def test_a_proposal_cannot_outlive_its_deadline(tmp_path):
    """Retries nest, so a retry COUNT is not a bound.

    The client retries a transient fault three times and the proposer
    resamples three times, so one proposal could occupy nine timeouts — 90
    minutes at the timeout run modmul_r6 used. A batch waits for every
    proposal before absorbing any, so that one call stalled a whole
    generation. This is the eval-side 2h04m hang wearing different clothes:
    each attempt got a fresh budget and nothing bounded the total.
    """
    from evoharness.core.proposer import SingleShotProposer
    from evoharness.core.routing import StaticRouter

    calls = {"n": 0}

    def slow_and_useless(messages, model, **kw):
        calls["n"] += 1
        time.sleep(0.05)
        return LLMResponse(text="no code here at all", model=model)

    proposer = SingleShotProposer(
        llm=LLMClient(transport=slow_and_useless, sleep=lambda s: None),
        model_router=StaticRouter(["m"]),
        max_resamples=10,
        deadline_s=0.12,
    )
    from evoharness.core import Candidate
    parent = Candidate(
        id="p", code=INITIAL, generation=0, parent_id=None,
        island_idx=0, operator="seed",
    )
    started = time.monotonic()
    result = proposer.propose("rewrite", parent, "sys", "user")
    elapsed = time.monotonic() - started

    assert result.proposal is None
    # It gave up on the deadline rather than walking all ten resamples.
    assert calls["n"] < 10
    assert elapsed < 0.5


def test_a_batch_spreads_across_islands_and_parents(tmp_path):
    """Both island rotation and the anti-monoculture penalty assumed one
    proposal per generation, and both broke silently when a generation
    started holding sixteen.

    Island choice was keyed on the generation, so every proposal in a batch
    got the same island. And children_count — the 1/(1+n) factor that stops
    the selector grinding on one parent — was charged on insert, which
    happens after the whole batch is already planned, so all sixteen saw
    zero. Run modmul_r7 put 17 of 20 candidates on one island and drew all
    15 offspring from a single parent, which ended the generation with
    children_count=15 that had never influenced a single choice.
    """
    from evoharness.core import PopulationConfig, PopulationStore

    cfg = SearchConfig(
        num_generations=1, operators=["rewrite"], operator_probs=[1.0],
        seed=7, eval_batch_size=9,
    )
    pop_cfg = PopulationConfig(num_islands=3)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg, pop_cfg=pop_cfg, store=store, grader=FlakyGrader(),
        llm=LLMClient(transport=make_rewrite_transport(), sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize increments"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        workdir=tmp_path,
    )
    loop.run(INITIAL, extra_seeds=[INITIAL, INITIAL])

    offspring = [c for c in store.all_candidates() if c.operator != "seed"]
    assert len(offspring) >= 6, "no batch to inspect"
    assert len({c.island_idx for c in offspring}) > 1, (
        "a whole batch landed on one island"
    )
    assert len({c.parent_id for c in offspring}) > 1, (
        "a whole batch came from one parent"
    )
