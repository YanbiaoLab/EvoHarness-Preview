"""Protocol §5 loop-side guarantees: candidates with no verdict are dropped
(never inserted, never counted as evaluations), a dead eval service trips the
circuit breaker instead of draining the run, and an ungradeable seed is fatal.
"""

from pathlib import Path

import pytest

from evoharness.evocore import (
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
from evoharness.evocore.remote import EvalInfraError

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
