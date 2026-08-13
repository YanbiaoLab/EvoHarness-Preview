"""The ShinkaEvolve adapter's score must stay finite and correctly ordered.

The first smoke run produced `combined_score: -inf` for a candidate that
failed the perturbation gate before any tier completed. ShinkaEvolve stored
that as 0.000, which would have ranked a disqualified program ABOVE a working
but slow one — the exact inversion the score exists to prevent. These tests
pin the ordering the search depends on.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "experiments" / "modmul" / "shinka" / "evaluate.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("modmul_shinka_evaluate", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = pytest.importorskip("torch") and _load_adapter()


class _Grade:
    """Minimal stand-in for serve.Grade — only the fields build_metrics reads."""

    def __init__(self, *, visible, passed=True, fault=None, fitness=0.0):
        self.visible_metrics = visible
        self.hidden_metrics = {}
        self.passed = passed
        self.fault = fault
        self.fitness = fitness
        self.stage_reached = 3
        self.execution_time = 1.0


def _healthy(seconds_per_tier: float, overall: float = 0.892) -> _Grade:
    visible = {"overall_accuracy": overall, "h90": 9, "params": 91_841}
    for tier in range(1, 11):
        visible[f"infer_s_tier_{tier}"] = seconds_per_tier
        visible[f"acc_tier_{tier}"] = 1.0 if tier <= 8 else 0.92
    return _Grade(visible=visible)


def test_score_is_finite_when_no_tier_ever_ran():
    """The failure that produced -inf: a gate rejected the candidate before a
    single tier reported a time, so there was nothing to extrapolate from."""
    grade = _Grade(
        visible={"overall_accuracy": 0.1, "h90": 1},
        passed=False,
        fault="perturbation-insensitive (L3)",
    )
    metrics = evaluate.build_metrics(grade, cases_per_tier=50)

    assert math.isfinite(metrics["combined_score"])
    assert math.isfinite(metrics["public"]["over_budget_factor"])
    assert math.isfinite(metrics["public"]["inference_seconds_projected_all_tiers"])


def test_a_failed_gate_ranks_below_every_passing_candidate():
    """The penalty is capped, so a passing candidate cannot fall below
    -LAMBDA*log2(MAX_OVER_BUDGET). A disqualified one must sit under that."""
    worst_passing = evaluate.build_metrics(
        _healthy(seconds_per_tier=1e6, overall=0.0), cases_per_tier=50
    )["combined_score"]
    disqualified = evaluate.build_metrics(
        _Grade(visible={"overall_accuracy": 0.99, "h90": 10}, passed=False,
               fault="perturbation-insensitive (L3)"),
        cases_per_tier=50,
    )["combined_score"]

    assert disqualified < worst_passing
    assert evaluate.SCORE_CRASHED < disqualified


def test_beating_the_deadline_early_earns_nothing():
    """The deadline is a constraint, not an objective. If extra speed scored
    points the search would trade accuracy for it forever."""
    fast = evaluate.build_metrics(_healthy(0.01), cases_per_tier=50)
    exact = evaluate.build_metrics(_healthy(0.27), cases_per_tier=50)

    assert fast["public"]["over_budget_factor"] < 1.0
    assert fast["combined_score"] == pytest.approx(exact["combined_score"])
    assert fast["combined_score"] == pytest.approx(0.892)


def test_being_over_the_deadline_costs_more_the_further_over_it_is():
    # The allowance is 136.4s for all ten tiers, so 10/20/40s per tier lands
    # at 0.73x, 1.47x and 2.93x of it — the last is where the real seed sits.
    results = [
        evaluate.build_metrics(_healthy(s), cases_per_tier=50) for s in (10, 20, 40)
    ]
    factors = [r["public"]["over_budget_factor"] for r in results]
    scores = [r["combined_score"] for r in results]

    assert factors[0] < 1.0 < factors[1] < factors[2]
    assert scores[0] > scores[1] > scores[2]
    # The measured seed is 2.9x over with overall 0.892; the penalty has to be
    # big enough to rank a within-budget candidate above it and small enough
    # that gutting accuracy for speed still loses. Computed from the raw ratio,
    # not the reported one — public rounds the factor to two decimals.
    allowed = evaluate.SECONDS_PER_PROBLEM * 50 * 10
    raw_factor = (40 * 10) / allowed
    assert scores[2] == pytest.approx(0.892 - 0.15 * math.log2(raw_factor))
    assert 0.15 < scores[0] - scores[2] < 0.35


def test_an_unreached_tier_is_projected_rather_than_ignored():
    """Tier 10 scores zero because the clock runs out, and it reports no time
    at all. Counting only observed seconds would make the worst candidates
    look cheapest — the projection is what keeps the pressure on."""
    visible = {"overall_accuracy": 0.892, "h90": 9}
    for tier in range(1, 10):                       # tier 10 never ran
        visible[f"infer_s_tier_{tier}"] = 2.0
    metrics = evaluate.build_metrics(_Grade(visible=visible), cases_per_tier=50)

    projected = metrics["public"]["inference_seconds_projected_all_tiers"]
    assert projected > 9 * 2.0                      # tier 10 was added in
    assert metrics["private"]["time_projection_used"] is True
    # Tier 10 has twice tier 9's operand width, so it costs about twice as much.
    assert projected == pytest.approx(9 * 2.0 + 2.0 * (4096 / 2048), rel=1e-6)
