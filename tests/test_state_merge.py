"""Population-level state merging: who gets merged, and what the loop does
with the plan."""

import numpy as np
import pytest

from evoharness.evocore.population import Candidate, EvalReport, IslandView
from evoharness.evoplus.feedback import BehaviorSignature
from evoharness.evoplus.merge import (
    Complementarity,
    StateMergePlanner,
    complementarity,
)


def _signature(bits: str) -> str:
    return BehaviorSignature(
        pass_vector=tuple(c == "1" for c in bits), error_histogram=()
    ).encode()


def _cand(cid, fitness, bits, island=0, metadata=None):
    return Candidate(
        id=cid,
        code=f"# {cid}",
        generation=1,
        parent_id=None,
        island_idx=island,
        operator="revise",
        report=EvalReport(fitness=fitness, passed=True),
        behavior_signature=_signature(bits) if bits else None,
        metadata=metadata or {},
    )


class _AlwaysFires:
    """np.random.Generator stand-in whose roll always passes."""

    @staticmethod
    def random():
        return 0.0


class _NeverFires:
    @staticmethod
    def random():
        return 1.0


# -- complementarity ------------------------------------------------------


def test_complementarity_counts_each_quadrant():
    base = BehaviorSignature((True, True, False, False), ())
    donor = BehaviorSignature((True, False, True, False), ())
    overlap = complementarity(base, donor)
    assert overlap == Complementarity(
        both_pass=1, base_only=1, donor_only=1, both_fail=1
    )
    assert overlap.is_complementary


def test_a_dominating_donor_is_not_a_merge():
    """If the donor solves everything the base does and more, take the donor.

    Merging is for two-sided disagreement; selection already handles strict
    improvement, and treating it as a merge would spend a slot re-deriving a
    candidate that already exists.
    """
    base = BehaviorSignature((True, False, False), ())
    donor = BehaviorSignature((True, True, True), ())
    overlap = complementarity(base, donor)
    assert overlap.donor_only == 2 and overlap.base_only == 0
    assert not overlap.is_complementary


def test_vectors_of_different_length_are_not_comparable():
    """Two candidates graded at different rungs describe different item sets."""
    assert (
        complementarity(
            BehaviorSignature((True, False), ()),
            BehaviorSignature((True, False, True), ()),
        )
        is None
    )


# -- planning -------------------------------------------------------------


def test_plans_a_merge_for_complementary_pair():
    island = IslandView(
        island_idx=0,
        candidates=[
            _cand("base", 0.9, "1101"),
            _cand("donor", 0.8, "1110"),
        ],
    )
    plan = StateMergePlanner(probability=1.0).plan(island, _AlwaysFires())
    assert plan is not None
    assert plan.base.id == "base"       # the fittest is the one to improve
    assert plan.donor.id == "donor"
    donors = plan.state_donors()
    assert [d["id"] for d in donors] == ["base", "donor"]
    assert donors[0]["weight"] + donors[1]["weight"] == pytest.approx(1.0)


def test_probability_roll_is_consumed_even_when_it_fails():
    """The roll happens before anything else, so enabling the planner shifts
    a seeded run's random stream in exactly one predictable place."""
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1101"), _cand("donor", 0.8, "1110")],
    )
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    assert StateMergePlanner(probability=0.0).plan(island, rng) is None
    assert rng.bit_generator.state != before


def test_no_merge_when_the_best_candidate_passes_everything():
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1111"), _cand("donor", 0.8, "1110")],
    )
    assert StateMergePlanner(probability=1.0).plan(island, _AlwaysFires()) is None


def test_no_merge_without_a_second_graded_candidate():
    island = IslandView(island_idx=0, candidates=[_cand("base", 0.9, "1101")])
    assert StateMergePlanner(probability=1.0).plan(island, _AlwaysFires()) is None


def test_weak_donors_are_refused():
    """A donor far below the base is noise, not knowledge."""
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1101"), _cand("donor", 0.1, "1110")],
    )
    planner = StateMergePlanner(probability=1.0, donor_fitness_floor=0.5)
    assert planner.plan(island, _AlwaysFires()) is None


def test_donor_solving_more_of_the_base_failures_wins():
    island = IslandView(
        island_idx=0,
        candidates=[
            _cand("base", 0.9, "1000"),
            _cand("small", 0.89, "1100"),   # fixes one, loses none
            _cand("big", 0.88, "0110"),     # fixes two, loses one
        ],
    )
    plan = StateMergePlanner(probability=1.0).plan(island, _AlwaysFires())
    assert plan is not None and plan.donor.id == "big"


def test_the_same_blend_is_never_bought_twice():
    """Re-buying a dead end is the failure the experience layer exists to
    stop; a planner that ignored its own history would reintroduce it."""
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1101"), _cand("donor", 0.8, "1110")],
    )
    planner = StateMergePlanner(probability=1.0, ratio_choices=(0.25, 0.75))
    first = planner.plan(island, _AlwaysFires())
    second = planner.plan(island, _AlwaysFires())
    third = planner.plan(island, _AlwaysFires())
    assert first is not None and second is not None
    assert first.ratio != second.ratio
    assert third is None            # both ratios exhausted


def test_attempted_blends_survive_a_checkpoint():
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1101"), _cand("donor", 0.8, "1110")],
    )
    planner = StateMergePlanner(probability=1.0, ratio_choices=(0.5,))
    assert planner.plan(island, _AlwaysFires()) is not None
    restored = StateMergePlanner(probability=1.0, ratio_choices=(0.5,))
    restored.set_state(planner.state())
    assert restored.plan(island, _AlwaysFires()) is None


def test_donor_ids_redirect_through_seed_copies():
    """A copy is a row that was never graded under its own id, so a domain
    keying state by id finds nothing there (SearchLoop redirects parents for
    exactly this reason)."""
    island = IslandView(
        island_idx=0,
        candidates=[
            _cand("base", 0.9, "1101"),
            _cand("copy", 0.8, "1110", metadata={"seed_copy_of": "original"}),
        ],
    )
    plan = StateMergePlanner(probability=1.0).plan(island, _AlwaysFires())
    assert plan is not None
    assert [d["id"] for d in plan.state_donors()] == ["base", "original"]


def test_unparseable_signature_is_ignored_not_fatal():
    island = IslandView(
        island_idx=0,
        candidates=[_cand("base", 0.9, "1101"), _cand("donor", 0.8, "1110")],
    )
    island.candidates[1].behavior_signature = "not-a-signature"
    assert StateMergePlanner(probability=1.0).plan(island, _AlwaysFires()) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probability": 1.5},
        {"ratio_choices": ()},
        {"ratio_choices": (0.0,)},
        {"ratio_choices": (1.0,)},
        {"min_gain": 0},
        {"donor_fitness_floor": 2.0},
    ],
)
def test_nonsense_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        StateMergePlanner(**kwargs)
