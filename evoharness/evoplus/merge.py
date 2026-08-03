# EvoHarness original research extension: population-level state merging.
# No upstream counterpart -- upstream recombines PROGRAM TEXT through the
# model; this recombines the trained state a candidate carries alongside its
# genome, without a model call.
"""Merging the carried state of complementary candidates.

The loop selects the single best candidate and lets the rest of the
population rot. But two candidates that fail DIFFERENT items each know
something the other does not, and for a domain whose candidates carry trained
state beside the genome, that knowledge can be combined directly -- no model
call, no retraining, seconds of compute.

Measured on the modular-arithmetic domain (2026-08-03): the run champion
missed one problem in the top tier and a sibling instance missed three, with
ZERO overlap between them. A blend of their weights solved all four, held at
a perfect score on the public benchmark and on five independently seeded test
sets, and beat both of its parents. Seven generations of search had not moved
that number; the merge took about twenty minutes of evaluation.

The split of responsibility is the point:

* THE FRAMEWORK decides WHO to merge. It needs no domain knowledge to do it,
  because it already records a per-item pass vector for every graded
  candidate (`BehaviorSignature`). Complementarity is visible there and
  nowhere else.
* THE DOMAIN decides HOW to merge, and whether it can at all, because only it
  knows what the carried state is. The plan reaches it as `state_donors` on
  `GradeContext`: an explicit list of candidate ids and blend weights. A
  domain that carries no state ignores the field and nothing changes.

Two properties of a merge child are deliberate and easy to get wrong:

* Its genome is byte-identical to the base's. The novelty gate must not see
  it as a duplicate proposal, because the mutation is not in the text.
* For the same reason, any content-addressed cache the domain keys on genome
  text will collide with the base recipe. The domain must not publish merged
  state into such a cache -- see the modular-arithmetic grader for what that
  costs if forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evoharness.evocore.population import Candidate, IslandView

from .feedback import BehaviorSignature


@dataclass(frozen=True)
class Complementarity:
    """How two candidates' per-item outcomes differ.

    `donor_only` is the entire reason to merge: items the donor solves and
    the base does not. `base_only` is what the merge puts at risk.
    """

    both_pass: int
    base_only: int
    donor_only: int
    both_fail: int

    @property
    def is_complementary(self) -> bool:
        """True when neither candidate dominates the other.

        A donor with `base_only == 0` solves everything the base does and
        more; that is not a merge, it is a better candidate, and selection
        already handles it.
        """
        return self.base_only > 0 and self.donor_only > 0


def complementarity(
    base: BehaviorSignature, donor: BehaviorSignature
) -> Complementarity | None:
    """Compare two pass vectors, or None when they are not comparable.

    Vectors of different lengths describe different item sets -- typically
    two candidates graded at different rungs -- and counting agreement
    between them would be meaningless.
    """
    if len(base.pass_vector) != len(donor.pass_vector):
        return None
    if not base.pass_vector:
        return None
    both_pass = base_only = donor_only = both_fail = 0
    for b, d in zip(base.pass_vector, donor.pass_vector):
        if b and d:
            both_pass += 1
        elif b:
            base_only += 1
        elif d:
            donor_only += 1
        else:
            both_fail += 1
    return Complementarity(both_pass, base_only, donor_only, both_fail)


def _state_id(cand: Candidate) -> str:
    """The id a domain's per-candidate state is actually filed under."""
    return str(cand.metadata.get("seed_copy_of") or cand.id)


@dataclass(frozen=True)
class MergePlan:
    """One proposed blend: the base, its donor, and the mixing ratio."""

    base: Candidate
    donor: Candidate
    ratio: float
    overlap: Complementarity

    @property
    def key(self) -> tuple[str, str, float]:
        return (self.base.id, self.donor.id, round(self.ratio, 4))

    def state_donors(self) -> list[dict]:
        """The domain-facing contract: ids and weights that sum to one.

        Ordered base first. A domain that can only honour one source should
        take the heaviest entry rather than guessing.

        Ids are redirected through `seed_copy_of` for the same reason
        SearchLoop redirects a parent's: a copy is a database row that was
        never graded under its own id, so a domain keying state by id finds
        nothing there however much the original had banked.
        """
        return [
            {"id": _state_id(self.base), "weight": round(1.0 - self.ratio, 6)},
            {"id": _state_id(self.donor), "weight": round(self.ratio, 6)},
        ]

    def audit(self) -> dict:
        """What was believed at planning time, for reading back later."""
        return {
            "merge_ratio": round(self.ratio, 6),
            "merge_base": self.base.id,
            "merge_donor": self.donor.id,
            "merge_donor_only": self.overlap.donor_only,
            "merge_base_only": self.overlap.base_only,
            "merge_both_fail": self.overlap.both_fail,
        }

    def title(self) -> str:
        return (
            f"Blend {self.ratio:.2f} of {self.donor.id[:8]} into "
            f"{self.base.id[:8]}"
        )

    def summary(self) -> str:
        overlap = self.overlap
        return (
            f"State merge, no genome change. The donor solves "
            f"{overlap.donor_only} item(s) the base fails; the base solves "
            f"{overlap.base_only} the donor fails; {overlap.both_fail} defeat "
            f"both. Blended at weight {self.ratio:.2f} toward the donor."
        )


@dataclass
class StateMergePlanner:
    """Finds a complementary donor for the strongest candidate on an island.

    Deliberately conservative. It proposes nothing unless a genuinely
    two-sided disagreement exists, it never proposes the same blend twice,
    and it is off unless a recipe turns it on -- a merge costs an evaluation
    slot, and on a domain that carries no state it would waste every one of
    them.
    """

    probability: float = 0.15
    ratio_choices: tuple[float, ...] = (0.25, 0.5, 0.75)
    min_gain: int = 1
    donor_fitness_floor: float = 0.5
    max_donor_candidates: int = 32
    attempted: set[tuple[str, str, float]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not self.ratio_choices:
            raise ValueError("ratio_choices must be non-empty")
        for ratio in self.ratio_choices:
            if not 0.0 < ratio < 1.0:
                raise ValueError("each ratio must lie strictly in (0, 1)")
        if self.min_gain < 1:
            raise ValueError("min_gain must be at least 1")
        if not 0.0 <= self.donor_fitness_floor <= 1.0:
            raise ValueError("donor_fitness_floor must be between 0 and 1")

    # -- planning ---------------------------------------------------------

    def _signature(self, cand: Candidate) -> BehaviorSignature | None:
        if not cand.behavior_signature:
            return None
        try:
            return BehaviorSignature.decode(cand.behavior_signature)
        except (ValueError, IndexError):
            return None

    def plan(self, island: IslandView, rng) -> MergePlan | None:
        """Propose a blend for this island, or None to fall through.

        The roll happens first and unconditionally so that enabling the
        planner perturbs the random stream in one predictable place rather
        than depending on how the population happens to look.
        """
        if rng.random() >= self.probability:
            return None

        passed = [c for c in island.passed_candidates if c.report is not None]
        if len(passed) < 2:
            return None
        base = max(passed, key=lambda c: c.fitness)
        base_sig = self._signature(base)
        if base_sig is None or all(base_sig.pass_vector):
            # Nothing left to win on the items we can see.
            return None

        floor = base.fitness * self.donor_fitness_floor
        pool = sorted(
            (c for c in passed if c.id != base.id and c.fitness >= floor),
            key=lambda c: -c.fitness,
        )[: self.max_donor_candidates]

        scored: list[tuple[int, float, Candidate, Complementarity]] = []
        for cand in pool:
            sig = self._signature(cand)
            if sig is None:
                continue
            overlap = complementarity(base_sig, sig)
            if overlap is None or not overlap.is_complementary:
                continue
            if overlap.donor_only < self.min_gain:
                continue
            scored.append((overlap.donor_only, cand.fitness, cand, overlap))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], -item[1]))

        # Walk donors and ratios in preference order and take the first blend
        # this run has not already bought. Re-buying a dead end is the
        # failure mode the experience layer exists to stop; a planner that
        # ignores its own history would reintroduce it here.
        for _gain, _fitness, donor, overlap in scored:
            for ratio in self.ratio_choices:
                plan = MergePlan(base, donor, float(ratio), overlap)
                if plan.key not in self.attempted:
                    self.attempted.add(plan.key)
                    return plan
        return None

    # -- checkpointing ----------------------------------------------------

    def state(self) -> dict:
        return {"attempted": [list(k) for k in sorted(self.attempted)]}

    def set_state(self, state: dict) -> None:
        self.attempted = {
            (str(k[0]), str(k[1]), float(k[2]))
            for k in state.get("attempted", [])
            if len(k) == 3
        }
