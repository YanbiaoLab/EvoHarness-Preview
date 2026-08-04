# EvoHarness original research extension: complementarity-driven inspiration.
# The selector ranks by fitness, so the model is always shown "second place".
# But the most informative reference is the candidate that SOLVES ITEMS THE
# PARENT FAILS -- and the framework already knows who that is, because every
# graded candidate carries a per-item pass vector.
"""Choosing a reference program for what it knows, not where it ranks."""

from __future__ import annotations

from dataclasses import dataclass

from evoharness.evocore.population import Candidate

from .feedback import BehaviorSignature
from .merge import complementarity


@dataclass
class ComplementaryInspiration:
    """Implements InspirationPolicy.

    Unlike a state merge, complementarity here is ONE-sided: a donor that
    solves items the parent fails is worth reading even if it is strictly
    stronger (a merge must exclude that case; an inspiration must not).
    Deterministic and rng-free, so enabling it does not shift a seeded
    run's random stream at all.
    """

    min_gain: int = 1
    max_named_items: int = 5

    def __post_init__(self) -> None:
        if self.min_gain < 1:
            raise ValueError("min_gain must be at least 1")
        if self.max_named_items < 0:
            raise ValueError("max_named_items must be nonnegative")

    @staticmethod
    def _signature(cand: Candidate) -> BehaviorSignature | None:
        if not cand.behavior_signature:
            return None
        try:
            return BehaviorSignature.decode(cand.behavior_signature)
        except (ValueError, IndexError):
            return None

    def pick(
        self, parent: Candidate, pool: list[Candidate]
    ) -> tuple[Candidate, str] | None:
        parent_sig = self._signature(parent)
        if parent_sig is None or all(parent_sig.pass_vector):
            # Nothing to learn: no signature, or no failures to fix.
            return None
        best = None
        best_key = None
        for cand in pool:
            if cand.id == parent.id or not cand.passed:
                continue
            sig = self._signature(cand)
            if sig is None:
                continue
            # Different lengths (candidates graded at different rungs)
            # return None here and the candidate is simply skipped.
            overlap = complementarity(parent_sig, sig)
            if overlap is None or overlap.donor_only < self.min_gain:
                continue
            key = (overlap.donor_only, cand.fitness, cand.id)
            if best_key is None or key > best_key:
                best, best_key = (cand, sig, overlap), key
        if best is None:
            return None
        cand, sig, overlap = best
        return cand, self._note(parent, parent_sig, sig, overlap)

    def _note(self, parent, parent_sig, donor_sig, overlap) -> str:
        solved = [
            index
            for index, (p, d) in enumerate(
                zip(parent_sig.pass_vector, donor_sig.pass_vector)
            )
            if d and not p
        ]
        names = self._item_names(parent, solved[: self.max_named_items])
        named = f" ({', '.join(names)})" if names else ""
        note = f"Solves {overlap.donor_only} item(s) this parent fails{named}"
        if overlap.base_only:
            note += f"; fails {overlap.base_only} the parent solves"
        return note + "."

    @staticmethod
    def _item_names(parent: Candidate, indices: list[int]) -> list[str]:
        """Item ids from the parent's own per-item feedback, positionally.

        Signatures are comparable only when item order is fixed (see
        BehaviorSignature), so positions name the same items on both sides.
        """
        report = parent.report
        feedback = report.structured_feedback if report else None
        items = (feedback or {}).get("items") or []
        names = []
        for index in indices:
            if index < len(items) and items[index].get("item_id"):
                names.append(str(items[index]["item_id"]))
        return names
