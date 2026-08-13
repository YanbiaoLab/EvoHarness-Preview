# EvoHarness original research extension (plan C2: behavioral-signature
# novelty). Complements the pre-evaluation embedding NoveltyGate: signatures
# only exist after grading, so this policy acts on archive admission and
# parent selection, not on proposal rejection.
"""C2: behavioral novelty via pass/fail signatures."""

from __future__ import annotations

from dataclasses import dataclass

from evoharness.core.population import Candidate, PopulationStore

from .feedback import BehaviorSignature, StructuredFeedback


class SignatureRecorder:
    """Implements LoopObserver. Always-on (all experiment groups): derives
    the behavior signature from structured feedback and stores its encoding,
    so E0/E1 populations remain comparable in post-hoc analysis. Must be
    registered BEFORE BehavioralNoveltyPolicy in the observer list."""

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        report = cand.report
        if report is None or not report.structured_feedback:
            return
        feedback = StructuredFeedback.from_json(report.structured_feedback)
        if feedback.items:
            cand.behavior_signature = feedback.signature().encode()


@dataclass
class CoverageStats:
    distinct_signatures: int
    duplicates_marked: int


class BehavioralNoveltyPolicy:
    """Implements LoopObserver + SamplingWeightPolicy (experiment group E2+).

    A candidate whose signature is within hamming_threshold of any existing
    signature on its island is marked behavior_duplicate: it is excluded from
    the archive (PopulationStore.refresh_archive) and its parent-selection
    weight is multiplied by duplicate_penalty."""

    def __init__(
        self, hamming_threshold: int = 0, duplicate_penalty: float = 0.25
    ):
        if not 0.0 <= duplicate_penalty <= 1.0:
            raise ValueError("duplicate_penalty must be in [0, 1]")
        self.hamming_threshold = hamming_threshold
        self.duplicate_penalty = duplicate_penalty
        self._duplicates_marked = 0

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        if cand.behavior_signature is None:
            return
        sig = BehaviorSignature.decode(cand.behavior_signature)
        for encoded in store.all_signatures(cand.island_idx):
            existing = BehaviorSignature.decode(encoded)
            if sig.hamming(existing) <= self.hamming_threshold:
                cand.behavior_duplicate = True
                self._duplicates_marked += 1
                return

    def weight_multiplier(self, cand: Candidate) -> float:
        return self.duplicate_penalty if cand.behavior_duplicate else 1.0

    def coverage_stats(self, store: PopulationStore) -> CoverageStats:
        return CoverageStats(
            distinct_signatures=len(set(store.all_signatures())),
            duplicates_marked=self._duplicates_marked,
        )


class RegressionSoftPenalty:
    """Implements LoopObserver + SamplingWeightPolicy (framework-evo backlog
    item 3: a SOFT validation gate).

    `children_count` already discounts a parent by how MANY children it has
    produced; nothing discounts it by how those children turned out. Live
    run 2026-07-24: one island re-selected the same parent for five
    consecutive generations while every child regressed, because it stayed
    the fittest candidate on the island.

    So: a parent whose recent children keep regressing is progressively
    down-weighted, and any single improving child clears the record. It is
    deliberately not a hard gate — the regressed candidate itself stays in
    the population and stays selectable. SkillOpt's strict-improvement gate
    would have deleted the g6 regression that later produced the run's best
    program; a penalty that decays but never reaches zero keeps that path
    open while stopping the loop from grinding on a dead parent.
    """

    def __init__(self, decay: float = 0.6, floor: float = 0.1):
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if not 0.0 < floor <= 1.0:
            raise ValueError("floor must be in (0, 1]")
        self.decay = decay
        self.floor = floor
        self.streaks: dict[str, int] = {}

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        if not cand.parent_id or cand.report is None:
            return
        parent = store.get(cand.parent_id)
        if parent is None or parent.report is None:
            return
        improved = cand.passed and cand.report.fitness > parent.report.fitness
        if improved:
            self.streaks.pop(cand.parent_id, None)
        else:
            self.streaks[cand.parent_id] = self.streaks.get(cand.parent_id, 0) + 1

    def weight_multiplier(self, cand: Candidate) -> float:
        streak = self.streaks.get(cand.id, 0)
        return max(self.floor, self.decay ** streak)

    def state(self) -> dict:
        return {"streaks": dict(self.streaks)}

    def set_state(self, state: dict) -> None:
        self.streaks = {str(k): int(v) for k, v in state.get("streaks", {}).items()}
