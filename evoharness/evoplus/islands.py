# EvoHarness original research extension: island health and soft restart.
# Upstream ShinkaEvolve has dynamic island spawning with stagnation detection;
# it was deliberately not ported (see population.py's porting note). This is
# the narrower, softer version the porting note was waiting for.
"""Detecting an island that has stopped contributing, and reviving it.

Islands exist to keep several lines of attack alive at once. When one of them
dies the loop does not notice: it keeps drawing parents from the dead island
in strict rotation, and every draw is a wasted evaluation slot.

Measured on the modular-arithmetic run r15: the third island produced a median
fitness of 0.23-0.29 for THIRTY consecutive generations while the other two
reached 0.95. Roughly a third of a three-day run was spent breeding from a
lineage that never once contributed a candidate worth keeping, and nothing in
the loop was able to say so.

The intervention is deliberately soft, for a reason this project has already
paid for once. A strict-improvement gate would have deleted the regression
that later produced one run's best program -- the breakthrough came out of a
lineage that looked dead at the time. So a restart here:

* never deletes, disables or rewrites anything the island already holds;
  every native candidate stays selectable, and a late bloomer can still bloom;
* only INJECTS a copy of a strong candidate from a healthy island, giving
  parent selection something else to draw from;
* demands a long patience window AND clear underperformance, not either one;
* is capped per island, so a genuinely hard island is not repeatedly paved
  over by whichever lineage happens to be ahead.

In other words it treats a dead island as under-supplied rather than wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evoharness.evocore.population import Candidate, PopulationStore


@dataclass(frozen=True)
class IslandRestart:
    """One injection, recorded so a run can be audited after the fact."""

    generation: int
    island_idx: int
    donor_island: int
    donor_id: str
    injected_id: str
    island_best: float
    global_best: float
    stalled_for: int


@dataclass(frozen=True)
class IslandHealth:
    """What the monitor believes about one island right now."""

    island_idx: int
    best_fitness: float
    last_improved_generation: int
    stalled_for: int
    restarts: int
    stalled: bool


@dataclass
class IslandHealthMonitor:
    """Implements LoopObserver; also called once at the end of a generation.

    Off unless a recipe constructs it. With a single island it does nothing,
    since there is nowhere to donate from.
    """

    patience: int = 10
    relative_floor: float = 0.75
    min_generation: int = 5
    max_restarts_per_island: int = 2
    best: dict[int, float] = field(default_factory=dict)
    improved_at: dict[int, int] = field(default_factory=dict)
    restarts: dict[int, int] = field(default_factory=dict)
    history: list[IslandRestart] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if not 0.0 <= self.relative_floor <= 1.0:
            raise ValueError("relative_floor must be between 0 and 1")
        if self.min_generation < 0:
            raise ValueError("min_generation must be nonnegative")
        if self.max_restarts_per_island < 0:
            raise ValueError("max_restarts_per_island must be nonnegative")

    # -- observation ------------------------------------------------------

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        """Record an island's high-water mark and when it was last raised.

        Only passing candidates count. A failing candidate carries no
        evidence that the island is alive, and letting failures reset the
        clock is exactly how a dead island stays invisible.
        """
        if not cand.passed:
            return
        island = cand.island_idx
        previous = self.best.get(island)
        if previous is None or cand.fitness > previous:
            self.best[island] = cand.fitness
            self.improved_at[island] = cand.generation

    # -- diagnosis --------------------------------------------------------

    def survey(self, generation: int, num_islands: int) -> list[IslandHealth]:
        """Per-island verdict. Pure; useful for metrics and for tests."""
        global_best = max(self.best.values(), default=0.0)
        report: list[IslandHealth] = []
        for idx in range(num_islands):
            best = self.best.get(idx, 0.0)
            improved = self.improved_at.get(idx, 0)
            stalled_for = generation - improved
            report.append(
                IslandHealth(
                    island_idx=idx,
                    best_fitness=best,
                    last_improved_generation=improved,
                    stalled_for=stalled_for,
                    restarts=self.restarts.get(idx, 0),
                    stalled=self._is_stalled(
                        idx, best, stalled_for, global_best, generation
                    ),
                )
            )
        return report

    def _is_stalled(
        self,
        island_idx: int,
        best: float,
        stalled_for: int,
        global_best: float,
        generation: int,
    ) -> bool:
        if generation < self.min_generation:
            return False
        if self.restarts.get(island_idx, 0) >= self.max_restarts_per_island:
            return False
        if stalled_for < self.patience:
            return False
        if global_best <= 0.0:
            return False
        # Holding the record is proof of life whatever the clock says.
        if best >= global_best:
            return False
        return best < global_best * self.relative_floor

    # -- intervention -----------------------------------------------------

    def maybe_restart(
        self, store: PopulationStore, generation: int, num_islands: int
    ) -> list[IslandRestart]:
        """Inject a healthy migrant into every island that has gone quiet."""
        if num_islands < 2 or self.max_restarts_per_island == 0:
            return []
        done: list[IslandRestart] = []
        for health in self.survey(generation, num_islands):
            if not health.stalled:
                continue
            donor = self._pick_donor(store, health.island_idx, num_islands)
            if donor is None:
                continue
            injected = self._inject(store, donor, health.island_idx)
            self.restarts[health.island_idx] = (
                self.restarts.get(health.island_idx, 0) + 1
            )
            # The clock restarts from here: the island has just been given
            # something new, and judging it on how long the OLD lineage was
            # quiet would fire again on the very next generation.
            self.improved_at[health.island_idx] = generation
            event = IslandRestart(
                generation=generation,
                island_idx=health.island_idx,
                donor_island=donor.island_idx,
                donor_id=donor.id,
                injected_id=injected.id,
                island_best=health.best_fitness,
                global_best=max(self.best.values(), default=0.0),
                stalled_for=health.stalled_for,
            )
            self.history.append(event)
            done.append(event)
        return done

    @staticmethod
    def _pick_donor(
        store: PopulationStore, island_idx: int, num_islands: int
    ) -> Candidate | None:
        best: Candidate | None = None
        for idx in range(num_islands):
            if idx == island_idx:
                continue
            for cand in store.island_view(idx).passed_candidates:
                if best is None or cand.fitness > best.fitness:
                    best = cand
        return best

    @staticmethod
    def _inject(
        store: PopulationStore, donor: Candidate, island_idx: int
    ) -> Candidate:
        """Copy a donor into the stalled island as a fresh, parentless entry.

        `seed_copy_of` is what lets a domain that carries per-candidate state
        find the donor's: a copy has its own id and nothing was ever published
        under it, so without the redirect every child of this injection would
        cold-start. SearchLoop reads exactly this key.
        """
        copy = Candidate(
            id=Candidate.new_id(),
            code=donor.code,
            generation=donor.generation,
            parent_id=None,
            island_idx=island_idx,
            operator="seed",
            workspace_kind=donor.workspace_kind,
            change_title=donor.change_title,
            change_summary=donor.change_summary,
            model_name=donor.model_name,
            report=donor.report,
            embedding=donor.embedding,
            behavior_signature=donor.behavior_signature,
            metadata={
                **donor.metadata,
                "seed_copy_of": donor.metadata.get("seed_copy_of", donor.id),
                "island_reseed_from": donor.id,
            },
        )
        store.insert(copy)
        return copy

    # -- checkpointing ----------------------------------------------------

    def state(self) -> dict:
        return {
            "best": {str(k): v for k, v in self.best.items()},
            "improved_at": {str(k): v for k, v in self.improved_at.items()},
            "restarts": {str(k): v for k, v in self.restarts.items()},
        }

    def set_state(self, state: dict) -> None:
        self.best = {
            int(k): float(v) for k, v in state.get("best", {}).items()
        }
        self.improved_at = {
            int(k): int(v) for k, v in state.get("improved_at", {}).items()
        }
        self.restarts = {
            int(k): int(v) for k, v in state.get("restarts", {}).items()
        }
