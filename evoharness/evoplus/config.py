# EvoHarness original: hyperparameters for the three research extensions.
# Note there are deliberately NO enable/disable switches here — which
# extensions are active is embodied by the chosen recipe file (recipes/),
# so an experiment group's definition is a file diff, not a flag set.
"""Hyperparameters for evoplus extensions (C1/C2/C3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlusConfig:
    # C1 structured feedback rendering
    feedback_top_k: int = 3
    feedback_examples_per_cat: int = 1
    # C2 behavioral novelty
    hamming_threshold: int = 0
    duplicate_penalty: float = 0.25
    # C3 experience store
    experience_interval: int = 5
    experience_max_bytes: int = 2048
    experience_top_m: int = 3
    experience_top_n: int = 2
    # Population-level state merging. Only meaningful for domains whose
    # candidates carry trained state; a recipe decides whether to mount it.
    merge_probability: float = 0.15
    merge_ratios: tuple[float, ...] = (0.25, 0.5, 0.75)
    merge_min_gain: int = 1
    merge_donor_fitness_floor: float = 0.5
    # Island revival. Patience is deliberately long: a lineage that looks
    # dead for a few generations is normal, and this project has already
    # seen a run's best program come out of one that looked dead.
    island_patience: int = 10
    island_relative_floor: float = 0.75
    island_min_generation: int = 5
    island_max_restarts: int = 2
