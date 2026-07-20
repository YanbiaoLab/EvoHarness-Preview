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
