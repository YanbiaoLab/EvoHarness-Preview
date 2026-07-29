"""Experiment recipes: one thin assembly module per group (verl-style
algorithm zoo). `diff recipes/e1_feedback.py recipes/e0_vanilla.py` IS the
ablation definition."""

from . import (
    b0_manual,
    e0_vanilla,
    e1_feedback,
    e2_behavior,
    e3g_global,
    e3r_retrieval,
    e4a_rejected,
    e5s_full,
)
from .common import RecipeContext, TaskBundle, assemble
from .config_io import load_experiment_config

_MODULES = [
    b0_manual,
    e0_vanilla,
    e1_feedback,
    e2_behavior,
    e3g_global,
    e3r_retrieval,
    e4a_rejected,
    e5s_full,
]

REGISTRY = {m.NAME: m for m in _MODULES}


def get_recipe(name: str):
    try:
        return REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown recipe {name!r}; available: {sorted(REGISTRY)}"
        ) from None


def list_recipes() -> dict[str, str]:
    return {name: mod.DESCRIPTION for name, mod in sorted(REGISTRY.items())}


__all__ = [
    "REGISTRY",
    "RecipeContext",
    "TaskBundle",
    "assemble",
    "get_recipe",
    "list_recipes",
    "load_experiment_config",
]
