# EvoHarness original (hydra-lite, verl-inspired): one YAML tree with
# sections {search, population, plus, proposal} plus CLI dot-path overrides like
# `search.seed=2`. Deliberately NOT Hydra — no plugin system needed at this
# project's scale; unknown keys are hard errors (typo guard).
"""Experiment config loading with dot-path overrides."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from evoharness.evocore import PopulationConfig, ProposalConfig, SearchConfig
from evoharness.evoplus.config import PlusConfig

_SECTIONS = {
    "search": SearchConfig,
    "population": PopulationConfig,
    "proposal": ProposalConfig,
    "plus": PlusConfig,
}


def _build_section(cls, values: dict):
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} keys: {sorted(unknown)} "
            f"(valid: {sorted(valid)})"
        )
    return cls(**values)


def _parse_value(raw: str):
    """CLI override values: try JSON (numbers/bools/lists), else string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def load_experiment_config(
    path: Path | str | None = None,
    overrides: list[str] | tuple[str, ...] = (),
) -> tuple[SearchConfig, PopulationConfig, PlusConfig, ProposalConfig]:
    """Load {search, population, plus, proposal}, then apply
    `section.key=value` overrides. Everything omitted takes the dataclass
    defaults (which are upstream-aligned for parity)."""
    tree: dict = {}
    if path is not None:
        text = Path(path).read_text()
        if str(path).endswith(".json"):
            tree = json.loads(text)
        else:
            import yaml

            tree = yaml.safe_load(text) or {}
    unknown_sections = set(tree) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"unknown config sections: {sorted(unknown_sections)} "
            f"(valid: {sorted(_SECTIONS)})"
        )

    values = {name: dict(tree.get(name, {})) for name in _SECTIONS}
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError(f"override {override!r} is not of form a.b=value")
        section, sep, field_name = key.partition(".")
        if not sep or section not in _SECTIONS:
            raise ValueError(
                f"override key {key!r} must start with one of {sorted(_SECTIONS)}"
            )
        values[section][field_name] = _parse_value(raw)

    return (
        _build_section(SearchConfig, values["search"]),
        _build_section(PopulationConfig, values["population"]),
        _build_section(PlusConfig, values["plus"]),
        _build_section(ProposalConfig, values["proposal"]),
    )
