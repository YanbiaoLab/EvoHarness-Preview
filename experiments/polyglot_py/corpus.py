"""The Polyglot Python exercise corpus: locate, parse, and split it.

HyperAgents evaluates the coding domain on Aider's polyglot-benchmark, one
Docker container per exercise across six languages. This box has neither a
reachable image registry nor the non-Python toolchains, so the corpus here is
the Python half of the same benchmark, run natively under pytest. The exercise
texts, stubs and tests are byte-identical to the originals; only the execution
substrate differs, so scores are comparable BETWEEN ARMS of this study and not
against the paper's 165-task Pass@1.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

_DOC_FILES = ("introduction.md", "instructions.md", "instructions.append.md")


@dataclass(frozen=True)
class Exercise:
    """One practice exercise: what the solver is told, and what judges it."""

    slug: str
    instructions: str
    stubs: dict[str, str]        # solution file name -> starting content
    tests: dict[str, str]        # test file name -> content
    reference: dict[str, str]    # example solution, never shown to the solver

    def as_task(self) -> dict:
        """The payload handed to the candidate solver."""

        return {
            "slug": self.slug,
            "instructions": self.instructions,
            "files": dict(self.stubs),
            "tests": dict(self.tests),
        }


def corpus_root() -> Path:
    raw = os.environ.get("POLYGLOT_ROOT")
    if not raw:
        raise RuntimeError(
            "POLYGLOT_ROOT is unset: point it at a clone of "
            "https://github.com/Aider-AI/polyglot-benchmark"
        )
    root = Path(raw)
    practice = root / "python" / "exercises" / "practice"
    if not practice.is_dir():
        raise RuntimeError(f"no python/exercises/practice under {root}")
    return practice


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _load_one(directory: Path) -> Exercise | None:
    config_path = directory / ".meta" / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    files = config.get("files") or {}

    stubs = {name: _read(directory / name) for name in files.get("solution", [])}
    tests = {name: _read(directory / name) for name in files.get("test", [])}
    reference = {name: _read(directory / name) for name in files.get("example", [])}
    if not stubs or not tests:
        return None
    # An exercise whose test file is missing from disk cannot judge anything.
    if any(not text.strip() for text in tests.values()):
        return None

    docs = directory / ".docs"
    instructions = "\n\n".join(
        text for text in (_read(docs / name) for name in _DOC_FILES) if text.strip()
    )
    if not instructions.strip():
        return None

    return Exercise(
        slug=directory.name,
        instructions=instructions,
        stubs=stubs,
        tests=tests,
        reference=reference,
    )


def load_exercises() -> list[Exercise]:
    """Every well-formed Python exercise, in a deterministic order."""

    practice = corpus_root()
    found = []
    for directory in sorted(practice.iterdir()):
        if not directory.is_dir():
            continue
        exercise = _load_one(directory)
        if exercise is not None:
            found.append(exercise)
    if not found:
        raise RuntimeError(f"no usable exercises under {practice}")
    return found


def split(
    exercises: list[Exercise], *, train_size: int, seed: int = 20260804
) -> tuple[list[Exercise], list[Exercise]]:
    """Fixed train/holdout split, identical for every arm of the study.

    The seed is a constant rather than the run's search seed on purpose: if
    each arm shuffled with its own seed, arms would be scored on different
    exercises and their fitnesses would not be comparable.
    """

    ordered = sorted(exercises, key=lambda item: item.slug)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    train = sorted(shuffled[:train_size], key=lambda item: item.slug)
    holdout = sorted(shuffled[train_size:], key=lambda item: item.slug)
    return train, holdout


def select(split_name: str, train_size: int) -> list[Exercise]:
    train, holdout = split(load_exercises(), train_size=train_size)
    if split_name == "train":
        return train
    if split_name == "holdout":
        return holdout
    raise ValueError(f"unknown split {split_name!r}")


__all__ = ["Exercise", "load_exercises", "split", "select", "corpus_root"]
