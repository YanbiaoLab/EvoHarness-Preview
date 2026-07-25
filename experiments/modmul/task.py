"""Framework-side task bundle for modmul (Round-1, multi-file genomes).

Consumer layer: this file IS the composition point for the task, so it may
import both the framework and the eval side.

The genome is a three-file workspace — model.py (inference contract, the
compliance-critical surface), arch.py (architecture), train.py (recipe) — so a
mutation can rewrite the architecture without re-emitting the contract.
Grading runs IN-PROCESS through WorkspaceGradeFnGrader: the heavy work already
happens in subprocesses inside grade_workspace, so eval_batch_size > 1 is safe.

Islands (heterogeneous seeding):
  0  limb_horner  — width-generic scan cell; the route to tiers 4+
  1  horner_cell  — fixed 16-bit cell; fast, honest tier-1..3 baseline
  2  serial_ar    — serial AR transformer; the diversity/control lineage

Env knobs:
  MODMUL_QUICK=1            laptop-scale rungs (see grade.QUICK_RUNGS)
  MODMUL_FORCE_ALL_RUNGS=1  skip promotion gates (baseline calibration)
"""

from __future__ import annotations

import os
from pathlib import Path

from evoharness import ScorableTask
from evoharness.evocore.workspace import GitWorkspace

_HERE = Path(__file__).resolve().parent
_SEEDS = _HERE / "seeds"
_GENOME_FILES = ("model.py", "arch.py", "train.py")

PRIMARY_SEED = "limb_horner"
EXTRA_SEEDS = ("horner_cell", "serial_ar")


def _workspace(name: str) -> GitWorkspace:
    return GitWorkspace.from_directory(
        _SEEDS / name,
        main_file="model.py",
        include_files=_GENOME_FILES,
    )


def make_task() -> ScorableTask:
    if os.environ.get("MODMUL_EVAL_URL"):
        raise RuntimeError(
            "MODMUL_EVAL_URL is set, but the modmul genome is now multi-file "
            "and eval protocol v1 only carries main_text() over the wire "
            "(see evocore/remote.py::_submit). Run the loop on the eval "
            "machine with the in-process grader, or unset the variable."
        )
    from modmul.grade import grade_workspace

    return ScorableTask.from_directory(
        _SEEDS / PRIMARY_SEED,
        grade_workspace,
        main_file="model.py",
        include_files=_GENOME_FILES,
        extra_seeds=[_workspace(name) for name in EXTRA_SEEDS],
        task_sys_msg=(_HERE / "sys_msg.md").read_text(),
        research_brief=(_HERE / "research_msg.md").read_text(),
    )
