"""Scoring one subgoal: compile it, then read Lean's own axiom report.

Shared by the demo, the CLI and anything else that dispatches a lemma, because
a second copy of these rules would drift and both copies would keep answering.
The three that matter are the same ones the P-0a fixture grader spells out:

**`passed` means "it compiles", not "it is proved."** `Candidate.archive_eligible`
is literally `return self.passed`, and every seed is a `sorry` skeleton. Define
`passed` as "proved" and the seed is not passed, so `SeedOnlySelector` finds no
parent, `_pick_island` skips the island, and the run emits no proposals at all
-- which reads exactly like a model that could not solve the problem.

**Fitness is graded**, because 0/1 gives the search nothing to climb. It only
orders the search: 1.0 comes from the axiom report and from nothing else.

**A broken toolchain is not a wrong answer.** No Lean, or a process killed by a
signal, means nothing was measured, so it raises rather than scoring zero. A
proof the candidate wrote that runs past the time limit IS a verdict about the
candidate, and is reported as `timeout`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .run_solver import SEED_MAIN, declaration_name
from .sketch import ERROR_RE, LeanRunner, SketchUnavailable

ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})

#: Getting a file past the elaborator is real progress over one that does not.
#: Without a floor every early candidate sits at exactly 0.0 and parent
#: selection has nothing to rank.
_COMPILES_FLOOR = 0.30


def make_grader(runner: LeanRunner | None = None) -> Callable:
    """A grade function bound to one way of invoking Lean.

    Parameterized rather than hard-coded, because a bare `lean` cannot see
    Mathlib: its olean search path comes from the lake environment. A grader
    that assumed bare `lean` would fail every real benchmark problem with
    "unknown module Mathlib" and score it as the candidate's fault.
    """

    runner = runner or LeanRunner()

    def grade(candidate_dir, ctx):
        from evoharness.serve import InfraError

        path = Path(candidate_dir) / SEED_MAIN
        if not path.is_file():
            return {
                "fitness": 0.0,
                "passed": False,
                "fault_kind": "invalid_candidate",
                "fault": f"no {SEED_MAIN} in the workspace",
            }

        source = path.read_text(encoding="utf-8")
        name = _target_name(source)
        try:
            returncode, output = runner.compile(source)
        except SketchUnavailable as exc:
            # Nothing was measured. Scoring it zero would blame the candidate
            # for the toolchain and put a fabricated point in the population.
            raise InfraError(str(exc)) from exc

        metrics = {"declaration": name}
        if returncode != 0:
            errors = [
                line for line in output.splitlines() if ERROR_RE.search(line)
            ]
            return {
                "fitness": 0.0,
                "passed": False,
                "fault_kind": "task_failure",
                "fault": "the file does not compile",
                "notes": "\n".join(errors)[:2000] or output[:2000],
                "visible_metrics": {**metrics, "error_count": len(errors)},
            }

        axioms = _axioms(output, name)
        forbidden = axioms - ALLOWED_AXIOMS - {"sorryAx"}
        if forbidden:
            # `native_decide` and friends. Compiles, prints, and is not a proof
            # under this policy.
            return {
                "fitness": 0.0,
                "passed": False,
                "fault_kind": "task_failure",
                "fault": f"forbidden axioms: {', '.join(sorted(forbidden))}",
                "visible_metrics": {**metrics, "axioms": ",".join(sorted(axioms))},
            }

        if "sorryAx" in axioms:
            # Compiles, still leans on `sorry`. Partial credit, and passed=True
            # so it can be a parent.
            return {
                "fitness": _COMPILES_FLOOR,
                "passed": True,
                "notes": output[:2000],
                "visible_metrics": {**metrics, "proved": 0},
            }

        return {
            "fitness": 1.0,
            "passed": True,
            "notes": f"Lean reports {name} depends only on allowed axioms.",
            "visible_metrics": {
                **metrics,
                "proved": 1,
                "axioms": ",".join(sorted(axioms)),
            },
        }

    return grade


def _target_name(source: str) -> str:
    """The declaration the axiom report is about.

    Read from inside the edit region, so a candidate that appended helper
    lemmas after it does not move the target: the goal is the lemma the seed
    declared, not whatever happens to be last in the file.
    """

    region = source
    if "EDIT-REGION-BEGIN" in region:
        region = region.split("EDIT-REGION-BEGIN", 1)[1]
    return declaration_name(region.split(":=", 1)[0])


def _axioms(output: str, name: str) -> frozenset[str]:
    if re.search(rf"'{re.escape(name)}' does not depend on any axioms", output):
        return frozenset()
    match = re.search(
        rf"'{re.escape(name)}' depends on axioms: \[([^\]]*)\]", output
    )
    if not match:
        # No report at all. Treating that as "proved" would hand a pass to any
        # candidate that deleted the `#print axioms` line.
        return frozenset({"sorryAx"})
    return frozenset(
        item.strip() for item in match.group(1).split(",") if item.strip()
    )


__all__ = ["ALLOWED_AXIOMS", "make_grader"]
