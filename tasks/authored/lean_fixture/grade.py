"""Score one Lean file by compiling it and reading Lean's own axiom report.

This is the P-0a grader for `todo/proof_vertical.md`. Four things it is built
to get right, each of which the plan calls load-bearing:

**`passed` means "it compiles", not "it is proved."** `Candidate.archive_eligible`
is literally `return self.passed` (`core/population.py`), and the seed is a
`sorry` skeleton. Define `passed` as "fully proved" and the seed itself is not
passed, so the island has no passed parent, so `_pick_island` skips it and the
run produces no proposals at all. Whether a proof is *finished* is a separate
question, carried by `fitness == 1.0` and the task's `criterion.solved_at`.

**Fitness is graded, because 0/1 gives search nothing to climb.** A file that
compiles with three `sorry`s left is measurably nearer than one that does not
compile. The gradient is deliberately soft and only ever orders the search --
the *fact* that a theorem is proved comes from Lean's axiom report and nothing
else. Inflating the soft score misdirects search; it cannot manufacture a proof.

**The proposition is pinned by Lean, not by us.** The seed's locked footer
restates the goal and applies `fixture_main` to it. Weakening the theorem makes
that footer fail to typecheck, so "prove something easier" is not a cheaper
route to a high score -- it is a compile error. We still check the footer is
textually present, because deleting it is the one attack typechecking cannot
catch by itself.

**A broken toolchain is not a wrong answer.** No `lean` on PATH, or the process
killed by a signal, means nothing was measured; that raises `InfraError` and the
candidate is dropped rather than scored zero. A candidate whose own proof runs
past the time limit is a different thing -- that IS a verdict about the
candidate, reported as `fault_kind="timeout"` (see NOTE below).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from evoharness.serve import InfraError

MAIN = "fixture.lean"
THEOREM = "fixture_main"

#: The axioms a finished proof may depend on. Same list as the official ETP
#: Stage-2 policy: classical logic is allowed, `sorryAx` is not, and neither is
#: anything `native_decide` drags in (`Lean.ofReduceBool`).
ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})

#: A short ASCII sentinel from the locked footer. Preflight looks for this one
#: line; the grader below checks the whole footer.
FOOTER_SENTINEL = "-- LOCKED FOOTER: do not edit or delete."

#: The footer, whitespace-normalized. Deleting it would leave a file that still
#: compiles and still prints axioms, while no longer pinning the proposition.
LOCKED_FOOTER = " ".join(
    """
    example : ∀ (a b c : Nat),
        (a + b) * c = a * c + b * c
        ∧ (a + b) + c = a + (b + c)
        ∧ a * 0 = 0 := fixture_main
    """.split()
)

#: Ceiling on one compile. The fixture itself checks in about a second, so
#: anything near this bound is a proof the candidate wrote, not this file.
TIMEOUT_S = float(os.environ.get("LEAN_FIXTURE_TIMEOUT_S", "120"))

#: Floor for "it compiles at all". Getting a file past the elaborator is real
#: progress over one that does not, and without a floor every early candidate
#: sits at exactly 0.0 and parent selection has nothing to rank.
_COMPILES_FLOOR = 0.10
#: How much of the range the closed-declaration ratio may earn. The remainder
#: below 1.0 is unreachable by design: only Lean's axiom report awards 1.0.
_CLOSED_SPAN = 0.60

_DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|example)\b",
    re.MULTILINE,
)
#: Lean emits both an untagged form (`: error: Type mismatch`) and a tagged one
#: (`: error(lean.unknownIdentifier): ...`) in the same version. Matching only
#: the first counts a file full of unknown identifiers as having zero errors.
_ERROR_RE = re.compile(r":\s*error(?:\([^)]*\))?:")
_AXIOMS_RE = re.compile(
    rf"'{re.escape(THEOREM)}' depends on axioms: \[([^\]]*)\]"
)
_NO_AXIOMS_RE = re.compile(
    rf"'{re.escape(THEOREM)}' does not depend on any axioms"
)


def _lean_binary() -> str:
    found = shutil.which("lean")
    if not found:
        # Nothing was measured. Scoring this zero would put a fabricated data
        # point in the population and look exactly like a model that failed.
        raise InfraError(
            "no `lean` on PATH; this task is scored by the Lean compiler and "
            "has no offline mode"
        )
    return found


def _toolchain_version(binary: str) -> str:
    try:
        done = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InfraError(f"`lean --version` failed: {exc}") from exc
    return done.stdout.strip().splitlines()[0] if done.stdout.strip() else ""


def _expected_toolchain() -> str:
    path = Path(__file__).resolve().parent / "lean-toolchain"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _toolchain_matches(declared: str, banner: str) -> bool:
    """Compare only the version, not the whole banner.

    `lean --version` prints the platform triple too, so matching the raw string
    would report a mismatch for the same toolchain on Linux and on macOS -- a
    false alarm on exactly the machine the long runs happen on.
    """

    if not declared or not banner:
        return False
    version = declared.rsplit(":v", 1)[-1] if ":v" in declared else declared
    return version in banner


def _compile(binary: str, path: Path) -> tuple[int, str]:
    """Run Lean over the candidate file; return (returncode, combined output).

    Raises `InfraError` only for faults on our side of the line: the process
    could not be started, or it was killed by a signal. A non-zero exit with
    diagnostics is a verdict about the candidate, not an infrastructure fault.
    """

    try:
        done = subprocess.run(
            [binary, path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return -1, "__timeout__"
    except OSError as exc:
        raise InfraError(f"could not run `lean`: {exc}") from exc
    if done.returncode < 0:
        raise InfraError(
            f"`lean` was killed by signal {-done.returncode}; nothing was "
            "measured"
        )
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _axioms(output: str) -> frozenset[str] | None:
    """The axiom set Lean reported for the main theorem, or None if it did not.

    None means the `#print axioms` command is gone from the file, which is a
    malformed submission rather than a wrong one.
    """

    if _NO_AXIOMS_RE.search(output):
        return frozenset()
    match = _AXIOMS_RE.search(output)
    if not match:
        return None
    return frozenset(
        name.strip() for name in match.group(1).split(",") if name.strip()
    )


def _closed_ratio(source: str, output: str) -> float:
    """Share of declarations Lean did NOT flag as using `sorry`.

    Deliberately crude, and gameable by declaring easy lemmas: this only orders
    the search (see the module docstring). The denominator counts declarations
    in the candidate's own region so the locked footer does not inflate it.
    """

    region = source
    if "EDIT-REGION-BEGIN" in source and "EDIT-REGION-END" in source:
        region = source.split("EDIT-REGION-BEGIN", 1)[1].split(
            "EDIT-REGION-END", 1
        )[0]
    total = len(_DECL_RE.findall(region))
    if total == 0:
        return 0.0
    with_sorry = output.count("declaration uses")
    return max(0.0, (total - min(with_sorry, total)) / total)


def _fail(reason: str, kind: str, metrics: dict, notes: str = "") -> dict:
    return {
        "fitness": 0.0,
        "passed": False,
        "fault_kind": kind,
        "fault": reason,
        "visible_metrics": metrics,
        "notes": notes[:4000],
        "n_units": 1,
        "trustworthy_units": 1,
    }


def grade(candidate_dir, ctx):
    candidate_dir = Path(candidate_dir)
    path = candidate_dir / MAIN

    binary = _lean_binary()
    actual_toolchain = _toolchain_version(binary)
    expected = _expected_toolchain()
    metrics: dict = {
        "lean_version": actual_toolchain,
        # Recorded rather than enforced. The toolchain file lives in the task
        # directory, so changing it already changes the task hash -- but a
        # machine whose `lean` differs from the declared one would otherwise
        # produce results under a task identity that does not describe them.
        "toolchain_declared": expected,
        "toolchain_match": int(_toolchain_matches(expected, actual_toolchain)),
    }

    if not path.is_file():
        return _fail(
            f"no {MAIN} in the workspace", "invalid_candidate", metrics
        )
    source = path.read_text(encoding="utf-8")
    if " ".join(source.split()).find(LOCKED_FOOTER) < 0:
        # Typechecking pins the proposition only while the footer is there.
        return _fail(
            "the locked footer is missing or altered",
            "invalid_candidate",
            metrics,
            "Restore the LOCKED FOOTER exactly as it appears in the seed. It "
            "is what pins the proposition; without it a weaker theorem would "
            "compile.",
        )

    started = time.monotonic()
    returncode, output = _compile(binary, path)
    metrics["compile_elapsed_s"] = round(time.monotonic() - started, 2)

    if output == "__timeout__":
        # NOTE: a verdict, not an infrastructure fault. The fixture checks in
        # about a second, so reaching a 120s ceiling means the candidate wrote
        # something expensive -- `decide` over a large type, a runaway
        # `simp`. `timeout` is a VERDICT fault in `evaluation/faults.py`: it is
        # recorded and repairable but never becomes a parent. Faults on OUR
        # side of the line (no binary, killed by a signal) raise `InfraError`
        # from `_compile` instead and are dropped without a verdict.
        metrics["timed_out"] = 1
        return _fail(
            f"compile exceeded {TIMEOUT_S:g}s", "timeout", metrics,
            "The proof did not finish checking in the time allowed. This is "
            "usually a tactic that searches too widely, not a missing idea.",
        )

    # The exit code is authoritative: Lean returns non-zero for any error. The
    # error count is informational, and is derived separately because the two
    # can disagree only if our pattern is wrong -- which is worth seeing rather
    # than letting a bad pattern quietly decide whether a file compiled.
    metrics["compile_ok"] = int(returncode == 0)
    metrics["error_count"] = len(_ERROR_RE.findall(output))

    if not metrics["compile_ok"]:
        return _fail(
            "the file does not compile", "task_failure", metrics, output
        )

    axioms = _axioms(output)
    if axioms is None:
        return _fail(
            f"no `#print axioms {THEOREM}` output", "invalid_candidate",
            metrics,
            f"The seed ends with `#print axioms {THEOREM}`. That line is how "
            "this task learns whether the theorem is actually proved; keep it.",
        )
    metrics["axioms"] = ",".join(sorted(axioms))

    forbidden = axioms - ALLOWED_AXIOMS - {"sorryAx"}
    if forbidden:
        # `native_decide` and friends. Compiles, prints, and is not a proof
        # under this task's policy.
        return _fail(
            f"forbidden axioms: {', '.join(sorted(forbidden))}",
            "task_failure", metrics,
            "Only propext, Quot.sound and Classical.choice are permitted.",
        )

    if "sorryAx" not in axioms:
        return {
            "fitness": 1.0,
            "passed": True,
            "visible_metrics": {**metrics, "proved": 1},
            "notes": f"Lean reports {THEOREM} depends only on allowed axioms.",
            "n_units": 1,
            "trustworthy_units": 1,
        }

    # Compiles, still leans on `sorry`. Partial credit, and `passed=True` so it
    # can be a parent -- see the module docstring on why that matters.
    ratio = _closed_ratio(source, output)
    return {
        "fitness": _COMPILES_FLOOR + _CLOSED_SPAN * ratio,
        "passed": True,
        "visible_metrics": {
            **metrics,
            "proved": 0,
            "closed_ratio": round(ratio, 3),
            "sorry_decls": output.count("declaration uses"),
        },
        "notes": (
            f"{THEOREM} still depends on sorryAx. Remaining diagnostics:\n"
            + output
        )[:4000],
        "n_units": 1,
        "trustworthy_units": 1,
    }
