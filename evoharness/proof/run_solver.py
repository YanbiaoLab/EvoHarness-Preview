"""Attacking one subgoal with one `api.run()`.

The stub proved the graph works. This is the same seam wired to the real
thing: every subgoal becomes a task with its own seed, its own run directory,
its own manifest and evidence, and its outcome is derived from that run's
report rather than decided here.

Each subgoal is a dynamically minted TaskSpec, and that is a feature. Its hash
covers the lemma's own statement, so the evidence for "this lemma was proved"
stands on its own and can be audited without the graph. The cost is one run
directory per node, so directories are keyed by goal: a graph of a hundred
nodes is otherwise a hundred anonymous ones. The durable index from a goal to
the directories it was attacked in is the store's `attempts.run_dir` column.

`search_profile_factory` is where P-2's ladder plugs in. L1 and L2 are
`BasicSearchProfile` with different proposal modes; L3 and L4 are
`EvolutionSearchProfile`. Nothing else in this layer changes when the rung
changes, which is what makes the ladder an experiment rather than a rewrite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .graph import Goal, Outcome
from .solver import AttemptResult

SEED_MAIN = "subgoal.lean"

_ATTEMPT_PREFIX = "attempt_"


def claim_attempt_dir(work_root: Path, goal_id: str) -> Path:
    """Create and return an unused attempt directory for one goal.

    Keyed by goal, and claimed by creating it rather than by counting. One
    attack is one process, so a counter held in a solver would start over for
    the next one and hand it a directory another goal is already recorded
    against: `api.run` finds a foreign checkpoint and refuses, and the earlier
    run's artifacts are overwritten while the graph still points at them.

    Creation is exclusive because scanning alone leaves a gap — two attacks on
    one goal read the same highest index — so the loser takes the next index
    instead of sharing a directory.

    :param work_root: directory holding one subdirectory per attacked goal.
    :param goal_id: the goal these attempts belong to.
    :returns: the freshly created, empty attempt directory.
    """
    parent = Path(work_root) / goal_id
    parent.mkdir(parents=True, exist_ok=True)
    used = [
        int(name)
        for child in parent.iterdir()
        if child.is_dir() and child.name.startswith(_ATTEMPT_PREFIX)
        and (name := child.name[len(_ATTEMPT_PREFIX):]).isdigit()
    ]
    index = max(used, default=-1) + 1
    while True:
        run_dir = parent / f"{_ATTEMPT_PREFIX}{index:04d}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            index += 1
        else:
            return run_dir

TASK_PROMPT = """\
Prove this Lean 4 lemma. Replace the `sorry` in `{main}`, between the
EDIT-REGION markers, with a real proof. Keep both markers and keep the
statement exactly as it is.

    {signature}

It is scored by the compiler and by Lean's own axiom report. Only propext,
Quot.sound and Classical.choice are permitted, so `sorry` and anything
`native_decide` pulls in are not proofs here, whatever the file looks like.
"""

_OPEN = "([{⦃"
_CLOSE = ")]}⦄"


@dataclass
class ApiRunSolver:
    """One `api.run()` per subgoal, against a task built for that lemma."""

    work_root: Path
    #: (output_dir: Path) -> RunSpec. Called per attempt so each run gets its
    #: own directory, manifest and budget file.
    run_spec_factory: Callable[[Path], object]
    #: () -> SearchProfile. The rung of P-2's ladder.
    search_profile_factory: Callable[[], object]
    #: The grade function subgoal tasks are scored by: compile, check axioms,
    #: and `passed` means "it compiles" (see the P-0a fixture grader for why).
    grade_func: Callable[..., object]
    transport: object = None
    preamble: str = ""
    level: str = "L2"

    def attack(self, goal: Goal, *, budget: float) -> AttemptResult:
        from evoharness import ResolvedTask, api

        run_dir = claim_attempt_dir(Path(self.work_root), goal.id)
        seed_dir = run_dir / "seed"
        seed_dir.mkdir()
        (seed_dir / SEED_MAIN).write_text(
            seed_text(goal, self.preamble), encoding="utf-8"
        )

        task = ResolvedTask.from_directory(
            seed_dir,
            self.grade_func,
            task_id=f"proof_subgoal_{goal.identity[:24]}",
            version="v1",
            main_file=SEED_MAIN,
            domain_prompt=TASK_PROMPT.format(
                main=SEED_MAIN, signature=goal.statement
            ),
        )

        try:
            report = api.run(
                task,
                self.run_spec_factory(run_dir / "run"),
                self.search_profile_factory(),
                transport=self.transport,
            )
        except Exception as exc:  # noqa: BLE001 - classified, not swallowed
            # A run that could not be driven says nothing about the goal. Same
            # line the whole layer draws: no verdict, so no push toward
            # exhaustion and no hunt for a different decomposition.
            return AttemptResult(
                Outcome.INFRA_FAILED,
                run_dir=str(run_dir),
                note=f"{type(exc).__name__}: {exc}"[:500],
            )

        return AttemptResult.from_run_report(
            report,
            run_dir=str(run_dir),
            solved_at=1.0,
            proof_text=best_proof_body(run_dir / "run", report),
        )


def seed_text(goal: Goal, preamble: str = "") -> str:
    """A COMPILING skeleton: the lemma with a `sorry` body.

    It has to compile, not merely exist. `passed` means "it compiles", so a
    seed that does not is not a passed candidate; `SeedOnlySelector` then finds
    no parent, `_pick_island` skips the island, and the run emits no proposals
    at all -- a failure that looks exactly like a model that could not solve it.
    """

    head = f"{preamble}\n\n" if preamble else ""
    return (
        f"{head}-- EDIT-REGION-BEGIN\n"
        f"{goal.statement} := by\n  sorry\n"
        f"-- EDIT-REGION-END\n\n"
        f"#print axioms {declaration_name(goal.statement)}\n"
    )


def declaration_name(signature: str) -> str:
    body = signature.strip()
    for keyword in ("theorem", "lemma", "example", "def"):
        if body.startswith(keyword + " "):
            body = body[len(keyword) + 1:].lstrip()
            break
    return body.split(None, 1)[0] if body else "unknown"


def proof_body(candidate_text: str) -> str | None:
    """The lemma's proof body -- what assembly splices in after `:=`.

    NOT the whole file. `render` re-emits the signature itself, so returning
    the file would produce a declaration with two signatures. Everything
    between the edit markers, after the top-level `:=`, is the body.
    """

    region = candidate_text
    if "EDIT-REGION-BEGIN" in region and "EDIT-REGION-END" in region:
        region = region.split("EDIT-REGION-BEGIN", 1)[1].split(
            "EDIT-REGION-END", 1
        )[0]
    depth = 0
    for index in range(len(region) - 1):
        char = region[index]
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif char == ":" and region[index + 1] == "=" and depth == 0:
            return _trim_dangling_comment(region[index + 2:]) or None
    return None


def _trim_dangling_comment(body: str) -> str:
    """Drop the comment opener the END marker leaves behind.

    Splitting on the marker TEXT keeps whatever prefixed it, so a Lean file
    ends the body with a stray `--`. Harmless to the compiler and pure noise in
    the assembled proof, which is the artifact a human reads.
    """

    lines = body.strip("\n").rstrip().splitlines()
    while lines and lines[-1].strip() in {"--", "//", "#", "/-", "(*"}:
        lines.pop()
    return "\n".join(lines).rstrip()


def best_proof_body(run_dir: Path, report) -> str | None:
    """Read the winning candidate back out of the run's own database.

    Read rather than reconstructed: the graph has to carry the exact text Lean
    accepted, because assembly compiles it again and any gap between "what we
    proved" and "what we stored" turns up there as a mystery.
    """

    if report.best_fitness is None or report.best_fitness < 1.0:
        return None
    database = Path(run_dir) / "run.db"
    if not database.is_file():
        return None

    from evoharness.core.config import PopulationConfig
    from evoharness.core.population import PopulationStore

    store = PopulationStore(PopulationConfig(), database)
    try:
        best = store.best()
        if best is None:
            return None
        return proof_body(best.workspace.main_text())
    finally:
        store.close()


__all__ = [
    "SEED_MAIN",
    "TASK_PROMPT",
    "ApiRunSolver",
    "best_proof_body",
    "declaration_name",
    "proof_body",
    "seed_text",
]
