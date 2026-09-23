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

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .graph import Goal, Outcome
from .solver import AttemptResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .verifier import VerifierBinding

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

#: The prompt when a verifier grades, built from the graph's policy rather than
#: fixed: the local prompt forbids native_decide because a local compile cannot
#: recheck it, and that stops being true when the verifier can.
VERIFIER_TASK_PROMPT = """\
Prove this Lean 4 lemma. Replace the `sorry` in `{main}`, between the
EDIT-REGION markers, with a real proof. Keep both markers and keep the
statement exactly as it is.

    {signature}

It is checked by an independent Lean verifier, which recompiles it and rechecks
every declaration in the kernel. {axioms} `sorry` is never a proof. {helpers}
"""

_AXIOM_TEXT = {
    "trusted": ("Only propext, Quot.sound and Classical.choice are permitted; "
                "native_decide is not."),
    "audited": ("propext, Quot.sound and Classical.choice are permitted, and so is "
                "native_decide on functions Lean already defines -- the verifier "
                "re-evaluates it, but cannot for functions you define here."),
    "claimed": "Any axiom is accepted, but the proof is labelled with what it rests on.",
}


def task_prompt(signature: str, *, minimum_trust: str | None = None,
                helpers_allowed: bool = False) -> str:
    """The prompt for one subgoal. `minimum_trust=None` is the local grader's."""

    if minimum_trust is None:
        return TASK_PROMPT.format(main=SEED_MAIN, signature=signature)
    name = declaration_name(signature)
    helpers = (
        f"Helper lemmas are allowed only if named under this one, e.g. `{name}.step`."
        if helpers_allowed else
        "Write the whole proof inside this one declaration: no other declarations."
    )
    return VERIFIER_TASK_PROMPT.format(
        main=SEED_MAIN, signature=signature, axioms=_AXIOM_TEXT[minimum_trust],
        helpers=helpers)


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
    #: Told (goal_id, run_dir) the moment a directory is claimed, before
    #: anything is written into it. The graph uses it to survive this process
    #: dying: an attempt that never returns is recorded by recovery, which
    #: otherwise has no way to learn where the work went. Optional -- the
    #: solver runs identically without it, it just leaves an orphan.
    on_attempt_dir: Callable[[str, str], None] | None = None
    #: When set, candidates are graded by the verifier and `grade_func` is not
    #: used: no local precheck, one judge. See `verifier.py`.
    verifier: "VerifierBinding | None" = None

    def attack(self, goal: Goal, *, budget: float) -> AttemptResult:
        from evoharness import ResolvedTask, api

        run_dir = claim_attempt_dir(Path(self.work_root), goal.id)
        if self.on_attempt_dir is not None:
            self.on_attempt_dir(goal.id, str(run_dir))

        grade_func = self.grade_func
        grader = None
        prompt = TASK_PROMPT.format(main=SEED_MAIN, signature=goal.statement)
        if self.verifier is not None:
            from .verifier import GoalUnresolvable, VerifierUnavailable

            try:
                contract = self.verifier.contract_for(goal)
            except VerifierUnavailable as exc:
                return AttemptResult(Outcome.INFRA_FAILED, run_dir=str(run_dir),
                                     note=f"the verifier did not answer: {exc}"[:500])
            except GoalUnresolvable as exc:
                # The statement itself does not elaborate in this environment:
                # nothing a candidate writes can fix that.
                return AttemptResult(Outcome.ENVIRONMENT_MISMATCH, run_dir=str(run_dir),
                                     note=f"the goal does not elaborate: {exc}"[:500])
            grader = self.verifier.grader(goal, contract, run_dir)
            grade_func = grader
            prompt = task_prompt(goal.statement,
                                 minimum_trust=self.verifier.policy.minimum_trust,
                                 helpers_allowed=self.verifier.allow_helpers)
            prompt += self.verifier.retrieve(goal, contract)

        seed_dir = run_dir / "seed"
        seed_dir.mkdir()
        (seed_dir / SEED_MAIN).write_text(
            seed_text(goal, self.preamble), encoding="utf-8"
        )

        task = ResolvedTask.from_directory(
            seed_dir,
            grade_func,
            task_id=f"proof_subgoal_{goal.identity[:24]}",
            version="v1",
            main_file=SEED_MAIN,
            domain_prompt=prompt,
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
            # exhaustion and no hunt for a different decomposition. Whatever
            # the verifier had already been asked is kept with it.
            return AttemptResult(
                Outcome.INFRA_FAILED,
                run_dir=str(run_dir),
                note=f"{type(exc).__name__}: {exc}"[:500],
                verifications=grader.ledger.verdicts() if grader else (),
                envelopes=grader.ledger.envelopes() if grader else (),
            )

        if grader is None:
            return AttemptResult.from_run_report(
                report,
                run_dir=str(run_dir),
                solved_at=1.0,
                proof_text=best_proof_body(run_dir / "run", report),
            )
        best = best_candidate_text(run_dir / "run", report)
        body, helpers = proof_parts(best, declaration_name(goal.statement)) if best else (None, ())
        return AttemptResult.from_run_report(
            report,
            run_dir=str(run_dir),
            solved_at=1.0,
            proof_text=body,
            auxiliary_declarations=helpers,
            verifications=grader.ledger.verdicts(),
            envelopes=grader.ledger.envelopes(),
            winner_envelope=grader.ledger.envelope_for_file(best) if best else None,
            verified_by_verifier=True,
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


#: A top-level declaration: a declaration keyword at column 0, after optional
#: attributes and modifiers. Top level means column 0 -- a nested `have` or a
#: `where` clause is indented -- so splitting on these never cuts a proof.
_TOP_DECLARATION = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance|structure|inductive|class|axiom|opaque|example)\b",
    re.M,
)


def proof_parts(candidate_text: str, root: str) -> tuple[str | None, tuple[str, ...]]:
    """(the goal's proof body, the helper declarations beside it).

    `proof_body` scans for the first top-level `:=`, which is right only when
    the goal's declaration is the only one: with a helper before it, the
    helper's body comes back as the goal's, and the goal's whole declaration
    trails behind as its tail. Here the edit region is cut into declarations
    first; the one named `root` gives the body, the rest are kept whole, in
    source order, to be emitted ahead of it when the proof is assembled.
    """

    region = candidate_text
    if "EDIT-REGION-BEGIN" in region and "EDIT-REGION-END" in region:
        region = region.split("EDIT-REGION-BEGIN", 1)[1].split("EDIT-REGION-END", 1)[0]
        # The BEGIN marker's line tail (after `-- EDIT-REGION-BEGIN`) is not a declaration.
        region = region.split("\n", 1)[1] if "\n" in region else ""
    starts = [m.start() for m in _TOP_DECLARATION.finditer(region)]
    if not starts:
        return None, ()
    pieces = [region[a:b] for a, b in zip(starts, [*starts[1:], len(region)])]
    body, helpers = None, []
    for piece in pieces:
        head = piece.split(":=", 1)[0]
        if declaration_name(head) == root and body is None:
            body = proof_body(piece)
        else:
            helpers.append(_trim_dangling_comment(piece))
    return body, tuple(helpers)


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

    text = best_candidate_text(run_dir, report)
    return proof_body(text) if text else None


def best_candidate_text(run_dir: Path, report) -> str | None:
    """The winning candidate's whole file, when the run reached 1.0."""

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
        return best.workspace.main_text()
    finally:
        store.close()


__all__ = [
    "SEED_MAIN",
    "TASK_PROMPT",
    "VERIFIER_TASK_PROMPT",
    "ApiRunSolver",
    "best_candidate_text",
    "best_proof_body",
    "declaration_name",
    "proof_body",
    "proof_parts",
    "seed_text",
    "task_prompt",
]
