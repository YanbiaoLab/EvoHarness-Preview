"""Final re-verification: put the proved lemmas back together and compile.

The graph saying a goal is PROVED is a claim about bookkeeping. This module is
where that claim is cashed.

Per-node green is not enough, and the reason is timing rather than logic. The
lemmas were proved in separate runs, minutes or hours apart. Between the first
and the last, the Mathlib version can move, a sibling lemma's signature can be
rewritten, a `sorry` can be filled in a way that typechecks alone and not in
company. Sketch validation catches a bad DECOMPOSITION at the moment it is
proposed; nothing catches DRIFT except compiling the finished article.

So the root's evidence is one thing and one thing only: the assembled file
compiles, and Lean reports it depends on no axiom outside the allowed set.

Assembly renders through `sketch.render` -- the same function that produced the
file the validator checked. The only difference is that the lemma bodies are
real proofs instead of `sorry`. Two assemblers would be free to disagree, and
their disagreement would be invisible until it was expensive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .graph import GoalStatus
from .sketch import ERROR_RE, SketchUnavailable, compile_lean, render

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ProofGraphStore

#: The axioms a finished proof may depend on: classical logic, and nothing
#: else. `sorryAx` is absent by construction, and so is whatever
#: `native_decide` drags in.
ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})

_NO_AXIOMS_RE = re.compile(r"'(\S+)' does not depend on any axioms")
_AXIOMS_RE = re.compile(r"'(\S+)' depends on axioms: \[([^\]]*)\]")


class AssemblyError(RuntimeError):
    """The graph cannot be turned into a file at all -- a wiring fault.

    Distinct from "the assembled file did not compile", which is a real
    finding about the proof. This one means a proved goal had no proof text,
    or an accepted decomposition had no sketch: states the rest of the layer
    is supposed to make impossible.
    """


@dataclass(frozen=True)
class AssemblyResult:
    ok: bool
    reason: str = ""
    text: str = ""
    axioms: frozenset[str] = field(default_factory=frozenset)

    @property
    def forbidden_axioms(self) -> frozenset[str]:
        return self.axioms - ALLOWED_AXIOMS


def assemble(store: "ProofGraphStore", goal_id: str) -> str:
    """Render the finished proof of one goal, recursively.

    Each subgoal contributes either the body of its own proved attempt or, if
    it was itself decomposed, its assembled text spliced in ahead of the
    parent. Depth-first so a lemma is always declared before it is used, which
    Lean requires.
    """

    goal = store.goal(goal_id)
    if goal.status is not GoalStatus.PROVED:
        raise AssemblyError(
            f"goal {goal_id} is {goal.status.value}, not proved"
        )

    direct = _direct_proof(store, goal_id)
    if direct is not None:
        # Closed without decomposing. `_direct_proof` yields the BODY, which is
        # what a subgoal contributes when spliced into a sketch; at the top of
        # the file it still needs its own declaration around it, or what comes
        # back is a bare term that compiles as nothing.
        return f"{goal.statement} := {direct}\n"

    completed = [
        decomposition
        for decomposition in store.decompositions_of(goal_id)
        if decomposition.status.value == "completed"
    ]
    if not completed:
        raise AssemblyError(
            f"goal {goal_id} is proved but has neither a proved attempt nor a "
            "completed decomposition"
        )
    decomposition = completed[0]
    sketch = store.sketch_of(decomposition.id)
    if sketch is None:
        raise AssemblyError(
            f"decomposition {decomposition.id} was accepted without a sketch; "
            "the graph knows which lemmas close the goal but not how"
        )

    bodies: dict[str, str] = {}
    preludes: list[str] = []
    for spec, subgoal_id in zip(sketch.subgoals, decomposition.subgoal_ids):
        sub_direct = _direct_proof(store, subgoal_id)
        if sub_direct is not None:
            bodies[spec.name] = sub_direct
            continue
        # The subgoal was itself decomposed. Its whole assembled text has to
        # appear before this file's declarations, and the lemma itself then
        # refers to the name that text defines.
        preludes.append(assemble(store, subgoal_id))
        bodies[spec.name] = spec.name + "_assembled"

    rendered = render(sketch, bodies=bodies)
    return "\n\n".join([*preludes, rendered.text])


def _direct_proof(store: "ProofGraphStore", goal_id: str) -> str | None:
    for attempt in store.attempts_of(goal_id):
        if attempt.outcome.value == "proved":
            if not attempt.proof_text:
                raise AssemblyError(
                    f"attempt {attempt.id} is proved but carries no proof text"
                )
            return attempt.proof_text
    return None


def verify(
    store: "ProofGraphStore",
    goal_id: str,
    *,
    lean: str = "lean",
    timeout_s: float = 300.0,
) -> AssemblyResult:
    """Assemble, compile, and read Lean's axiom report.

    This is the only thing that makes the root's PROVED status mean anything.
    A failure here is a real finding: the graph believed something the compiler
    does not.
    """

    goal = store.goal(goal_id)
    text = assemble(store, goal_id)
    name = _root_name(store, goal_id)
    text = text.rstrip() + f"\n\n#print axioms {name}\n"

    try:
        returncode, output = compile_lean(text, lean=lean, timeout_s=timeout_s)
    except SketchUnavailable as exc:
        # Nothing was measured. Reporting ok=False would say the assembled
        # proof is wrong, which is a much stronger claim than "we could not
        # check it".
        raise AssemblyError(f"could not run the final check: {exc}") from exc

    if returncode != 0:
        errors = "\n".join(
            line for line in output.splitlines() if ERROR_RE.search(line)
        )
        return AssemblyResult(
            ok=False,
            reason=(
                "the assembled proof does not compile -- the per-node greens "
                f"did not survive being put together:\n{errors or output}"
            )[:4000],
            text=text,
        )

    axioms = _axioms(output, name)
    if axioms is None:
        return AssemblyResult(
            ok=False,
            reason=f"no `#print axioms {name}` output",
            text=text,
        )
    result = AssemblyResult(ok=True, text=text, axioms=axioms)
    if result.forbidden_axioms:
        return AssemblyResult(
            ok=False,
            reason=(
                "the assembled proof depends on axioms outside the policy: "
                + ", ".join(sorted(result.forbidden_axioms))
            ),
            text=text,
            axioms=axioms,
        )
    _ = goal  # kept for symmetry with the error paths above
    return result


def _root_name(store: "ProofGraphStore", goal_id: str) -> str:
    """The declaration `#print axioms` should be asked about.

    Taken from the goal's own statement rather than from a sketch. A goal the
    solver closed directly has no decomposition and therefore no sketch, and
    requiring one turned the easiest possible outcome -- the model just proved
    it -- into a crash.
    """

    name = _declaration_name(store.goal(goal_id).statement)
    if name:
        return name
    for decomposition in store.decompositions_of(goal_id):
        sketch = store.sketch_of(decomposition.id)
        if sketch is not None:
            return sketch.parent_name
    raise AssemblyError(
        f"goal {goal_id} has no readable declaration name"
    )


def _declaration_name(signature: str) -> str:
    body = signature.strip()
    for keyword in ("theorem", "lemma", "example", "def"):
        if body.startswith(keyword + " "):
            body = body[len(keyword) + 1:].lstrip()
            break
    else:
        return ""
    return body.split(None, 1)[0] if body else ""


def _axioms(output: str, name: str) -> frozenset[str] | None:
    for match in _NO_AXIOMS_RE.finditer(output):
        if match.group(1) == name:
            return frozenset()
    for match in _AXIOMS_RE.finditer(output):
        if match.group(1) == name:
            return frozenset(
                item.strip()
                for item in match.group(2).split(",")
                if item.strip()
            )
    return None


__all__ = [
    "ALLOWED_AXIOMS",
    "AssemblyError",
    "AssemblyResult",
    "assemble",
    "verify",
]
