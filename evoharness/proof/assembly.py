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

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .graph import (
    Certification,
    Decomposition,
    DecompositionStatus,
    GoalStatus,
)
from .policy import AxiomPolicy
from .sketch import ERROR_RE, LeanRunner, SketchUnavailable, render

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .store import ProofGraphStore

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
    #: The trust level the axiom report puts the finished proof at, read
    #: through the graph's `AxiomPolicy`. Empty when nothing was compiled.
    trust: str = ""
    #: Every axiom that on its own keeps the proof below the policy --
    #: `sorryAx`, user-declared axioms, and native_decide axioms a local
    #: compile could not recheck. Computed by the policy, not by a set kept here.
    forbidden: frozenset[str] = field(default_factory=frozenset)

    @property
    def forbidden_axioms(self) -> frozenset[str]:
        return self.forbidden


class AmbiguousRoute(AssemblyError):
    """Several completed routes close this goal and none of them was named.

    Not a wiring fault and not a finding about the proof: a question. Picking
    for the caller is what the first version did -- it took the oldest -- and
    age has no bearing on which proof anyone wants. Worse, the choice was
    invisible: two routes, one file, and nothing saying a choice happened.
    """

    def __init__(self, goal_id: str, candidates: "list[tuple[str, tuple[str, ...]]]"):
        self.goal_id = goal_id
        self.candidates = candidates
        listing = "; ".join(
            f"{route_id} -> {', '.join(names) or 'no subgoals'}"
            for route_id, names in candidates
        )
        super().__init__(
            f"goal {goal_id} has {len(candidates)} completed routes and none "
            f"was named; assemble with one of: {listing}"
        )


def assemble(
    store: "ProofGraphStore",
    goal_id: str,
    *,
    routes: "Iterable[str]" = (),
    preamble: str = "",
) -> str:
    """Render the finished proof of one goal, recursively.

    Each subgoal contributes either the body of its own proved attempt or, if
    it was itself decomposed, its own rendered subtree spliced in ahead of the
    parent. Depth-first so a lemma is always declared before it is used, which
    Lean requires.

    `routes` names decompositions to assemble through -- one per goal, matched
    to its goal by the decomposition itself, so the caller passes ids and not
    pairs. A goal with several completed routes and no named one raises
    `AmbiguousRoute` rather than picking.

    Imports are hoisted here and emitted once. They used to come out of
    `render`, once per sketch, which put an `import` in the middle of the file
    the moment any subgoal had a subtree of its own -- and `import` is only
    legal at the top. Deduplicated by line and kept in first-seen order, so a
    subtree that needs something extra keeps it.

    `preamble` is the board's own header, and it comes first. Sketches carry
    a copy of it, so a proof assembled through a route got its imports from
    there -- but a goal the solver closed directly has no sketch, and its
    file came out with no header at all: a Mathlib proof that compiled in its
    own run failed to parse on its first `∑` once assembled.
    """

    pins = _pin_map(store, routes)
    header: list[str] = [line for line in preamble.strip().splitlines()]
    used: set[str] = set()
    text, _ = _assemble_body(store, goal_id, pins, header, used)
    unused = sorted(pins[goal] for goal in set(pins) - used)
    if unused:
        # Silence here would be the worst outcome: the caller named a route,
        # got a file assembled through a different one, and has nothing in
        # front of it saying so.
        raise AssemblyError(
            "these routes were named but never applied -- their goals are not "
            f"on the route being assembled: {', '.join(unused)}"
        )
    if header:
        return "\n".join(header) + "\n\n" + text
    return text


def route_for(
    store: "ProofGraphStore", goal_id: str, pins: "Mapping[str, str]"
) -> "Decomposition | None":
    """The decomposition this goal is assembled through, or None if it is not.

    None has one meaning only: nothing was chosen because nothing was there to
    choose -- the goal was closed by a solver directly. It is not "the caller
    did not say", which raises instead.
    """

    completed = [
        decomposition
        for decomposition in store.decompositions_of(goal_id)
        if decomposition.status is DecompositionStatus.COMPLETED
    ]
    pinned = pins.get(goal_id)
    if pinned is not None:
        for decomposition in completed:
            if decomposition.id == pinned:
                return decomposition
        raise AssemblyError(
            f"{pinned} is not a completed decomposition of goal {goal_id}"
        )
    if not completed:
        return None
    if len(completed) > 1:
        raise AmbiguousRoute(
            goal_id,
            [
                (decomposition.id, _subgoal_names(store, decomposition))
                for decomposition in completed
            ],
        )
    return completed[0]


def _subgoal_names(
    store: "ProofGraphStore", decomposition
) -> tuple[str, ...]:
    """What a route is FOR, in a form a person choosing can read.

    Ids alone would make the refusal unanswerable without another query, and
    the caller being refused is usually a model that has to answer in one
    turn.
    """

    sketch = store.sketch_of(decomposition.id)
    if sketch is not None:
        return tuple(spec.name for spec in sketch.subgoals)
    return tuple(
        _declaration_name(store.goal(subgoal_id).statement)
        for subgoal_id in decomposition.subgoal_ids
    )


def _pin_map(
    store: "ProofGraphStore", routes: "Iterable[str]"
) -> dict[str, str]:
    """Route ids -> {goal_id: route_id}, refusing anything that cannot apply.

    Checked here rather than where each is used, because a route naming a goal
    that is nowhere in this tree would otherwise be silently ignored: the
    caller would believe it had chosen, and get the file it was trying not to
    get.
    """

    pins: dict[str, str] = {}
    for route_id in routes:
        decomposition = store.decomposition(route_id)
        if decomposition.status is not DecompositionStatus.COMPLETED:
            raise AssemblyError(
                f"route {route_id} is {decomposition.status.value}, not "
                "completed; only a route whose subgoals are all proved can be "
                "assembled"
            )
        already = pins.get(decomposition.goal_id)
        if already is not None and already != route_id:
            raise AssemblyError(
                f"two routes named for goal {decomposition.goal_id}: "
                f"{already} and {route_id}"
            )
        pins[decomposition.goal_id] = route_id
    return pins


def _assemble_body(
    store: "ProofGraphStore",
    goal_id: str,
    pins: "Mapping[str, str]",
    preamble: list[str],
    used: set[str],
) -> tuple[str, str]:
    """(text, the name that text declares) for one goal, imports stripped.

    The declared name is returned rather than assumed because the parent
    splices this text in place of a lemma it will not emit itself. If the two
    ever disagreed, the file would reference something nothing defines, and
    Lean would report it as an unknown identifier far from the cause.
    """

    goal = store.goal(goal_id)
    if goal.status is not GoalStatus.PROVED:
        raise AssemblyError(
            f"goal {goal_id} is {goal.status.value}, not proved"
        )

    if goal_id not in pins:
        direct = _direct_proof(store, goal_id)
        if direct is not None:
            # Closed without decomposing. `_direct_proof` yields the BODY,
            # which is what a subgoal contributes when spliced into a sketch;
            # at the top of the file it still needs its own declaration around
            # it, or what comes back is a bare term that compiles as nothing.
            return f"{goal.statement} := {direct}\n", _declaration_name(
                goal.statement
            )

    decomposition = route_for(store, goal_id, pins)
    if goal_id in pins:
        used.add(goal_id)
    if decomposition is None:
        raise AssemblyError(
            f"goal {goal_id} is proved but has neither a proved attempt nor a "
            "completed decomposition"
        )
    sketch = store.sketch_of(decomposition.id)
    if sketch is None:
        raise AssemblyError(
            f"decomposition {decomposition.id} was accepted without a sketch; "
            "the graph knows which lemmas close the goal but not how"
        )
    for line in sketch.preamble.splitlines():
        if line not in preamble:
            preamble.append(line)

    bodies: dict[str, str] = {}
    preludes: list[str] = []
    omit: set[str] = set()
    for spec, subgoal_id in zip(sketch.subgoals, decomposition.subgoal_ids):
        if subgoal_id not in pins:
            sub_direct = _direct_proof(store, subgoal_id)
            if sub_direct is not None:
                bodies[spec.name] = sub_direct
                continue
        # The subgoal was itself decomposed. Its subtree declares the lemma,
        # so this sketch must not declare it a second time.
        text, declared = _assemble_body(
            store, subgoal_id, pins, preamble, used
        )
        if declared != spec.name:
            raise AssemblyError(
                f"subgoal {subgoal_id} assembles as `{declared}` but its "
                f"parent's sketch calls it `{spec.name}`; the spliced text "
                "would not define what the parent cites"
            )
        preludes.append(text)
        omit.add(spec.name)

    rendered = render(sketch, bodies=bodies, preamble=False, omit=omit)
    return "\n\n".join([*preludes, rendered.text]), sketch.parent_name


def _direct_proof(store: "ProofGraphStore", goal_id: str) -> str | None:
    for attempt in store.attempts_of(goal_id):
        if attempt.outcome.value == "proved":
            if not attempt.proof_text:
                raise AssemblyError(
                    f"attempt {attempt.id} is proved but carries no proof text"
                )
            return attempt.proof_text
    return None


def _top_route(
    store: "ProofGraphStore", goal_id: str, pins: "Mapping[str, str]"
) -> "Decomposition | None":
    """The route `assemble` will use for the goal itself.

    Mirrors the precedence in `_assemble_body` rather than re-deriving it: a
    goal the solver closed directly is assembled from that proof even when it
    also has routes, so recording a route there would name one that was not
    used.
    """

    if goal_id not in pins and _direct_proof(store, goal_id) is not None:
        return None
    return route_for(store, goal_id, pins)


def verify(
    store: "ProofGraphStore",
    goal_id: str,
    *,
    runner: LeanRunner | None = None,
    routes: "Iterable[str]" = (),
    policy: AxiomPolicy | None = None,
    preamble: str = "",
) -> AssemblyResult:
    """Assemble, compile, and read Lean's axiom report through the policy.

    This is the only thing that makes the root's PROVED status mean anything.
    A failure here is a real finding: the graph believed something the compiler
    does not -- with one exception spelled out in the reason: native_decide
    axioms, which a local compile has no way to recheck.
    """

    policy = policy or AxiomPolicy()

    goal = store.goal(goal_id)
    text = assemble(store, goal_id, routes=routes, preamble=preamble)
    name = _root_name(store, goal_id)
    text = text.rstrip() + f"\n\n#print axioms {name}\n"

    try:
        returncode, output = (runner or LeanRunner()).compile(text)
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
    # No native_decide recheck is available here: such axioms classify as
    # `claimed` and block the certificate. Failing closed is the point.
    trust = policy.classify(axioms)
    forbidden = policy.blocking(axioms)
    if forbidden:
        unverified = policy.unverified_native(axioms)
        reason = (
            "the assembled proof depends on axioms outside the policy "
            f"(minimum_trust={policy.minimum_trust}): "
            + ", ".join(sorted(forbidden))
        )
        if unverified:
            reason += (
                "; native_decide axioms need the verifier's content recheck and "
                "a local compile cannot certify them: "
                + ", ".join(sorted(unverified))
            )
        return AssemblyResult(
            ok=False,
            reason=reason,
            text=text,
            axioms=axioms,
            trust=trust,
            forbidden=forbidden,
        )
    _ = goal  # kept for symmetry with the error paths above
    return AssemblyResult(ok=True, text=text, axioms=axioms, trust=trust)


def certify(
    store: "ProofGraphStore",
    goal_id: str,
    *,
    runner: LeanRunner | None = None,
    routes: "Iterable[str]" = (),
    policy: AxiomPolicy | None = None,
    preamble: str = "",
) -> tuple[AssemblyResult, Certification]:
    """Verify, and record the verdict on the goal.

    `verify` alone leaves no trace: the compile happens, the caller reads the
    answer, and the graph still says only PROVED -- a status derived from the
    route closing, which is a weaker claim than the finished file compiling.
    Recording it is what lets a later reader tell the two apart, and what
    makes a failed assembly visible as the finding it is rather than as an
    absence.

    Nothing is recorded when nothing was measured: `verify` raises
    `AssemblyError` for that, and it propagates.

    The route is recorded with the rest. `text_sha256` promises the compiled
    file is reproducible from the graph, and that promise held only while the
    selection rule was fixed; now that a caller may choose, the same graph
    renders several different files and a hash nobody can regenerate reads
    like provenance while being none.
    """

    result = verify(store, goal_id, runner=runner, routes=routes, policy=policy,
                    preamble=preamble)
    decomposition = _top_route(store, goal_id, _pin_map(store, routes))
    certification = store.record_certification(
        goal_id,
        ok=result.ok,
        axioms=result.axioms,
        reason=result.reason,
        text_sha256=hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        decomposition_id=decomposition.id if decomposition else "",
        trust=result.trust,
    )
    return result, certification


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
    "AmbiguousRoute",
    "AssemblyError",
    "AssemblyResult",
    "assemble",
    "route_for",
    "verify",
]
