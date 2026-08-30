"""Asking a model for a route, and refusing to trust it any further than Lean.

The controller does not care where a decomposition comes from -- P-3 handed it
a table, this hands it a model. What changes is only who proposes; what does
NOT change is who decides. The proposal arrives as text, is parsed into a
`Sketch`, and then goes to `LeanSketchValidator` like any other.

That ordering is the whole point. The model is answering "which way is worth
trying", never "is this correct". A proposal it is completely confident about
and Lean rejects is rejected.

Parsing is deliberately strict. A malformed proposal is refused rather than
patched up: a half-understood sketch that happens to compile is a worse
outcome than no sketch at all, because it enters the graph carrying an
implication nobody checked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .graph import Goal
from .sketch import Sketch, SubgoalSpec

PROMPT = """\
You are decomposing one Lean 4 goal into lemmas.

GOAL (a declaration signature, everything before `:=`):

{signature}

Propose a decomposition: a small number of lemmas that are each strictly
easier than the goal, plus the proof term that closes the goal from them.

Reply with ONE json object in a ```json fenced block, and nothing else:

{{
  "lemmas": [
    {{"name": "<lean declaration name>",
      "signature": "theorem <name> <binders> : <proposition>"}}
  ],
  "parent_body": "<a Lean term or `by ...` block closing the goal, referring
                   to the lemmas by name>"
}}

Rules that will get a proposal thrown away:
- A lemma that restates the goal. Decomposing has to make progress; a subgoal
  equivalent to its own ancestor is the classic way to burn a budget without
  moving.
- `sorry` anywhere in `parent_body`. The lemmas carry the holes; the parent
  body must close the goal outright, given them.
- Anything but the json block.
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
#: Names Lean will accept and that cannot collide with the parent's.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


class ProposalError(ValueError):
    """The model's reply could not be read as a decomposition."""


def parse_proposal(
    text: str, *, goal: Goal, parent_name: str, preamble: str = ""
) -> Sketch:
    """Turn a model reply into a Sketch, or refuse it.

    Every rejection here is cheap. Accepting a proposal we only half
    understood is not: it would reach the graph as an implication that nobody,
    model or compiler, actually checked.
    """

    blocks = _FENCE_RE.findall(text)
    raw = blocks[-1] if blocks else text
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"reply is not json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProposalError("reply must be a json object")

    lemmas = payload.get("lemmas")
    body = payload.get("parent_body")
    if not isinstance(lemmas, list) or not lemmas:
        raise ProposalError("`lemmas` must be a non-empty list")
    if not isinstance(body, str) or not body.strip():
        raise ProposalError("`parent_body` must be a non-empty string")
    if "sorry" in body:
        # Caught here as well as in the validator. The validator would reject
        # it too, but that costs a Lean compile to learn something the text
        # already said.
        raise ProposalError("`parent_body` contains `sorry`")

    specs: list[SubgoalSpec] = []
    seen: set[str] = set()
    for item in lemmas:
        if not isinstance(item, dict):
            raise ProposalError("each lemma must be an object")
        name = str(item.get("name", "")).strip()
        signature = str(item.get("signature", "")).strip()
        if not _NAME_RE.match(name):
            raise ProposalError(f"unusable lemma name: {name!r}")
        if name == parent_name:
            raise ProposalError("a lemma may not reuse the parent's name")
        if name in seen:
            raise ProposalError(f"duplicate lemma name: {name}")
        if not signature:
            raise ProposalError(f"lemma {name} has no signature")
        if ":=" in signature:
            raise ProposalError(
                f"lemma {name} signature must stop before `:=`"
            )
        seen.add(name)
        specs.append(
            # Identity is filled in by the caller, which owns the hasher: this
            # module must not decide what counts as the same lemma.
            SubgoalSpec(name=name, identity="", signature=signature)
        )

    return Sketch(
        parent_name=parent_name,
        parent_signature=goal.statement,
        parent_body=body.strip(),
        subgoals=tuple(specs),
        preamble=preamble,
    )


@dataclass
class ModelDecompositionSource:
    """A `DecompositionSource` backed by anything that turns a prompt into text.

    `ask` is the seam: a plain LLM transport, a dsh agent session with Mathlib
    retrieval, or a canned reply in a test. The controller sees none of it.
    """

    ask: Callable[[str], str]
    #: Fills identities. Owned here rather than in parsing, because "are these
    #: the same lemma" is a correctness question and belongs with the hasher.
    hasher: object
    parent_name: str = "goal"
    preamble: str = ""
    #: How many times one goal may be offered a route. Beyond this the source
    #: is out of ideas and says so, rather than looping.
    max_proposals: int = 1

    errors: list[str] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)

    def propose(self, goal: Goal) -> Sketch | None:
        used = self._counts.get(goal.identity, 0)
        if used >= self.max_proposals:
            return None
        self._counts[goal.identity] = used + 1

        reply = self.ask(PROMPT.format(signature=goal.statement))
        try:
            sketch = parse_proposal(
                reply,
                goal=goal,
                parent_name=self.parent_name,
                preamble=self.preamble,
            )
        except ProposalError as exc:
            # Not a graph event: nothing was proposed, so there is nothing to
            # mark rejected. Recorded so a run that produced no decompositions
            # can be told apart from one whose model kept replying badly.
            self.errors.append(f"{goal.identity}: {exc}")
            return None

        identities = self.hasher.hash_many(
            [spec.signature for spec in sketch.subgoals]
        )
        return Sketch(
            parent_name=sketch.parent_name,
            parent_signature=sketch.parent_signature,
            parent_body=sketch.parent_body,
            subgoals=tuple(
                SubgoalSpec(
                    name=spec.name, identity=identity, signature=spec.signature
                )
                for spec, identity in zip(sketch.subgoals, identities)
            ),
            preamble=sketch.preamble,
        )


__all__ = [
    "PROMPT",
    "ModelDecompositionSource",
    "ProposalError",
    "parse_proposal",
]
