"""The memoization key, and the one rule that keeps it from manufacturing
proofs.

Identity is what turns the tree into a graph. Two branches that need the same
lemma should find one node, not two -- that reuse is the whole of LEAP's
memoization ablation (40.0% -> 56.7% on its Advanced set). It is also what
gives the acyclicity check teeth: minting a fresh id per proposal makes an
id-based cycle check vacuous, because every restatement is a new node and
nothing ever closes a loop.

**Merging two goals means "the proof of A also proves B".** If they are not in
fact the same, an unproved goal has just been marked proved. That is the only
path in this whole layer that can manufacture a false conclusion, so:

- computing an identity may FAIL, and failure resolves to "not the same" --
  never to "the same";
- the cost is asymmetric and stays that way. A missed merge costs one
  re-proof. A wrong merge costs correctness. **Do not add an LLM that judges
  whether two lemmas are equivalent** in order to raise the merge rate: it is a
  natural-looking improvement that removes the guard entirely.

Two hashers ship. `ExactTextHasher` is the default and merges only statements
that are identical after whitespace normalization -- conservative, and it can
never be wrong. `LeanExprHasher` merges alpha-equivalent statements by asking
Lean to elaborate them, which is what you want once there is real cross-branch
reuse to harvest.

Scope, stated so nobody assumes more: `LeanExprHasher` gives alpha-equivalence
with binder names erased and metadata stripped. It is NOT full definitional
identity (`isDefEq`), which unfolds definitions and would merge more. The
conservative end is the safe end.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .sketch import LeanRunner, SketchUnavailable, compile_lean

#: Erase binder names before serializing. Lean's own `Expr.hash` is already
#: alpha-invariant -- de Bruijn indices see to that -- but it is a UInt64, and
#: one collision is one wrong merge. So we serialize the whole structure and
#: hash it ourselves. `repr` alone will not do either: it keeps binder names,
#: so `∀ a b c` and `∀ x y z` would differ.
_HELPER = r"""import Lean
open Lean

partial def eraseBinderNames : Expr → Expr
  | .forallE _ d b bi => .forallE `_ (eraseBinderNames d) (eraseBinderNames b) bi
  | .lam _ d b bi     => .lam `_ (eraseBinderNames d) (eraseBinderNames b) bi
  | .letE _ t v b nd  => .letE `_ (eraseBinderNames t) (eraseBinderNames v) (eraseBinderNames b) nd
  | .app f a          => .app (eraseBinderNames f) (eraseBinderNames a)
  | .mdata _ e        => eraseBinderNames e
  | .proj s i e       => .proj s i (eraseBinderNames e)
  | e                 => e

open Elab Command Term in
elab "identity_of" tag:str t:term : command => do
  liftTermElabM do
    try
      let e ← elabTerm t none
      let e ← instantiateMVars e
      -- An unknown identifier does not raise: elaboration succeeds and leaves
      -- a sorry or a metavariable behind. Hashing that would hand a real key
      -- to a statement Lean never understood.
      if e.hasSorry || e.hasExprMVar then
        logInfo m!"EVOSKIP {tag.getString}"
      else
        let n := eraseBinderNames e
        logInfo m!"EVOIDENT {tag.getString} {(toString (repr n)).replace "\n" " "}"
    catch _ =>
      logInfo m!"EVOSKIP {tag.getString}"
"""

_IDENT_RE = re.compile(r"EVOIDENT (\S+) (.*)")

#: Anything Lean could not elaborate gets one of these. Unique per call, so it
#: can never collide with another goal -- failing toward "not the same".
_UNRESOLVED = "unresolved"


class IdentityHasher(Protocol):
    """Batched on purpose: `LeanExprHasher` pays a fixed `import Lean` cost of
    several seconds per invocation, so asking one statement at a time would
    make the graph unusable."""

    #: Recorded alongside identities so a later reader knows which rule
    #: produced them. Two hashers' keys are not interchangeable.
    name: str

    def hash_many(self, statements: Sequence[str]) -> list[str]:
        ...


@dataclass
class ExactTextHasher:
    """Merge only statements that are identical after whitespace normalization.

    The default. It misses alpha-variants, which costs re-proofs, and it cannot
    merge two things that are not the same, which costs nothing. Until there is
    measured cross-branch reuse to win, that is the right side to be wrong on.
    """

    name: str = "exact-text"

    def hash_many(self, statements: Sequence[str]) -> list[str]:
        return [self._one(statement) for statement in statements]

    @staticmethod
    def _one(statement: str) -> str:
        normalized = " ".join(statement.split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"text:{digest}"


@dataclass
class LeanExprHasher:
    """Merge alpha-equivalent statements by elaborating them in Lean.

    One Lean process for the whole batch. `import Lean` pulls in the compiler
    frontend and costs roughly five seconds, which is fine once per graph write
    and ruinous once per goal.
    """

    #: `import Lean` needs no lake project, so the bare runner is right by
    #: default even when the goals themselves import Mathlib.
    runner: LeanRunner = field(default_factory=lambda: LeanRunner(timeout_s=300.0))
    name: str = "lean-expr"

    def hash_many(self, statements: Sequence[str]) -> list[str]:
        if not statements:
            return []
        tags = [f"g{index}" for index in range(len(statements))]
        lines = [_HELPER]
        for tag, statement in zip(tags, statements):
            lines.append(f'identity_of "{tag}" ({_proposition(statement)})')
        source = "\n".join(lines) + "\n"

        try:
            _, output = self.runner.compile(source)
        except SketchUnavailable:
            # The toolchain, not the statements. Every goal in the batch gets a
            # unique key: nothing merges, and nothing is wrongly merged.
            return [_unresolved() for _ in statements]

        found: dict[str, str] = {}
        for match in _IDENT_RE.finditer(output):
            tag, body = match.group(1), " ".join(match.group(2).split())
            found[tag] = "lean:" + hashlib.sha256(
                body.encode("utf-8")
            ).hexdigest()
        # A statement Lean could not elaborate simply has no line. It must not
        # borrow anybody else's key.
        return [found.get(tag) or _unresolved() for tag in tags]


def _unresolved() -> str:
    return f"{_UNRESOLVED}:{uuid.uuid4().hex}"


def is_unresolved(identity: str) -> bool:
    """True for a key that stands for "we could not tell". Useful for reporting
    how much reuse a run gave up, which is the honest cost of failing safe."""

    return identity.startswith(_UNRESOLVED + ":")


def _proposition(statement: str) -> str:
    """Turn a declaration signature into the proposition it states.

        theorem foo (a b : Nat) : a + b = b + a
        -> ∀ (a b : Nat), a + b = b + a

    Binders move to the left of the colon so the elaborated term is the whole
    statement rather than its conclusion -- two lemmas with the same conclusion
    and different hypotheses are not the same lemma.
    """

    body = statement.strip()
    for keyword in ("theorem", "lemma", "example", "def"):
        if body.startswith(keyword + " "):
            body = body[len(keyword) + 1:].lstrip()
            break
    # Drop the declaration name.
    parts = body.split(None, 1)
    body = parts[1] if len(parts) == 2 else ""
    binders, conclusion = _split_on_top_level_colon(body)
    if not conclusion:
        return body.strip()
    if not binders:
        return conclusion
    return f"∀ {binders}, {conclusion}"


#: Brackets Lean uses for binders: explicit, implicit, strict implicit, and
#: instance.
_OPEN = "([{⦃"
_CLOSE = ")]}⦄"


def _split_on_top_level_colon(body: str) -> tuple[str, str]:
    """Split binders from conclusion at the colon that is not inside a binder.

    `str.partition(":")` splits on the FIRST colon, which in
    `(a b c : Nat) : ...` is the one inside the binder group. That produced
    nonsense Lean could not elaborate, and because an unelaborable statement
    resolves to "unresolved", every goal silently stopped merging -- a failure
    that costs efficiency quietly rather than breaking loudly.
    """

    depth = 0
    for index, char in enumerate(body):
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif char == ":" and depth == 0:
            return body[:index].strip(), body[index + 1:].strip()
    return "", body.strip()


__all__ = [
    "ExactTextHasher",
    "IdentityHasher",
    "LeanExprHasher",
    "is_unresolved",
]
