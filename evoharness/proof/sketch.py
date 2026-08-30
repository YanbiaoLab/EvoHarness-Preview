"""What a decomposition actually is, and what earns it ACCEPTED.

The state machine already refuses to let a merely PROPOSED decomposition
complete. This module decides which ones get past that gate.

A decomposition is not a list of subgoal statements. It is a Lean file:

    theorem lemma_1 : A := sorry
    theorem lemma_2 : B := sorry

    theorem parent : A ∧ B := ⟨lemma_1, lemma_2⟩

Two things have to hold, and Lean is asked both rather than told either.

**It compiles.** The parent's proof term typechecks GIVEN the lemma
statements. That typecheck IS the implication "all subgoals proved => parent
proved"; without it the graph asserts something nobody verified.

**`sorry` appears only in the proposed lemmas.** A sketch whose parent body
still contains `sorry` has not reduced the goal, only moved it -- and it would
compile perfectly well.

`render` is shared with `assembly.py`: the file validated here and the file
compiled at final re-verification are built by the same code, differing only in
whether the lemma bodies are `sorry` or real proofs. Two separate assemblers
would be free to drift, and that drift is the failure this layer exists to
prevent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .graph import Goal

#: Lean 4.30 emits both `: error:` and the tagged `: error(lean.foo):` form in
#: the same version. Matching only the first reads a file full of unknown
#: identifiers as having no errors at all.
ERROR_RE = re.compile(r":\s*error(?:\([^)]*\))?:")

#: ``file.lean:LINE:COL: warning: declaration uses `sorry```. The position is
#: the declaration's own, which is why rendering records line numbers rather
#: than parsing them back out of the source.
_SORRY_RE = re.compile(r":(\d+):\d+:\s*warning: declaration uses")


class SketchUnavailable(RuntimeError):
    """The sketch could not be checked -- not a verdict on the sketch.

    Returning ok=False here would mark the decomposition rejected-by-verifier,
    which is terminal and never revisited. A dead toolchain would then delete a
    viable route from the graph permanently.
    """


@dataclass(frozen=True)
class Validation:
    """Whether a proposed decomposition may be trusted to compose back."""

    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class SubgoalSpec:
    """One lemma a decomposition proposes."""

    #: The Lean declaration name, unique within the sketch.
    name: str
    #: The memoization key. `identity.py` computes it; nothing here reads it.
    identity: str
    #: The declaration up to but excluding `:=`, e.g.
    #: "theorem lemma_distrib (a b c : Nat) : (a + b) * c = a * c + b * c"
    signature: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "identity": self.identity,
            "signature": self.signature,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SubgoalSpec":
        return cls(
            name=data["name"],
            identity=data["identity"],
            signature=data["signature"],
        )


@dataclass(frozen=True)
class Sketch:
    """A proposed route from one goal to a set of subgoals.

    Self-contained on purpose: assembly reads it back from the database long
    after the Goal object that produced it is gone.
    """

    parent_name: str
    parent_signature: str
    #: The proof term closing the parent, referring to the subgoals by name.
    parent_body: str
    subgoals: tuple[SubgoalSpec, ...]
    #: Imports and options. Empty for the core-Lean fixture.
    preamble: str = ""

    def to_json(self) -> dict:
        return {
            "parent_name": self.parent_name,
            "parent_signature": self.parent_signature,
            "parent_body": self.parent_body,
            "subgoals": [spec.to_json() for spec in self.subgoals],
            "preamble": self.preamble,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Sketch":
        return cls(
            parent_name=data["parent_name"],
            parent_signature=data["parent_signature"],
            parent_body=data["parent_body"],
            subgoals=tuple(
                SubgoalSpec.from_json(item) for item in data["subgoals"]
            ),
            preamble=data.get("preamble", ""),
        )


@dataclass(frozen=True)
class RenderedSketch:
    text: str
    #: declaration name -> the 1-based line its declaration starts on. Recorded
    #: while writing rather than recovered by parsing: we laid the file out, so
    #: guessing where things landed would be inventing an uncertainty.
    declaration_lines: Mapping[str, int]


def render(
    sketch: Sketch, *, bodies: Mapping[str, str] | None = None
) -> RenderedSketch:
    """Lay the sketch out as a Lean file.

    `bodies=None` gives every lemma a `sorry` body -- the sketch, for
    validation. Supplying bodies gives the assembled proof, for final
    re-verification. One function for both, so the thing checked and the thing
    finally compiled cannot drift apart.
    """

    bodies = dict(bodies or {})
    lines: list[str] = []
    declaration_lines: dict[str, int] = {}

    if sketch.preamble:
        lines.extend(sketch.preamble.splitlines())
        lines.append("")

    def emit(name: str, signature: str, body: str) -> None:
        declaration_lines[name] = len(lines) + 1
        if "\n" in body:
            lines.append(f"{signature} :=")
            lines.extend(body.splitlines())
        else:
            lines.append(f"{signature} := {body}")
        lines.append("")

    for spec in sketch.subgoals:
        emit(spec.name, spec.signature, bodies.get(spec.name, "sorry"))
    emit(sketch.parent_name, sketch.parent_signature, sketch.parent_body)

    return RenderedSketch(
        text="\n".join(lines).rstrip() + "\n",
        declaration_lines=dict(declaration_lines),
    )


def compile_lean(
    text: str, *, lean: str = "lean", timeout_s: float = 120.0
) -> tuple[int, str]:
    """Run Lean over a rendered file. Returns (returncode, combined output).

    Raises `SketchUnavailable` only for faults on our side of the line: no
    binary, a signal kill, a timeout on work that should take a second. A
    non-zero exit with diagnostics is a verdict about the file.
    """

    binary = shutil.which(lean)
    if not binary:
        raise SketchUnavailable(f"no `{lean}` on PATH")
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "sketch.lean"
        path.write_text(text, encoding="utf-8")
        try:
            done = subprocess.run(
                [binary, path.name],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise SketchUnavailable(
                f"lean exceeded {timeout_s:g}s"
            ) from exc
        except OSError as exc:
            raise SketchUnavailable(f"could not run `{lean}`: {exc}") from exc
    if done.returncode < 0:
        raise SketchUnavailable(
            f"`{lean}` was killed by signal {-done.returncode}"
        )
    return done.returncode, (done.stdout or "") + (done.stderr or "")


@dataclass
class LeanSketchValidator:
    """Compile the sketch and ask Lean the two questions."""

    lean: str = "lean"
    timeout_s: float = 120.0

    def __call__(self, goal: Goal, sketch: Sketch) -> Validation:
        if sketch.parent_signature.strip() != goal.statement.strip():
            return Validation(
                ok=False,
                reason="the sketch closes a different proposition than the goal",
            )

        rendered = render(sketch)
        returncode, output = compile_lean(
            rendered.text, lean=self.lean, timeout_s=self.timeout_s
        )

        if returncode != 0:
            errors = "\n".join(
                line for line in output.splitlines() if ERROR_RE.search(line)
            )
            return Validation(
                ok=False,
                reason=f"the sketch does not compile:\n{errors or output}"[:2000],
            )

        line_to_name = {
            line: name for name, line in rendered.declaration_lines.items()
        }
        with_sorry = {
            line_to_name.get(int(match.group(1)), f"<line {match.group(1)}>")
            for match in _SORRY_RE.finditer(output)
        }
        expected = {spec.name for spec in sketch.subgoals}

        if sketch.parent_name in with_sorry:
            return Validation(
                ok=False,
                reason=(
                    "the parent body still contains `sorry`: this sketch moves "
                    "the goal rather than reducing it"
                ),
            )
        unexpected = with_sorry - expected
        if unexpected:
            return Validation(
                ok=False,
                reason=f"`sorry` outside the proposed lemmas: {sorted(unexpected)}",
            )
        return Validation(
            ok=True, reason=f"{len(expected)} lemmas, parent closed"
        )


__all__ = [
    "ERROR_RE",
    "LeanSketchValidator",
    "RenderedSketch",
    "Sketch",
    "SketchUnavailable",
    "SubgoalSpec",
    "Validation",
    "compile_lean",
    "render",
]
