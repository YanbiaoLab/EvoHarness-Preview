"""A candidate file, split once into what the verifier checks.

EvoHarness handles whole Lean files: a preamble (imports included), an edit
region holding the lemma, and a trailing `#print axioms`. The verifier takes a
declaration body with no imports -- its environment is imported once, up
front -- plus a structured context. Getting from one to the other is a
contract, not a detail, because the verifier's certification and publication
records are immutable: whatever this split binds to in the first version stays
bound for every proof published under it.

The contract is `SourceEnvelope`. This module produces one per candidate,
**once**; the caller persists it and verify, assembly and publish reuse that
same instance. They compare source byte for byte and context dict for dict,
so re-splitting -- even one that only reorders JSON keys -- fails at
publication with a message that points the wrong way.

Rules this split enforces, all by text, none by running Lean:

**R1 -- the preamble comes from the graph's scope, not from the file.** The file
must begin with exactly the scope's preamble; only whitespace may sit between
it and the edit region. A candidate that edits the preamble changes what every
statement means, and that is not a change the candidate gets to make. The cost,
stated rather than hidden: in this phase a candidate cannot add its own imports.

**2d -- trailing commands are whitelisted, not guessed at.** After the edit region
only `#print axioms`, `#print` and `#check` lines are allowed; they are removed
from the body and kept in `stripped_commands`. Anything else is refused, not
stripped: removing `#exit` would make Lean elaborate the text it was told to
ignore, which changes what the file says.

The imports are separated, not judged. Whether an environment covers them is
the verifier's call; the envelope records them so that call can be made.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .run_solver import declaration_name
from .scope import ENVELOPE_SCHEMA_VERSION

BEGIN = "-- EDIT-REGION-BEGIN"
END = "-- EDIT-REGION-END"

#: The verifier's contract schema version. It is part of every stable hash's
#: domain string, separately from the envelope's own `schema_version` field --
#: the two are equal today and need not stay so.
CONTRACT_SCHEMA_VERSION = 1

#: The namespace every stable hash is taken in. Fixed by the verifier's
#: contract: it is hashed input, so changing one byte of it changes every hash
#: and nothing published would verify again.
_HASH_NAMESPACE = "leanground"

#: Commands that may follow the edit region and are stripped from the body.
#: Each is a query about the file that adds nothing to it.
STRIPPABLE = ("#print axioms", "#print", "#check")


#: A command that ends in `in` applies only to the command after it.
_DANGLING_IN = re.compile(r"(^|\s)in$")


class EnvelopeError(ValueError):
    """The candidate file breaks the split's rules. A verdict on the candidate."""


def _stable_hash(domain: str, value: Any) -> str:
    """The verifier's stable hash, reimplemented byte for byte.

    Pinned by the hash vectors in `tests/fixtures/envelope_vectors.json`, which
    come from the verifier's own implementation.
    """

    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = f"{_HASH_NAMESPACE}:{domain}:v{CONTRACT_SCHEMA_VERSION}\n{canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceEnvelope:
    """Field for field the verifier's `SourceEnvelope` contract."""

    candidate_file_sha256: str
    imports: tuple[str, ...]
    ctx: Mapping[str, Any]
    body: str
    primary_root: str
    expected_roots: tuple[str, ...]
    stripped_commands: tuple[str, ...] = ()
    schema_version: int = ENVELOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.expected_roots or self.primary_root not in self.expected_roots:
            raise ValueError("primary_root must be among expected_roots")
        if len(set(self.expected_roots)) != len(self.expected_roots):
            raise ValueError("expected_roots must not repeat")

    @property
    def envelope_hash(self) -> str:
        """Derived, never stored as a field, so it cannot disagree with the content."""

        return _stable_hash("source-envelope", asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "envelope_hash": self.envelope_hash}

    def to_json(self) -> str:
        """The exact text to persist. Canonical, so storing it twice stores it once."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceEnvelope":
        fields_ = {k: v for k, v in data.items() if k != "envelope_hash"}
        for key in ("imports", "expected_roots", "stripped_commands"):
            if key in fields_:
                fields_[key] = tuple(fields_[key])
        envelope = cls(**fields_)
        claimed = data.get("envelope_hash")
        if claimed is not None and claimed != envelope.envelope_hash:
            raise ValueError("envelope_hash does not match the content")
        return envelope


@dataclass(frozen=True)
class PreambleContext:
    """A graph's preamble, split once: the imports, and the context the rest becomes.

    Per graph, not per candidate -- the preamble is fixed by the scope. The same
    `ctx` must be used to resolve the goal and to verify its proofs: the
    verifier's goal identity hashes the context, and publication compares the
    two as dicts.
    """

    imports: tuple[str, ...]
    ctx: Mapping[str, Any] = field(default_factory=dict)


def preamble_context(preamble: str) -> PreambleContext:
    """Imports off the top; everything after them, verbatim, into `ctx.raw`.

    `import` is only legal before the first command, so the header ends at the
    first line that is neither an import nor blank nor a line comment. No
    remainder gives `{}`, not `{"raw": []}`: the two hash differently, and there
    must be one way to say "no context".
    """

    lines = preamble.strip().splitlines()
    imports: list[str] = []
    rest_start = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("--"):
            rest_start = index
            break
    rest = "\n".join(lines[rest_start:]).strip()
    last = next((line.strip() for line in reversed(rest.splitlines())
                 if line.strip() and not line.strip().startswith("--")), "")
    if _DANGLING_IN.search(last):
        # `open Nat in` at the end of the context scopes nothing: after the
        # split no command follows it, so the context cannot be elaborated on
        # its own, and every candidate under the graph would fail for a reason
        # that is not theirs -- the graph would exhaust its goals on a preamble
        # fault. Refuse the preamble instead, once.
        raise EnvelopeError(
            f"the preamble ends with `{last}`, which scopes only the next "
            "command; after the split there is none. Put the `open` in the "
            "preamble without `in`, or inside the edit region"
        )
    return PreambleContext(tuple(imports), {"raw": [rest]} if rest else {})


def split_source(
    text: str, *, preamble: str, root: str, name_prefix: str = ""
) -> SourceEnvelope:
    """Split one candidate file. Raises `EnvelopeError` when a rule is broken.

    `preamble` is the graph's (from its scope), `root` the short name of the
    goal's declaration, `name_prefix` what the verifier reported for the
    namespace when it resolved the goal -- `""` for the root namespace,
    otherwise ending in `.`. The verifier reports roots fully qualified, so
    `expected_roots` must be too.
    """

    preamble = preamble.strip()
    if not text.startswith(preamble):
        raise EnvelopeError(
            "the file does not begin with the graph's preamble; a candidate may "
            "not change it"
        )
    after = text[len(preamble):]
    begin = after.find(BEGIN)
    if begin < 0:
        raise EnvelopeError(f"no `{BEGIN}` marker")
    if after[:begin].strip():
        raise EnvelopeError(
            "text between the preamble and the edit region; only the edit "
            "region may change"
        )
    region_and_tail = after[begin + len(BEGIN):]
    end = region_and_tail.find(END)
    if end < 0:
        raise EnvelopeError(f"no `{END}` marker")
    body = region_and_tail[:end].strip("\n").rstrip()
    if not body.strip():
        raise EnvelopeError("the edit region is empty")

    declared = declaration_name(body.split(":=", 1)[0])
    if declared != root:
        raise EnvelopeError(
            f"the edit region must open with the goal's own declaration `{root}`, "
            f"found `{declared}`"
        )

    stripped: list[str] = []
    for line in region_and_tail[end + len(END):].splitlines():
        command = line.strip()
        if not command or command.startswith("--"):
            continue
        if any(command == c or command.startswith(c + " ") for c in STRIPPABLE):
            stripped.append(command)
            continue
        raise EnvelopeError(
            f"after the edit region only {', '.join(STRIPPABLE)} are allowed; "
            f"refusing rather than stripping: {command[:80]!r}"
        )

    context = preamble_context(preamble)
    primary = name_prefix + root
    return SourceEnvelope(
        candidate_file_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        imports=context.imports,
        ctx=dict(context.ctx),
        body=body,
        primary_root=primary,
        expected_roots=(primary,),
        stripped_commands=tuple(stripped),
    )


__all__ = [
    "BEGIN",
    "END",
    "STRIPPABLE",
    "EnvelopeError",
    "PreambleContext",
    "SourceEnvelope",
    "preamble_context",
    "split_source",
]
