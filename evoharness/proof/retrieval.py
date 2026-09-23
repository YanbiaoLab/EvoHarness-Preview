"""Retrieval from the verifier's library, before a goal is attacked.

What comes back goes into the prompt, and is also kept: the request id, each
declaration's id and environment, its elaborated type, trust and axioms, and
what matched. Kept because a proof that leaned on a retrieved lemma is only as
good as that lemma's standing, and because the library can shrink under a
running graph (see below).

**This side may choose which hits reach the prompt; it may not restate what the
verifier said about them.** Trust and axioms are the verifier's answer and are
carried through unchanged.

**The trust floor is the graph's.** Retrieval asks for `minimum_trust` from the
graph's scope, so the prompt never suggests a lemma the graph would refuse to
accept a proof resting on.

**The library can shrink.** When the verifier folds published declarations
into a new environment, it moves them out of the old one, and a graph pinned
to the old environment stops seeing them -- silently. Pinning the environment
keeps that from happening while the pin lives; `vanished` is the check that
notices when it happened anyway, so the run can stop rather than keep working
from a view that is no longer there.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contract import GoalContract

#: Dotted identifiers in a statement: `Nat.succ_le`, `Finset.sum`. What the
#: library is searched by, since the verifier's search is by name.
_IDENT = re.compile(r"\b[A-Z][A-Za-z0-9_']*(?:\.[A-Za-z0-9_']+)+\b")


@dataclass(frozen=True)
class RetrievalEvidence:
    """One retrieved declaration, as the verifier described it."""

    declaration_id: str
    base_id: str
    name: str
    kind: str
    type: str | None
    trust: int | None
    axioms: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_hit(cls, hit: Mapping[str, Any]) -> "RetrievalEvidence":
        return cls(
            declaration_id=hit["declaration_id"], base_id=hit["base_id"], name=hit["name"],
            kind=hit["kind"], type=hit.get("type"), trust=hit.get("trust"),
            axioms=tuple(hit.get("axioms") or ()), evidence=dict(hit.get("evidence") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "declaration_id": self.declaration_id, "base_id": self.base_id, "name": self.name,
            "kind": self.kind, "type": self.type, "trust": self.trust,
            "axioms": list(self.axioms), "evidence": dict(self.evidence),
        }


class RetrievalProvider(Protocol):
    def retrieve(self, goal, contract: GoalContract) -> tuple[str, list[RetrievalEvidence]]:
        """(request id, hits). Raises only when there was no answer at all."""


def queries_for(statement: str, *, limit: int = 8) -> tuple[str, ...]:
    """What to search the library by: the dotted names the statement mentions."""

    seen: list[str] = []
    for match in _IDENT.finditer(statement):
        if match.group(0) not in seen:
            seen.append(match.group(0))
    return tuple(seen[:limit])


@dataclass
class VerifierRetrievalProvider:
    client: Any
    #: The graph's floor, from its scope. Not a separate setting: see above.
    minimum_trust: str
    limit: int = 10

    def retrieve(self, goal, contract: GoalContract) -> tuple[str, list[RetrievalEvidence]]:
        queries = queries_for(goal.statement)
        request_id = f"evo-retrieval-{uuid.uuid4().hex[:16]}"
        if not queries:
            return request_id, []
        hits = self.client.retrieve({
            "schema_version": 1, "request_id": request_id, "base": dict(contract.base),
            "queries": list(queries), "kind": "theorem", "min_trust": self.minimum_trust,
            "limit": self.limit,
        })
        return request_id, [RetrievalEvidence.from_hit(h) for h in hits]


def prompt_section(hits: Sequence[RetrievalEvidence]) -> str:
    """The hits as the model sees them. Name and type; trust as the verifier said."""

    if not hits:
        return ""
    lines = ["", "Declarations from the verifier's library that may help "
             "(checked there; use them by name):"]
    for hit in hits:
        typ = f" : {hit.type}" if hit.type else ""
        lines.append(f"  - {hit.name}{typ}")
    return "\n".join(lines) + "\n"


def vanished(client: Any, stored: Sequence[tuple[Mapping[str, Any], RetrievalEvidence]],
             *, minimum_trust: str) -> list[str]:
    """Declarations retrieved earlier that the same environment no longer shows.

    `stored` pairs each hit with the environment it was retrieved in. Asked
    again by name in that environment; a hit that is not among the answers has
    left the view.
    """

    gone = []
    for base, hit in stored:
        answers = client.retrieve({
            "schema_version": 1, "request_id": f"evo-visibility-{uuid.uuid4().hex[:12]}",
            "base": dict(base), "queries": [hit.name], "kind": "all",
            "min_trust": minimum_trust, "limit": 20,
        })
        if hit.declaration_id not in {a["declaration_id"] for a in answers}:
            gone.append(hit.name)
    return gone


__all__ = [
    "RetrievalEvidence",
    "RetrievalProvider",
    "VerifierRetrievalProvider",
    "prompt_section",
    "queries_for",
    "vanished",
]
