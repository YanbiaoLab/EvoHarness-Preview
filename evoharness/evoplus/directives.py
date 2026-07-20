# EvoHarness original (hitl_design.md, tier-1 cut): human directives as
# plain plugins over a control file — the engine never learns about HITL.
# v1 implements the two directive kinds that need no core changes:
#   guidance      -> HumanDirectiveContributor (PromptContributor, with TTL)
#   lineage_veto  -> LineageVetoPolicy (SamplingWeightPolicy, incl. descendants)
# freeze_region / branch_from / operator overrides need loop/PatchEngine
# hooks and are deliberately deferred.
"""Human-in-the-loop directives: control file + the two v1 plugins."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evoharness.evocore.interfaces import MutationContext
from evoharness.evocore.population import Candidate, PopulationStore

DIRECTIVE_KINDS = ("guidance", "lineage_veto")


@dataclass
class Directive:
    id: str
    kind: str
    text: str = ""
    candidate_ids: list[str] = field(default_factory=list)
    ttl_generations: int | None = None  # guidance only; None = no expiry

    @classmethod
    def from_json(cls, d: dict) -> "Directive":
        if d.get("kind") not in DIRECTIVE_KINDS:
            raise ValueError(f"unknown directive kind {d.get('kind')!r}")
        return cls(
            id=str(d["id"]),
            kind=d["kind"],
            text=d.get("text", ""),
            candidate_ids=list(d.get("candidate_ids", [])),
            ttl_generations=d.get("ttl_generations"),
        )


class DirectiveBook:
    """Reads control/directives.json, reloading on mtime change. TTL is
    counted from the generation at which the directive is first seen by the
    loop (stamped in memory; a restart re-stamps, which errs on the side of
    keeping human guidance alive)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._mtime: float | None = None
        self._directives: list[Directive] = []
        self._first_seen: dict[str, int] = {}
        self.version = 0

    def _reload_if_changed(self) -> None:
        if not self.path.exists():
            self._directives, self.version = [], 0
            return
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime:
            return
        self._mtime = mtime
        data = json.loads(self.path.read_text())
        self.version = int(data.get("version", 0))
        self._directives = [
            Directive.from_json(d) for d in data.get("directives", [])
        ]

    def active(self, generation: int) -> list[Directive]:
        self._reload_if_changed()
        out = []
        for d in self._directives:
            if d.kind == "guidance" and d.ttl_generations is not None:
                first = self._first_seen.setdefault(d.id, generation)
                if generation - first >= d.ttl_generations:
                    continue
            out.append(d)
        return out

    def vetoed_ids(self, generation: int) -> set[str]:
        return {
            cid
            for d in self.active(generation)
            if d.kind == "lineage_veto"
            for cid in d.candidate_ids
        }


class HumanDirectiveContributor:
    """Implements PromptContributor: active guidance becomes a prompt section
    placed ahead of feedback/experience (human word outranks machine word)."""

    def __init__(self, book: DirectiveBook):
        self.book = book

    def contribute(self, ctx: MutationContext) -> str | None:
        lines = [
            f"- {d.text}"
            for d in self.book.active(ctx.generation)
            if d.kind == "guidance" and d.text.strip()
        ]
        if not lines:
            return None
        return (
            "# Reviewer directives\n"
            "A human reviewer watching this run asks you to follow these "
            "instructions; they take precedence over other suggestions:\n"
            + "\n".join(lines)
        )


class LineageVetoPolicy:
    """Implements SamplingWeightPolicy: a vetoed candidate and ALL of its
    descendants get zero parent-selection weight. Ancestry is resolved
    through the store (bounded walk; lineages are shallow at this scale)."""

    def __init__(self, book: DirectiveBook, store: PopulationStore,
                 max_depth: int = 200):
        self.book = book
        self.store = store
        self.max_depth = max_depth
        self._generation_hint = 10**9  # veto has no TTL; any generation works

    def weight_multiplier(self, cand: Candidate) -> float:
        vetoed = self.book.vetoed_ids(self._generation_hint)
        if not vetoed:
            return 1.0
        node, depth = cand, 0
        while node is not None and depth < self.max_depth:
            if node.id in vetoed:
                return 0.0
            node = self.store.get(node.parent_id) if node.parent_id else None
            depth += 1
        return 1.0


def append_directive(path: Path | str, directive: dict) -> dict:
    """Server-side helper: validate, assign id, bump version, write the
    control file atomically. Returns the stored directive."""
    path = Path(path)
    data = {"version": 0, "directives": []}
    if path.exists():
        data = json.loads(path.read_text())
    data["version"] = int(data.get("version", 0)) + 1
    directive = dict(directive)
    directive["id"] = f"d{data['version']}"
    Directive.from_json(directive)  # validate
    data.setdefault("directives", []).append(directive)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return directive
