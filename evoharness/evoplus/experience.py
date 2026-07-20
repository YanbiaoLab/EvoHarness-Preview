# EvoHarness original research extension (plan C3: retrievable experience
# store, repositioned after discovering upstream's meta-recommendations
# already flow back into prompts). Differentiation vs upstream: experience
# is indexed by the parent's failure modes and retrieved per-mutation
# ("retrieval" mode); "global" mode reproduces the upstream-style single
# shared cheatsheet as an ablation arm.
"""C3: failure-mode-indexed experience store and its prompt contributor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from evoharness.evocore.interfaces import MutationContext
from evoharness.evocore.llm import LLMClient
from evoharness.evocore.population import Candidate, PopulationStore

from .feedback import StructuredFeedback


@dataclass
class ExperienceEntry:
    parent_error_categories: list[str]
    operator: str
    change_title: str
    change_summary: str
    fitness_delta: float
    success: bool
    generation: int

    def to_json(self) -> dict:
        return {
            "parent_error_categories": self.parent_error_categories,
            "operator": self.operator,
            "change_title": self.change_title,
            "change_summary": self.change_summary,
            "fitness_delta": self.fitness_delta,
            "success": self.success,
            "generation": self.generation,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ExperienceEntry":
        return cls(**d)

    def render(self) -> str:
        verb = "helped" if self.success else "did NOT help"
        title = self.change_title or self.operator
        line = f"- [{self.operator}] {title} — {verb} (Δfitness {self.fitness_delta:+.4g})"
        if self.change_summary:
            line += f": {self.change_summary}"
        return line


class ExperienceStore:
    """Implements LoopObserver: records one entry per graded mutation whose
    parent has a report. Persisted as JSON lines for post-hoc analysis."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.entries: list[ExperienceEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.entries.append(ExperienceEntry.from_json(json.loads(line)))

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        if not cand.parent_id or cand.report is None:
            return
        parent = store.get(cand.parent_id)
        if parent is None or parent.report is None:
            return
        categories: list[str] = []
        if parent.report.structured_feedback:
            categories = StructuredFeedback.from_json(
                parent.report.structured_feedback
            ).top_error_categories()
        delta = cand.report.fitness - parent.report.fitness
        entry = ExperienceEntry(
            parent_error_categories=categories,
            operator=cand.operator,
            change_title=cand.change_title,
            change_summary=cand.change_summary,
            fitness_delta=delta,
            success=cand.report.passed and delta > 0,
            generation=cand.generation,
        )
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry.to_json()) + "\n")

    def query(
        self, categories: list[str], top_m: int = 3, top_n: int = 2
    ) -> tuple[list[ExperienceEntry], list[ExperienceEntry]]:
        """Entries whose parent shared at least one failure category.
        Returns (top_m effective mutations, top_n ineffective lessons).
        Exact category matching: the store is small (~150 entries/run), no
        vector retrieval needed."""
        wanted = set(categories)
        matched = [
            e for e in self.entries
            if wanted and wanted.intersection(e.parent_error_categories)
        ]
        wins = sorted(
            (e for e in matched if e.success),
            key=lambda e: -e.fitness_delta,
        )[:top_m]
        losses = sorted(
            (e for e in matched if not e.success),
            key=lambda e: e.fitness_delta,
        )[:top_n]
        return wins, losses


@dataclass
class Cheatsheet:
    version: int
    generation: int
    text: str


_DISTILL_SYS = (
    "You maintain a compact evolution cheatsheet for a program-search loop. "
    "From the mutation history below, distill: proven successful patterns, "
    "confirmed dead ends, and promising unexplored directions. "
    "Be specific and non-repetitive. Hard limit: 2000 characters."
)


class ExperienceContributor:
    """Implements PromptContributor. Modes (experiment matrix):
    off       -> contributes nothing (E0/E1/E2)
    global    -> one shared LLM-distilled cheatsheet, refreshed every
                 `interval` generations (upstream-style arm, E3g)
    retrieval -> per-mutation lookup by the parent's top error categories,
                 rendered directly without an extra LLM call (E3r)"""

    def __init__(
        self,
        store: ExperienceStore,
        mode: str = "off",
        llm: LLMClient | None = None,
        model: str = "",
        interval: int = 5,
        max_bytes: int = 2048,
        top_m: int = 3,
        top_n: int = 2,
    ):
        if mode not in ("off", "global", "retrieval"):
            raise ValueError(f"unknown experience mode {mode!r}")
        if mode == "global" and llm is None:
            raise ValueError("global mode requires an LLM client")
        self.store = store
        self.mode = mode
        self.llm = llm
        self.model = model
        self.interval = interval
        self.max_bytes = max_bytes
        self.top_m = top_m
        self.top_n = top_n
        self.cheatsheets: list[Cheatsheet] = []

    def state(self) -> dict:
        return {
            "cheatsheets": [
                {"version": c.version, "generation": c.generation, "text": c.text}
                for c in self.cheatsheets
            ]
        }

    def set_state(self, state: dict) -> None:
        self.cheatsheets = [
            Cheatsheet(**c) for c in state.get("cheatsheets", [])
        ]

    def contribute(self, ctx: MutationContext) -> str | None:
        if self.mode == "off":
            return None
        if self.mode == "retrieval":
            return self._retrieve(ctx)
        return self._global(ctx)

    def _retrieve(self, ctx: MutationContext) -> str | None:
        categories: list[str] = []
        report = ctx.parent.report
        if report is not None and report.structured_feedback:
            categories = StructuredFeedback.from_json(
                report.structured_feedback
            ).top_error_categories()
        if not categories:
            return None
        wins, losses = self.store.query(categories, self.top_m, self.top_n)
        if not wins and not losses:
            return None
        lines = [
            "# Experience from similar failure modes",
            f"(parent's failure categories: {', '.join(categories)})",
        ]
        lines += [e.render() for e in wins]
        lines += [e.render() for e in losses]
        text = "\n".join(lines)
        return text[: self.max_bytes]

    def _global(self, ctx: MutationContext) -> str | None:
        current = self.cheatsheets[-1] if self.cheatsheets else None
        stale = (
            current is None
            or ctx.generation - current.generation >= self.interval
        )
        if stale and self.store.entries:
            history = "\n".join(
                e.render() for e in self.store.entries[-50:]
            )
            previous = current.text if current else "(none yet)"
            user = (
                f"Previous cheatsheet:\n{previous}\n\n"
                f"Recent mutation history:\n{history}\n\n"
                "Produce the updated cheatsheet."
            )
            resp = self.llm.query(_DISTILL_SYS, user, self.model)
            self.cheatsheets.append(
                Cheatsheet(
                    version=len(self.cheatsheets) + 1,
                    generation=ctx.generation,
                    text=resp.text.strip()[: self.max_bytes],
                )
            )
            current = self.cheatsheets[-1]
        if current is None:
            return None
        return (
            f"# Evolution cheatsheet (v{current.version})\n{current.text}"
        )
