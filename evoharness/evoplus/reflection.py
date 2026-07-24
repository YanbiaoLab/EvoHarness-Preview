# EvoHarness original research extension (L1 attribution layer, design in
# todo/experience_buffer_design.md §2b). Post-evaluation LLM reflection:
# graded mutations are attributed in batches (improved / regressed / noise)
# into per-mutation lessons plus one bounded shared scratchpad.
# Reference systems: SkillOpt reflect->aggregate (textual gradient over
# failure trajectories), ShinkaEvolve MetaSummarizer (interval-batched meta
# memory, bounded scratchpad). The "noise" verdict is our counterpart of
# SAR's Execution Lapse: with a small single-shot validation set, a delta
# inside the noise floor must not become a lesson.
"""L1: batched LLM attribution over the experience evidence buffer."""

from __future__ import annotations

import json
import logging

from evoharness.evocore.interfaces import BudgetLike
from evoharness.evocore.llm import LLMClient
from evoharness.evocore.population import Candidate, PopulationStore

from .experience import ExperienceEntry, ExperienceStore
from .feedback import StructuredFeedback

logger = logging.getLogger(__name__)

VERDICTS = ("improved", "regressed", "noise")

_REFLECT_SYS = """\
You are the attribution analyst for an evolutionary program-search loop.
You receive a batch of code mutations with measured outcomes, plus the
current shared scratchpad.

For EACH mutation decide a verdict:
- "improved": the edit itself plausibly caused the fitness gain.
- "regressed": the edit itself plausibly caused the loss.
- "noise": the outcome is within evaluation noise (small delta, few
  behavior flips) — no lesson should be drawn. Prefer "noise" when the
  evidence is weak: a wrong lesson is worse than no lesson.

For each mutation also write:
- "why": one sentence attributing the outcome to something concrete in
  the edit (not a restatement of the delta).
- "advice": one actionable sentence for future mutations that face a
  similar failure mode.
- "tags": 1-3 lowercase-slug failure-mode tags. REUSE an existing tag
  whenever one fits; invent a new one only for a genuinely new mode.

Then rewrite the shared scratchpad (hard limit {max_bytes} characters)
with exactly three sections: "Successful patterns", "Ineffective
approaches", "Unexplored directions". Merge with the previous version,
deduplicate, stay specific. Exclude noise-verdict mutations. If many
proposals were rejected as near-duplicates, reflect that under
"Unexplored directions".

Respond with ONLY a JSON object, no prose, no code fences:
{{"lessons": [{{"child_id": "...", "verdict": "...", "why": "...",
"advice": "...", "tags": ["..."]}}], "scratchpad": "..."}}
"""

_CONSOLIDATE_SYS = """\
You maintain the lesson memory of an evolutionary program-search loop.
The lessons below have accumulated tag drift and redundancy. Produce:
1. "tag_map": merge synonymous failure-mode tags — map each alias to ONE
   canonical lowercase slug. Identity mappings may be omitted.
2. "rewrites": for lessons marked REDEEMED (their regression later led to
   an improvement), rewrite "advice" as a stepping-stone lesson: the
   direction was worth pursuing despite the short-term loss; recommend
   keeping the direction with smaller, safer steps.
Do not invent new lessons or touch unlisted ones. Respond with ONLY a
JSON object:
{"tag_map": {"alias": "canonical"}, "rewrites": [{"child_id": "...",
"advice": "..."}]}
"""

def _extract_json(text: str) -> dict:
    """Tolerate code fences / prose around the JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reflection response")
    return json.loads(text[start : end + 1])


def _signature(cand: Candidate | None):
    if cand is None or cand.report is None:
        return None
    if not cand.report.structured_feedback:
        return None
    return StructuredFeedback.from_json(
        cand.report.structured_feedback
    ).signature()

class MutationReflector:
    """Implements LoopObserver (register AFTER the ExperienceStore so the
    entry exists when this runs). Every `batch_size` unlessoned evaluated
    entries, ONE LLM call writes a lesson per mutation and rewrites the
    scratchpad. Any LLM/parse failure skips the batch — entries stay
    pending for the next attempt; reflection must never kill the run."""

    def __init__(
            self,
            store: ExperienceStore,
            llm: LLMClient,
            model: str = "",
            batch_size: int = 8,
            max_batch: int = 16,
            max_scratchpad_bytes: int = 2000,
            noise_hamming: int = 1,
            max_examples: int = 2,
            consolidate_threshold: int = 40,
            budget: BudgetLike | None = None,
    ):
        self.store = store
        self.llm = llm
        self.model = model
        self.batch_size = batch_size
        self.max_batch = max_batch
        self.max_scratchpad_bytes = max_scratchpad_bytes
        self.noise_hamming = noise_hamming
        self.max_examples = max_examples
        self.consolidate_threshold = consolidate_threshold
        self.budget = budget
        self.consolidated_at = 0
        self.scratchpad = ""
        self.scratchpad_version = 0

    def state(self) -> dict:
        return {
            "scratchpad": self.scratchpad,
            "scratchpad_version": self.scratchpad_version,
            "consolidated_at": self.consolidated_at,
        }

    def set_state(self, state: dict) -> None:
        self.scratchpad = state.get("scratchpad", "")
        self.scratchpad_version = int(state.get("scratchpad_version", 0))
        self.consolidated_at = int(state.get("consolidated_at", 0))

    def on_candidate_graded(
            self, cand: Candidate, store: PopulationStore
    ) -> None:
        pending = self.store.pending_reflection()
        if len(pending) < self.batch_size:
            return
        self._reflect(pending[: self.max_batch], store)

    def _reflect(
            self, batch: list[ExperienceEntry], pop: PopulationStore
    ) -> None:
        try:
            user = self._build_user(batch, pop)
            system = _REFLECT_SYS.format(max_bytes=self.max_scratchpad_bytes)
            resp = self.llm.query(system, user, self.model)
            self._charge(resp)
            parsed = _extract_json(resp.text)
            lessons = self._normalize(parsed.get("lessons", []), batch)
        except Exception as exc:  # noqa: BLE001 — reflection is best-effort
            logger.warning("reflection batch skipped: %s", exc)
            return
        for child_id, lesson in lessons.items():
            self.store.attach_lesson(child_id, lesson)
        scratchpad = str(parsed.get("scratchpad", "")).strip()
        if scratchpad:
            self.scratchpad = scratchpad[: self.max_scratchpad_bytes]
            self.scratchpad_version += 1
        # Consolidation check runs AFTER the fresh lessons are attached, so
        # the watermark comparison sees this batch's contribution.
        lessoned = sum(
            1 for e in self.store.entries
            if e.kind == "evaluated" and e.lesson is not None
        )
        if lessoned - self.consolidated_at >= self.consolidate_threshold:
            self._consolidate()

    def _normalize(
            self, raw_lessons: list, batch: list[ExperienceEntry]
    ) -> dict[str, dict]:
        """Keep only well-formed lessons for ids actually in the batch;
        hallucinated ids and bad verdicts are dropped, not fatal."""
        batch_ids = {e.child_id for e in batch}
        out: dict[str, dict] = {}
        for raw in raw_lessons:
            if not isinstance(raw, dict):
                continue
            child_id = str(raw.get("child_id", ""))
            verdict = str(raw.get("verdict", "")).strip().lower()
            if child_id not in batch_ids or verdict not in VERDICTS:
                continue
            tags = [
                str(t).strip().lower().replace(" ", "-")
                for t in raw.get("tags", [])
                if str(t).strip()
            ]
            out[child_id] = {
                "verdict": verdict,
                "why": str(raw.get("why", ""))[:300],
                "advice": str(raw.get("advice", ""))[:300],
                "tags": tags[:3],
            }
        return out

    def _charge(self, resp) -> None:
        if self.budget is not None:
            self.budget.charge(getattr(resp, "cost", 0.0))

    def _consolidate(self) -> None:
        """SkillOpt-style consolidation (design §2c): merge synonymous tags,
        rewrite redeemed regressions as stepping stones. Best-effort — a
        failure leaves the watermark unchanged so the next batch retries."""
        pool = self.store.lesson_entries()
        if not pool:
            return
        lines = []
        for e in pool:
            lesson = e.lesson or {}
            flag = f" REDEEMED by {e.redeemed_by}" if e.redeemed_by else ""
            lines.append(
                f"- {e.child_id} | {lesson.get('verdict')} | "
                f"tags: {', '.join(lesson.get('tags', []))} | "
                f"advice: {lesson.get('advice', '')}{flag}"
            )
        try:
            resp = self.llm.query(
                _CONSOLIDATE_SYS, "## Lessons\n" + "\n".join(lines), self.model
            )
            self._charge(resp)
            parsed = _extract_json(resp.text)
            tag_map = {
                str(k).strip().lower(): str(v).strip().lower()
                for k, v in dict(parsed.get("tag_map", {})).items()
            }
            rewrites = {
                str(r.get("child_id", "")): str(r.get("advice", ""))[:300]
                for r in parsed.get("rewrites", [])
                if isinstance(r, dict) and r.get("advice")
            }
        except Exception as exc:  # noqa: BLE001 — best-effort, retry later
            logger.warning("consolidation skipped: %s", exc)
            return
        for e in pool:
            lesson = dict(e.lesson)
            tags, seen = [], set()
            for t in lesson.get("tags", []):
                canon = tag_map.get(t, t)
                if canon and canon not in seen:
                    seen.add(canon)
                    tags.append(canon)
            changed = tags != lesson.get("tags", [])
            # Rewrites apply ONLY to redeemed entries: the LLM must not
            # rewrite arbitrary advice.
            if e.redeemed_by and e.child_id in rewrites:
                lesson["advice"] = rewrites[e.child_id]
                changed = True
            if changed:
                lesson["tags"] = tags
                self.store.attach_lesson(e.child_id, lesson)
        self.consolidated_at = sum(
            1 for e in self.store.entries
            if e.kind == "evaluated" and e.lesson is not None
        )

    # -- prompt assembly -----------------------------------------------------------

    def _build_user(
            self, batch: list[ExperienceEntry], pop: PopulationStore
    ) -> str:
        blocks = [self._entry_block(e, pop) for e in batch]
        vocab = sorted({
            t
            for e in self.store.entries
            if e.lesson
            for t in e.lesson.get("tags", [])
        })
        window_start = min(e.generation for e in batch)
        rejections = [
            e for e in self.store.entries
            if e.generation >= window_start
               and e.kind in ("rejected_novelty", "proposal_failed")
        ]
        n_dup = sum(1 for e in rejections if e.kind == "rejected_novelty")
        n_fail = len(rejections) - n_dup
        parts = [
            "## Previous scratchpad\n" + (self.scratchpad or "(none yet)"),
            "## Existing tags (reuse before inventing)\n"
            + (", ".join(vocab) or "(none yet)"),
            f"## Rejections in this window\n{n_dup} near-duplicate "
            f"proposals rejected, {n_fail} proposals failed to produce "
            "a valid edit.",
            "## Mutations to attribute\n" + "\n\n".join(blocks),
        ]
        return "\n\n".join(parts)

    def _entry_block(self, e: ExperienceEntry, pop: PopulationStore) -> str:
        lines = [
            f"### mutation {e.child_id}",
            f"operator: {e.operator}; generation {e.generation}; "
            f"island {e.island_idx}",
        ]
        if e.change_title:
            lines.append(f"title: {e.change_title}")
        if e.change_summary:
            lines.append(f"diff: {e.change_summary}")
        lines.append(f"fitness delta: {e.fitness_delta:+.4g}")
        child = pop.get(e.child_id)
        parent = pop.get(e.parent_id) if e.parent_id else None
        child_sig, parent_sig = _signature(child), _signature(parent)
        if child_sig is not None and parent_sig is not None:
            n_items = len(child_sig.pass_vector)
            flips = parent_sig.hamming(child_sig)
            lines.append(
                f"behavior: {flips} of {n_items} items flipped pass/fail "
                f"vs parent (single-shot eval, noise floor ~{1 / n_items:.3f})"
            )
            if (
                    abs(e.fitness_delta) <= 1 / n_items
                    and flips <= self.noise_hamming
            ):
                lines.append(
                    "NOTE: delta and behavior change are within the noise "
                    "floor — strongly consider verdict \"noise\"."
                )
        failures = self._failures(child)
        if failures:
            lines.append("child failures: " + " | ".join(failures))
        return "\n".join(lines)

    def _failures(self, cand: Candidate | None) -> list[str]:
        if cand is None or cand.report is None:
            return []
        if not cand.report.structured_feedback:
            return []
        feedback = StructuredFeedback.from_json(
            cand.report.structured_feedback
        )
        out = []
        for item in feedback.items:
            if item.passed:
                continue
            desc = f"item {item.item_id}"
            if item.error_category:
                desc += f" [{item.error_category}]"
            if item.expected or item.predicted:
                desc += (
                    f": expected {item.expected!r}, got {item.predicted!r}"
                )
            out.append(desc[:160])
            if len(out) >= self.max_examples:
                break
        return out

