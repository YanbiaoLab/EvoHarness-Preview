# EvoHarness original research extension (plan C3, v2 L0 evidence layer).
# Differentiation vs upstream (ShinkaEvolve): pre-evaluation rejections
# (novelty gate, failed proposals) are captured and fed back into mutation
# prompts as negative experience — upstream only logs them for accounting.
# Failure-mode indexing was intentionally dropped (2026-07-24): it required
# per-item error categories only some domains can produce; retrieval by
# failure mode moves to the reflection layer's open-vocabulary tags
# (todo/experience_buffer_design.md §2b'). This layer stays mechanical.
"""C3/L0: evidence buffer of mutation outcomes and its prompt contributor."""

from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from evoharness.core.interfaces import MutationContext, RejectionEvent
from evoharness.core.llm import LLMClient
from evoharness.core.population import Candidate, PopulationStore


def _summarize_patch(patch: str, *, max_added: int, max_chars: int) -> str:
    """Render a unified-diff patch as 'file +A/-D; ... | added: <key lines>'."""
    per_file: dict[str, list[int]] = {}
    added: list[str] = []
    current = "?"
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:].strip()
            per_file.setdefault(current, [0, 0])
        elif line.startswith("diff --git"):
            current = line.split(" b/")[-1].strip()
            per_file.setdefault(current, [0, 0])
        elif line.startswith("+") and not line.startswith("+++"):
            per_file.setdefault(current, [0, 0])[0] += 1
            content = line[1:].strip()
            if len(content) > 8:
                added.append(content)
        elif line.startswith("-") and not line.startswith("---"):
            per_file.setdefault(current, [0, 0])[1] += 1
    if not per_file:
        return ""
    # Reserve room for the `| added:` tail before spending budget on the file
    # list. The earlier version concatenated an unbounded file list with the tail
    # and truncated the whole string, so a patch touching many files pushed the
    # tail off the cliff — and the tail is the only part that says WHAT changed.
    # A workspace polluted with stray temp files was enough to consume the entire
    # budget with filenames, leaving later generations able to see that a change
    # failed but not what it was. The file list is the low-density part; it is
    # what should be elided.
    tail = (" | added: " + " ¶ ".join(added[:max_added])) if added else ""
    budget = max(40, max_chars - len(tail))
    parts = [f"{p} +{a}/-{d}" for p, (a, d) in per_file.items()]
    desc = ""
    for i, part in enumerate(parts):
        nxt = f"{desc}; {part}" if desc else part
        if len(nxt) > budget:
            desc = (desc or part[:budget]) + f"; +{len(parts) - i} more files"
            break
        desc = nxt
    return (desc + tail)[:max_chars]


def describe_change(
    parent: Candidate,
    child: Candidate,
    *,
    max_added: int = 3,
    max_chars: int = 240,
) -> str:
    """Describe the parent->child mutation from its canonical git patch.

    Git-backed candidates ARE a patch stack, so the mutation's exact diff is
    child.workspace.patches[len(parent.patches):] — the real `git diff` the
    proposer produced. Read it directly instead of re-diffing texts (agent
    self-reported titles/summaries are unreliable and empty here). Falls back
    to a synthesized text diff only for non-git workspaces.
    """
    try:
        cw, pw = child.workspace, parent.workspace
    except Exception:
        return ""
    if getattr(cw, "kind", None) == "git" and getattr(pw, "kind", None) == "git":
        new = cw.patches[len(pw.patches):]
        return _summarize_patch(
            "\n".join(new), max_added=max_added, max_chars=max_chars
        )
    try:
        parent_texts, child_texts = pw.texts(), cw.texts()
    except Exception:
        return ""
    lines: list[str] = []
    for path in sorted(set(parent_texts) | set(child_texts)):
        a = parent_texts.get(path, "").splitlines()
        b = child_texts.get(path, "").splitlines()
        if a == b:
            continue
        lines.append(f"+++ b/{path}")
        lines += list(difflib.unified_diff(a, b, lineterm=""))
    return _summarize_patch(
        "\n".join(lines), max_added=max_added, max_chars=max_chars
    )


# Substrings that mark an infrastructure failure. The test is whether the reason
# describes the machine or the candidate: endpoint refusals, protocol errors,
# timeouts and dropped connections say nothing about the mutation, while
# "no valid edit produced" or "turn limit reached" are real search signal.
_INFRA_FAILURE_MARKS = (
    "backend-error", "backend_error", "protocol_error",
    "llm query failed", "http error", "connection", "timeout",
    "insufficient_balance", "forbidden", "overload",
)


def _is_infra_failure(reason: str) -> bool:
    low = reason.lower()
    return any(m in low for m in _INFRA_FAILURE_MARKS)


def _describe_proposal(
    parent: Candidate,
    workspace,
    code: str | None,
    *,
    max_added: int = 3,
    max_chars: int = 240,
) -> str:
    """describe_change for a proposal that never became a Candidate: git
    patch tail when both sides are git-backed, else a main-text diff."""
    try:
        pw = parent.workspace
    except Exception:
        return ""
    if (
        workspace is not None
        and getattr(workspace, "kind", None) == "git"
        and getattr(pw, "kind", None) == "git"
    ):
        new = workspace.patches[len(pw.patches):]
        return _summarize_patch(
            "\n".join(new), max_added=max_added, max_chars=max_chars
        )
    if code is None:
        return ""
    try:
        a = pw.main_text().splitlines()
    except Exception:
        return ""
    b = code.splitlines()
    if a == b:
        return ""
    lines = ["+++ b/main"] + list(difflib.unified_diff(a, b, lineterm=""))
    return _summarize_patch(
        "\n".join(lines), max_added=max_added, max_chars=max_chars
    )


def _report_evidence(
    parent_report,
    child_report,
    *,
    max_metrics: int = 4,
    max_chars: int = 300,
) -> str:
    """Compress WHY a mutation scored what it did, domain-agnostically.

    Motivating autopsy (modmul r12): four candidates raised the Horner radix
    and their buffer entries read "did NOT help (Δfitness -0.78): arch.py
    +1/-1" — the WHAT. The WHY (inference time halved exactly as projected
    while accuracy above tier 1 collapsed to 2-3%) was sitting in the graded
    report's fault string and visible_metrics the whole time, and never
    reached a single future proposal. Fitness deltas say that something
    died; the evidence says what killed it, and only the second prevents
    the next attempt from dying the same way.

    Mechanically: the fault first, then the metrics whose RELATIVE change
    vs the parent is largest (an accuracy 1.0 -> 0.02 outranks a timing
    121 -> 51), then changed short strings (a warm_start flip is evidence
    too). The metrics stay opaque key-values — nothing here knows any
    domain's names, which is what keeps this in the framework layer.
    """
    parts: list[str] = []
    fault = (getattr(child_report, "fault", None) or "").strip()
    if fault:
        parts.append("fault: " + " ".join(fault.split())[:140])
    parent_m = getattr(parent_report, "visible_metrics", None) or {}
    child_m = getattr(child_report, "visible_metrics", None) or {}
    numeric: list[tuple[float, str, float, float]] = []
    worded: list[str] = []
    for key in child_m:
        pv, cv = parent_m.get(key), child_m[key]
        if isinstance(pv, bool) or isinstance(cv, bool):
            continue
        if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
            if pv != cv:
                rel = abs(pv - cv) / max(abs(pv), abs(cv), 1e-9)
                numeric.append((rel, key, float(pv), float(cv)))
        elif (
            isinstance(pv, str) and isinstance(cv, str) and pv != cv
            and len(pv) <= 24 and len(cv) <= 24
        ):
            worded.append(f"{key} {pv}->{cv}")
    numeric.sort(key=lambda t: (-t[0], t[1]))
    shifted = [f"{k} {pv:g}->{cv:g}" for _, k, pv, cv in numeric[:max_metrics]]
    shifted += worded[: max(0, max_metrics - len(shifted))]
    if shifted:
        parts.append("shifted: " + ", ".join(shifted))
    return " | ".join(parts)[:max_chars]


@dataclass
class ExperienceEntry:
    # Schema v2 (design §3). `kind` discriminates: "evaluated" rows come
    # from graded children (v1 rows load as this kind via from_json
    # defaults), the other two are pre-evaluation rejections.
    operator: str = ""
    change_title: str = ""
    change_summary: str = ""
    fitness_delta: float = 0.0
    success: bool = False
    generation: int = 0
    kind: str = "evaluated"  # | "rejected_novelty" | "proposal_failed"
    island_idx: int = 0
    parent_id: str = ""
    child_id: str = ""
    redeemed_by: str | None = None
    max_similarity: float = 0.0
    most_similar_id: str = ""
    # L1 attribution (design §2b), written back by MutationReflector:
    # {"verdict": "improved|regressed|noise", "why", "advice", "tags": [...]}
    lesson: dict | None = None
    # Schema v3: compact WHY from the graded report (fault + largest metric
    # shifts vs the parent) — see _report_evidence. Older rows load as "".
    evidence: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "ExperienceEntry":
        # Filter to known fields: v1 rows carry dropped keys (e.g.
        # parent_error_categories) and future rows may carry new ones.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def render(self, *, show_origin: bool = False) -> str:
        origin = (
            f" (gen {self.generation}, island {self.island_idx})"
            if show_origin
            else ""
        )
        if self.kind == "rejected_novelty":
            target = self.most_similar_id or "an existing candidate"
            line = (
                f"- [{self.operator}] REJECTED before eval: near-duplicate "
                f"of {target} (similarity {self.max_similarity:.3f}){origin}"
            )
            if self.change_summary:
                line += f": {self.change_summary}"
            return line
        if self.kind == "proposal_failed":
            reason = self.change_summary or "no valid edit produced"
            return f"- [{self.operator}] PROPOSAL FAILED{origin}: {reason}"
        verb = "helped" if self.success else "did NOT help"
        title = self.change_title or self.operator
        line = (
            f"- [{self.operator}] {title} — {verb} "
            f"(Δfitness {self.fitness_delta:+.4g}){origin}"
        )
        if self.change_summary:
            line += f": {self.change_summary}"
        if self.evidence:
            line += f"\n  evidence: {self.evidence}"
        return line


class ExperienceStore:
    """Implements LoopObserver + RejectionObserver. One JSONL row per event.
    The file stays append-only: "redemption" rows are write-back updates
    applied last-write-wins on load (design §6)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.entries: list[ExperienceEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "redemption":
                    self._apply_redemption(row["child_id"], row["redeemed_by"])
                elif row.get("kind") == "lesson":
                    self._apply_lesson(row["child_id"], row["lesson"])
                else:
                    self.entries.append(ExperienceEntry.from_json(row))

    def _append_row(self, row: dict) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _apply_redemption(self, child_id: str, redeemer_id: str) -> bool:
        hit = False
        for e in self.entries:
            if (
                e.kind == "evaluated"
                and e.child_id == child_id
                and not e.success
                and e.redeemed_by is None
            ):
                e.redeemed_by = redeemer_id
                hit = True
        return hit

    def _apply_lesson(self, child_id: str, lesson: dict) -> bool:
        hit = False
        for e in self.entries:
            if e.kind == "evaluated" and e.child_id == child_id:
                e.lesson = lesson
                hit = True
        return hit

    def attach_lesson(self, child_id: str, lesson: dict) -> None:
        """L1 write-back: same append-only mechanics as redemption rows."""
        if self._apply_lesson(child_id, lesson):
            self._append_row(
                {"kind": "lesson", "child_id": child_id, "lesson": lesson}
            )

    def pending_reflection(self) -> list[ExperienceEntry]:
        """Evaluated entries no reflection batch has covered yet (FIFO)."""
        return [
            e for e in self.entries
            if e.kind == "evaluated" and e.lesson is None
        ]

    def parent_tags(self, parent_id: str) -> list[str]:
        """Failure-mode tags the reflector assigned to the parent's own
        creation entry — the retrieval key after categories were dropped
        (design §2b')."""
        for e in self.entries:
            if e.kind == "evaluated" and e.child_id == parent_id and e.lesson:
                return list(e.lesson.get("tags", []))
        return []

    def lesson_entries(self) -> list[ExperienceEntry]:
        """Evaluated entries carrying a non-noise lesson (L1 output)."""
        return [
            e for e in self.entries
            if e.kind == "evaluated"
            and e.lesson is not None
            and e.lesson.get("verdict") != "noise"
        ]

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        if not cand.parent_id or cand.report is None:
            return
        parent = store.get(cand.parent_id)
        if parent is None or parent.report is None:
            return
        delta = cand.report.fitness - parent.report.fitness
        entry = ExperienceEntry(
            kind="evaluated",
            operator=cand.operator,
            change_title=cand.change_title,
            # Diff-derived description first; agent self-report is unreliable.
            change_summary=describe_change(parent, cand) or cand.change_summary,
            fitness_delta=delta,
            success=cand.passed and delta > 0,
            generation=cand.generation,
            island_idx=cand.island_idx,
            parent_id=parent.id,
            child_id=cand.id,
            evidence=_report_evidence(parent.report, cand.report),
        )
        self.entries.append(entry)
        self._append_row(entry.to_json())
        # Redemption (design §6): an improving child retroactively clears its
        # parent's own "regressed" entry from the negative sections — the
        # g6->g8 stepping-stone pattern must not poison the prompt.
        if entry.success and self._apply_redemption(parent.id, cand.id):
            self._append_row(
                {"kind": "redemption", "child_id": parent.id, "redeemed_by": cand.id}
            )

    def on_proposal_rejected(self, event: RejectionEvent) -> None:
        parent = event.parent
        common = dict(
            operator=event.operator,
            generation=event.generation,
            island_idx=parent.island_idx,
            parent_id=parent.id,
        )
        if event.kind == "novelty":
            entry = ExperienceEntry(
                kind="rejected_novelty",
                change_title=event.change_title,
                change_summary=_describe_proposal(
                    parent, event.proposal_workspace, event.proposal_code
                ),
                max_similarity=event.max_similarity,
                most_similar_id=event.most_similar_id or "",
                **common,
            )
        else:
            # Infrastructure failures are not search signal, so they stay out of
            # the store. The distinction is load-bearing: "this change did not
            # help" is about the candidate and later generations should learn it;
            # "the endpoint did not answer" is about the machine and says nothing.
            # Mixing them starves the reflector — a run whose store was 70%
            # endpoint errors distilled to "no measured successful pattern yet"
            # even though it contained that run's single largest gain.
            if _is_infra_failure(event.failure_reason or ""):
                return
            entry = ExperienceEntry(
                kind="proposal_failed",
                change_summary=(event.failure_reason or "")[:240],
                **common,
            )
        self.entries.append(entry)
        self._append_row(entry.to_json())

    _PATTERN_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    @classmethod
    def _pattern_key(cls, entry: ExperienceEntry) -> tuple[str, str] | None:
        """(files touched, first identifier added) — the convergence key.

        Derived from the diff-summary format _summarize_patch emits
        ("arch.py +1/-1 | added: RADIX_BITS = 2"), so it is exactly as
        domain-agnostic as the diff itself. Entries whose summary carries
        no added line get no key and never cluster.
        """
        summary = entry.change_summary or ""
        files_part, sep, added = summary.partition(" | added: ")
        if not sep:
            return None
        files = ",".join(sorted(
            chunk.split(" ")[0]
            for chunk in files_part.split("; ") if chunk.strip()
        ))
        match = cls._PATTERN_IDENT.search(added.split(" ¶ ")[0])
        if not match:
            return None
        return (files, match.group(0))

    def loss_clusters(self, top_n: int = 2) -> list[list[ExperienceEntry]]:
        """Unredeemed evaluated failures, grouped by convergence key.

        Four independent candidates raising the same constant and dying the
        same way is ONE pattern with four bodies, not four unrelated lines
        — and repeated independent failure is the strongest negative signal
        the buffer holds, precisely the one that per-instance rendering
        dilutes (with top_n=2, two of the four r12 radix deaths displaced
        every other failure mode and the count "4" appeared nowhere).
        Returns clusters worst-first by (size, worst delta); singletons are
        clusters of one.
        """
        losses = [
            e for e in self.entries
            if e.kind == "evaluated" and not e.success and e.redeemed_by is None
        ]
        groups: dict[object, list[ExperienceEntry]] = {}
        for index, entry in enumerate(losses):
            key = self._pattern_key(entry) or ("", f"solo-{index}")
            groups.setdefault(key, []).append(entry)
        return sorted(
            groups.values(),
            key=lambda g: (-len(g), min(e.fitness_delta for e in g)),
        )[:top_n]

    @staticmethod
    def render_cluster(cluster: list[ExperienceEntry]) -> str:
        if len(cluster) == 1:
            return cluster[0].render()
        worst = min(cluster, key=lambda e: e.fitness_delta)
        best = max(cluster, key=lambda e: e.fitness_delta)
        title = worst.change_title or worst.operator
        line = (
            f"- [{worst.operator}] {title} — tried {len(cluster)} times "
            f"INDEPENDENTLY, helped 0 times (Δfitness "
            f"{worst.fitness_delta:+.4g}..{best.fitness_delta:+.4g})"
        )
        if worst.change_summary:
            line += f": {worst.change_summary}"
        if worst.evidence:
            line += f"\n  evidence (worst instance): {worst.evidence}"
        return line

    def recent_wins(self, top_m: int = 3) -> list[ExperienceEntry]:
        """Best evaluated improvements, best-first."""
        wins = [e for e in self.entries if e.kind == "evaluated" and e.success]
        return sorted(wins, key=lambda e: -e.fitness_delta)[:top_m]

    def recent_losses(self, top_n: int = 2) -> list[ExperienceEntry]:
        """Worst unredeemed evaluated regressions, worst-first."""
        losses = [
            e for e in self.entries
            if e.kind == "evaluated" and not e.success and e.redeemed_by is None
        ]
        return sorted(losses, key=lambda e: e.fitness_delta)[:top_n]

    def recent_negatives(
        self,
        generation: int,
        window: int = 10,
        top_r: int = 5,
        island_idx: int | None = None,
    ) -> list[ExperienceEntry]:
        """Design §5b: pre-eval rejections + unredeemed regressions inside
        the sliding window; same-island entries first, then recency."""
        recent = [
            e for e in self.entries
            if e.generation >= generation - window
            and (
                e.kind in ("rejected_novelty", "proposal_failed")
                or (
                    e.kind == "evaluated"
                    and not e.success
                    and e.redeemed_by is None
                    # Lessoned regressions render via the lessons section;
                    # noise-verdict ones must not render anywhere (§2b).
                    and e.lesson is None
                )
            )
        ]
        if island_idx is not None:
            recent.sort(key=lambda e: (e.island_idx != island_idx, -e.generation))
        else:
            recent.sort(key=lambda e: -e.generation)
        return recent[:top_r]


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
    off                -> contributes nothing (E0/E1/E2)
    global             -> one shared LLM-distilled cheatsheet (E3g)
    retrieval          -> recent wins/losses from the evidence buffer (E3r)
    retrieval+rejected -> E3r plus the negative-experience section (E4a)
    lessons            -> L1 lessons matched on the parent's failure tags,
                          mechanical fallback before the first batch (E5r)
    lessons+scratchpad -> E5r plus one sampled scratchpad direction (E5s)"""

    _MODES = (
        "off",
        "global",
        "retrieval",
        "retrieval+rejected",
        "lessons",
        "lessons+scratchpad",
    )

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
        window: int = 10,
        top_r: int = 5,
        max_reject_bytes: int = 1024,
        reflector: object | None = None,
    ):
        if mode not in self._MODES:
            raise ValueError(f"unknown experience mode {mode!r}")
        if mode == "global" and llm is None:
            raise ValueError("global mode requires an LLM client")
        if mode == "lessons+scratchpad" and reflector is None:
            raise ValueError(
                "lessons+scratchpad mode requires a MutationReflector"
            )
        self.store = store
        self.mode = mode
        self.llm = llm
        self.model = model
        self.interval = interval
        self.max_bytes = max_bytes
        self.top_m = top_m
        self.top_n = top_n
        self.window = window
        self.top_r = top_r
        self.max_reject_bytes = max_reject_bytes
        self.reflector = reflector
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
        if self.mode == "global":
            return self._global(ctx)
        sections: list[str] = []
        if self.mode in ("lessons", "lessons+scratchpad"):
            # L1 lessons are the main course; mechanical rendering only
            # covers the cold start before the first reflection batch.
            main = self._lessons(ctx) or self._retrieve(ctx)
        else:
            main = self._retrieve(ctx)
        if main:
            sections.append(main)
        if self.mode in ("retrieval+rejected", "lessons", "lessons+scratchpad"):
            rejected = self._rejected(ctx)
            if rejected:
                sections.append(rejected)
        if self.mode == "lessons+scratchpad":
            hint = self._scratchpad_hint(ctx)
            if hint:
                sections.append(hint)
        return "\n\n".join(sections) or None

    def _retrieve(self, ctx: MutationContext) -> str | None:
        wins = self.store.recent_wins(self.top_m)
        # Clusters, not instances: N independent same-shape failures render
        # as one line carrying the count, instead of two of them displacing
        # every other failure mode (see loss_clusters).
        clusters = self.store.loss_clusters(self.top_n)
        if not wins and not clusters:
            return None
        lines = ["# Experience from this run (past mutation outcomes)"]
        # Losses first, wins worst->best: the strongest positive example
        # lands last, where LLM recency bias weighs it most.
        lines += [self.store.render_cluster(c) for c in clusters]
        lines += [e.render() for e in reversed(wins)]
        return "\n".join(lines)[: self.max_bytes]

    def _rejected(self, ctx: MutationContext) -> str | None:
        entries = self.store.recent_negatives(
            ctx.generation,
            window=self.window,
            top_r=self.top_r,
            island_idx=ctx.parent.island_idx,
        )
        if not entries:
            return None
        lines = [
            "# Recently ineffective or rejected edits "
            "(avoid repeating without variation)"
        ]
        lines += [e.render(show_origin=True) for e in entries]
        return "\n".join(lines)[: self.max_reject_bytes]

    def _lessons(self, ctx: MutationContext) -> str | None:
        pool = self.store.lesson_entries()
        if not pool:
            return None
        tags = set(self.store.parent_tags(ctx.parent.id))

        def overlap(e: ExperienceEntry) -> int:
            return len(tags.intersection(e.lesson.get("tags", [])))

        wins = [e for e in pool if e.lesson.get("verdict") == "improved"]
        losses = [
            e for e in pool
            if e.lesson.get("verdict") == "regressed"
            and e.redeemed_by is None
        ]
        if tags:
            wins.sort(key=lambda e: (-overlap(e), -e.fitness_delta))
            losses.sort(key=lambda e: (-overlap(e), e.fitness_delta))
        else:
            wins.sort(key=lambda e: -e.fitness_delta)
            losses.sort(key=lambda e: e.fitness_delta)
        wins, losses = wins[: self.top_m], losses[: self.top_n]
        if not wins and not losses:
            return None
        note = (
            "(matched on the parent's failure tags: "
            f"{', '.join(sorted(tags))})"
            if tags
            else "(parent not attributed yet; most decisive lessons shown)"
        )
        lines = ["# Lessons from past mutations", note]
        # Losses first, wins worst->best (strongest positive advice last).
        lines += [self._render_lesson(e, ctx.generation) for e in losses]
        lines += [self._render_lesson(e, ctx.generation) for e in reversed(wins)]
        return "\n".join(lines)[: self.max_bytes]

    @staticmethod
    def _render_lesson(e: ExperienceEntry, generation: int | None = None) -> str:
        lesson = e.lesson or {}
        verb = "helped" if lesson.get("verdict") == "improved" else "hurt"
        tags = ", ".join(lesson.get("tags", [])) or "-"
        # Age in plain words, not a bare number: a lesson drawn from a
        # program several generations removed may describe code that has
        # since mutated away, and its specificity makes it sound more
        # authoritative than it deserves.
        age = ""
        if generation is not None:
            gap = generation - e.generation
            if gap >= 5:
                age = (
                    f"; learned {gap} generations ago — the program has "
                    "moved since, verify before relying on it"
                )
            elif gap > 0:
                age = f"; learned {gap} generation{'s' if gap > 1 else ''} ago"
        line = (
            f"- [{e.operator}] {e.change_title or e.operator} — {verb} "
            f"(Δfitness {e.fitness_delta:+.4g}; tags: {tags}{age})"
        )
        if lesson.get("why"):
            line += f"\n  why: {lesson['why']}"
        if lesson.get("advice"):
            line += f"\n  advice: {lesson['advice']}"
        return line

    def _scratchpad_hint(self, ctx: MutationContext) -> str | None:
        pad = getattr(self.reflector, "scratchpad", "") or ""
        if not pad.strip():
            return None
        bullets: list[str] = []
        unexplored: list[str] = []
        in_unexplored = False
        for raw in pad.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("-", "*")):
                text = line.lstrip("-* ").strip()
                if text:
                    (unexplored if in_unexplored else bullets).append(text)
            else:  # section header line
                in_unexplored = "unexplored" in line.lower()
        pool = unexplored or bullets
        if not pool:
            return None
        # One sampled direction per proposal (Shinka's sample-1): different
        # parents draw different directions so proposals don't herd, while
        # the (generation, parent) seed keeps reruns reproducible.
        rng = random.Random(f"{ctx.generation}:{ctx.parent.id}")
        return (
            "# Direction hint (from the evolution scratchpad)\n- "
            + rng.choice(pool)
        )

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


class LessonDirectiveContributor:
    """Implements PromptContributor (framework-evo backlog item 6: promote a
    high-confidence lesson from passive context to the mutation's explicit
    directive). Fires only when the parent's own creation lesson qualifies
    (non-noise verdict, non-empty advice), and only with `probability` per
    proposal — undirected mutations keep exploring. The draw is seeded by
    (generation, parent) so reruns and checkpoints stay reproducible."""

    def __init__(
        self,
        store: ExperienceStore,
        probability: float = 0.5,
        max_bytes: int = 700,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.store = store
        self.probability = probability
        self.max_bytes = max_bytes

    def contribute(self, ctx: MutationContext) -> str | None:
        lesson = None
        for e in self.store.entries:
            if e.kind == "evaluated" and e.child_id == ctx.parent.id and e.lesson:
                lesson = e.lesson
                break
        if not lesson:
            return None
        verdict = lesson.get("verdict")
        advice = str(lesson.get("advice", "")).strip()
        if verdict == "noise" or not advice:
            return None
        rng = random.Random(f"directive:{ctx.generation}:{ctx.parent.id}")
        if rng.random() >= self.probability:
            return None
        if verdict == "regressed":
            framing = (
                "The edit that produced the current program HURT fitness"
            )
            action = "Apply the directive to repair this in the current mutation"
        else:
            framing = (
                "The edit that produced the current program HELPED fitness"
            )
            action = (
                "Continue in this direction with one meaningful variation "
                "(do not repeat the same edit verbatim)"
            )
        why = str(lesson.get("why", "")).strip()
        lines = [
            "# Mutation directive (apply THIS in the current mutation)",
            f"{framing}." + (f" Attribution: {why}" if why else ""),
            f"Directive: {advice}",
            f"{action}; keep unrelated parts of the program unchanged. "
            "If the directive conflicts with what you observe, deviate and "
            "say why in SUMMARY.",
        ]
        return "\n".join(lines)[: self.max_bytes]
