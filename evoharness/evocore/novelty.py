# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/core/novelty_judge.py (rejection sampling flow),
#           shinka/embed/embedding.py (cosine similarity),
#           shinka/prompts/prompts_novelty.py (judge verdict convention)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Behavior-aligned port. The embedding function and the LLM judge are
# injected callables so the gate stays testable and provider-agnostic.
"""Pre-evaluation novelty gate: embedding similarity + optional LLM judge."""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .population import IslandView

EmbedFn = Callable[[str], list[float]]
# judge(existing_code, proposed_code) -> raw LLM text; verdict must start
# with NOVEL / NOT NOVEL (upstream convention).
JudgeFn = Callable[[str, str], str]


@dataclass
class GateVerdict:
    accepted: bool
    embedding: list[float] | None = None
    max_similarity: float = 0.0
    most_similar_id: str | None = None
    judged: bool = False


def hashing_embedding(text: str, dims: int = 512, ngram: int = 5) -> list[float]:
    """Provider-free character-n-gram hashing embedding.

    Upstream embeds with a paid model. At the 0.99 similarity threshold the
    gate only needs to recognise near-identical programs, which n-gram
    overlap captures well — so the default gate costs nothing and works
    offline. Swap in a real embedder when semantic novelty matters.
    """
    vec = np.zeros(dims, dtype=float)
    normalized = " ".join(text.split())
    if not normalized:
        return [0.0] * dims
    for i in range(max(1, len(normalized) - ngram + 1)):
        chunk = normalized[i : i + ngram]
        # crc32, not hash(): str hashing is salted per process, which would
        # make embeddings persisted before a resume incomparable after it.
        vec[zlib.crc32(chunk.encode("utf-8")) % dims] += 1.0
    norm = float(np.linalg.norm(vec))
    return list(vec / norm) if norm else list(vec)


def novelty_text(workspace, fallback: str) -> str:
    """The text the gate should judge: the WHOLE workspace, not main_text.

    Multi-file candidates often mutate a non-main file (IMO edits live in
    prompts.py while solver.py never moves), so judging main_text alone
    scored 45/55 real sibling pairs above the 0.99 threshold — the gate
    would have rejected ~82% of proposals. Over the same candidates the
    whole-workspace rendering puts only 4/55 pairs above it.
    """
    try:
        texts = workspace.texts()
    except Exception:
        return fallback
    if not texts:
        return fallback
    return "\n".join(f"# {path}\n{texts[path]}" for path in sorted(texts))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_judge_verdict(text: str) -> bool:
    """True if the judge deems the proposal novel (upstream: response must
    start with NOVEL, tolerating markdown bold)."""
    head = text.strip().upper()
    if head.startswith("**"):
        head = head[2:].lstrip()
    return head.startswith("NOVEL") and not head.startswith("NOT NOVEL")


def content_digest(text: str) -> str:
    """Whitespace-insensitive identity of a candidate's content."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


class NoveltyGate:
    """Rejects proposals that add nothing new to the island.

    Two modes:

    "identity" (default) rejects only a proposal whose workspace is, up to
    whitespace, one an island candidate already has. That is the failure
    worth spending a retry on: the proposer burned a whole session and
    changed nothing.

    "similarity" is the upstream-parity path — embed and reject above a
    cosine threshold. It needs an embedder whose geometry matches the
    question being asked. Ours does not: a mutation is by construction
    almost character-identical to its own parent, and the parent sits in
    the same island, so an n-gram gate at 0.99 rejected 40% of perfectly
    good small edits (measured live, 2026-07-24) — each one costing a full
    agent session. Use this mode only with a semantic embedder.

    The retry loop (max_novelty_attempts) lives in SearchLoop.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        threshold: float = 0.99,
        judge_fn: JudgeFn | None = None,
        mode: str = "identity",
    ):
        if mode not in ("identity", "similarity"):
            raise ValueError(f"unknown novelty mode {mode!r}")
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.judge_fn = judge_fn
        self.mode = mode

    def _candidate_text(self, cand) -> str:
        try:
            return novelty_text(cand.workspace, cand.code)
        except Exception:
            return cand.code

    def check(self, code: str, island: IslandView) -> GateVerdict:
        # The embedding is still recorded: it costs nothing, it is the
        # observable proof the gate ran at all, and it keeps the door open
        # for a semantic embedder later.
        embedding = list(self.embed_fn(code))
        if self.mode == "identity":
            digest = content_digest(code)
            for cand in island.candidates:
                if content_digest(self._candidate_text(cand)) == digest:
                    return GateVerdict(False, embedding, 1.0, cand.id)
            return GateVerdict(True, embedding, 0.0, None)

        vec = np.asarray(embedding, dtype=float)
        max_sim, similar = 0.0, None
        for cand in island.candidates:
            if cand.embedding is None:
                continue
            sim = cosine_similarity(vec, np.asarray(cand.embedding, dtype=float))
            if sim > max_sim:
                max_sim, similar = sim, cand
        if similar is None or max_sim <= self.threshold:
            return GateVerdict(True, embedding, max_sim,
                               similar.id if similar else None)
        if self.judge_fn is not None:
            verdict = parse_judge_verdict(self.judge_fn(similar.workspace.main_text(), code))
            return GateVerdict(verdict, embedding, max_sim, similar.id, judged=True)
        return GateVerdict(False, embedding, max_sim, similar.id)
