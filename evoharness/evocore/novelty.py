# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/core/novelty_judge.py (rejection sampling flow),
#           shinka/embed/embedding.py (cosine similarity),
#           shinka/prompts/prompts_novelty.py (judge verdict convention)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Behavior-aligned port. The embedding function and the LLM judge are
# injected callables so the gate stays testable and provider-agnostic.
"""Pre-evaluation novelty gate: embedding similarity + optional LLM judge."""

from __future__ import annotations

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


class NoveltyGate:
    """Rejects proposals too similar to existing candidates in the island.

    Flow ([parity]): embed the proposal, compare against all embedded
    candidates in the island; accept if max cosine similarity <= threshold;
    otherwise consult the LLM judge if enabled, else reject. The retry loop
    (max_novelty_attempts, re-sampling a parent) lives in SearchLoop.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        threshold: float = 0.99,
        judge_fn: JudgeFn | None = None,
    ):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.judge_fn = judge_fn

    def check(self, code: str, island: IslandView) -> GateVerdict:
        embedding = list(self.embed_fn(code))
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
