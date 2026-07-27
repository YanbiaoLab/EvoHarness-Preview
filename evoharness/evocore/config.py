# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/database/dbase.py (DatabaseConfig), shinka/core/config.py
#           (EvolutionConfig), shinka/defaults.py
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Defaults are aligned with upstream for parity (see docs/naming_map.md for
# the naming correspondence). Intentional deviations are noted inline.
"""Configuration dataclasses for the evocore engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ProposalConfig:
    """Orthogonal proposal-generation arm and its per-proposal limits."""

    mode: str = "single_shot"
    model: str | None = None
    # Raised from upstream parity (12/40/300s/32k). Measured on run e5s_r2:
    # sessions used 11.4 turns and 12.4 tool calls on average, so none of
    # these bound — but a proposal cut off mid-edit produces a broken
    # workspace, and the compaction that max_input_tokens drives rewrites
    # history and drops the provider prompt cache. Both are expensive
    # failure modes to sit close to, and the headroom costs nothing when
    # unused.
    max_turns: int = 48
    max_tool_calls: int = 120
    timeout_s: float = 5400.0
    max_cost_usd: float | None = None
    max_repair_rounds: int = 3
    max_input_tokens: int = 131_072
    max_parallel_tools: int = 4
    # Mirrors the validated keep-recent-5 window used by production
    # tool-using agents; older results of expiring tools are cleared.
    recent_tool_results_to_keep: int = 5
    # Fraction of max_input_tokens at which stale results are cleared in
    # bulk. Compaction rewrites history and drops the provider prompt
    # cache, so it must be rare and decisive rather than per-turn.
    compact_trigger_ratio: float = 0.6
    hybrid_agent_probability: float = 0.1
    hybrid_stagnation_generations: int = 5

    def __post_init__(self) -> None:
        if self.mode not in {
            "single_shot",
            "conversational",
            "agentic",
            "hybrid",
        }:
            raise ValueError(
                "proposal mode must be single_shot, conversational, agentic, "
                "or hybrid"
            )
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("proposal model must be non-empty when present")
        for name, minimum in (
            ("max_turns", 1),
            ("max_tool_calls", 0),
            ("max_repair_rounds", 0),
            ("max_input_tokens", 1),
            ("max_parallel_tools", 1),
            ("recent_tool_results_to_keep", 0),
            ("hybrid_stagnation_generations", 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(f"{name} must be at least {minimum}")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("proposal timeout_s must be positive and finite")
        if self.max_cost_usd is not None and (
            isinstance(self.max_cost_usd, bool)
            or not isinstance(self.max_cost_usd, (int, float))
            or not math.isfinite(self.max_cost_usd)
            or self.max_cost_usd < 0
        ):
            raise ValueError(
                "proposal max_cost_usd must be nonnegative and finite"
            )
        if (
            isinstance(self.hybrid_agent_probability, bool)
            or not isinstance(
                self.hybrid_agent_probability,
                (int, float),
            )
            or not math.isfinite(self.hybrid_agent_probability)
            or not 0 <= self.hybrid_agent_probability <= 1
        ):
            raise ValueError(
                "hybrid_agent_probability must be between 0 and 1"
            )


@dataclass
class PopulationConfig:
    """Population store, islands, migration and parent selection.

    Field defaults mirror upstream DatabaseConfig so that parity runs need no
    overrides. Note migration is OFF by default upstream (migration_rate=0.0).
    """

    num_islands: int = 2
    archive_size: int = 40
    elite_selection_ratio: float = 0.3
    num_archive_inspirations: int = 1
    num_top_k_inspirations: int = 1
    migration_interval: int = 10
    migration_rate: float = 0.0
    island_elitism: bool = True
    enforce_island_separation: bool = True
    parent_strategy: str = "weighted"  # weighted|power_law|beam|seed_only|latest
    weighted_lambda: float = 10.0
    power_alpha: float = 1.0
    beam_width: int = 5
    archive_update_strategy: str = "fitness"  # crowding intentionally not ported


@dataclass
class SearchConfig:
    """Main loop configuration (subset of upstream EvolutionConfig)."""

    num_generations: int = 150
    task_sys_msg: str = ""
    language: str = "python"
    llm_models: list[str] = field(default_factory=lambda: ["gpt-5.1"])
    llm_temperature: float = 0.75
    llm_max_tokens: int = 4096
    operators: list[str] = field(
        default_factory=lambda: ["revise", "rewrite", "recombine"]
    )
    operator_probs: list[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])
    max_op_resamples: int = 3
    max_novelty_attempts: int = 3
    similarity_threshold: float = 0.99
    # "identity" rejects only proposals identical to an island candidate —
    # the agent-burned-a-session-and-changed-nothing case. "similarity"
    # is the upstream cosine path and needs a semantic embedder to be
    # meaningful (see NoveltyGate).
    novelty_mode: str = "identity"
    novelty_llm_judge: bool = False  # deviation: simplified, off by default
    # Upstream-parity default. Turn OFF for harnesses whose proposer emits
    # duplicate programs by construction (deterministic mock transports),
    # where every proposal would be a legitimate near-duplicate rejection.
    novelty_enabled: bool = True
    repair_enabled: bool = True
    # Throttle for the repair-first policy: with a failed candidate pending,
    # repair is chosen with this probability, else a normal proposal proceeds.
    # 1.0 = upstream-parity always-repair. Guards against "repair storms"
    # (gpu_run1: 36/60 proposals were repairs, crowding out exploration).
    repair_probability: float = 1.0
    seed: int = 0
    # Circuit breaker for remote evaluation (no upstream counterpart):
    # proposals are paid for BEFORE grading, so a dead eval service would
    # otherwise burn the whole LLM budget on candidates that get dropped.
    max_consecutive_infra_failures: int = 5
    # Batched dispatch (WS-2, no upstream counterpart): candidates proposed
    # per generation and graded CONCURRENTLY — required when one evaluation
    # is minute-scale (remote GPU training). 1 = upstream-parity serial loop.
    eval_batch_size: int = 1
    # How many proposal LLM calls may be in flight at once. 1 keeps the
    # strictly sequential behaviour. Raising it overlaps only the network
    # wait: planning still runs one at a time so rng draws keep their order,
    # and absorption runs in planning order, so a seeded run still
    # reproduces. Worth raising once eval_batch_size is large — at 16 the
    # sixteen sequential calls became the dominant cost of a generation.
    # Caveat: this preserves the choices this loop makes, not order-dependent
    # state inside a proposer, transport or rotating model router — those see
    # the calls finish in whatever order they finish.
    proposal_concurrency: int = 1
