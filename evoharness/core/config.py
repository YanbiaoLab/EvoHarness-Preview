# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/database/dbase.py (DatabaseConfig), shinka/core/config.py
#           (EvolutionConfig), shinka/defaults.py
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Defaults are aligned with upstream for parity (see docs/naming_map.md for
# the naming correspondence). Intentional deviations are noted inline.
"""Configuration dataclasses for the core engine."""

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
    # Total tokens one proposal may spend across every backend run, repairs
    # included. Off by default so existing runs keep their behaviour.
    #
    # Turns do not stand in for this. An agentic turn resends the whole
    # context and the context grows, so spend rises faster than the turn
    # count. Measured 2026-08-28 on ETP run18: two proposals spent 2.48M and
    # 2.46M tokens while staying inside 110 turns / 600 tool calls / 150
    # minutes — every declared limit respected, nothing to stop them. At ~70k
    # tokens a turn, `max_turns=110` is really a 7.7M token budget that
    # nobody had written down.
    max_tokens_per_proposal: int | None = None
    max_repair_rounds: int = 3
    max_input_tokens: int = 131_072
    max_parallel_tools: int = 4
    # Ceiling the `run` tool clamps every requested timeout to. The default
    # suits a shell command; a domain whose verification step runs for
    # minutes needs it raised, or the agent has to poll. Measured on ETP run
    # 15: a self-check takes 60-150s against a 60s cap, so each one cost two
    # or three extra model calls -- and at that point a call resends 130-200k
    # tokens of context, which is where the run's budget actually went.
    run_timeout_cap_s: float = 60.0
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
        if (
            isinstance(self.run_timeout_cap_s, bool)
            or not isinstance(self.run_timeout_cap_s, (int, float))
            or not math.isfinite(self.run_timeout_cap_s)
            or self.run_timeout_cap_s <= 0
        ):
            raise ValueError(
                "proposal run_timeout_cap_s must be positive and finite"
            )
        if self.max_tokens_per_proposal is not None and (
            isinstance(self.max_tokens_per_proposal, bool)
            or not isinstance(self.max_tokens_per_proposal, int)
            or self.max_tokens_per_proposal < 1
        ):
            raise ValueError(
                "max_tokens_per_proposal must be a positive integer when set"
            )
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
    archive_update_strategy: str = "fitness"  # fitness|feature_buckets
    # Feature-bucketed archive (MAP-Elites on one axis). A top-N-by-fitness
    # archive collapses onto whatever region of the search space the current
    # leaders occupy: once every leader sits against a resource ceiling, no
    # cheaper program is left to breed from, and reclaiming the resource
    # means starting from something that has none to give. Reserving slots
    # per bucket keeps one elite alive in each region.
    #
    # `archive_feature_metric` names a visible_metrics key (e.g. a byte
    # count); leaving it empty keeps today's behaviour exactly.
    archive_feature_metric: str = ""
    archive_feature_bucket: float = 0.0
    # Ranking metric inside a bucket. Gates that zero out `fitness` erase the
    # ordering among gated candidates -- all of them tie at 0 and the bucket
    # elite becomes arbitrary. Point this at the pre-gate score to keep them
    # comparable. Empty means "use fitness".
    archive_feature_quality: str = ""
    archive_feature_reserve: int = 0
    # Display only: the hard cap on the axis, so the proposal prompt can state
    # remaining headroom instead of a bare number. None = unknown, report the
    # raw cost.
    archive_feature_cap: float | None = None
    archive_feature_unit: str = ""


@dataclass
class SearchConfig:
    """Main loop configuration (subset of upstream EvolutionConfig)."""

    num_generations: int = 150
    # Stop as soon as the population holds a candidate at least this good.
    # Off by default, and it has to be: on an open-ended search there is no
    # such number, and a wrong one ends the run at the first lucky
    # generation. Set it where the task has a ceiling that means "solved" —
    # a single problem the judge either accepts or does not — because past
    # that point the remaining generations can only spend money re-solving
    # something already solved. An authored task that declares
    # `criterion.solved_at` fills this in for its own runs.
    stop_at_fitness: float | None = None
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
