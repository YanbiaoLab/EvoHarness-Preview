# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/core/async_runner.py (LLM query -> parse -> apply flow
#           with resampling on malformed output)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# The Proposer seam itself is an EvoHarness extension: proposal generation
# is a pluggable strategy (single-shot / conversational / agentic, with an
# optional outer hybrid selector), so the
# "how smart is one mutation" axis becomes an ablation instead of a
# framework commitment. SingleShotProposer preserves upstream single-call
# semantics and is the parity default.
"""Proposal generation strategies."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .population import Candidate
from .llm import LLMClient
from .operators import (
    PatchEngine,
    apply_rewrite,
    parse_change_header,
    parse_file_blocks,
)
from .routing import ModelRouter
from .workspace import Workspace, WorkspaceError

if TYPE_CHECKING:
    from .operators import PromptBuilder
    from .population import PopulationStore

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """A parsed, validated mutation ready for evaluation.

    code is always the new main-file text. workspace, when present, is the
    complete child genome produced by a multi-file or agentic proposal.
    metadata carries provider-neutral proposal provenance."""

    code: str
    title: str
    summary: str
    model: str
    workspace: Workspace | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ProposeResult:
    """Outcome of one complete proposal attempt series."""

    proposal: Proposal | None
    llm_cost: float = 0.0
    attempts: int = 0
    failure_reason: str | None = None
    trace_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.proposal is not None

    def __post_init__(self) -> None:
        if (
            isinstance(self.llm_cost, bool)
            or not isinstance(self.llm_cost, (int, float))
            or not math.isfinite(self.llm_cost)
            or self.llm_cost < 0
        ):
            raise ValueError("llm_cost must be nonnegative and finite")

        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 0
        ):
            raise ValueError("attempts must be a nonnegative integer")

        if self.proposal is None:
            if (
                not isinstance(self.failure_reason, str)
                or not self.failure_reason.strip()
            ):
                raise ValueError(
                    "failed propose result requires failure_reason"
                )
        elif self.failure_reason is not None:
            raise ValueError(
                "successful propose result cannot have failure_reason"
            )

        if self.trace_path is not None and (
            not isinstance(self.trace_path, str)
            or not self.trace_path.strip()
        ):
            raise ValueError("trace_path must be non-empty when present")


class Proposer(ABC):
    """Turns an assembled (system, user) prompt into validated code."""

    @abstractmethod
    def propose(
        self, operator: str, parent: Candidate, system: str, user: str
    ) -> ProposeResult: ...


@dataclass(frozen=True)
class ProposalLane:
    """A complete prompt-and-proposer lane selected before prompt assembly."""

    name: str
    prompt_builder: "PromptBuilder"
    proposer: Proposer

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("proposal lane name must be non-empty")


class HybridProposalSelector:
    """Route proposals without adding hybrid behavior to either proposer.

    The agentic lane is selected for low-probability exploration or after a
    configured number of generations without a strict global-best gain. The
    SearchLoop RNG owns randomness, so checkpointed RNG state preserves exact
    resume behavior.
    """

    def __init__(
        self,
        single_shot: ProposalLane,
        agentic: ProposalLane,
        *,
        agent_probability: float = 0.1,
        stagnation_generations: int = 5,
    ) -> None:
        if (
            isinstance(agent_probability, bool)
            or not isinstance(agent_probability, (int, float))
            or not math.isfinite(agent_probability)
            or not 0 <= agent_probability <= 1
        ):
            raise ValueError("agent_probability must be between 0 and 1")
        if (
            isinstance(stagnation_generations, bool)
            or not isinstance(stagnation_generations, int)
            or stagnation_generations < 1
        ):
            raise ValueError("stagnation_generations must be at least 1")
        self.single_shot = single_shot
        self.agentic = agentic
        self.agent_probability = float(agent_probability)
        self.stagnation_generations = stagnation_generations

    def select(
        self,
        generation: int,
        store: "PopulationStore",
        rng: np.random.Generator,
    ) -> ProposalLane:
        last_improvement = self._last_improvement_generation(store)
        is_stagnant = (
            generation - last_improvement >= self.stagnation_generations
        )
        if is_stagnant or rng.random() < self.agent_probability:
            return self.agentic
        return self.single_shot

    @staticmethod
    def _last_improvement_generation(store: "PopulationStore") -> int:
        best_fitness = -math.inf
        last_improvement = 0
        for candidate in store.all_candidates():
            if not candidate.passed:
                continue
            if candidate.fitness > best_fitness:
                best_fitness = candidate.fitness
                last_improvement = candidate.generation
        return last_improvement


class SingleShotProposer(Proposer):
    """One LLM call per attempt, up to max_resamples attempts; malformed
    output is retried blind ([parity] with upstream — unlike the session
    lanes, the parse error is not fed back into a continued conversation)."""

    def __init__(
        self,
        llm: LLMClient,
        model_router: ModelRouter,
        language: str = "python",
        max_resamples: int = 3,
    ):
        self.llm = llm
        self.model_router = model_router
        self.language = language
        self.max_resamples = max_resamples
        self.patch_engine = PatchEngine()

    def propose(
        self, operator: str, parent: Candidate, system: str, user: str
    ) -> ProposeResult:
        cost = 0.0
        for attempt in range(1, self.max_resamples + 1):
            model = self.model_router.pick()
            try:
                resp = self.llm.query(system, user, model)
            except RuntimeError as e:
                logger.warning("LLM unavailable for %s: %s", model, e)
                continue
            cost += resp.cost
            title, summary = parse_change_header(resp.text)
            parent_code = parent.workspace.main_text()
            # Multi-file lane (M2.5): FILE blocks in the answer become a
            # complete child genome, shipped via Proposal.workspace — the
            # same lane M3 agent sessions use. revise stays main-file-only:
            # ORIGINAL/UPDATED patches carry no file paths.
            if operator != "revise":
                edits = parse_file_blocks(resp.text)
                if edits:
                    try:
                        child_ws = parent.workspace.with_files(edits)
                    except WorkspaceError as e:
                        # ../escape or growing a single-file genome: a
                        # REJECTED proposal, not a crash — resample.
                        logger.warning(
                            "multi-file proposal rejected (attempt %d): %s",
                            attempt, e,
                        )
                        continue
                    if child_ws.texts() == parent.workspace.texts():
                        # A "mutation" that changed nothing. It is not a
                        # neutral candidate: it ties its parent's fitness for
                        # free — and where evaluation is cached, it ties it
                        # exactly — so it lands at the top of the population
                        # having contributed nothing, and can be selected as a
                        # parent. Observed in run modmul_r1.
                        logger.warning(
                            "proposal rejected (%s, attempt %d): "
                            "identical to parent",
                            operator, attempt,
                        )
                        continue
                    return ProposeResult(
                        Proposal(child_ws.main_text(), title, summary, model,
                                 workspace=child_ws),
                        llm_cost=cost,
                        attempts=attempt,
                    )
            if operator == "revise":
                outcome = self.patch_engine.apply(parent_code, resp.text)
            else:  # rewrite | recombine | repair share full-program output
                outcome = apply_rewrite(parent_code, resp.text, self.language)
            if outcome.ok:
                return ProposeResult(
                    Proposal(outcome.new_code, title, summary, model),
                    llm_cost=cost,
                    attempts=attempt,
                )
            # WARNING, not INFO: run modmul_r1 lost 14 of 16 proposals and
            # the log recorded nothing about why, so the failure could
            # only be guessed at afterwards.
            logger.warning(
                "proposal rejected (%s, attempt %d): %s",
                operator, attempt, outcome.error,
            )
        return ProposeResult(
            None,
            llm_cost=cost,
            attempts=self.max_resamples,
            failure_reason="resample-exhausted",
        )
