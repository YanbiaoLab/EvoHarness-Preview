"""HyperAgents-harness adapter for the EvoProof strategy.

Run this file through ``domains.harness`` from the vendored HyperAgents root.
The harness supplies the exact same model and problem inputs as its baselines;
all algorithmic decisions remain in :mod:`strategy`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.imo_proof.strategy import ProofStrategy  # noqa: E402

from agent.base_agent import AgentSystem  # type: ignore[import-not-found]  # noqa: E402
from agent.llm_withtools import chat_with_agent  # type: ignore[import-not-found]  # noqa: E402


class TaskAgent(AgentSystem):
    """Expose EvoProof through HyperAgents' benchmark TaskAgent contract."""

    def forward(self, inputs: dict[str, Any]):
        problem = inputs.get("problem")
        histories: list[dict[str, Any]] = []

        def ask(stage: str, prompt: str) -> str:
            self.log(f">>>>>>> EvoProof stage: {stage}")
            history = chat_with_agent(
                prompt,
                model=self.model,
                msg_history=[],
                logging=self.log,
                tools_available=[],
            )
            histories.extend(history)
            return history[-1].get("text", "")

        run = ProofStrategy(ask).solve(problem)
        self.log(
            f">>>>>>> EvoProof completed: calls={len(run.calls)}, "
            f"audit={run.audit.verdict}"
        )
        return run.final_solution, histories
