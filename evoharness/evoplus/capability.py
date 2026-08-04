# EvoHarness original: declared-versus-observed feedback capability.
# Audit finding (2026-07-30): which feedback layers are active is embodied by
# the recipe, but nothing records whether the layers a run MOUNTED ever
# actually received their inputs. A task that silently returns bare scalars
# downgrades the whole feedback stack and no artifact says so.
"""Recording which feedback channels a run actually exercised.

The liveness lesson, applied to feedback: a mounted plugin is not a working
plugin. One audit day found eight defects behind a green test suite, every
one a mechanism that existed but never fired. This ledger is the runtime
counterpart for the feedback stack -- it counts, per graded candidate, which
channels carried anything, so the manifest can state observed capability
next to declared capability and the gap is visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evoharness.evocore.population import Candidate, PopulationStore


@dataclass
class CapabilityLedger:
    """Implements LoopObserver. Always-on observability, not an ablation
    variable: it changes no behaviour, only what the run can say about
    itself afterwards."""

    graded: int = 0
    channels: dict[str, int] = field(default_factory=dict)

    def on_candidate_graded(
        self, cand: Candidate, store: PopulationStore
    ) -> None:
        report = cand.report
        if report is None:
            return
        self.graded += 1
        self._mark("scalar", True)
        self._mark("structured_feedback", bool(report.structured_feedback))
        self._mark("artifacts", bool(report.artifacts_ref))
        self._mark("uncertainty", report.sem > 0.0 or report.n_units > 0)
        self._mark("behavior_signature", bool(cand.behavior_signature))
        self._mark("hidden_metrics", bool(report.hidden_metrics))

    def _mark(self, channel: str, present: bool) -> None:
        if present:
            self.channels[channel] = self.channels.get(channel, 0) + 1

    def summary(self, declared: list[str] | None = None) -> dict:
        """What the run can honestly claim, channel by channel.

        `silent` names the channels that were declared by the assembly but
        never carried anything across the whole run -- the explicit-downgrade
        record the audit asked for. Reported, not raised: a run whose task
        provides less than the recipe hoped is degraded, not wrong.
        """
        observed = {
            channel: count for channel, count in sorted(self.channels.items())
        }
        result = {"graded": self.graded, "observed": observed}
        if declared is not None:
            result["declared"] = sorted(declared)
            result["silent"] = sorted(
                channel for channel in declared if channel not in observed
            )
        return result

    # -- checkpointing ----------------------------------------------------

    def state(self) -> dict:
        return {"graded": self.graded, "channels": dict(self.channels)}

    def set_state(self, state: dict) -> None:
        self.graded = int(state.get("graded", 0))
        self.channels = {
            str(k): int(v) for k, v in state.get("channels", {}).items()
        }
