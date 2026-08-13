# EvoHarness original: make a hard resource budget visible at proposal time.
#
# A model editing a program under a cap can measure its own size; what it cannot
# see is what the rest of the population costs. Once every leader sits against
# the cap, "free some room" reads as pure loss -- the edit spends score and buys
# nothing the model can point at. Showing the cheaper elites that already exist,
# with what they score, turns reclamation into a move with a known destination:
# recombine toward the compact one instead of shaving the expensive one.
"""ResourceLedgerContributor: the population's spread along a cost axis."""

from __future__ import annotations

from evoharness.core.interfaces import MutationContext

_HEADER = "# Resource budget\n"


def _metric(cand, key: str) -> float | None:
    report = getattr(cand, "report", None)
    if report is None or not key:
        return None
    value = report.visible_metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class ResourceLedgerContributor:
    """Implements PromptContributor. Renders parent and inspirations by cost.

    Inert unless the parent reports `metric`, and silent when every candidate
    on hand costs the same -- a one-row table states no relationship the model
    could act on.

    `quality_metric` should name the score BEFORE any gate zeroes it, so a
    candidate that was rejected for a regression still shows what it achieved;
    that is the whole reason it is worth listing as a donor.
    """

    def __init__(
        self,
        metric: str,
        *,
        store=None,
        cap: float | None = None,
        quality_metric: str = "",
        unit: str = "",
        max_rows: int = 8,
    ):
        self.metric = metric
        self.store = store
        self.cap = cap
        self.quality_metric = quality_metric
        self.unit = unit
        self.max_rows = max_rows

    def _from_store(self) -> list:
        """Bucket elites straight from the population.

        Reading only the sampled inspirations would show nothing: inspiration
        selection ranks on `fitness`, so when a gate zeroes every cheap
        candidate the draw is all ceiling-pinned programs and the table has one
        distinct cost. The whole point is to surface what selection cannot.
        """
        if self.store is None:
            return []
        try:
            return self.store._bucket_elites(
                [c for c in self.store.all_candidates() if c.passed]
            )
        except (AttributeError, TypeError):
            return []

    def _quality(self, cand) -> float | None:
        value = _metric(cand, self.quality_metric)
        if value is not None:
            return value
        report = getattr(cand, "report", None)
        return None if report is None else report.fitness

    def contribute(self, ctx: MutationContext) -> str | None:
        own = _metric(ctx.parent, self.metric)
        if own is None:
            return None
        pool = {ctx.parent.id: ctx.parent}
        for cand in (list(ctx.archive_inspirations) + list(ctx.top_k_inspirations)
                     + self._from_store()):
            if _metric(cand, self.metric) is not None:
                pool.setdefault(cand.id, cand)
        rows = sorted(pool.values(), key=lambda c: (_metric(c, self.metric), c.id))
        if len({_metric(c, self.metric) for c in rows}) < 2:
            return None

        unit = f" {self.unit}" if self.unit else ""
        lines = [_HEADER]
        if self.cap is not None:
            lines.append(
                f"Cap {self.cap:,.0f}{unit}. This parent spends {own:,.0f}, "
                f"leaving {self.cap - own:,.0f}.\n"
            )
        else:
            lines.append(f"This parent spends {own:,.0f}{unit}.\n")
        lines.append(
            "\nWhat the population holds along this axis. Every row is a real "
            "program you may recombine with, not a hypothetical:\n\n"
        )
        lines.append(f"    {'cost':>12}  {'headroom':>10}  {'score':>8}  who\n")
        for cand in rows[: self.max_rows]:
            cost = _metric(cand, self.metric)
            head = "-" if self.cap is None else f"{self.cap - cost:,.0f}"
            quality = self._quality(cand)
            score = "-" if quality is None else f"{quality:.4f}"
            who = "this parent" if cand.id == ctx.parent.id else cand.id[:12]
            lines.append(f"    {cost:>12,.0f}  {head:>10}  {score:>8}  {who}\n")
        return "".join(lines)
