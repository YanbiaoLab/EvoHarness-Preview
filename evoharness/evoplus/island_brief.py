# EvoHarness original: per-island prompt specialization.
#
# Islands exist to preserve diversity, but the fitness gradient pulls every island
# onto the same slope: given the same prompt and the same fast feedback, each one
# independently picks whichever direction is currently easiest to climb, and the
# population converges in behaviour even while staying separated in genes.
# Island isolation separates genomes; it does not separate tasks. This contributor
# separates the task too, so a distinct line of exploration survives at the
# population level rather than depending on any single proposal being adventurous.
"""IslandBriefContributor: give each island its own standing directive."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from evoharness.core.interfaces import MutationContext

logger = logging.getLogger(__name__)

# A brief is the one prompt section carrying measured evidence a human decided the
# island needs. Truncating it silently drops whichever fact was written last, and
# the loss is invisible from both ends -- the file on disk still reads complete.
# Budget generously and say so when the cap bites.
DEFAULT_MAX_BYTES = 8192

_HEADER = "# This island's assignment\n"


class IslandBriefContributor:
    """Implements PromptContributor. Injects a per-island directive.

    Inert unless enabled: a missing file, or an island with no entry, returns
    None. This follows the HITL directives.json pattern — the mechanism is always
    mounted and a file in the run directory decides whether it does anything, so
    other tasks and recipes need no changes.

    The file is re-read on every call, so a directive can be changed mid-run
    (same as DirectiveBook). An island observed to be spinning can be pointed
    somewhere else without restarting.
    """

    def __init__(self, path: Path | str, max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes

    def _briefs(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def contribute(self, ctx: MutationContext) -> str | None:
        idx = getattr(ctx.parent, "island_idx", None)
        if idx is None or idx < 0:
            return None
        text = (self._briefs().get(str(idx)) or "").strip()
        if not text:
            return None
        raw = text.encode()
        if len(raw) > self.max_bytes:
            logger.warning(
                "island %s brief truncated: %d bytes over the %d cap; "
                "the tail of %s was dropped",
                idx, len(raw) - self.max_bytes, self.max_bytes, self.path,
            )
            text = raw[: self.max_bytes].decode(errors="ignore")
        return _HEADER + text
