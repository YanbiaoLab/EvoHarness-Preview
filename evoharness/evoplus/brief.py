# EvoHarness original: tier-1 external knowledge injection (frozen research
# brief). The brief is produced OFFLINE at task-authoring time with any deep
# research tool, committed next to the task, and hashed into the run
# manifest — it is part of the task definition, shared by ALL experiment
# groups, never a per-run variable (reproducibility guardrail, plan sec. 7).
"""StaticBriefContributor: inject a frozen, task-authored research brief."""

from __future__ import annotations

from pathlib import Path

from evoharness.core.interfaces import MutationContext

DEFAULT_MAX_BYTES = 4096

_HEADER = (
    "# Domain research brief\n"
    "Curated background on techniques relevant to this task, gathered "
    "before the run. Use it to inform improvements; it may be incomplete "
    "or partially outdated.\n\n"
)


class StaticBriefContributor:
    """Implements PromptContributor. Stateless: the same frozen text is
    injected into every mutation prompt (methods-level knowledge only —
    never instance/answer material, see plan guardrails)."""

    def __init__(self, text: str, max_bytes: int = DEFAULT_MAX_BYTES):
        text = text.strip()
        if len(text.encode()) > max_bytes:
            text = text.encode()[:max_bytes].decode(errors="ignore")
        self.text = text

    @classmethod
    def from_file(
        cls, path: Path | str, max_bytes: int = DEFAULT_MAX_BYTES
    ) -> "StaticBriefContributor":
        return cls(Path(path).read_text(), max_bytes)

    def contribute(self, ctx: MutationContext) -> str | None:
        if not self.text:
            return None
        return _HEADER + self.text
