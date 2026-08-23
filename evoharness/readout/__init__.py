"""Read-only views over a run directory.

The run directory that `SearchLoop` writes is the single source of truth; this
package shapes it for whoever is asking — a dsh tool, a human, an audit — and
never writes to it.

It exists as a library rather than as a web backend because the consumers are
no longer only a console: a session tool that had to parse `run.db` and
`metrics.jsonl` itself would be reimplementing Python-side internals in
another language, and they would drift.
"""

from .detail import (
    CandidateRow,
    Generation,
    candidate_detail,
    population,
    run_detail,
    trajectory,
)
from .governance import (
    GovernanceError,
    card,
    pending_cards,
    recent_decisions,
)
from .peer import DEFAULT_MAX_CHARS, UnknownCandidate, peer_view
from .status import (
    STALL_AFTER_S,
    ReadoutError,
    RunDirectory,
    RunStatus,
    list_runs,
    run_status,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "STALL_AFTER_S",
    "CandidateRow",
    "Generation",
    "GovernanceError",
    "ReadoutError",
    "RunDirectory",
    "RunStatus",
    "UnknownCandidate",
    "candidate_detail",
    "card",
    "list_runs",
    "peer_view",
    "pending_cards",
    "population",
    "recent_decisions",
    "run_detail",
    "run_status",
    "trajectory",
]
