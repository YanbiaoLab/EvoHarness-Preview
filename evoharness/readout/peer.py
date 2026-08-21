"""What one candidate may learn about another.

Separate from `detail` because the audience differs in the way that matters:
`detail` answers a human or a host session and hands over the whole evaluation
report. This answers a CANDIDATE, running inside its own runtime, about a
program it was offered as a reference.

The difference is not enforced by filtering. It is enforced by construction —
this module builds a fixed set of keys and never touches the report as a
whole, so a field added to `EvalReport` later cannot arrive here by default.
Filtering a full dump has the opposite property, and the field a task author
most wants withheld is exactly the one a forgotten allowlist entry releases.

The shape mirrors the in-process `InspectCandidateTool` so that a candidate
gets the same answer whichever runtime it proposes in.
"""

from __future__ import annotations

from pathlib import Path

from evoharness.core import PopulationStore

from .status import ReadoutError, RunDirectory

#: Matches InspectCandidateTool's default. A candidate that cannot fit a
#: reference program in its context is not helped by being sent all of it.
DEFAULT_MAX_CHARS = 20_000


class UnknownCandidate(ReadoutError):
    """No candidate with that id, or no such file inside it."""


def _open_store(run: RunDirectory) -> PopulationStore:
    db = run.path / "run.db"
    if not db.exists():
        raise ReadoutError(f"{run.path} has no run.db")
    return PopulationStore.open_readonly(db)


def peer_view(
    run_dir: Path | str,
    candidate_id: str,
    path: str | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    """One evaluated candidate as a peer candidate may see it.

    With no `path`, an inventory: what the program scored, what it changed,
    and which files it has. With a `path`, that file's text.

    Deliberately absent, and to stay absent: `hidden_metrics`, which a task
    author withholds from the optimizer on purpose; `stdout_log` and
    `stderr_log`, raw grader output and the likeliest place for a holdout set
    to leak; and anything about the run as a whole.
    """

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise UnknownCandidate("candidate_id must be a non-empty string")
    if path is not None and not isinstance(path, str):
        raise UnknownCandidate("path must be text when given")

    store = _open_store(RunDirectory(Path(run_dir)))
    try:
        candidate = store.get(candidate_id)
        if candidate is None:
            # Name what a real id looks like. The first live in-process
            # session called the equivalent tool with candidate_id="list",
            # reading the description's "list that candidate's files" as a
            # command word.
            raise UnknownCandidate(
                f"no candidate with id {candidate_id!r}; ids are opaque and "
                "must be copied from a reference-program listing"
            )
        try:
            texts = candidate.workspace.texts()
        except Exception as exc:  # noqa: BLE001 — surfaced as a refusal
            raise ReadoutError(
                f"candidate {candidate_id} has no readable workspace"
            ) from exc

        if path is None:
            return {
                "ok": True,
                "candidate_id": candidate_id,
                "fitness": candidate.fitness,
                "change_title": candidate.change_title,
                "files": {
                    name: len(text.splitlines())
                    for name, text in sorted(texts.items())
                },
            }

        if path not in texts:
            raise UnknownCandidate(
                f"candidate {candidate_id} has no file {path}; "
                f"available: {', '.join(sorted(texts))}"
            )
        content = texts[path]
        truncated = len(content) > max_chars
        return {
            "ok": True,
            "candidate_id": candidate_id,
            "path": path,
            "content": content[:max_chars],
            "truncated": truncated,
        }
    finally:
        store.close()
