"""Reading one finished attempt back, narrowly.

The board says how an attempt ended in one word. That is enough to decide
whether a goal is hard and not nearly enough to decide what to try next, and
the gap is not hypothetical: in the PB-Advanced-006 session the model closed
it itself, with twenty-five `glob`/`read`/`grep` calls into
`.evo/runs/*/attempt_*/run/`, including hand-written regexes over the
solver's own `events.jsonl`. It got what it needed by parsing internal files
nobody promised would keep their shape -- and the day they change, that path
returns nothing and the model has no way to notice it has gone blind.

So: a supported view over the same facts. Two rules shape it.

**Narrow by construction, not by deletion.** Every key here is built
explicitly. Taking the evaluation report and removing `hidden_metrics`,
`stdout_log` and `stderr_log` would work today and leak the next field
somebody adds to it. `readout/peer.py` exists for the same reason and is
built the same way.

**A run that was cut has not concluded anything.** `interrupted`,
`infra-failed` and `budget-exhausted` leave the solver holding whatever it
happened to hold at the moment it stopped, which is frequently a line it was
about to abandon. Presented flatly beside a `task-failed`, that debris reads
as a finding, and the reader builds a story on it. `conclusive` says which
kind this is, in the payload rather than in documentation.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .graph import Outcome

#: Outcomes where the solver ran its course and the result says something
#: about the goal. Everything else says something about the run.
CONCLUSIVE = frozenset({Outcome.PROVED, Outcome.TASK_FAILED, Outcome.TIMEOUT})

#: `<temp dir>/sketch.lean:16:29: error: ...` -> `16:29: error: ...`. The path
#: is a temp directory that no longer exists; the line and column are the only
#: parts a reader can act on, and they line up with `code` when it is asked
#: for.
_LEAN_PATH = re.compile(r"^\S*\.lean:")


class InspectError(RuntimeError):
    """The attempt cannot be read back -- a wiring fault, not a finding."""


def attempt_view(
    store,
    goal_id: str,
    attempt_id: str | None = None,
    *,
    include_code: bool = False,
    max_candidates: int = 3,
    max_errors: int = 12,
) -> dict:
    """What one attempt tried, and what Lean said about it.

    `attempt_id` defaults to the most recent attempt on the goal, because the
    board shows attempts in order and the last one is what a caller deciding
    its next move is asking about.
    """

    attempts = store.attempts_of(goal_id)
    if not attempts:
        raise InspectError(f"goal {goal_id} has no recorded attempts")
    if attempt_id is None:
        attempt = attempts[-1]
    else:
        matching = [a for a in attempts if a.id == attempt_id]
        if not matching:
            raise InspectError(
                f"{attempt_id} is not an attempt on goal {goal_id}"
            )
        attempt = matching[0]

    view = {
        "attempt_id": attempt.id,
        "goal_id": goal_id,
        "outcome": attempt.outcome.value,
        "conclusive": attempt.outcome in CONCLUSIVE,
        "note": attempt.note[:400],
        "run_dir": attempt.run_dir,
        "tried": [],
        "effort": None,
    }
    if attempt.outcome not in CONCLUSIVE:
        view["caveat"] = (
            f"this run ended as `{attempt.outcome.value}`, which is not a "
            "verdict on the goal: it was stopped rather than finished. What "
            "follows is where it happened to be, not what it concluded."
        )
    if attempt.run_dir is None:
        view["caveat"] = (
            "this attempt was recorded without a run directory, so nothing "
            "of what it tried can be read back. Attempts interrupted before "
            "the graph kept that pointer are the only ones like this."
        )
        return view

    database = Path(attempt.run_dir) / "run" / "run.db"
    if not database.exists():
        view["caveat"] = f"no run database under {attempt.run_dir}"
        return view

    view["tried"], view["effort"] = _candidates(
        database,
        include_code=include_code,
        max_candidates=max_candidates,
        max_errors=max_errors,
    )
    return view


def _candidates(
    database: Path,
    *,
    include_code: bool,
    max_candidates: int,
    max_errors: int,
) -> tuple[list[dict], dict | None]:
    """The last few things the solver tried, newest first.

    Newest first because a reader deciding what to do next cares most about
    where the run ended up; oldest first would put the seed at the top, which
    it already knows.
    """

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT generation, operator, change_title, change_summary,"
            " report, metadata, code FROM candidates"
            " ORDER BY generation DESC, rowid DESC LIMIT ?",
            (max_candidates,),
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise InspectError(f"cannot read {database}: {exc}") from exc
    finally:
        connection.close()

    tried: list[dict] = []
    effort: dict | None = None
    for index, row in enumerate(rows):
        report = _loads(row["report"])
        entry = {
            "generation": row["generation"],
            "operator": row["operator"],
            "title": (row["change_title"] or "")[:200],
            "summary": (row["change_summary"] or "")[:600],
            "passed": bool(report.get("passed")),
            "fitness": report.get("fitness"),
            "fault": (report.get("fault") or "")[:200],
            "lean_errors": _errors(report.get("notes"), max_errors),
        }
        if include_code and index == 0:
            entry["code"] = _main_file(row["code"])
        tried.append(entry)
        if effort is None:
            effort = _effort(_loads(row["metadata"]))
    return tried, effort


def _errors(notes, limit: int) -> list[str]:
    """Lean's own messages, stripped of a temp path nobody can visit.

    The compiler's text is the one part of a failed attempt that is a fact
    rather than a story: the solver's summary is what it believed it was
    doing, and a wrong belief carried forward is worth less than nothing.
    """

    if not isinstance(notes, str) or not notes.strip():
        return []
    out = []
    for line in notes.splitlines():
        line = line.strip()
        if line:
            out.append(_LEAN_PATH.sub("", line)[:300])
    return out[:limit]


def _effort(metadata: dict) -> dict | None:
    """What the attempt spent, in the units that are actually measured.

    No dollars: nothing here is priced, and a zero would read as free rather
    than as unpriced. Tokens and seconds are observations.
    """

    if not metadata:
        return None
    return {
        "turns": metadata.get("turns"),
        "tool_calls": metadata.get("tool_calls"),
        "repair_rounds": metadata.get("repair_rounds"),
        "elapsed_s": metadata.get("agent_elapsed_s"),
        "prompt_tokens": metadata.get("prompt_tokens"),
        "completion_tokens": metadata.get("completion_tokens"),
        "termination": metadata.get("termination"),
    }


def _main_file(code) -> str | None:
    payload = _loads(code)
    files = payload.get("base_files")
    if isinstance(files, dict):
        for name, text in files.items():
            if name.endswith(".lean") and isinstance(text, str):
                return text
    return None


def _loads(raw) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["CONCLUSIVE", "InspectError", "attempt_view"]
