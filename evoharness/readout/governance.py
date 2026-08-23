"""What is waiting for a person to decide, and what they already decided.

The first two layers of DH-5: a session can be told a card exists and can
show what it says. It cannot answer one. That boundary is the whole design,
and it is enforced here the way the peer view enforces its own — by
construction. This module opens the ledger read-only and contains no code
that writes, so there is no answer path to disable, guard, or forget to
guard.

The audience is a person, through a session they are sitting in, so nothing
is withheld: unlike `peer`, whose reader is a candidate under evaluation,
this reader is the one the evidence is FOR. The narrow thing here is the set
of operations, not the set of fields.

Why not reuse `InboxStore`: its constructor demands a non-empty set of
authorized actors, because answering is what it is for. Building one to read
with would mean holding an object that could answer, and its research-store
companion runs schema DDL on construction — so a mistyped path would
manufacture an empty ledger and report an empty queue rather than an error.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .status import ReadoutError

LEDGER_NAME = "research.sqlite3"


class GovernanceError(ReadoutError):
    """The research ledger could not be read."""


def _connect(research_root: Path | str) -> sqlite3.Connection:
    """Open the ledger read-only.

    `mode=ro` is refused by the driver if the file is missing, which is the
    answer wanted: a mistyped research root is an error, not an empty inbox.
    A caller told "no cards pending" for a path that does not exist would
    conclude there is nothing to decide.
    """

    ledger = Path(research_root) / LEDGER_NAME
    if not ledger.is_file():
        raise GovernanceError(f"no research ledger at {ledger}")
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return connection


def _summarize(payload: dict) -> dict:
    """One card as a queue entry: enough to choose, not enough to answer."""

    return {
        "request_id": payload.get("request_id"),
        "kind": payload.get("kind"),
        "experiment_id": payload.get("experiment_id"),
        "question": payload.get("question"),
        "recommended_action": payload.get("recommended_action"),
        # What happens if nobody acts. A queue that shows only the question
        # sorts by arrival; this is what lets a reader sort by consequence.
        "default_action": payload.get("default_action"),
        "consequence_of_waiting": payload.get("consequence_of_waiting"),
        "created_at": payload.get("created_at"),
    }


def pending_cards(research_root: Path | str) -> list[dict]:
    """Every decision request with no decision against it, oldest first."""

    with _connect(research_root) as connection:
        rows = connection.execute(
            """
            SELECT r.payload_json
            FROM decision_requests AS r
            LEFT JOIN research_decisions AS d
              ON d.request_id = r.request_id
            WHERE d.request_id IS NULL
            ORDER BY r.sequence
            """
        ).fetchall()
    return [_summarize(json.loads(row[0])) for row in rows]


def card(research_root: Path | str, request_id: str) -> dict:
    """One card in full, with its decision when it has one.

    Everything the request carries is included. The reader is the person the
    card is addressed to, so the question is what they need in order to judge,
    not what is safe to disclose.
    """

    if not isinstance(request_id, str) or not request_id.strip():
        raise GovernanceError("request_id must be a non-empty string")

    with _connect(research_root) as connection:
        row = connection.execute(
            "SELECT payload_json FROM decision_requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise GovernanceError(f"unknown decision request {request_id!r}")
        request = json.loads(row[0])
        answered = connection.execute(
            "SELECT payload_json FROM research_decisions WHERE request_id = ?",
            (request_id,),
        ).fetchone()

    return {
        "request": request,
        # Present and null rather than absent: "not yet answered" is the
        # state a reader most needs to see, and an absent key reads as a
        # rendering that forgot to include it.
        "decision": json.loads(answered[0]) if answered else None,
    }


def recent_decisions(
    research_root: Path | str, limit: int = 20
) -> list[dict]:
    """The most recently answered cards, newest first.

    Who signed and why, which is the record an audit reads. Ordered by the
    ledger's own sequence rather than by timestamp: the sequence is what the
    store assigned, and two decisions in the same second have an order.
    """

    if limit < 1:
        raise GovernanceError("limit must be at least 1")

    with _connect(research_root) as connection:
        rows = connection.execute(
            """
            SELECT payload_json FROM research_decisions
            ORDER BY sequence DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]
