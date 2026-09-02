"""A certification is the record that the assembled proof compiled as one file.

PROVED is derived: a route closed, or a direct attempt succeeded. Neither says
the finished article was ever compiled together, and a reader handed only the
status takes the weaker claim for the stronger. The certification is the
stronger claim, kept as its own record so the two cannot be confused.
"""

from __future__ import annotations

import pytest

from evoharness.proof.store import ProofGraphStore


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def test_a_goal_starts_uncertified(store):
    goal = store.upsert_goal("text:g", "theorem g : True")

    assert store.latest_certification(goal.id) is None
    assert store.certifications_of(goal.id) == []


def test_a_certification_round_trips(store):
    goal = store.upsert_goal("text:g", "theorem g : True")

    written = store.record_certification(
        goal.id,
        ok=True,
        axioms=["propext", "Classical.choice"],
        text_sha256="abc",
    )
    read = store.latest_certification(goal.id)

    assert read == written
    assert read.ok is True
    assert read.axioms == frozenset({"propext", "Classical.choice"})
    assert read.text_sha256 == "abc"


def test_a_failed_certification_is_kept_as_a_finding(store):
    """Recording only passes would show a graph the compiler disagreed with
    as merely "not yet certified"."""
    goal = store.upsert_goal("text:g", "theorem g : True")

    store.record_certification(goal.id, ok=False, reason="does not compile")
    read = store.latest_certification(goal.id)

    assert read is not None
    assert read.ok is False
    assert read.reason == "does not compile"


def test_certifications_accumulate_in_order(store):
    """The graph can change under a certification, so the history of what
    was compiled when is what tells a stale pass from a current one."""
    goal = store.upsert_goal("text:g", "theorem g : True")

    store.record_certification(goal.id, ok=False, reason="first")
    store.record_certification(goal.id, ok=True)

    history = store.certifications_of(goal.id)
    assert [c.ok for c in history] == [False, True]
    assert store.latest_certification(goal.id).ok is True


def test_a_certification_belongs_to_one_goal(store):
    a = store.upsert_goal("text:a", "theorem a : True")
    b = store.upsert_goal("text:b", "theorem b : True")

    store.record_certification(a.id, ok=True)

    assert store.latest_certification(b.id) is None


def test_an_existing_graph_gains_the_table_on_open(tmp_path):
    """A graph written before certifications existed must still open, and
    must report its goals as uncertified rather than fail to read them."""
    path = tmp_path / "graph.db"
    first = ProofGraphStore(path)
    goal = first.upsert_goal("text:g", "theorem g : True")
    first._conn.execute("DROP TABLE certifications")
    first._conn.commit()
    first.close()

    reopened = ProofGraphStore(path)
    try:
        assert reopened.latest_certification(goal.id) is None
    finally:
        reopened.close()
