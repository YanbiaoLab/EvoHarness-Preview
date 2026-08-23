"""Reading the governance queue, and being unable to answer it.

DH-5's first two layers: a session may be told a card exists and may show
what it says. The line is `InboxStore.answer` — dsh can read, draft and
notify; it cannot sign. What is asserted here is mostly that the reading path
has no writing in it, which is a property of what the module contains rather
than of a flag it checks.
"""

import sqlite3

import pytest

from evoharness.readout import governance
from evoharness.readout.governance import (
    GovernanceError,
    card,
    pending_cards,
    recent_decisions,
)
from evoharness.research import (
    DecisionRequest,
    ExperimentSpec,
    InboxStore,
    ResearchStore,
)


@pytest.fixture
def research_root(tmp_path):
    root = tmp_path / "research"
    store = ResearchStore(root)
    store.put_experiment(ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="raise the budget", created_at=1.0,
    ))
    return root, store


@pytest.fixture
def inbox(research_root):
    _, store = research_root
    return InboxStore(store, authorized_actors=frozenset({"zk"}))


def make_request(**changes):
    values = {
        "kind": "budget-expansion",
        "experiment_id": "exp-1",
        "question": "double the evaluation budget?",
        "alternatives": ("approve", "veto"),
        "recommended_action": "approve",
        "consequence_of_waiting": "the run stalls at generation 12",
        "subject_hash": "sha256:deadbeef",
        "created_at": 100.0,
    }
    values.update(changes)
    return DecisionRequest(**values)


def test_a_pending_card_is_listed_with_what_it_costs_to_wait(
    research_root, inbox
):
    root, _ = research_root
    request_id = inbox.submit(make_request())

    (entry,) = pending_cards(root)
    assert entry["request_id"] == request_id
    assert entry["question"] == "double the evaluation budget?"
    # The queue has to be sortable by consequence, not only by arrival: what
    # happens if nobody acts is the reason one card outranks another.
    assert entry["consequence_of_waiting"] == "the run stalls at generation 12"
    assert entry["default_action"] == "veto"


def test_an_answered_card_leaves_the_queue(research_root, inbox):
    root, _ = research_root
    request_id = inbox.submit(make_request())
    inbox.answer(request_id, action="approve", actor="zk", reason="fine")

    assert pending_cards(root) == []


def test_a_card_shows_its_decision_once_it_has_one(research_root, inbox):
    root, _ = research_root
    request_id = inbox.submit(make_request())

    unanswered = card(root, request_id)
    # Present and null, not absent: "nobody has signed this" is the state a
    # reader most needs, and a missing key reads as a view that forgot it.
    assert unanswered["decision"] is None
    assert unanswered["request"]["question"] == "double the evaluation budget?"

    inbox.answer(request_id, action="approve", actor="zk", reason="evidence holds")
    answered = card(root, request_id)
    assert answered["decision"]["actor"] == "zk"
    assert answered["decision"]["reason"] == "evidence holds"


def test_decisions_are_readable_newest_first(research_root, inbox):
    root, _ = research_root
    first = inbox.submit(make_request(question="first?"))
    second = inbox.submit(make_request(question="second?"))
    inbox.answer(first, action="approve", actor="zk", reason="a")
    inbox.answer(second, action="veto", actor="zk", reason="b")

    actions = [d["action"] for d in recent_decisions(root)]
    assert actions == ["veto", "approve"]


def test_the_reading_path_cannot_write(research_root, inbox):
    """The ledger is opened read-only, so a write fails at the driver.

    Belt and braces around the real design, which is that this module has no
    writing code at all: a guard that has to be remembered is one that can be
    forgotten, and a connection that physically cannot write cannot be.
    """

    root, _ = research_root
    inbox.submit(make_request())

    connection = governance._connect(root)  # noqa: SLF001
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM decision_requests")
    finally:
        connection.close()


def test_the_module_exposes_no_way_to_answer():
    """DH-5's boundary is one method. Asserting its absence keeps a later
    convenience from quietly crossing it."""

    surface = {name for name in dir(governance) if not name.startswith("_")}
    assert not {n for n in surface if "answer" in n or "submit" in n}


def test_a_missing_research_root_is_an_error_not_an_empty_queue(tmp_path):
    """"Nothing to decide" and "I could not look" must not print the same.

    A caller told the queue is empty concludes there is nothing waiting; the
    research store's own constructor would have created the ledger and made
    that true.
    """

    with pytest.raises(GovernanceError, match="no research ledger"):
        pending_cards(tmp_path / "typo")
    assert not (tmp_path / "typo").exists()


def test_an_unknown_request_id_is_refused(research_root):
    root, _ = research_root
    with pytest.raises(GovernanceError, match="unknown decision request"):
        card(root, "no-such-card")
