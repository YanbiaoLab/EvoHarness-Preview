"""签字这条路:谁能签、签什么、签完留下什么。

治理闭环此前是断的——`evoweb` 删了,会话里的工具按设计只读,所以卡片进得
去出不来。这是唯一的耐久签署入口,而它先于会话签署存在是刻意的:会话审批
阻塞一次工具调用,而这些决定跨天,所以"正好有人开着会话"永远只能是机会主
义路径,不能是唯一路径。

用例分两类:**拒绝**要有可操作的理由,**放行**要真的写进账本。只测前者的
话,一个永远拒绝的实现也能全绿。
"""

import json

import pytest

from evoharness.research import (
    DecisionRequest,
    ExperimentSpec,
    InboxError,
    InboxStore,
    ResearchStore,
)
from evoharness.research.answer import (
    ACTORS_FILE,
    AnswerRefused,
    answer_card,
    retype_the_id,
)
from evoharness.research.__main__ import build_parser


ACTOR = "zk"


@pytest.fixture
def research_root(tmp_path):
    root = tmp_path / "research"
    store = ResearchStore(root)
    store.put_experiment(ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="answer a card", created_at=1.0,
    ))
    inbox = InboxStore(store, authorized_actors=frozenset({ACTOR}))
    inbox.submit(DecisionRequest(
        kind="budget-expansion",
        experiment_id="exp-1",
        question="double the evaluation budget?",
        alternatives=("approve", "veto"),
        recommended_action="approve",
        consequence_of_waiting="the run stops at the current cap",
        subject_hash="deadbeef",
        created_at=1.0,
    ))
    (root / ACTORS_FILE).write_text(f"# who may sign\n{ACTOR}\n", encoding="utf-8")
    return root


def only_request(research_root):
    store = ResearchStore(research_root)
    inbox = InboxStore(store, authorized_actors=frozenset({ACTOR}))
    return inbox.pending()[0]


def sign(research_root, request_id, **changes):
    values = {
        "action": "approve",
        "reason": "the evidence supports it",
        "actor": ACTOR,
        "confirm": lambda _id: True,
        "show": lambda *_args: None,
    }
    values.update(changes)
    return answer_card(research_root, request_id, **values)


def test_a_signed_card_reaches_the_ledger(research_root):
    """The liveness control. Every refusal below would be satisfied by an
    implementation that refuses everything, and this is what says otherwise."""

    request = only_request(research_root)
    recorded = sign(research_root, request.request_id)

    assert recorded["action"] == "approve"
    assert recorded["actor"] == ACTOR
    assert recorded["request_id"] == request.request_id

    store = ResearchStore(research_root)
    inbox = InboxStore(store, authorized_actors=frozenset({ACTOR}))
    assert inbox.pending() == []
    assert inbox.decision(request.request_id).action == "approve"


def test_the_decision_records_which_interface_signed_it(research_root):
    """An audit that cannot tell a terminal signature from a session one
    cannot weigh them, and they do not weigh the same."""

    request = only_request(research_root)
    assert sign(research_root, request.request_id)["source"] == "cli"


def test_an_unlisted_identity_cannot_sign(research_root):
    request = only_request(research_root)
    with pytest.raises(InboxError, match="not authorized"):
        sign(research_root, request.request_id, actor="somebody-else")


def test_the_command_line_offers_no_way_to_choose_the_actor():
    """An --actor flag is a card nominating its own approver, wearing argv:
    whoever runs the command would pick the name in the audit record."""

    import argparse

    parser = build_parser()
    sub = [
        action for action in parser._actions  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction)  # noqa: SLF001
    ][0]
    flags = {
        action.dest for action in sub.choices["answer"]._actions  # noqa: SLF001
    }
    assert "actor" not in flags
    assert "reason" in flags


def test_an_action_the_card_does_not_allow_is_refused(research_root):
    request = only_request(research_root)
    # `branch` belongs to the generic policy, not to budget-expansion. The
    # allowed set is security policy, not card data.
    with pytest.raises(InboxError, match="not allowed"):
        sign(research_root, request.request_id, action="branch")


def test_a_card_cannot_be_signed_twice(research_root):
    request = only_request(research_root)
    sign(research_root, request.request_id)
    with pytest.raises(InboxError, match="already answered"):
        sign(research_root, request.request_id)


def test_a_failed_confirmation_writes_nothing(research_root):
    """Irreversible, so the abort has to leave the queue untouched rather than
    leave a half-written decision behind."""

    request = only_request(research_root)
    with pytest.raises(AnswerRefused, match="aborted"):
        sign(research_root, request.request_id, confirm=lambda _id: False)

    store = ResearchStore(research_root)
    inbox = InboxStore(store, authorized_actors=frozenset({ACTOR}))
    assert [r.request_id for r in inbox.pending()] == [request.request_id]
    assert inbox.decision(request.request_id) is None


def test_confirmation_needs_the_id_not_a_yes():
    """A yes/no prompt is answered by reflex; producing the identifier cannot
    be done without having read what is on screen."""

    assert retype_the_id("req-1", read=lambda _p: "req-1") is True
    assert retype_the_id("req-1", read=lambda _p: "y") is False
    assert retype_the_id("req-1", read=lambda _p: "") is False


def test_an_empty_reason_is_refused(research_root):
    request = only_request(research_root)
    with pytest.raises(AnswerRefused, match="reason is required"):
        sign(research_root, request.request_id, reason="   ")


def test_a_missing_actor_file_says_what_to_create(research_root):
    (research_root / ACTORS_FILE).unlink()
    request_id = "whatever"
    with pytest.raises(AnswerRefused) as caught:
        sign(research_root, request_id)
    assert ACTORS_FILE in str(caught.value)


def test_an_actor_file_of_only_comments_is_not_an_empty_allowlist(research_root):
    """Refusing everyone is the safe reading of a file that lists nobody; the
    alternative is an inbox with no authorized signer, which InboxStore
    already refuses to construct — but with a message about its own arguments
    rather than about the file the operator has to fix."""

    (research_root / ACTORS_FILE).write_text("# nobody yet\n", encoding="utf-8")
    with pytest.raises(AnswerRefused, match="lists no identities"):
        sign(research_root, "whatever")


def test_a_mistyped_research_root_is_an_error_not_an_empty_queue(tmp_path):
    """ResearchStore runs its schema DDL on construction, so reaching it first
    would manufacture an empty ledger and report nothing pending."""

    with pytest.raises(AnswerRefused, match="no research ledger"):
        sign(tmp_path / "typo", "whatever")
    assert not (tmp_path / "typo" / "research.sqlite3").exists()


def test_the_card_is_shown_before_it_is_signed(research_root):
    """A signer who has not seen the default cannot tell approving apart from
    walking away."""

    request = only_request(research_root)
    shown = []
    sign(research_root, request.request_id, show=shown.append)

    rendered = "\n".join(shown)
    assert request.request_id in rendered
    assert "the run stops at the current cap" in rendered
    assert "if nobody acts" in rendered


def test_the_old_records_stay_readable(research_root):
    """`source` has a default so a ledger written before it existed still
    loads — and "unknown" is distinguishable from "cli", which is the point."""

    from evoharness.research import ResearchDecision

    legacy = ResearchDecision.from_json(json.loads(json.dumps({
        "schema_version": 1,
        "experiment_id": "exp-1",
        "action": "approve",
        "reason": "signed before sources were recorded",
        "actor": ACTOR,
        "created_at": 1.0,
        "request_id": "req-old",
    })))
    assert legacy.source == "unknown"
