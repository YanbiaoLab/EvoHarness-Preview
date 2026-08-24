"""I-5 契约:卡片自足、动作绑定、append-only 回答、白名单路由、记分卡。"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from evoharness.research import (
    AUTO_EXECUTABLE,
    AssessmentGuard,
    Claim,
    ClaimKind,
    CoreScorecard,
    DecisionRequest,
    ExperimentSpec,
    InboxError,
    InboxStore,
    ResearchScorecard,
    ResearchDecision,
    ResearchStore,
    ResearchStoreError,
    breakthrough_request,
    compare_measurements,
    protocol_change_request,
    requires_human,
)
from tests.test_claim_assessment import _env
from tests.test_measurement_migration import NEW_NS, OLD_NS, _panel


def _report():
    return compare_measurements(
        _panel({"a": 0.9, "b": 0.7, "c": 0.5}, OLD_NS),
        _panel({"a": 0.8, "b": 0.6, "c": 0.4}, NEW_NS),
    )


def _frozen_experiment(store):
    spec = ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="migrate measurement", created_at=1.0,
    )
    store.put_experiment(spec)
    return spec


def _inbox(store, *actors):
    return InboxStore(store, authorized_actors=frozenset(actors or ("zk",)))


def test_card_answers_the_four_questions_without_a_transcript():
    """发生了什么/证据够不够/要决定什么/继续要花多少——卡片自足。"""
    card = protocol_change_request(
        _report(), experiment_id="exp-1", created_at=1.0,
        estimated_costs=(("request-more-evidence", 25.0),),
    )
    assert card.question                        # 发生了什么、要决定什么
    assert card.uncertainty                     # 证据够不够
    assert card.consequence_of_waiting          # 不决定会怎样
    assert card.estimated_costs                 # 继续要花多少
    assert card.alternatives and card.recommended_action


def test_wrong_action_for_card_kind_is_rejected(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store)
    request_id = inbox.submit(
        protocol_change_request(
            _report(), experiment_id="exp-1", created_at=1.0
        )
    )
    with pytest.raises(InboxError, match="not allowed"):
        inbox.answer(
            request_id, action="approve", actor="zk",
            reason="lgtm",
        )   # 协议变更卡上没有裸 approve——必须用 approve-protocol-change


def test_answer_records_actor_reason_timestamp_and_request_id(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store)
    request_id = inbox.submit(
        protocol_change_request(
            _report(), experiment_id="exp-1", created_at=1.0
        )
    )
    decision = inbox.answer(
        request_id, action="approve-protocol-change", actor="zk",
        reason="panel clean: 0 flips",
        now=lambda: 42.0,
    )
    assert decision.request_id == request_id
    assert decision.actor == "zk" and decision.created_at == 42.0
    (recorded,) = store.decisions("exp-1")
    assert recorded["request_id"] == request_id
    assert inbox.pending() == []                # 回答后出队


def test_double_answer_is_rejected(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store)
    request_id = inbox.submit(
        protocol_change_request(
            _report(), experiment_id="exp-1", created_at=1.0
        )
    )
    inbox.answer(request_id, action="veto", actor="zk", reason="not yet")
    with pytest.raises(InboxError, match="already answered"):
        inbox.answer(request_id, action="veto", actor="zk", reason="again")


def test_resubmission_is_idempotent(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store)
    card = protocol_change_request(
        _report(), experiment_id="exp-1", created_at=1.0
    )
    assert inbox.submit(card) == inbox.submit(card)
    assert len(inbox.pending()) == 1


def test_breakthrough_card_requires_supported_objective():
    guard = AssessmentGuard()
    unsupported = guard.assess(
        Claim(ClaimKind.OBJECTIVE_MET, "cand"),
        _env("cand", 1.0, objective_met=None),
    )
    with pytest.raises(ValueError, match="SUPPORTED"):
        breakthrough_request(
            unsupported, experiment_id="exp-1",
            candidate_id="cand", created_at=1.0,
        )


def test_breakthrough_never_defaults_to_announced():
    guard = AssessmentGuard()
    supported = guard.assess(
        Claim(ClaimKind.OBJECTIVE_MET, "cand"),
        _env("cand", 1.0, objective_met=True),
    )
    card = breakthrough_request(
        supported, experiment_id="exp-1",
        candidate_id="cand", created_at=1.0,
    )
    assert card.default_action == "veto"
    assert card.approval_required is True


def test_unknown_event_kinds_require_a_human():
    """白名单外一律进人工队列——fail closed 的路由方向。"""
    assert not requires_human("re-evaluate-within-approved-budget")
    assert requires_human("announce-breakthrough")
    assert requires_human("modify-measurement")
    assert requires_human("anything-the-router-has-never-seen")
    assert "announce-breakthrough" not in AUTO_EXECUTABLE


def test_request_kind_owns_approval_policy():
    with pytest.raises(TypeError):
        DecisionRequest(
            kind="protocol-change",
            experiment_id="exp-1",
            question="change protocol?",
            alternatives=("approve",),
            recommended_action="approve",
            allowed_actions=("approve",),
            approval_required=False,
        )

    card = protocol_change_request(
        _report(), experiment_id="exp-1", created_at=1.0,
    )
    assert card.allowed_actions == (
        "approve-protocol-change", "request-more-evidence", "veto",
    )
    assert card.approval_required is True
    assert card.default_action == "veto"


def test_tampered_policy_fields_are_rejected():
    payload = protocol_change_request(
        _report(), experiment_id="exp-1", created_at=1.0,
    ).to_json()
    payload["approval_required"] = False
    with pytest.raises(ValueError, match="policy fields"):
        DecisionRequest.from_json(payload)


def test_unauthorized_actor_cannot_answer(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store, "zk")
    request_id = inbox.submit(
        protocol_change_request(_report(), experiment_id="exp-1", created_at=1.0)
    )
    with pytest.raises(InboxError, match="not authorized"):
        inbox.answer(
            request_id, action="veto", actor="agent", reason="self approval"
        )


def test_research_store_cannot_bypass_inbox_for_request_decision(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    with pytest.raises(ResearchStoreError, match="InboxStore.answer"):
        store.append_decision(ResearchDecision(
            experiment_id="exp-1", action="approve-protocol-change",
            reason="bypass", actor="agent", created_at=1.0,
            request_id="forged-request",
        ))


def test_concurrent_answers_have_one_winner(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store, "alice", "bob")
    request_id = inbox.submit(
        protocol_change_request(_report(), experiment_id="exp-1", created_at=1.0)
    )

    def answer(actor):
        try:
            return inbox.answer(
                request_id, action="veto", actor=actor, reason="race"
            )
        except InboxError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(answer, ("alice", "bob")))

    assert sum(result is not None for result in results) == 1
    assert len(store.decisions("exp-1")) == 1
    assert inbox.pending() == []


def test_research_scorecard_counts_and_stays_honest(tmp_path):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = _inbox(store)
    inbox.submit(protocol_change_request(
        _report(), experiment_id="exp-1", created_at=1.0
    ))
    card = ResearchScorecard.build(
        tmp_path / "research", pending_requests=len(inbox.pending()),
    )
    assert card.experiments == 1
    assert card.pending_requests == 1
    assert card.discovery_latency_s is None     # 不伪造算不出的指标


def test_empty_research_root_yields_a_scorecard_not_none(tmp_path):
    card = ResearchScorecard.build(
        tmp_path / "nowhere", pending_requests=0,
    )
    assert card is not None and card.experiments == 0


def test_core_scorecard_prefers_finalized_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_report = {
        "stopped_reason": "running", "generations_completed": 2,
        "evaluations": 2, "history": [], "best_fitness": 0.5,
        "total_eval_cost": 1.0,
    }
    final_report = {**checkpoint_report, "stopped_reason": "completed"}
    (run_dir / "checkpoint.json").write_text(json.dumps({
        "run_report": checkpoint_report,
    }))
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed", "report": final_report,
    }))

    card = CoreScorecard.from_run_dir(run_dir)
    assert card.stopped_reason == "completed"
