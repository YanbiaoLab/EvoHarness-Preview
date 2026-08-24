"""I-4 后半:迁移面板指标、同 namespace 拒绝、人工审批强制。"""

import pytest

from evoharness.evaluation import (
    Coverage,
    EvidenceEnvelope,
    ScoreNamespace,
)
from evoharness.research import (
    MigrationApprovalRequired,
    MigrationPanelError,
    ExperimentSpec,
    InboxStore,
    ReferenceRecord,
    ReferenceStore,
    ResearchStore,
    compare_measurements,
    establish_baseline,
    protocol_change_request,
)

OLD_NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m1",
    evaluator_hash="e1", universe_hash="u1",
)
NEW_NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m2",
    evaluator_hash="e1", universe_hash="u1",
)


def _env(candidate_id, fitness, namespace, cost=1.0):
    return EvidenceEnvelope(
        evidence_id="auto",
        candidate_id=candidate_id,
        namespace=namespace,
        evaluation_valid=True,
        admissible=True,
        fitness=fitness,
        fault_kind=None,
        coverage=Coverage(10, 10, 10),
        budget_used_usd=cost,
    )


def _panel(scores, namespace, cost=1.0):
    return [_env(c, f, namespace, cost) for c, f in scores.items()]


def test_order_preserving_migration_reports_clean_metrics():
    old = _panel({"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}, OLD_NS)
    new = _panel({"a": 0.8, "b": 0.6, "c": 0.4, "d": 0.2}, NEW_NS, cost=2.0)
    report = compare_measurements(old, new, top_k=2)

    assert report.n_pairs == 4
    assert report.spearman == pytest.approx(1.0)
    assert report.champion_changed is False
    assert report.top_k_overlap == 1.0
    assert report.decision_flips == 0
    assert report.pairwise_inversions == 0
    assert report.cost_old_usd == pytest.approx(4.0)
    assert report.cost_new_usd == pytest.approx(8.0)
    assert report.requires_human_approval is True   # 永远要求人工审批


def test_champion_swap_is_surfaced_not_averaged_away():
    """秩相关只跌一点,但冠军换人了——面板必须显式报告,不许被均值掩盖。"""
    old = _panel({"a": 0.9, "b": 0.8, "c": 0.5, "d": 0.3}, OLD_NS)
    new = _panel({"a": 0.8, "b": 0.9, "c": 0.5, "d": 0.3}, NEW_NS)
    report = compare_measurements(old, new, top_k=2)

    assert report.champion_changed is True
    assert report.champion_old == "a" and report.champion_new == "b"
    assert report.decision_flips >= 1
    assert "b" in report.flipped_candidates
    assert report.pairwise_inversions >= 1
    assert ("a", "b") in report.inverted_pairs_sample


def test_same_namespace_panel_is_rejected():
    with pytest.raises(MigrationPanelError, match="nothing migrated"):
        compare_measurements(
            _panel({"a": 0.9, "b": 0.7}, OLD_NS),
            _panel({"a": 0.8, "b": 0.6}, OLD_NS),
        )


def test_mixed_namespace_panel_is_rejected():
    mixed = [_env("a", 0.9, OLD_NS), _env("b", 0.7, NEW_NS)]
    with pytest.raises(MigrationPanelError, match="mixes"):
        compare_measurements(mixed, _panel({"a": 0.8, "b": 0.6}, NEW_NS))


def _record(candidate_id, namespace, *, fitness=0.5, evidence_id="record"):
    return ReferenceRecord(
        candidate_id=candidate_id,
        evidence_id=evidence_id,
        namespace=namespace, fitness=fitness, policy_hash="p",
        assessor_hash="a", reasons=(), promoted_at=1.0,
    )


def _report():
    return compare_measurements(
        _panel({"a": 0.9, "b": 0.7, "c": 0.5}, OLD_NS),
        _panel({"a": 0.8, "b": 0.6, "c": 0.4}, NEW_NS),
    )


def _approved_inbox(tmp_path, report, *, action="approve-protocol-change"):
    research = ResearchStore(tmp_path / "research")
    research.put_experiment(ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t", run_ref="r", search_ref="s", intervention="migrate",
        created_at=1.0,
    ))
    inbox = InboxStore(research, authorized_actors={"zk"})
    request_id = inbox.submit(protocol_change_request(
        report, experiment_id="exp-1", created_at=1.0,
    ))
    inbox.answer(
        request_id, action=action, reason="panel reviewed", actor="zk",
        now=lambda: 2.0,
    )
    return inbox, request_id


def test_baseline_requires_protocol_change_approval(tmp_path):
    report = _report()
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record(
        "a", OLD_NS, fitness=0.9,
        evidence_id=report.champion_old_evidence_id,
    ), expected_current=None)
    inbox, request_id = _approved_inbox(tmp_path, report, action="veto")
    with pytest.raises(MigrationApprovalRequired):
        establish_baseline(
            store, _record(
                "a", NEW_NS, fitness=0.8,
                evidence_id=report.champion_new_evidence_id,
            ), report=report, inbox=inbox,
            request_id=request_id, experiment_id="exp-1",
        )
    assert store.current().namespace == OLD_NS   # 没被偷渡


def test_approved_migration_rebases_and_records_the_basis(tmp_path):
    report = _report()
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record(
        "a", OLD_NS, fitness=0.9,
        evidence_id=report.champion_old_evidence_id,
    ), expected_current=None)
    inbox, request_id = _approved_inbox(tmp_path, report)
    rebased = establish_baseline(
        store, _record(
            "a", NEW_NS, fitness=0.8,
            evidence_id=report.champion_new_evidence_id,
        ), report=report, inbox=inbox,
        request_id=request_id, experiment_id="exp-1",
        now=lambda: 42.0,
    )
    assert store.current().namespace == NEW_NS
    assert rebased.previous_candidate_id == "a"
    # 历史里必须能找到带 migration 依据的那一行
    history = (tmp_path / "reference_history.jsonl").read_text().splitlines()
    assert any('"migration"' in line for line in history)


def test_record_must_match_report_namespace(tmp_path):
    store = ReferenceStore(tmp_path / "reference.json")
    report = _report()
    inbox, request_id = _approved_inbox(tmp_path, report)
    with pytest.raises(MigrationPanelError, match="does not match"):
        establish_baseline(
            store, _record("a", OLD_NS, fitness=0.9), report=report, inbox=inbox,
            request_id=request_id, experiment_id="exp-1",
        )


def test_approval_cannot_be_replayed_for_another_report(tmp_path):
    approved_report = _report()
    inbox, request_id = _approved_inbox(tmp_path, approved_report)
    different_report = compare_measurements(
        _panel({"a": 0.9, "b": 0.7, "c": 0.5}, OLD_NS),
        _panel({"a": 0.6, "b": 0.8, "c": 0.4}, NEW_NS),
    )
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record(
        "a", OLD_NS, fitness=0.9,
        evidence_id=different_report.champion_old_evidence_id,
    ), expected_current=None)

    with pytest.raises(MigrationApprovalRequired, match="different subject"):
        establish_baseline(
            store, _record(
                "b", NEW_NS, fitness=0.8,
                evidence_id=different_report.champion_new_evidence_id,
            ), report=different_report,
            inbox=inbox,
            request_id=request_id, experiment_id="exp-1",
        )


def test_approved_report_cannot_select_a_non_champion_record(tmp_path):
    report = _report()
    inbox, request_id = _approved_inbox(tmp_path, report)
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record(
        "a", OLD_NS, fitness=0.9,
        evidence_id=report.champion_old_evidence_id,
    ), expected_current=None)

    with pytest.raises(MigrationPanelError, match="approved panel champion"):
        establish_baseline(
            store, _record("b", NEW_NS, fitness=0.6), report=report,
            inbox=inbox, request_id=request_id, experiment_id="exp-1",
        )
