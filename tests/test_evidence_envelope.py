"""I-2 契约:故障分类、coverage、namespace、信封准入政策。"""

import copy

import pytest

from evoharness.core.population import EvalReport
from evoharness.evaluation import (
    Coverage,
    EvidenceEnvelope,
    EvidenceProtocolError,
    FaultKind,
    NamespaceMismatch,
    ScoreNamespace,
    classify_fault,
    envelope_for_infra_error,
    envelope_from_report,
    may_enter_population,
    may_rank,
    may_support_objective,
)
from evoharness.evaluation.evidence import decode_evidence, encode_evidence
from evoharness.evaluation.producer import envelope_from_report as build_evidence

NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m1",
    evaluator_hash="e1", universe_hash="u1",
)


def test_fault_classification_vocabulary():
    assert classify_fault(passed=True, fault_kind=None) is None
    assert classify_fault(passed=False, fault_kind=None) is FaultKind.TASK_FAILURE
    assert classify_fault(passed=False, fault_kind="timeout") is FaultKind.TIMEOUT
    # 词表外的值 fail closed,不静默变成任务失败
    assert classify_fault(passed=False, fault_kind="weird") is FaultKind.UNKNOWN


def test_undeclared_plan_is_never_complete():
    assert Coverage(0, 100, 100).complete is False


def test_trustworthy_cannot_exceed_executed():
    with pytest.raises(ValueError):
        Coverage(planned_units=10, executed_units=5, trustworthy_units=6)


def test_executed_units_cannot_overrun_a_declared_universe():
    with pytest.raises(ValueError, match="cannot exceed planned"):
        Coverage(planned_units=10, executed_units=11, trustworthy_units=10)


def test_universe_change_breaks_comparability():
    other = ScoreNamespace(
        criterion_hash="c1", measurement_hash="m1",
        evaluator_hash="e1", universe_hash="u2",
    )
    with pytest.raises(NamespaceMismatch, match="universe_hash"):
        NS.require_comparable(other)


def test_verdict_report_becomes_admissible_evidence():
    report = EvalReport(
        fitness=0.8, passed=True, n_units=100, trustworthy_units=100
    )
    env = envelope_from_report(
        report, candidate_id="c", namespace=NS, planned_units=100
    )
    assert env.evaluation_valid and env.admissible and env.fault_kind is None
    assert env.coverage.complete
    assert may_enter_population(env)


def test_timeout_is_a_distinct_state_not_task_failure():
    report = EvalReport(
        fitness=0.0, passed=False,
        fault="evaluation timeout", fault_kind="timeout",
    )
    env = envelope_from_report(
        report, candidate_id="c", namespace=NS, planned_units=10
    )
    assert env.fault_kind is FaultKind.TIMEOUT
    assert env.fault_kind is not FaultKind.TASK_FAILURE
    assert may_enter_population(env)   # 仍入种群、触发 repair(wire 语义不变)


@pytest.mark.parametrize(
    "fault_kind", ["missing", "infra_error", "protocol_error", "unknown"]
)
def test_no_verdict_fault_cannot_be_smuggled_through_a_report(fault_kind):
    with pytest.raises(EvidenceProtocolError):
        envelope_from_report(
            EvalReport(fitness=0.0, passed=False, fault_kind=fault_kind),
            candidate_id="c",
            namespace=NS,
            planned_units=10,
        )


def test_infra_evidence_carries_no_verdict():
    env = envelope_for_infra_error(
        candidate_id="c", namespace=NS, planned_units=10, error="judge 503"
    )
    assert env.evaluation_valid is False
    assert env.fitness is None and env.admissible is None
    assert not may_enter_population(env)


def test_ranking_across_namespaces_is_a_type_error():
    a = envelope_from_report(
        EvalReport(fitness=0.5, passed=True), candidate_id="a",
        namespace=NS, planned_units=0,
    )
    b = envelope_from_report(
        EvalReport(fitness=0.9, passed=True), candidate_id="b",
        namespace=ScoreNamespace(
            criterion_hash="c1", measurement_hash="m2",
            evaluator_hash="e1", universe_hash="u1",
        ),
        planned_units=0,
    )
    with pytest.raises(NamespaceMismatch):
        may_rank(a, b)


def test_partial_coverage_cannot_be_ranked_even_in_one_namespace():
    complete = envelope_from_report(
        EvalReport(
            fitness=0.5, passed=True, n_units=10, trustworthy_units=10
        ),
        candidate_id="a", namespace=NS, planned_units=10,
    )
    partial = envelope_from_report(
        EvalReport(
            fitness=0.9, passed=True, n_units=3, trustworthy_units=3
        ),
        candidate_id="b", namespace=NS, planned_units=10,
    )
    with pytest.raises(ValueError, match="incomplete"):
        may_rank(complete, partial)


def test_contradictory_envelope_is_rejected_at_construction():
    with pytest.raises(EvidenceProtocolError):
        EvidenceEnvelope(
            evidence_id="auto",
            candidate_id="c",
            namespace=NS,
            evaluation_valid=True,
            admissible=True,
            fitness=0.5,
            fault_kind=FaultKind.INFRA_ERROR,
            coverage=Coverage(10, 10, 10),
        )


def test_wire_booleans_are_not_truthiness_coerced():
    env = envelope_from_report(
        EvalReport(
            fitness=0.5, passed=True, n_units=10, trustworthy_units=10
        ),
        candidate_id="c", namespace=NS, planned_units=10,
    )
    payload = copy.deepcopy(env.to_json())
    payload["evaluation_valid"] = "false"
    with pytest.raises(TypeError, match="must be bool"):
        EvidenceEnvelope.from_json(payload)
    with pytest.raises(TypeError, match="passed must be bool"):
        EvalReport.from_json({"fitness": 0.5, "passed": "false"})
    with pytest.raises(TypeError, match="n_units must be an integer"):
        EvalReport.from_json(
            {"fitness": 0.5, "passed": True, "n_units": "10"}
        )


def test_evidence_identity_is_content_addressed_and_payload_is_frozen():
    report = EvalReport(
        fitness=0.5,
        passed=True,
        n_units=10,
        trustworthy_units=10,
        visible_metrics={"items": [1, 2]},
    )
    first = envelope_from_report(
        report, candidate_id="c", namespace=NS, planned_units=10
    )
    second = envelope_from_report(
        report, candidate_id="c", namespace=NS, planned_units=10
    )
    assert first.evidence_id == second.evidence_id == first.content_hash()
    with pytest.raises(TypeError):
        first.observations["visible_metrics"] = {}


def test_split_factory_and_codec_preserve_the_public_wire_contract():
    env = build_evidence(
        EvalReport(
            fitness=0.5,
            passed=True,
            n_units=10,
            trustworthy_units=10,
        ),
        candidate_id="c",
        namespace=NS,
        planned_units=10,
    )
    payload = encode_evidence(env)
    assert decode_evidence(payload) == env
    assert EvidenceEnvelope.from_json(payload) == env


def test_objective_claim_requires_a_verifier_identity():
    with pytest.raises(EvidenceProtocolError, match="verifier identity"):
        EvidenceEnvelope(
            evidence_id="auto",
            candidate_id="c",
            namespace=NS,
            evaluation_valid=True,
            admissible=True,
            fitness=1.0,
            fault_kind=None,
            coverage=Coverage(10, 10, 10),
            objective_met=True,
        )


def test_passed_true_is_never_objective_met():
    """passed 是评估层裁决;objective_met 只能由 SuccessPolicy/Verifier 写入。"""
    report = EvalReport(
        fitness=1.0, passed=True, n_units=100, trustworthy_units=100
    )
    env = envelope_from_report(
        report, candidate_id="c", namespace=NS, planned_units=100
    )
    assert env.objective_met is None
    assert not may_support_objective(env)


def test_fault_kind_survives_wire_roundtrip():
    report = EvalReport(fitness=0.0, passed=False, fault_kind="timeout")
    assert EvalReport.from_json(report.to_json()).fault_kind == "timeout"


def test_legacy_wire_report_without_fault_kind_still_loads():
    legacy = {"fitness": 0.5, "passed": False}
    assert EvalReport.from_json(legacy).fault_kind is None


def test_declared_success_must_report_trustworthy_units():
    with pytest.raises(EvidenceProtocolError, match="trustworthy_units"):
        envelope_from_report(
            EvalReport(fitness=1.0, passed=True, n_units=10),
            candidate_id="c",
            namespace=NS,
            planned_units=10,
        )
