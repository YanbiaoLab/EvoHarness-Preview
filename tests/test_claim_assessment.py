"""I-4 契约:Claim 类型系统、fail closed、CAS 晋升与依据落盘。"""

import pytest

from evoharness.evaluation import (
    Coverage,
    EvidenceEnvelope,
    NamespaceMismatch,
    ScoreNamespace,
)
from evoharness.research import (
    AssessmentGuard,
    Claim,
    ClaimKind,
    InsufficientEvidence,
    PromotionPolicy,
    ReferenceConflict,
    ReferenceRecord,
    ReferenceStore,
    Verdict,
    require_supported,
)

NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m1",
    evaluator_hash="e1", universe_hash="u1",
)
OTHER_NS = ScoreNamespace(
    criterion_hash="c1", measurement_hash="m2",
    evaluator_hash="e1", universe_hash="u1",
)


def _env(candidate_id, fitness, *, complete=True, namespace=NS,
         objective_met=None, unit_results=None, sem=0.0):
    planned = 10
    executed = planned if complete else 3
    observations = {}
    if unit_results is not None:
        observations["unit_results"] = unit_results
    return EvidenceEnvelope(
        evidence_id="auto",
        candidate_id=candidate_id,
        namespace=namespace,
        evaluation_valid=True,
        admissible=True,
        fitness=fitness,
        fault_kind=None,
        coverage=Coverage(planned, executed, executed),
        objective_met=objective_met,
        objective_verifier_hash=(
            "verifier-v1" if objective_met is not None else None
        ),
        observations=observations,
        provenance={"sem": sem} if sem else {},
    )


GUARD = AssessmentGuard()


def test_partial_success_supports_existence_but_not_aggregate():
    """局部成功只能支持存在性 Claim(I-4 验收原文)。"""
    subject = _env("cand", 0.9, complete=False)
    reference = _env("ref", 0.5, complete=True)
    gain = GUARD.assess(
        Claim(ClaimKind.HAS_ANY_GAIN, "cand", reference_id="ref"),
        subject, reference,
    )
    better = GUARD.assess(
        Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand", reference_id="ref"),
        subject, reference,
    )
    assert gain.status is Verdict.SUPPORTED
    assert better.status is Verdict.UNKNOWN


def test_finite_sample_never_contradicts_existence():
    subject = _env("cand", 0.3, complete=True)
    reference = _env("ref", 0.5, complete=True)
    gain = GUARD.assess(
        Claim(ClaimKind.HAS_ANY_GAIN, "cand", reference_id="ref"),
        subject, reference,
    )
    assert gain.status is Verdict.UNKNOWN     # 不是 CONTRADICTED


def test_no_regression_unknown_without_unit_results():
    """只有聚合分数时 NoRegression 必须 unknown(验收原文)。"""
    assessment = GUARD.assess(
        Claim(ClaimKind.NO_REGRESSION, "cand", reference_id="ref"),
        _env("cand", 0.9), _env("ref", 0.5),
    )
    assert assessment.status is Verdict.UNKNOWN


def test_no_regression_flip_contradicts_and_full_hold_supports():
    ref_units = {"u1": True, "u2": True, "u3": False}
    flip = GUARD.assess(
        Claim(ClaimKind.NO_REGRESSION, "cand", reference_id="ref"),
        _env("cand", 0.9, unit_results={"u1": True, "u2": False, "u3": True}),
        _env("ref", 0.5, unit_results=ref_units),
    )
    hold = GUARD.assess(
        Claim(ClaimKind.NO_REGRESSION, "cand", reference_id="ref"),
        _env("cand", 0.9, unit_results={"u1": True, "u2": True, "u3": False}),
        _env("ref", 0.5, unit_results=ref_units),
    )
    assert flip.status is Verdict.CONTRADICTED
    assert hold.status is Verdict.SUPPORTED


def test_passed_never_sets_objective():
    assessment = GUARD.assess(
        Claim(ClaimKind.OBJECTIVE_MET, "cand"),
        _env("cand", 1.0, objective_met=None),
    )
    assert assessment.status is Verdict.UNKNOWN
    with pytest.raises(InsufficientEvidence):
        require_supported(assessment)


def test_cross_namespace_assessment_is_a_type_error():
    with pytest.raises(NamespaceMismatch):
        GUARD.assess(
            Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand",
                  reference_id="ref"),
            _env("cand", 0.9),
            _env("ref", 0.5, namespace=OTHER_NS),
        )


def test_noisy_advantage_within_floor_is_underpowered_not_a_verdict():
    """v2 噪声下限:IMO 终选教训——点估计比较会把幸运种子当成优势。"""
    subject = _env("cand", 0.55, sem=0.1)
    reference = _env("ref", 0.5, sem=0.1)     # floor = sqrt(0.02) ≈ 0.141
    better = GUARD.assess(
        Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand", reference_id="ref"),
        subject, reference,
    )
    gain = GUARD.assess(
        Claim(ClaimKind.HAS_ANY_GAIN, "cand", reference_id="ref"),
        subject, reference,
    )
    assert better.status is Verdict.UNKNOWN   # 不是 SUPPORTED 也不是 CONTRADICTED
    assert gain.status is Verdict.UNKNOWN
    assert any("underpowered" in r for r in better.reasons)


def test_advantage_beyond_noise_floor_still_supports():
    subject = _env("cand", 0.9, sem=0.05)
    reference = _env("ref", 0.5, sem=0.05)    # floor ≈ 0.0707,优势 0.4
    better = GUARD.assess(
        Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand", reference_id="ref"),
        subject, reference,
    )
    assert better.status is Verdict.SUPPORTED


def test_noise_free_behaviour_is_unchanged_from_v1():
    """sem 未上报时 floor=0,判定与 v1 完全一致(向后兼容)。"""
    worse = GUARD.assess(
        Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand", reference_id="ref"),
        _env("cand", 0.5), _env("ref", 0.5),
    )
    assert worse.status is Verdict.CONTRADICTED


def test_z_participates_in_assessor_identity():
    assert AssessmentGuard(z=2.0).assessor_hash != GUARD.assessor_hash


def test_assessor_hash_is_recorded_and_direction_sensitive():
    assessment = GUARD.assess(
        Claim(ClaimKind.CANDIDATE_VALID, "cand"), _env("cand", 1.0)
    )
    assert assessment.assessor_hash == GUARD.assessor_hash
    assert (
        AssessmentGuard(direction="minimize").assessor_hash
        != GUARD.assessor_hash
    )


# --- PromotionPolicy / ReferenceStore ---


def _policy(tmp_path, **kwargs):
    store = ReferenceStore(tmp_path / "reference.json")
    return PromotionPolicy(GUARD, store, **kwargs), store


def test_bootstrap_then_promote_then_reject(tmp_path):
    policy, store = _policy(tmp_path)
    first = policy.consider(_env("gen0", 0.5), None)
    assert first.action == "promote"
    assert store.current().candidate_id == "gen0"

    better = policy.consider(_env("gen1", 0.8), _env("gen0", 0.5))
    assert better.action == "promote"
    assert store.current().candidate_id == "gen1"
    assert store.current().previous_candidate_id == "gen0"

    worse = policy.consider(_env("gen2", 0.6), _env("gen1", 0.8))
    assert worse.action == "reject"
    assert store.current().candidate_id == "gen1"


def test_unknown_holds_instead_of_promoting(tmp_path):
    """unknown fail closed:部分覆盖的高分不能晋升(验收原文)。"""
    policy, store = _policy(tmp_path)
    policy.consider(_env("gen0", 0.5), None)
    decision = policy.consider(
        _env("gen1", 0.99, complete=False), _env("gen0", 0.5)
    )
    assert decision.action == "hold"
    assert store.current().candidate_id == "gen0"


def test_promotion_records_full_basis(tmp_path):
    policy, store = _policy(tmp_path)
    policy.consider(_env("gen0", 0.5), None)
    record = store.current()
    assert record.evidence_id
    assert record.policy_hash == policy.policy_hash
    assert record.assessor_hash == GUARD.assessor_hash


def _record(candidate_id, namespace=NS):
    return ReferenceRecord(
        candidate_id=candidate_id, evidence_id=f"ev-{candidate_id}",
        namespace=namespace, fitness=0.5, policy_hash="p",
        assessor_hash="a", reasons=(), promoted_at=1.0,
    )


def test_cas_conflict_on_stale_expected_current(tmp_path):
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record("gen0"), expected_current=None)
    with pytest.raises(ReferenceConflict):
        store.promote(_record("gen1"), expected_current=None)  # 过期视图


def test_namespace_migration_cannot_sneak_through_promotion(tmp_path):
    store = ReferenceStore(tmp_path / "reference.json")
    store.promote(_record("gen0"), expected_current=None)
    with pytest.raises(NamespaceMismatch):
        store.promote(
            _record("gen1", namespace=OTHER_NS), expected_current="gen0"
        )
