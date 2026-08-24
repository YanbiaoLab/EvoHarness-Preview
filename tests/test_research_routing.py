"""I-5 接线契约:每个事件都有去处,自动执行也留痕,卡片不阻塞 run。"""

import json

import pytest

from evoharness.evaluation import ScoreNamespace
from evoharness.research import (
    AssessmentGuard,
    Claim,
    ClaimKind,
    ExperimentOutcome,
    ExperimentSpec,
    InboxStore,
    PromotionPolicy,
    ReferenceRecord,
    ReferenceStore,
    ResearchRouter,
    ResearchStore,
    RoutingError,
    Verdict,
    breakthrough_request,
    budget_expansion_card_for,
)
from tests.test_claim_assessment import NS, _env

CLOCK = 1000.0


def _frozen_experiment(store, experiment_id="exp-1"):
    spec = ExperimentSpec(
        experiment_id=experiment_id, hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="route research events", created_at=1.0,
    )
    store.put_experiment(spec)
    return spec


def _router(tmp_path, *, actors=("zk",)):
    store = ResearchStore(tmp_path / "research")
    _frozen_experiment(store)
    inbox = InboxStore(store, authorized_actors=frozenset(actors))
    return ResearchRouter(inbox, now=lambda: CLOCK), store, inbox


def _outcome(**overrides):
    defaults = dict(
        experiment_id="exp-1", spec_hash="h", run_id="run-1",
        stopped_reason="completed", generations_planned=4,
        generations_completed=4, evaluations=8, infra_drops=0,
        evidence_count=8, best_fitness=0.9, eval_cost_usd=2.0,
        created_at=CLOCK,
    )
    defaults.update(overrides)
    return ExperimentOutcome(**defaults)


def _write_evidence(run_dir, envelopes):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence.jsonl").write_text(
        "\n".join(json.dumps(env.to_json()) for env in envelopes) + "\n"
    )
    return run_dir


# -- fail-closed dispatch --------------------------------------------------


def test_auto_executable_event_must_not_carry_a_card(tmp_path):
    router, _store, _inbox = _router(tmp_path)
    with pytest.raises(RoutingError, match="auto-executable"):
        router.route(
            "record-outcome", experiment_id="exp-1",
            card=breakthrough_request(
                _supported_objective(), experiment_id="exp-1",
                candidate_id="cand", created_at=CLOCK,
            ),
        )


def test_event_needing_a_human_without_a_card_is_an_error(tmp_path):
    router, _store, _inbox = _router(tmp_path)
    with pytest.raises(RoutingError, match="no card was built"):
        router.route("announce-breakthrough", experiment_id="exp-1")


def test_wrong_card_kind_for_the_event_is_refused(tmp_path):
    router, _store, _inbox = _router(tmp_path)
    with pytest.raises(RoutingError, match="'budget-expansion' card"):
        router.route(
            "expand-budget", experiment_id="exp-1",
            card=breakthrough_request(
                _supported_objective(), experiment_id="exp-1",
                candidate_id="cand", created_at=CLOCK,
            ),
        )


def test_auto_execution_still_leaves_a_trace(tmp_path):
    router, store, inbox = _router(tmp_path)
    event = router.route(
        "record-outcome", experiment_id="exp-1", detail="run-1 completed"
    )
    assert event.disposition == "auto" and not event.request_id
    (recorded,) = store.routed_events("exp-1")
    assert recorded["event_kind"] == "record-outcome"
    assert inbox.pending() == []            # 自动执行不占人的队列


# -- breakthrough ----------------------------------------------------------


def _supported_objective(candidate_id="cand"):
    return AssessmentGuard().assess(
        Claim(ClaimKind.OBJECTIVE_MET, candidate_id),
        _env(candidate_id, 1.0, objective_met=True),
    )


def test_supported_objective_queues_a_breakthrough_card(tmp_path):
    router, store, inbox = _router(tmp_path)
    (event,) = router.route_assessment(
        _supported_objective(), experiment_id="exp-1", candidate_id="cand",
    )
    assert event.disposition == "card"
    (card,) = inbox.pending()
    assert card.kind == "breakthrough" and card.default_action == "veto"
    assert [a.claim_kind for a in store.assessments("exp-1")] == [
        "objective_met"
    ]


def test_routing_the_same_assessment_twice_stays_one_card(tmp_path):
    router, store, inbox = _router(tmp_path)
    assessment = _supported_objective()
    router.route_assessment(
        assessment, experiment_id="exp-1", candidate_id="cand"
    )
    router.route_assessment(
        assessment, experiment_id="exp-1", candidate_id="cand"
    )
    assert len(inbox.pending()) == 1
    assert len(store.assessments("exp-1")) == 1


def test_unsupported_objective_asks_nobody_anything(tmp_path):
    router, _store, inbox = _router(tmp_path)
    unknown = AssessmentGuard().assess(
        Claim(ClaimKind.OBJECTIVE_MET, "cand"),
        _env("cand", 1.0, objective_met=None),
    )
    assert unknown.status is Verdict.UNKNOWN
    assert router.route_assessment(
        unknown, experiment_id="exp-1", candidate_id="cand"
    ) == ()
    assert inbox.pending() == []


# -- conflicting evidence --------------------------------------------------


def test_two_assessors_disagreeing_notify_without_queueing_a_card(tmp_path):
    router, store, inbox = _router(tmp_path)
    claim = Claim(ClaimKind.BETTER_THAN_REFERENCE, "cand", "ref")
    subject, reference = _env("cand", 0.9), _env("ref", 0.5)
    supported = AssessmentGuard().assess(claim, subject, reference)
    # 同一个 claim,方向配错的评估器给出相反判决——这正是要被人看见的冲突。
    contradicted = AssessmentGuard(direction="minimize").assess(
        claim, subject, reference
    )
    assert supported.status is Verdict.SUPPORTED
    assert contradicted.status is Verdict.CONTRADICTED

    router.route_assessment(supported, experiment_id="exp-1", candidate_id="cand")
    (event,) = router.route_assessment(
        contradicted, experiment_id="exp-1", candidate_id="cand"
    )
    assert event.disposition == "notification"
    assert event.event_kind == "conflicting-evidence"
    assert inbox.pending() == []            # 通知但不阻塞
    assert len(store.routed_events("exp-1")) == 1


# -- budget ----------------------------------------------------------------


def test_budget_exhaustion_asks_for_more_than_the_current_budget():
    card = budget_expansion_card_for(
        _outcome(
            stopped_reason="budget", generations_completed=1,
            eval_cost_usd=3.0,
        ),
        budget_usd=10.0,
    )
    assert card is not None and card.kind == "budget-expansion"
    requested = json.loads(card.payload)["requested_budget_usd"]
    assert requested == pytest.approx(10.0 + 3.0 * 3)   # 3 代未跑,每代 3 USD
    assert card.default_action == "veto"


def test_budget_exhausted_before_the_first_generation_doubles():
    card = budget_expansion_card_for(
        _outcome(
            stopped_reason="budget", generations_completed=0,
            eval_cost_usd=9.5,
        ),
        budget_usd=10.0,
    )
    assert json.loads(card.payload)["requested_budget_usd"] == 20.0
    assert "无速率可外推" in card.question


def test_a_finished_run_asks_for_nothing():
    assert budget_expansion_card_for(_outcome(), budget_usd=10.0) is None
    assert budget_expansion_card_for(
        _outcome(stopped_reason="budget"), budget_usd=10.0
    ) is None                                # 预算停但代数已跑满
    assert budget_expansion_card_for(
        _outcome(stopped_reason="budget", generations_completed=1),
        budget_usd=None,
    ) is None                                # 没有上限就没有可扩的预算


# -- ranking flip ----------------------------------------------------------


def _reference_store(tmp_path):
    return ReferenceStore(tmp_path / "reference.json")


def _record(candidate_id, fitness, namespace: ScoreNamespace = NS):
    return ReferenceRecord(
        candidate_id=candidate_id, evidence_id=f"ev-{candidate_id}",
        namespace=namespace, fitness=fitness,
        policy_hash="p", assessor_hash="a",
        reasons=("seeded",), promoted_at=1.0,
    )


def test_champion_change_queues_a_ranking_flip_card(tmp_path):
    router, _store, inbox = _router(tmp_path)
    references = _reference_store(tmp_path)
    references.promote(_record("old", 0.5), expected_current=None)
    policy = PromotionPolicy(
        AssessmentGuard(), references,
        router=router, experiment_id="exp-1",
    )

    decision = policy.consider(_env("new", 0.9), _env("old", 0.5))

    assert decision.action == "promote"
    (card,) = inbox.pending()
    assert card.kind == "ranking-flip"
    subject = json.loads(card.payload)
    assert subject["previous_candidate_id"] == "old"
    assert subject["promoted_candidate_id"] == "new"
    assert references.current().candidate_id == "new"   # 晋升不等人


def test_bootstrap_is_not_a_ranking_flip(tmp_path):
    router, _store, inbox = _router(tmp_path)
    references = _reference_store(tmp_path)
    policy = PromotionPolicy(
        AssessmentGuard(), references,
        router=router, experiment_id="exp-1",
    )
    assert policy.consider(_env("first", 0.5), None).action == "promote"
    assert inbox.pending() == []            # 从无到有,没有排名被推翻


def test_held_promotion_flips_nothing(tmp_path):
    router, _store, inbox = _router(tmp_path)
    references = _reference_store(tmp_path)
    references.promote(_record("old", 0.9), expected_current=None)
    policy = PromotionPolicy(
        AssessmentGuard(), references,
        router=router, experiment_id="exp-1",
    )
    assert policy.consider(_env("new", 0.1), _env("old", 0.9)).action == "reject"
    assert inbox.pending() == []


def test_routing_a_promotion_needs_the_experiment_it_belongs_to(tmp_path):
    router, _store, _inbox = _router(tmp_path)
    with pytest.raises(ValueError, match="experiment"):
        PromotionPolicy(AssessmentGuard(), _reference_store(tmp_path),
                        router=router)


def test_router_does_not_change_promotion_identity(tmp_path):
    router, _store, _inbox = _router(tmp_path)
    references = _reference_store(tmp_path)
    plain = PromotionPolicy(AssessmentGuard(), references)
    routed = PromotionPolicy(
        AssessmentGuard(), references, router=router, experiment_id="exp-1"
    )
    assert plain.policy_hash == routed.policy_hash


# -- the whole run ---------------------------------------------------------


def test_route_run_covers_outcome_and_evidence(tmp_path):
    router, store, inbox = _router(tmp_path)
    run_dir = _write_evidence(tmp_path / "run", [
        _env("cand-a", 1.0, objective_met=True),
        _env("cand-b", 0.4, objective_met=None),
    ])

    events = router.route_run(
        _outcome(stopped_reason="budget", generations_completed=1),
        run_dir=run_dir, budget_usd=10.0,
    )

    assert [e.disposition for e in events] == ["auto", "card", "card"]
    kinds = {card.kind for card in inbox.pending()}
    assert kinds == {"budget-expansion", "breakthrough"}
    # 没有 verifier 判决的候选不产生任何评估记录
    assert [a.claim_hash for a in store.assessments("exp-1")] == [
        Claim(ClaimKind.OBJECTIVE_MET, "cand-a").hash
    ]


def test_run_experiment_routes_its_outcome(tmp_path):
    from evoharness import BasicSearchProfile, spec_hashes
    from evoharness.research import run_experiment
    from tasks.demo_counter import make_task
    from tests.test_public_specs import _run

    task = make_task()
    run_spec = _run(tmp_path / "run")
    profile = BasicSearchProfile(
        num_trajectories=1, proposal_mode="single_shot",
    )
    refs = spec_hashes(task.spec, run_spec, profile)
    store = ResearchStore(tmp_path / "research")
    inbox = InboxStore(store, authorized_actors=frozenset({"zk"}))
    router = ResearchRouter(inbox, now=lambda: CLOCK)
    experiment = ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref=refs["task_hash"], run_ref=refs["run_hash"],
        search_ref=refs["search_hash"],
        intervention="route research events", created_at=1.0,
    )

    outcome = run_experiment(
        store, experiment,
        task=task, run_spec=run_spec, profile=profile,
        router=router, now=lambda: 42.0,
    )

    (routed,) = store.routed_events("exp-1")
    assert routed["event_kind"] == "record-outcome"
    assert outcome.run_id in routed["detail"]
