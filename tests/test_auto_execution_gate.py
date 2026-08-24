"""I-5 契约:白名单只是门的第一道。协议漂移、账不全、预算没人批,都不放行。"""

import json

import pytest

from evoharness.research import (
    AutoExecutionDenied,
    AutoExecutionGate,
    ExperimentOutcome,
    ExperimentSpec,
    InboxStore,
    ResearchRouter,
    ResearchStore,
    budget_expansion_request,
)

RE_EVAL = "re-evaluate-within-approved-budget"


def _experiment(**overrides):
    defaults = dict(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="re-evaluate the champion", created_at=1.0,
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


def _store(tmp_path, **overrides):
    store = ResearchStore(tmp_path / "research")
    store.put_experiment(_experiment(**overrides))
    return store


def _inbox(store):
    return InboxStore(store, authorized_actors=frozenset({"zk"}))


def _outcome(run_id="run-1", eval_cost_usd=1.0):
    return ExperimentOutcome(
        experiment_id="exp-1", spec_hash="h", run_id=run_id,
        stopped_reason="completed", generations_planned=1,
        generations_completed=1, evaluations=1, infra_drops=0,
        evidence_count=1, best_fitness=0.5, eval_cost_usd=eval_cost_usd,
        created_at=1.0,
    )


def _metered_run(runs_root, run_id, spent_usd):
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "budget.json").write_text(json.dumps({
        "spent_usd": spent_usd, "hard_cap_usd": 100.0, "n_charges": 3,
    }))
    return run_dir


def _approved(tmp_path, *, estimated_cost_usd=10.0):
    return _store(
        tmp_path, approved_by="zk", estimated_cost_usd=estimated_cost_usd
    )


# -- the whitelist is only the first condition -----------------------------


def test_events_outside_the_whitelist_never_auto_execute(tmp_path):
    gate = AutoExecutionGate(_approved(tmp_path))
    decision = gate.check("announce-breakthrough", experiment_id="exp-1")
    assert not decision.allowed
    assert "whitelist" in decision.reasons[0]


def test_an_unfrozen_protocol_blocks_even_bookkeeping(tmp_path):
    gate = AutoExecutionGate(ResearchStore(tmp_path / "research"))
    decision = gate.check("record-outcome", experiment_id="ghost")
    assert not decision.allowed
    assert "not frozen" in decision.reasons[0]


def test_a_tampered_spec_is_not_a_frozen_protocol(tmp_path):
    store = _approved(tmp_path)
    spec_path = tmp_path / "research" / "experiments" / "exp-1" / "spec.json"
    payload = json.loads(spec_path.read_text())
    payload["intervention"] = "something else entirely"
    spec_path.write_text(json.dumps(payload))

    decision = AutoExecutionGate(store).check(
        "record-outcome", experiment_id="exp-1"
    )
    assert not decision.allowed
    assert "spec_hash" in decision.reasons[0]


def test_bookkeeping_needs_only_a_frozen_protocol(tmp_path):
    gate = AutoExecutionGate(_store(tmp_path))     # 未批预算也可以记录
    assert gate.check("record-outcome", experiment_id="exp-1").allowed
    assert gate.check(
        "quarantine-hard-regression", experiment_id="exp-1"
    ).allowed


def test_a_free_action_that_projects_cost_is_a_misclassification(tmp_path):
    decision = AutoExecutionGate(_store(tmp_path)).check(
        "record-outcome", experiment_id="exp-1", projected_cost_usd=5.0,
    )
    assert not decision.allowed
    assert "whitelist entry is wrong" in decision.reasons[0]


# -- spending needs a trusted ledger and a pre-approved budget -------------


def test_spending_without_an_approved_budget_is_denied(tmp_path):
    runs_root = tmp_path / "runs"
    store = _store(tmp_path)                       # 冻结了,但没人批过成本
    _metered_run(runs_root, "run-1", 1.0)
    store.append_outcome(_outcome())

    decision = AutoExecutionGate(store, runs_root=runs_root).check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=1.0,
    )
    assert not decision.allowed
    assert any("no pre-approved budget" in r for r in decision.reasons)


def test_eval_cost_alone_is_not_a_trusted_total(tmp_path):
    """没有 metered 总额就只有评测成本——提案花费无从对账,少算的账不是账。"""
    runs_root = tmp_path / "runs"
    store = _approved(tmp_path)
    store.append_outcome(_outcome(eval_cost_usd=1.0))

    gate = AutoExecutionGate(store, _inbox(store), runs_root=runs_root)
    ledger = gate.budget("exp-1")
    assert ledger.spent_usd == 1.0 and not ledger.trusted
    decision = gate.check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=0.5
    )
    assert not decision.allowed
    assert any("no metered total" in r for r in decision.reasons)


def test_a_metered_run_inside_an_approved_budget_runs_unattended(tmp_path):
    runs_root = tmp_path / "runs"
    store = _approved(tmp_path)
    _metered_run(runs_root, "run-1", 4.0)
    store.append_outcome(_outcome())

    gate = AutoExecutionGate(store, _inbox(store), runs_root=runs_root)
    ledger = gate.budget("exp-1")
    assert ledger.trusted and ledger.spent_usd == 4.0   # 总额,不是评测的 1.0
    assert ledger.remaining_usd == 6.0
    assert gate.check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=6.0
    ).allowed


def test_spending_past_the_approved_ceiling_is_denied(tmp_path):
    runs_root = tmp_path / "runs"
    store = _approved(tmp_path)
    _metered_run(runs_root, "run-1", 9.5)
    store.append_outcome(_outcome())

    decision = AutoExecutionGate(
        store, _inbox(store), runs_root=runs_root
    ).check(RE_EVAL, experiment_id="exp-1", projected_cost_usd=1.0)
    assert not decision.allowed
    assert any("approved 10.0000 USD" in r for r in decision.reasons)


def test_an_approved_expansion_raises_the_ceiling(tmp_path):
    runs_root = tmp_path / "runs"
    store = _approved(tmp_path)
    _metered_run(runs_root, "run-1", 9.5)
    store.append_outcome(_outcome())
    inbox = _inbox(store)
    request_id = inbox.submit(budget_expansion_request(
        experiment_id="exp-1", current_budget_usd=10.0,
        requested_budget_usd=25.0, justification="two more generations",
        created_at=1.0,
    ))
    gate = AutoExecutionGate(store, inbox, runs_root=runs_root)
    assert not gate.check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=1.0
    ).allowed                                   # 卡还没人回答,天花板没动

    inbox.answer(
        request_id, action="approve", actor="zk", reason="worth finishing"
    )
    assert gate.budget("exp-1").approved_usd == 25.0
    assert gate.check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=1.0
    ).allowed


def test_a_vetoed_expansion_does_not_raise_the_ceiling(tmp_path):
    runs_root = tmp_path / "runs"
    store = _approved(tmp_path)
    _metered_run(runs_root, "run-1", 9.5)
    store.append_outcome(_outcome())
    inbox = _inbox(store)
    request_id = inbox.submit(budget_expansion_request(
        experiment_id="exp-1", current_budget_usd=10.0,
        requested_budget_usd=25.0, justification="two more generations",
        created_at=1.0,
    ))
    inbox.answer(request_id, action="veto", actor="zk", reason="not worth it")

    gate = AutoExecutionGate(store, inbox, runs_root=runs_root)
    assert gate.budget("exp-1").approved_usd == 10.0


def test_a_negative_projected_cost_is_refused(tmp_path):
    gate = AutoExecutionGate(_approved(tmp_path))
    assert not gate.check(
        RE_EVAL, experiment_id="exp-1", projected_cost_usd=-1.0
    ).allowed


# -- the router refuses to downgrade silently ------------------------------


def test_the_router_raises_instead_of_downgrading(tmp_path):
    store = ResearchStore(tmp_path / "research")
    store.put_experiment(_experiment())
    router = ResearchRouter(_inbox(store))

    with pytest.raises(AutoExecutionDenied, match=RE_EVAL):
        router.route(
            RE_EVAL, experiment_id="exp-1", projected_cost_usd=1.0
        )
    assert store.routed_events("exp-1") == []   # 被拒的事件不算发生过
