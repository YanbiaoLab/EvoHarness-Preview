"""活性纪律:接了路由器但没跑,必须与从没接过区分开。"""

import json
import time

from evoharness.research import (
    ExperimentOutcome,
    ExperimentSpec,
    InboxStore,
    ResearchRouter,
    ResearchStore,
    budget_expansion_request,
)
from scripts.audit import audit_research
from tests.test_claim_assessment import _env

# 卡片的年龄是按墙上时钟算的,所以路由器的假时钟也得是"刚刚"。
RECENT = time.time()


def _run_dir(tmp_path, *, experiment_id="exp-1", envelopes=(), name="run-1"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed",
        "experiment_ref": {"experiment_id": experiment_id},
    }))
    if envelopes:
        (run_dir / "evidence.jsonl").write_text(
            "\n".join(json.dumps(e.to_json()) for e in envelopes) + "\n"
        )
    return run_dir


def _research(tmp_path):
    store = ResearchStore(tmp_path / "research")
    store.put_experiment(ExperimentSpec(
        experiment_id="exp-1", hypothesis_id="hyp-1", goal_id="goal-1",
        task_ref="t1", run_ref="r1", search_ref="s1",
        intervention="audit the router", created_at=1.0,
    ))
    return store


def _findings(run_dir, research_root):
    found = []
    audit_research(str(run_dir), str(research_root) if research_root else None,
                   lambda status, mechanism, detail: found.append(
                       (status, mechanism, detail)))
    return {mechanism: (status, detail) for status, mechanism, detail in found}


def _outcome(run_id="run-1"):
    return ExperimentOutcome(
        experiment_id="exp-1", spec_hash="h", run_id=run_id,
        stopped_reason="completed", generations_planned=1,
        generations_completed=1, evaluations=1, infra_drops=0,
        evidence_count=1, best_fitness=0.5, eval_cost_usd=1.0,
        created_at=1.0,
    )


def _router(store):
    return ResearchRouter(
        InboxStore(store, authorized_actors=frozenset({"zk"})),
        now=lambda: RECENT,
    )


def test_a_run_outside_any_experiment_is_not_audited(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"status": "completed"}))
    assert _findings(run_dir, tmp_path / "research") == {}


def test_a_wired_but_silent_router_is_dead(tmp_path):
    store = _research(tmp_path)
    store.append_outcome(_outcome())
    findings = _findings(_run_dir(tmp_path), tmp_path / "research")
    assert findings["research routing"][0] == "DEAD"
    assert "router unmounted" in findings["research routing"][1]


def test_a_run_that_bypassed_the_router_is_dead(tmp_path):
    store = _research(tmp_path)
    _router(store).route_run(
        _outcome(run_id="some-other-run"), run_dir=tmp_path / "elsewhere"
    )
    findings = _findings(_run_dir(tmp_path), tmp_path / "research")
    assert findings["research routing"][0] == "DEAD"
    assert "bypassed the router" in findings["research routing"][1]


def test_a_routed_run_reports_its_dispositions(tmp_path):
    store = _research(tmp_path)
    run_dir = _run_dir(tmp_path, envelopes=[_env("cand", 1.0, objective_met=True)])
    _router(store).route_run(_outcome(), run_dir=run_dir)

    findings = _findings(run_dir, tmp_path / "research")
    assert findings["research routing"][0].strip() == "ok"
    assert "'auto': 1" in findings["research routing"][1]
    assert "'card': 1" in findings["research routing"][1]
    assert findings["research assessment"][0].strip() == "ok"
    assert findings["research inbox"][0] == "info"      # 一张新卡,还不算积压


def test_verifier_verdicts_that_were_never_assessed_are_dead(tmp_path):
    store = _research(tmp_path)
    run_dir = _run_dir(tmp_path, envelopes=[_env("cand", 1.0, objective_met=True)])
    # 只记了 outcome,评估通道从没走到——证据有判决,却没有任何 claim 被评。
    _router(store).route("record-outcome", experiment_id="exp-1", run_id="run-1")

    findings = _findings(run_dir, tmp_path / "research")
    assert findings["research assessment"][0] == "DEAD"
    assert "no claim was ever assessed" in findings["research assessment"][1]


def test_an_unanswered_queue_eventually_warns(tmp_path):
    store = _research(tmp_path)
    run_dir = _run_dir(tmp_path)
    router = _router(store)
    router.route_run(_outcome(), run_dir=run_dir)
    router.inbox.submit(budget_expansion_request(
        experiment_id="exp-1", current_budget_usd=10.0,
        requested_budget_usd=20.0, justification="stale card",
        created_at=time.time() - 30 * 86400,
    ))

    findings = _findings(run_dir, tmp_path / "research")
    assert findings["research inbox"][0] == "WARN"
    assert "oldest 30 days" in findings["research inbox"][1]


def test_a_missing_research_root_is_reported_not_ignored(tmp_path):
    _research(tmp_path)
    findings = _findings(_run_dir(tmp_path), None)
    assert findings["research routing"][0] == "WARN"
    assert "no research root given" in findings["research routing"][1]
