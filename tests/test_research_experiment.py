"""I-3 契约:实验冻结、append-only、ref 校验与 run 级对账。"""

import json

import pytest

from evoharness import BasicSearchProfile, spec_hashes
from evoharness.research import (
    ExperimentRefMismatch,
    ExperimentSpec,
    ResearchDecision,
    ResearchStore,
    ResearchStoreError,
    run_experiment,
)
from tests.test_public_specs import _run

REFS = {"task_hash": "t1", "run_hash": "r1", "search_hash": "s1"}


def _experiment(refs, **overrides):
    defaults = dict(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        goal_id="goal-1",
        task_ref=refs["task_hash"],
        run_ref=refs["run_hash"],
        search_ref=refs["search_hash"],
        intervention="switch parent strategy to power_law",
        prediction_claims=('{"kind": "HasAnyGain"}',),
        stopping_rule="2 generations",
        created_at=1000.0,
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


def test_prediction_change_changes_experiment_identity():
    base = _experiment(REFS)
    revised = _experiment(
        REFS, prediction_claims=('{"kind": "NoRegression"}',)
    )
    assert base.hash != revised.hash
    # created_at 和 approved_by 是记录不是身份
    assert base.hash == _experiment(REFS, created_at=2000.0).hash
    assert base.hash == _experiment(REFS, approved_by="zk").hash


def test_store_is_write_once_for_specs(tmp_path):
    store = ResearchStore(tmp_path / "research")
    spec = _experiment(REFS)
    store.put_experiment(spec)
    store.put_experiment(spec)                    # 幂等重放
    with pytest.raises(ResearchStoreError):
        store.put_experiment(
            _experiment(REFS, intervention="something else")
        )
    assert store.load_experiment("exp-1").hash == spec.hash


def test_records_only_attach_to_frozen_experiments(tmp_path):
    store = ResearchStore(tmp_path / "research")
    decision = ResearchDecision(
        experiment_id="ghost", action="approve",
        reason="r", actor="zk", created_at=1.0,
    )
    with pytest.raises(ResearchStoreError):
        store.append_decision(decision)


def test_decisions_are_append_only(tmp_path):
    store = ResearchStore(tmp_path / "research")
    store.put_experiment(_experiment(REFS))
    for reason in ("first", "second"):
        store.append_decision(ResearchDecision(
            experiment_id="exp-1", action="approve",
            reason=reason, actor="zk", created_at=1.0,
        ))
    assert [d["reason"] for d in store.decisions("exp-1")] == [
        "first", "second",
    ]


def test_ref_mismatch_refuses_to_run(tmp_path):
    from tasks.demo_counter import make_task

    task = make_task()
    run_spec = _run(tmp_path / "run")
    profile = BasicSearchProfile(
        num_trajectories=1, proposal_mode="single_shot",
    )
    refs = spec_hashes(task.spec, run_spec, profile)
    drifted = _experiment({**refs, "run_hash": "stale"})
    store = ResearchStore(tmp_path / "research")
    with pytest.raises(ExperimentRefMismatch, match="run_hash"):
        run_experiment(
            store, drifted,
            task=task, run_spec=run_spec, profile=profile,
        )
    assert store.outcomes("exp-1") == []          # 拒跑就没有 outcome


def test_end_to_end_outcome_reconciles_plan_vs_reality(tmp_path):
    from tasks.demo_counter import make_task

    task = make_task()
    run_spec = _run(tmp_path / "run")
    profile = BasicSearchProfile(
        num_trajectories=2, proposal_mode="single_shot",
    )
    refs = spec_hashes(task.spec, run_spec, profile)
    store = ResearchStore(tmp_path / "research")

    outcome = run_experiment(
        store, _experiment(refs),
        task=task, run_spec=run_spec, profile=profile,
        now=lambda: 42.0,
    )

    assert outcome.generations_planned == 2
    assert outcome.generations_completed == 2
    assert outcome.evidence_count == outcome.evaluations + outcome.infra_drops
    (recorded,) = store.outcomes("exp-1")
    assert recorded == outcome
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["experiment_ref"]["experiment_id"] == "exp-1"
    assert manifest["experiment_ref"]["spec_hash"] == _experiment(refs).hash
