"""Verifier answers, per candidate and per attempt.

Pinned here:

1. Every (status, reason) pair the verifier's contract lists maps to an
   outcome on purpose, and the pinned copy of that list still equals the
   verifier's own when `EVO_VERIFIER_CONTRACT` points at it.
2. The new no-verdict outcomes are in `NO_VERDICT_OUTCOMES`. Leaving one out
   means a goal that never exhausts and is retried forever.
3. The aggregation rule: certified first, then "only no-verdict answers" ends
   as the most severe no-verdict, then the run's stopping reason.
4. A solved score is PROVED only if the winning candidate's own envelope
   carries a certification.
5. The store keeps each candidate's answer with its attempt, in one write,
   and hands back the ones the verifier never answered.
6. The controller stops at once on a broken environment, and never pushes a
   goal toward exhaustion on any no-verdict outcome.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from evoharness.proof.contract import GoalContract, StatementError, proposition_of
from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import (
    CAPABILITY_OUTCOMES,
    NO_VERDICT_OUTCOMES,
    NO_VERDICT_SEVERITY,
    GoalStatus,
    GraphError,
    Outcome,
)
from evoharness.proof.sketch import Validation
from evoharness.proof.solver import AttemptResult, StubSolver
from evoharness.proof.store import ProofGraphStore
from evoharness.proof.verdict import (
    UNVERIFIED,
    CandidateVerdict,
    attempt_outcome,
    classify,
    is_known,
)

FIXTURE = Path(__file__).parent / "fixtures" / "verification_reasons.json"
TABLE = json.loads(FIXTURE.read_text())["by_status"]
PAIRS = [(status, reason) for status, reasons in TABLE.items()
         for reason in (reasons or [None])]


# -- 1. the pinned reason table ----------------------------------------------


@pytest.mark.parametrize("status,reason", PAIRS, ids=[f"{s}/{r}" for s, r in PAIRS])
def test_every_contract_pair_is_mapped_on_purpose(status, reason):
    assert is_known(status, reason)
    assert classify(status, reason) in {Outcome.PROVED} | CAPABILITY_OUTCOMES | NO_VERDICT_OUTCOMES


def test_schema_versions_match_the_verifier():
    """A graph's scope records these versions; they are only worth recording if
    they are the verifier's."""

    from evoharness.proof.scope import ENVELOPE_SCHEMA_VERSION, GOALKEY_SCHEMA_VERSION

    path = os.environ.get("EVO_VERIFIER_CONTRACT")
    if not path or not Path(path).is_file():
        pytest.skip("set EVO_VERIFIER_CONTRACT to the verifier's contract file")
    defs = json.loads(Path(path).read_text())["$defs"]
    assert defs["GoalKey"]["properties"]["schema_version"] == {"const": GOALKEY_SCHEMA_VERSION}
    assert defs["SourceEnvelope"]["properties"]["schema_version"] == {"const": ENVELOPE_SCHEMA_VERSION}


def test_pinned_reasons_still_equal_the_verifier():
    path = os.environ.get("EVO_VERIFIER_CONTRACT")
    if not path or not Path(path).is_file():
        pytest.skip("set EVO_VERIFIER_CONTRACT to the verifier's contract file")
    upstream = json.loads(Path(path).read_text())["x-verification-reasons"]["by_status"]
    assert TABLE == upstream


@pytest.mark.parametrize("status,reason,outcome", [
    ("goal_mismatch", "proposition_differs", Outcome.TASK_FAILED),
    ("policy_rejected", "sorry_axiom", Outcome.TASK_FAILED),
    ("policy_rejected", "trust_below_minimum", Outcome.TASK_FAILED),
    ("policy_rejected", "root_contract", Outcome.CONTRACT_REJECTED),
    ("policy_rejected", "namespace_prefix", Outcome.CONTRACT_REJECTED),
    ("stale_input", "base_changed", Outcome.STALE_INPUT),
    ("environment_mismatch", "missing_imports", Outcome.ENVIRONMENT_MISMATCH),
    ("environment_mismatch", "context_error", Outcome.ENVIRONMENT_MISMATCH),
    ("task_failed", "kernel_replay", Outcome.TASK_FAILED),
])
def test_the_split_statuses_land_where_they_belong(status, reason, outcome):
    assert classify(status, reason) is outcome


def test_an_unknown_answer_is_no_verdict_not_a_task_failure():
    """The verifier's contract moved past this one: a fault here, not the goal's."""

    assert classify("brand_new_status") is Outcome.CONTRACT_REJECTED
    assert classify("policy_rejected", "brand_new_reason") is Outcome.CONTRACT_REJECTED
    assert not is_known("policy_rejected", "brand_new_reason")
    assert classify(UNVERIFIED) is Outcome.INFRA_FAILED


# -- 2. the sets -------------------------------------------------------------


def test_every_new_outcome_carries_no_verdict():
    new = {Outcome.CONTRACT_REJECTED, Outcome.STALE_INPUT, Outcome.ENVIRONMENT_MISMATCH}
    assert new <= NO_VERDICT_OUTCOMES
    assert not (NO_VERDICT_OUTCOMES & CAPABILITY_OUTCOMES)
    # Budget is not an answer from the verifier, so it has no rank here.
    assert set(NO_VERDICT_SEVERITY) == NO_VERDICT_OUTCOMES - {Outcome.BUDGET_EXHAUSTED}


# -- 3. aggregation ----------------------------------------------------------


def v(status, reason=None, **kw):
    return CandidateVerdict(candidate_id=kw.pop("cid", f"c-{status}-{reason}"),
                            status=status, reason=reason, **kw)


def test_a_certified_candidate_wins_whatever_else_happened():
    assert attempt_outcome([v("infra_failed", "worker_failure"),
                            v("verified", certification_id="x")]) is Outcome.PROVED


def test_only_no_verdict_answers_end_as_the_most_severe():
    assert attempt_outcome([v("infra_failed", "worker_failure"),
                            v("environment_mismatch", "context_error")]) is Outcome.ENVIRONMENT_MISMATCH
    assert attempt_outcome([v("infra_failed", "worker_failure"),
                            v("policy_rejected", "root_contract")]) is Outcome.CONTRACT_REJECTED


def test_one_verdict_on_a_candidate_hands_back_to_the_run():
    assert attempt_outcome([v("infra_failed", "worker_failure"),
                            v("task_failed", "elaboration_error")]) is None
    assert attempt_outcome([]) is None


def report(fitness, reason="converged"):
    return SimpleNamespace(best_fitness=fitness, stopped_reason=reason, total_llm_cost=1.0)


def test_a_run_that_only_met_a_broken_environment_is_not_a_task_failure():
    result = AttemptResult.from_run_report(
        report(0.0), verifications=[v("environment_mismatch", "context_error", cid="a"),
                                    v("environment_mismatch", "context_error", cid="b")])
    assert result.outcome is Outcome.ENVIRONMENT_MISMATCH
    assert "context_error x2" in result.note
    assert len(result.verifications) == 2


# -- 4. binding --------------------------------------------------------------


def test_a_solved_score_needs_the_winners_own_certification():
    certified = v("verified", certification_id="cert-1", envelope_hash="env-win")
    proved = AttemptResult.from_run_report(
        report(1.0), proof_text="rfl", verifications=[certified],
        winner_envelope="env-win", verified_by_verifier=True)
    assert proved.outcome is Outcome.PROVED

    elsewhere = AttemptResult.from_run_report(
        report(1.0), proof_text="rfl", verifications=[certified],
        winner_envelope="env-other", verified_by_verifier=True)
    assert elsewhere.outcome is Outcome.CONTRACT_REJECTED
    assert elsewhere.proof_text is None

    local = AttemptResult.from_run_report(report(1.0), proof_text="rfl")
    assert local.outcome is Outcome.PROVED, "a local-compile run keeps the old rule"


# -- 5. the store ------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def test_verdicts_are_stored_with_their_attempt(store):
    goal = store.upsert_goal("id:g", "theorem g : True")
    pending = v(UNVERIFIED, cid="c1", envelope_hash="env-1", note="verifier unreachable")
    failed = v("task_failed", "elaboration_error", cid="c2", job_id="j2")
    attempt = store.record_attempt(goal.id, Outcome.INFRA_FAILED,
                                   verifications=[pending, failed])
    assert store.verifications_of(attempt.id) == [pending, failed]
    assert store.unverified_candidates() == [(attempt.id, pending)]


def test_goal_contract_is_written_once_and_never_rebound(store):
    goal = store.upsert_goal("id:g", "theorem g : True")
    contract = GoalContract(goal_key="k1", goal_key_obj={"key": "k1"}, base={"base_id": "b"},
                            name_prefix="", proposition="True")
    store.record_goal_contract(goal.id, contract)
    store.record_goal_contract(goal.id, contract)
    assert store.goal_contract(goal.id) == contract
    with pytest.raises(GraphError, match="already bound"):
        store.record_goal_contract(goal.id, GoalContract(
            goal_key="k2", goal_key_obj={"key": "k2"}, base={}, name_prefix="",
            proposition="True"))


# -- 6. the controller -------------------------------------------------------


ROOT = ("sha256:root", "theorem root : True")


def controller(store, solver):
    return ProofController(store, solver, decompositions=FixedDecompositions({}),
                           validate_sketch=lambda g, s: Validation(ok=True),
                           max_capability_attempts=1)


@pytest.mark.parametrize("outcome", [Outcome.ENVIRONMENT_MISMATCH, Outcome.STALE_INPUT])
def test_a_scope_wide_fault_stops_the_run_at_once(store, outcome):
    root = store.upsert_goal(*ROOT)
    report_ = controller(store, StubSolver({}, default=outcome)).solve(root.id, budget=100)
    assert report_.stopped_reason == outcome.value
    assert report_.attempts == 1
    assert store.goal(root.id).status is GoalStatus.OPEN


def test_contract_rejections_count_toward_the_no_verdict_streak(store):
    root = store.upsert_goal(*ROOT)
    solver = StubSolver({}, default=Outcome.CONTRACT_REJECTED)
    ctl = controller(store, solver)
    ctl.max_consecutive_infra = 2
    report_ = ctl.solve(root.id, budget=100)
    assert report_.stopped_reason == "infra"
    assert store.goal(root.id).status is GoalStatus.OPEN, "no verdict never exhausts"


# -- the statement -> proposition split ----------------------------------------


@pytest.mark.parametrize("statement,prop", [
    ("theorem t : 1 = 1", "1 = 1"),
    ("theorem t (a b : Nat) (h : a < b) : a ≤ b", "∀ (a b : Nat) (h : a < b), a ≤ b"),
    ("lemma t {α : Type} [inst : Inhabited α] (f : α → α) : f = f",
     "∀ {α : Type} [inst : Inhabited α] (f : α → α), f = f"),
    ("theorem t (s : Set Nat) (h : ∀ x ∈ s, x = 0 ∧ (x : Int) = 0) : True",
     "∀ (s : Set Nat) (h : ∀ x ∈ s, x = 0 ∧ (x : Int) = 0), True"),
])
def test_proposition_quantifies_the_binders_over_the_conclusion(statement, prop):
    assert proposition_of(statement) == prop


def test_a_statement_without_a_type_is_refused():
    with pytest.raises(StatementError):
        proposition_of("def f := 1")
    with pytest.raises(StatementError):
        proposition_of("theorem t (a : Nat)")
