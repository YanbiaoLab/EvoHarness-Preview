"""Grading through the verifier.

Pinned here, against a fake verifier (no Lean, no network unless stated):

1. The request is the one mapping from an envelope: body, context, roots,
   imports; the idempotency key follows the candidate, not the call.
2. The grader keeps the local grader's meanings -- `passed` is "compiles",
   1.0 only from a certification, a broken judge raises `InfraError` -- and
   records every answer in the attempt's ledger, envelope first.
3. Multi-declaration candidates are refused by text, before any request.
4. The client tells "no answer" from "refused as malformed".
5. `ApiRunSolver` with a verifier: the attempt's outcome is aggregated from the
   ledger and bound to the winning candidate's own certification.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from evoharness.proof.contract import GoalContract
from evoharness.proof.envelope import BEGIN, END, split_source
from evoharness.proof.graph import Goal, Outcome
from evoharness.proof.policy import AxiomPolicy
from evoharness.proof.run_solver import SEED_MAIN, ApiRunSolver, task_prompt
from evoharness.proof.verdict import UNVERIFIED
from evoharness.proof.verifier import (
    VerificationLedger,
    VerifierBinding,
    VerifierClient,
    VerifierGrader,
    VerifierRejected,
    VerifierUnavailable,
    ensure_contracts,
    extra_declarations,
    verification_request,
)
from evoharness.serve import InfraError

PREAMBLE = "import Mathlib\nopen Nat"
CONTEXT = {"raw": ["open Nat"]}
STATEMENT = "theorem t (n : Nat) : n + 0 = n"


def contract(**kw):
    data = dict(goal_key="gk", goal_key_obj={"key": "gk", "schema_version": 2},
                base={"base_id": "b", "fingerprint": "f" * 64}, name_prefix="",
                proposition="∀ (n : Nat), n + 0 = n", context=CONTEXT)
    data.update(kw)
    return GoalContract(**data)


def candidate(body: str) -> str:
    return f"{PREAMBLE}\n\n{BEGIN}\n{body}\n{END}\n\n#print axioms t\n"


class FakeVerifier:
    """Answers `verify` from a script keyed by a substring of the source."""

    def __init__(self, answers=None, *, down=False):
        self.answers = answers or {}
        self.down = down
        self.requests = []

    def verify(self, request):
        self.requests.append(request)
        if self.down:
            raise VerifierUnavailable("connection refused")
        for needle, (status, reason) in self.answers.items():
            if needle in request["source"]:
                cert = None
                if status == "verified":
                    cert = {"certification_id": "cert-" + needle, "trust": "trusted",
                            "goal_key": request["goal"]["key"]}
                return {"status": status, "reason": reason, "certification": cert,
                        "messages": [f"{status}/{reason}"], "worker_result": {}}
        return {"status": "task_failed", "reason": "elaboration_error",
                "certification": None, "messages": ["[error] nope"], "worker_result": {}}


def grade(tmp_path, body, verifier, **kw):
    workdir = tmp_path / "cand"
    workdir.mkdir(exist_ok=True)
    (workdir / SEED_MAIN).write_text(candidate(body))
    ledger = VerificationLedger(tmp_path / "ledger")
    grader = VerifierGrader(client=verifier, contract=contract(), root="t", preamble=PREAMBLE,
                            policy=AxiomPolicy(), ledger=ledger, **kw)
    return grader, grader(workdir, SimpleNamespace(candidate_id="c1"))


# -- 1. the request ----------------------------------------------------------


def test_the_request_is_the_one_mapping_from_an_envelope():
    envelope = split_source(candidate("theorem t (n : Nat) : n + 0 = n := rfl"),
                            preamble=PREAMBLE, root="t")
    request = verification_request(envelope, contract(), minimum_trust="audited")
    assert request["source"] == envelope.body
    assert request["context"] == CONTEXT
    assert request["imports"] == ["import Mathlib"]
    assert request["expected_root"] == "t" and request["auxiliary_roots"] == []
    assert request["goal"] == {"key": "gk", "schema_version": 2}
    again = verification_request(envelope, contract(), minimum_trust="audited")
    assert request["idempotency_key"] == again["idempotency_key"]
    stricter = verification_request(envelope, contract(), minimum_trust="trusted")
    assert stricter["idempotency_key"] != request["idempotency_key"]


def test_a_request_under_another_context_is_refused_here():
    envelope = split_source(candidate("theorem t (n : Nat) : n + 0 = n := rfl"),
                            preamble=PREAMBLE, root="t")
    with pytest.raises(ValueError, match="context"):
        verification_request(envelope, contract(context={}), minimum_trust="audited")


# -- 2. the grader -----------------------------------------------------------


def test_a_certification_is_the_only_way_to_one(tmp_path):
    grader, out = grade(tmp_path, "theorem t (n : Nat) : n + 0 = n := by simp  -- GOOD",
                        FakeVerifier({"GOOD": ("verified", None)}))
    assert out["fitness"] == 1.0 and out["passed"] is True
    assert out["visible_metrics"]["certification_id"] == "cert-GOOD"
    [verdict] = grader.ledger.verdicts()
    assert verdict.status == "verified" and verdict.certification_id == "cert-GOOD"
    assert verdict.envelope_hash == grader.ledger.envelopes()[0].envelope_hash


def test_sorry_compiles_and_can_be_a_parent(tmp_path):
    _, out = grade(tmp_path, "theorem t (n : Nat) : n + 0 = n := by\n  sorry",
                   FakeVerifier({"sorry": ("policy_rejected", "sorry_axiom")}))
    assert out["passed"] is True and 0 < out["fitness"] < 1.0
    assert "fault_kind" not in out


def test_proving_something_else_is_the_candidates_fault(tmp_path):
    _, out = grade(tmp_path, "theorem t (n : Nat) : n + 0 = n := WEAK",
                   FakeVerifier({"WEAK": ("goal_mismatch", "proposition_differs")}))
    assert out["fitness"] == 0.0 and out["fault_kind"] == "task_failure"


@pytest.mark.parametrize("status,reason", [
    ("environment_mismatch", "context_error"),
    ("policy_rejected", "root_contract"),
    ("infra_failed", "worker_failure"),
])
def test_no_verdict_raises_and_is_recorded(tmp_path, status, reason):
    workdir = tmp_path / "cand"
    workdir.mkdir()
    (workdir / SEED_MAIN).write_text(candidate("theorem t (n : Nat) : n + 0 = n := X"))
    grader = VerifierGrader(client=FakeVerifier({"X": (status, reason)}), contract=contract(),
                            root="t", preamble=PREAMBLE, policy=AxiomPolicy())
    with pytest.raises(InfraError):
        grader(workdir, None)
    [verdict] = grader.ledger.verdicts()
    assert (verdict.status, verdict.reason) == (status, reason)


def test_an_unreachable_verifier_keeps_the_candidate_for_later(tmp_path):
    workdir = tmp_path / "cand"
    workdir.mkdir()
    (workdir / SEED_MAIN).write_text(candidate("theorem t (n : Nat) : n + 0 = n := rfl"))
    ledger = VerificationLedger(tmp_path / "ledger")
    grader = VerifierGrader(client=FakeVerifier(down=True), contract=contract(), root="t",
                            preamble=PREAMBLE, policy=AxiomPolicy(), ledger=ledger)
    with pytest.raises(InfraError):
        grader(workdir, None)
    [verdict] = ledger.verdicts()
    assert verdict.status == UNVERIFIED and verdict.envelope_hash
    # On disk before the request was made: a crash mid-request leaves it behind.
    lines = (tmp_path / "ledger" / "envelopes.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["envelope_hash"] == verdict.envelope_hash


# -- 3. declarations beside the goal -------------------------------------------


def test_an_extra_declaration_is_refused_without_asking(tmp_path):
    verifier = FakeVerifier()
    body = "theorem helper : True := trivial\ntheorem t (n : Nat) : n + 0 = n := rfl"
    _, out = grade(tmp_path, body, verifier)
    assert out["fault_kind"] == "invalid_candidate" and "helper" in out["fault"]
    assert verifier.requests == []


def test_a_helper_named_under_the_goal_passes_only_when_allowed(tmp_path):
    body = "theorem t.step : True := trivial\ntheorem t (n : Nat) : n + 0 = n := rfl"
    assert extra_declarations(body, "t", allow_named_under_root=False) == ["theorem t.step"]
    assert extra_declarations(body, "t", allow_named_under_root=True) == []
    assert extra_declarations("theorem t : True := trivial\n  -- def x in a comment", "t",
                              allow_named_under_root=False) == []


# -- 4. the client -------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"error": "x", "kind": "k"}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("code,error", [(400, VerifierRejected), (503, VerifierUnavailable),
                                        (500, VerifierUnavailable)])
def test_the_client_tells_refusal_from_no_answer(code, error):
    handler = type("H", (_Handler,), {"status": code})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(error):
            VerifierClient(f"http://127.0.0.1:{server.server_address[1]}").verify({})
    finally:
        server.shutdown()
        server.server_close()


def test_nobody_listening_is_no_answer():
    with pytest.raises(VerifierUnavailable):
        VerifierClient("http://127.0.0.1:9", timeout_s=2).verify({})


# -- contracts ---------------------------------------------------------------


def test_contracts_are_resolved_in_one_batch_and_kept(tmp_path):
    from evoharness.proof.store import ProofGraphStore

    class Resolver:
        calls = 0

        def resolve_batch(self, items, *, base):
            Resolver.calls += 1
            return [
                {"ok": True, "resolved": {"goal_key": {"key": f"k{i}"}, "name_prefix": ""}}
                if "False" not in item["proposition"] else {"ok": False, "error": "no"}
                for i, item in enumerate(items)
            ]

    store = ProofGraphStore(tmp_path / "graph.db")
    a = store.upsert_goal("id:a", "theorem a : 1 = 1")
    b = store.upsert_goal("id:b", "theorem b : False")
    out = ensure_contracts(store, Resolver(), [a, b], base={"fingerprint": "f"},
                           preamble=PREAMBLE)
    assert out[a.id].goal_key == "k0" and out[a.id].context == CONTEXT
    assert isinstance(out[b.id], ValueError)
    ensure_contracts(store, Resolver(), [a], base={"fingerprint": "f"}, preamble=PREAMBLE)
    assert Resolver.calls == 1, "a stored contract is not resolved again"
    store.close()


# -- 5. the solver -------------------------------------------------------------


def test_the_prompt_follows_the_graphs_policy():
    local = task_prompt(STATEMENT)
    assert "native_decide" in local and "not proofs here" in local
    audited = task_prompt(STATEMENT, minimum_trust="audited")
    assert "native_decide on functions Lean already defines" in audited
    assert "no other declarations" in audited
    assert "`t.step`" in task_prompt(STATEMENT, minimum_trust="trusted", helpers_allowed=True)


@pytest.fixture
def fake_run(monkeypatch):
    """`api.run` that grades a fixed list of candidate bodies and reports the best."""

    from evoharness import api
    import evoharness.proof.run_solver as run_solver

    state = {}

    def run(task, run_spec, profile, transport=None):
        grade_fn = task.grader._grade_func if hasattr(task.grader, "_grade_func") else None
        best, best_text = 0.0, None
        infra = 0
        for i, body in enumerate(state["bodies"]):
            d = Path(state["tmp"]) / f"c{i}"
            d.mkdir(parents=True, exist_ok=True)
            text = candidate(body)
            (d / SEED_MAIN).write_text(text)
            try:
                out = grade_fn(d, SimpleNamespace(candidate_id=f"c{i}"))
            except InfraError:
                infra += 1
                continue
            if out["fitness"] >= best:
                best, best_text = out["fitness"], text
        state["best_text"] = best_text
        return SimpleNamespace(best_fitness=best, total_llm_cost=0.0,
                               stopped_reason="eval_infra" if infra and best < 1 else "converged")

    monkeypatch.setattr(api, "run", run)
    monkeypatch.setattr(run_solver, "best_candidate_text",
                        lambda run_dir, report: state.get("best_text") if report.best_fitness >= 1 else None)
    return state


def solver(tmp_path, verifier):
    binding = VerifierBinding(client=verifier, preamble=PREAMBLE, policy=AxiomPolicy(),
                              contract_for=lambda goal: contract())
    return ApiRunSolver(work_root=tmp_path / "runs", run_spec_factory=lambda out: None,
                        search_profile_factory=lambda: None, grade_func=None,
                        preamble=PREAMBLE, verifier=binding)


GOAL = Goal(id="g1", identity="id:t", statement=STATEMENT)


def test_a_certified_winner_proves_the_attempt(tmp_path, fake_run):
    fake_run.update(tmp=tmp_path / "c", bodies=[
        "theorem t (n : Nat) : n + 0 = n := by\n  sorry",
        "theorem t (n : Nat) : n + 0 = n := by simp  -- GOOD"])
    verifier = FakeVerifier({"GOOD": ("verified", None), "sorry": ("policy_rejected", "sorry_axiom")})
    result = solver(tmp_path, verifier).attack(GOAL, budget=10)
    assert result.outcome is Outcome.PROVED
    assert result.proof_text.strip().startswith("by simp")
    assert {v.candidate_id for v in result.verifications} == {"c0", "c1"}
    assert len(result.envelopes) == 2


def test_a_broken_environment_ends_the_attempt_as_no_verdict(tmp_path, fake_run):
    fake_run.update(tmp=tmp_path / "c", bodies=["theorem t (n : Nat) : n + 0 = n := X"])
    verifier = FakeVerifier({"X": ("environment_mismatch", "context_error")})
    result = solver(tmp_path, verifier).attack(GOAL, budget=10)
    assert result.outcome is Outcome.ENVIRONMENT_MISMATCH
    assert "environment_mismatch/context_error x1" in result.note


def test_an_unresolvable_goal_is_an_environment_answer(tmp_path, fake_run):
    from evoharness.proof.verifier import GoalUnresolvable

    def refuse(goal):
        raise GoalUnresolvable("unknown identifier")

    s = solver(tmp_path, FakeVerifier())
    s.verifier.contract_for = refuse
    assert s.attack(GOAL, budget=10).outcome is Outcome.ENVIRONMENT_MISMATCH
