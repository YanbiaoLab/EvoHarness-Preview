"""Assembly, sketches, publication, retrieval and recovery through the verifier.

Against a fake verifier: what is pinned is what this side sends and how it
reads the answers, not Lean.

1. The assembled file goes out without its preamble, with every lemma declared
   as a root, qualified by the namespace the root goal was resolved under.
2. Helpers named under a goal are kept apart from its proof body, stored with
   the attempt, and assembled ahead of the declaration that uses them.
3. A sketch is checked in `sketch` mode; an answer with no verdict leaves it
   undecided instead of rejecting it.
4. Publication sends the certified envelope back byte for byte, under a key
   derived from the certification, so a retry is the same request.
5. Retrieval reaches the prompt and the graph, at the graph's own trust floor.
6. A dead attempt's verifier answers are kept with its INTERRUPTED record.
"""

from __future__ import annotations

import json

import pytest

from evoharness.proof.contract import GoalContract
from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import Goal, Outcome
from evoharness.proof.policy import AxiomPolicy
from evoharness.proof.retrieval import VerifierRetrievalProvider, prompt_section, queries_for
from evoharness.proof.run_solver import proof_parts
from evoharness.proof.sketch import Sketch, SketchUnavailable, SubgoalSpec, Validation
from evoharness.proof.solver import StubSolver
from evoharness.proof.store import ProofGraphStore
from evoharness.proof.verdict import CandidateVerdict
from evoharness.proof.verified import (
    VerifierSketchValidator,
    assembly_envelope,
    certify_with_verifier,
    declared_roots,
    publish,
)
from evoharness.proof.verifier import VerifierBinding

PREAMBLE = "import Mathlib\nopen Nat"
BASE = {"base_id": "b", "base_key": 1, "fingerprint": "f" * 64}


class FakeVerifier:
    def __init__(self, status="verified", reason=None):
        self.status, self.reason = status, reason
        self.requests, self.published = [], []

    def resolve_batch(self, items, *, base):
        return [{"ok": True, "resolved": {"goal_key": {"key": f"k{i}"}, "name_prefix": "Ns."}}
                for i, _ in enumerate(items)]

    def verify(self, request):
        self.requests.append(request)
        cert = None
        if self.status == "verified" and request["mode"] != "sketch":
            cert = {"certification_id": "cert-1", "goal_key": request["goal"]["key"],
                    "source_sha256": "s" * 64, "verifier_runtime_fingerprint": "r" * 64,
                    "axiom_set": ["propext"], "trust": "trusted", "evidence_hash": "e" * 64}
        return {"job_id": request["job_id"], "status": self.status, "reason": self.reason,
                "certification": cert, "messages": ["m"], "worker_result": {}}

    verify_async = verify

    def publish(self, request):
        self.published.append(request)
        return {"publication_id": "p1", "status": "admitted", "admitted_node_ids": [7],
                "certification_id": request["certification_id"], "base": BASE, "cached": False}

    def retrieve(self, request):
        self.requests.append(request)
        return [{"declaration_id": "b:1", "base_id": "b", "name": "Nat.succ_le", "kind": "thm",
                 "type": "a < b ↔ a + 1 ≤ b", "trust": 3, "axioms": [], "evidence": {}}]


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def proved(store, statement, body, helpers=()):
    goal = store.upsert_goal(f"id:{statement}", statement)
    store.record_attempt(goal.id, Outcome.PROVED, proof_text=body, auxiliary_declarations=helpers)
    store.propagate(goal.id, max_capability_attempts=3)
    return goal


# -- 1. the assembled file -----------------------------------------------------


def test_declared_roots_are_name_minimal_and_qualified():
    body = ("theorem t.step : True := trivial\ntheorem t : True := t.step\n"
            "theorem t_other : True := trivial\n")
    assert declared_roots(body, "Ns.") == ("Ns.t", "Ns.t_other")


def test_assembly_goes_out_without_its_preamble_and_with_every_lemma(store):
    root = store.upsert_goal("id:root", "theorem root : True ∧ True")
    sketch = Sketch(parent_name="root", parent_signature="theorem root : True ∧ True",
                    parent_body="⟨l1, l2⟩",
                    subgoals=(SubgoalSpec("l1", "id:l1", "theorem l1 : True"),
                              SubgoalSpec("l2", "id:l2", "theorem l2 : True")),
                    preamble=PREAMBLE)
    decomposition = store.add_decomposition(
        root.id, [(s.identity, s.signature) for s in sketch.subgoals], sketch=sketch)
    from evoharness.proof.graph import DecompositionStatus

    store.set_decomposition_status(decomposition.id, DecompositionStatus.ACCEPTED)
    for sub in decomposition.subgoal_ids:
        store.record_attempt(sub, Outcome.PROVED, proof_text="trivial")
        store.propagate(sub, max_capability_attempts=3)
    envelope = assembly_envelope(store, root.id, preamble=PREAMBLE, name_prefix="Ns.")
    assert "import" not in envelope.body
    assert envelope.imports == ("import Mathlib",)
    assert envelope.ctx == {"raw": ["open Nat"]}
    assert envelope.primary_root == "Ns.root"
    assert set(envelope.expected_roots) == {"Ns.root", "Ns.l1", "Ns.l2"}

    verifier = FakeVerifier()
    result, cert = certify_with_verifier(store, root.id, client=verifier, base=BASE,
                                         preamble=PREAMBLE)
    assert result.ok and cert.external["certification_id"] == "cert-1"
    [request] = [r for r in verifier.requests if r.get("mode") == "assembly"]
    assert sorted(request["auxiliary_roots"]) == ["Ns.l1", "Ns.l2"]
    assert request["route_id"] == decomposition.id
    assert store.latest_certification(root.id).external["envelope_hash"] == envelope.envelope_hash


def test_a_sketch_with_a_foreign_preamble_is_refused(store):
    root = store.upsert_goal("id:root", "theorem root : True")
    sketch = Sketch(parent_name="root", parent_signature="theorem root : True",
                    parent_body="l1", subgoals=(SubgoalSpec("l1", "id:l1", "theorem l1 : True"),),
                    preamble="import Mathlib\nopen Real")
    decomposition = store.add_decomposition(root.id, [("id:l1", "theorem l1 : True")], sketch=sketch)
    from evoharness.proof.assembly import AssemblyError
    from evoharness.proof.graph import DecompositionStatus

    store.set_decomposition_status(decomposition.id, DecompositionStatus.ACCEPTED)
    store.record_attempt(decomposition.subgoal_ids[0], Outcome.PROVED, proof_text="trivial")
    store.propagate(decomposition.subgoal_ids[0], max_capability_attempts=3)
    with pytest.raises(AssemblyError, match="preamble"):
        assembly_envelope(store, root.id, preamble=PREAMBLE, name_prefix="")


def test_an_assembly_the_verifier_refuses_is_recorded_as_failed(store):
    goal = proved(store, "theorem t : True", "trivial")
    result, cert = certify_with_verifier(store, goal.id, client=FakeVerifier("task_failed",
                                                                              "elaboration_error"),
                                         base=BASE, preamble=PREAMBLE)
    assert not result.ok and "task_failed/elaboration_error" in result.reason
    assert cert.ok is False and cert.external is None


# -- 2. helpers ----------------------------------------------------------------


def test_helpers_are_split_off_stored_and_assembled_first(store):
    text = ("import Mathlib\n\n-- EDIT-REGION-BEGIN\n"
            "theorem t.step : True := trivial\n"
            "theorem t : True := t.step\n-- EDIT-REGION-END\n")
    body, helpers = proof_parts(text, "t")
    assert body.strip() == "t.step"
    assert helpers == ("theorem t.step : True := trivial",)
    goal = proved(store, "theorem t : True", body, helpers)
    assert store.attempts_of(goal.id)[0].auxiliary_declarations == helpers
    envelope = assembly_envelope(store, goal.id, preamble=PREAMBLE, name_prefix="")
    assert envelope.body.index("theorem t.step") < envelope.body.index("theorem t :")
    assert envelope.expected_roots == ("t",), "a helper named under the goal is not a root"


# -- 3. sketches ---------------------------------------------------------------


def contract_for(goal):
    return GoalContract(goal_key="k0", goal_key_obj={"key": "k0"}, base=BASE, name_prefix="",
                        proposition="True", context={"raw": ["open Nat"]})


SKETCH = Sketch(parent_name="root", parent_signature="theorem root : True", parent_body="l1",
                subgoals=(SubgoalSpec("l1", "id:l1", "theorem l1 : True"),), preamble=PREAMBLE)
GOAL = Goal(id="g", identity="id:root", statement="theorem root : True")


def test_a_sketch_is_checked_in_sketch_mode():
    verifier = FakeVerifier()
    check = VerifierSketchValidator(verifier, PREAMBLE, AxiomPolicy(), contract_for)
    assert check(GOAL, SKETCH).ok
    [request] = verifier.requests
    assert request["mode"] == "sketch" and request["auxiliary_roots"] == ["l1"]
    assert "sorry" in request["source"] and "import" not in request["source"]


def test_a_sketch_verdict_rejects_and_no_verdict_defers():
    reject = VerifierSketchValidator(FakeVerifier("policy_rejected", "parent_uses_sorry"),
                                     PREAMBLE, AxiomPolicy(), contract_for)
    verdict = reject(GOAL, SKETCH)
    assert isinstance(verdict, Validation) and not verdict.ok
    assert "parent_uses_sorry" in verdict.reason
    defer = VerifierSketchValidator(FakeVerifier("environment_mismatch", "context_error"),
                                    PREAMBLE, AxiomPolicy(), contract_for)
    with pytest.raises(SketchUnavailable):
        defer(GOAL, SKETCH)


# -- 4. publication ------------------------------------------------------------


def test_publication_sends_the_certified_envelope_back_under_a_stable_key(store):
    goal = proved(store, "theorem t : True", "trivial")
    verifier = FakeVerifier()
    _, cert = certify_with_verifier(store, goal.id, client=verifier, base=BASE, preamble=PREAMBLE)
    envelope = store.envelope(cert.external["envelope_hash"])
    publish(store, cert.id, client=verifier)
    publish(store, cert.id, client=verifier)
    first, second = verifier.published
    assert first == second, "a retry is the same request"
    assert first["idempotency_key"] == "publish:cert-1"
    assert first["source"] == envelope.body and first["context"] == dict(envelope.ctx)
    assert first["provenance"]["envelope_hash"] == envelope.envelope_hash
    assert store.publication_of(cert.id)["admitted_node_ids"] == [7]


def test_a_local_certification_is_not_publishable(store):
    from evoharness.proof.verified import PublicationError

    goal = proved(store, "theorem t : True", "trivial")
    cert = store.record_certification(goal.id, ok=True, axioms=["propext"], trust="trusted")
    with pytest.raises(PublicationError, match="not a verifier certification"):
        publish(store, cert.id, client=FakeVerifier())


# -- 5. retrieval --------------------------------------------------------------


def test_retrieval_asks_at_the_graphs_floor_and_reaches_the_prompt(store):
    verifier = FakeVerifier()
    provider = VerifierRetrievalProvider(verifier, minimum_trust="trusted")
    goal = store.upsert_goal("id:x", "theorem x (a b : Nat) (h : Nat.succ a ≤ b) : a < b")
    assert queries_for(goal.statement) == ("Nat.succ",)
    seen = []
    binding = VerifierBinding(client=verifier, preamble=PREAMBLE, policy=AxiomPolicy("trusted"),
                              contract_for=contract_for, retrieval=provider,
                              on_retrieval=lambda g, rid, c, hits: seen.append((g.id, rid, hits)))
    section = binding.retrieve(goal, contract_for(goal))
    assert "Nat.succ_le" in section
    assert verifier.requests[-1]["min_trust"] == "trusted"
    [(gid, rid, hits)] = seen
    store.record_retrieval(gid, rid, base=BASE, hits=hits)
    [(base, hit)] = store.retrieved()
    assert base == BASE and hit.name == "Nat.succ_le" and hit.trust == 3
    assert prompt_section([]) == ""


# -- 6. recovery ---------------------------------------------------------------


def test_a_dead_attempts_answers_are_kept_with_its_interruption(store, tmp_path):
    goal = store.upsert_goal("id:g", "theorem g : True")
    run_dir = tmp_path / "attempt_0000"
    (run_dir / "verifier").mkdir(parents=True)
    verdict = CandidateVerdict(candidate_id="c1", status="verified", certification_id="cert-9",
                               envelope_hash="env-9")
    (run_dir / "verifier" / "verdicts.jsonl").write_text(json.dumps(verdict.to_json()) + "\n")
    assert store.claim(goal.id, "dead-worker", ttl_s=-1)
    store.note_attempt_dir(goal.id, str(run_dir))
    controller = ProofController(store, StubSolver({}), decompositions=FixedDecompositions({}),
                                 validate_sketch=lambda g, s: Validation(ok=True))
    assert controller.recover() == [goal.id]
    [attempt] = store.attempts_of(goal.id)
    assert attempt.outcome is Outcome.INTERRUPTED, "a record, never a verdict"
    assert store.verifications_of(attempt.id) == [verdict]


def test_a_heartbeat_can_stop_the_run(store):
    goal = store.upsert_goal("id:g", "theorem g : True")
    controller = ProofController(store, StubSolver({}), decompositions=FixedDecompositions({}),
                                 validate_sketch=lambda g, s: Validation(ok=True),
                                 heartbeat=lambda: "retrieval_view_shrank")
    report = controller.solve(goal.id, budget=10)
    assert report.stopped_reason == "retrieval_view_shrank" and report.attempts == 0


