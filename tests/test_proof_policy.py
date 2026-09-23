"""The axiom policy: one classification, shared with the verifier, failing closed.

Three things are pinned here:

1. `AxiomPolicy` classifies every case in the shared fixture exactly as the
   verifier does. The fixture is a pinned copy of the verifier's trust
   classification cases; when `EVO_VERIFIER_CONTRACT` points at the verifier's
   contract file, the copy's data must still equal it.
2. A native_decide axiom is never believed on its name. Without a positive
   recheck result it is `claimed`: the name shape does not show that the
   asserted `Bool` evaluates to `true`, and this side has no Lean to recheck
   with.
3. The grader, the final assembly check and the certificate record all read the
   same policy, and the certificate keeps the trust level it was granted at.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from evoharness.proof.assembly import verify
from evoharness.proof.grade import make_grader
from evoharness.proof.graph import Outcome
from evoharness.proof.policy import AxiomPolicy, is_native_decide_axiom
from evoharness.proof.run_solver import SEED_MAIN
from evoharness.proof.store import ProofGraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "trust_classification.json"
TABLE = json.loads(FIXTURE.read_text())
NATIVE = "nd_probe._native.native_decide.ax_1"


def _holds(case):
    if "native_holds" not in case:
        return None
    return {a: case["native_holds"] for a in case["axioms"] if is_native_decide_axiom(a)}


# -- 1. the shared classification --------------------------------------------


@pytest.mark.parametrize("case", TABLE["cases"], ids=[c["id"] for c in TABLE["cases"]])
def test_policy_matches_the_shared_fixture(case):
    assert AxiomPolicy().classify(case["axioms"], _holds(case)) == case["trust"]


def test_policy_version_matches_the_fixture():
    from evoharness.proof.policy import LEVELS, POLICY_VERSION

    assert TABLE["policy_version"] == POLICY_VERSION
    assert TABLE["levels"] == dict(LEVELS)


def _upstream_contract() -> Path | None:
    """The verifier's contract file, when `EVO_VERIFIER_CONTRACT` points at one."""

    path = os.environ.get("EVO_VERIFIER_CONTRACT")
    return Path(path) if path and Path(path).is_file() else None


def test_pinned_fixture_still_equals_the_verifier():
    """The copy is only worth pinning if drift from the original is caught.

    Data fields only: the copy carries its own description.
    """

    path = _upstream_contract()
    if path is None:
        pytest.skip("set EVO_VERIFIER_CONTRACT to the verifier's contract file")
    upstream = json.loads(path.read_text())["x-trust-classification"]
    for key in ("policy_version", "levels", "cases"):
        assert TABLE[key] == upstream[key], key


# -- 2. thresholds and the native_decide rule --------------------------------


def test_thresholds():
    audited, trusted = AxiomPolicy("audited"), AxiomPolicy("trusted")
    assert audited.accepts("trusted") and audited.accepts("audited")
    assert not audited.accepts("claimed") and not audited.accepts("tainted")
    assert not trusted.accepts("audited")
    with pytest.raises(ValueError):
        AxiomPolicy("tainted")  # no policy may accept sorry


def test_a_native_decide_name_alone_never_reaches_audited():
    policy = AxiomPolicy()
    assert policy.classify([NATIVE]) == "claimed"
    assert policy.classify([NATIVE], {NATIVE: False}) == "claimed"
    assert policy.classify([NATIVE], {NATIVE: True}) == "audited"
    assert policy.unverified_native([NATIVE]) == {NATIVE}
    assert policy.unverified_native([NATIVE], {NATIVE: True}) == frozenset()


# -- 3. the grader -----------------------------------------------------------


class ReportingRunner:
    """Compiles nothing; answers with the axiom report a test asks for."""

    def __init__(self, axioms: list[str]):
        self.axioms = axioms

    def compile(self, text: str) -> tuple[int, str]:
        if not self.axioms:
            return 0, "'lemma1' does not depend on any axioms"
        return 0, f"'lemma1' depends on axioms: [{', '.join(self.axioms)}]"


def _grade(tmp_path, axioms, policy=None):
    (tmp_path / SEED_MAIN).write_text(
        "-- EDIT-REGION-BEGIN\ntheorem lemma1 : True := trivial\n-- EDIT-REGION-END\n"
    )
    return make_grader(ReportingRunner(axioms), policy)(tmp_path, None)


def test_grader_proves_on_standard_axioms(tmp_path):
    out = _grade(tmp_path, ["propext", "Classical.choice"])
    assert out["fitness"] == 1.0
    assert out["visible_metrics"]["trust"] == "trusted"


def test_grader_accepts_compiler_axioms_under_audited_only(tmp_path):
    axioms = ["Lean.ofReduceBool", "Lean.trustCompiler"]
    assert _grade(tmp_path, axioms)["fitness"] == 1.0
    strict = _grade(tmp_path, axioms, AxiomPolicy("trusted"))
    assert strict["fitness"] == 0.0 and strict["fault_kind"] == "task_failure"


def test_grader_never_proves_on_an_unrechecked_native_axiom(tmp_path):
    """Partial credit, a parent, not a fault -- and never 1.0 from here."""

    out = _grade(tmp_path, ["propext", NATIVE])
    assert out["fitness"] < 1.0
    assert out["passed"] is True
    assert "fault_kind" not in out
    assert "verifier" in out["notes"]


def test_grader_fails_a_user_declared_axiom(tmp_path):
    out = _grade(tmp_path, ["myAx"])
    assert out["fitness"] == 0.0 and out["fault_kind"] == "task_failure"


def test_grader_keeps_sorry_as_partial_credit(tmp_path):
    out = _grade(tmp_path, ["sorryAx", "propext"])
    assert out["fitness"] < 1.0 and out["passed"] is True


# -- 4. the final certificate --------------------------------------------------


def _proved_store(tmp_path) -> tuple[ProofGraphStore, str]:
    store = ProofGraphStore(tmp_path / "graph.db")
    goal = store.upsert_goal("text:lemma1", "theorem lemma1 : True")
    store.record_attempt(goal.id, Outcome.PROVED, proof_text="trivial")
    store.propagate(goal.id, max_capability_attempts=3)
    return store, goal.id


def test_assembly_certifies_with_its_trust_level(tmp_path):
    from evoharness.proof.assembly import certify

    store, goal_id = _proved_store(tmp_path)
    result, cert = certify(
        store, goal_id, runner=ReportingRunner(["Lean.ofReduceBool", "Lean.trustCompiler"])
    )
    assert result.ok and result.trust == "audited"
    assert cert.trust == "audited"
    assert store.latest_certification(goal_id).trust == "audited"


def test_assembly_refuses_an_unrechecked_native_axiom(tmp_path):
    store, goal_id = _proved_store(tmp_path)
    result = verify(store, goal_id, runner=ReportingRunner([NATIVE]))
    assert result.ok is False
    assert NATIVE in result.forbidden_axioms
    assert "verifier" in result.reason


def test_old_certification_rows_read_back_with_unknown_trust(tmp_path):
    """A row written before `trust` existed must read as None, not as ''."""

    store, goal_id = _proved_store(tmp_path)
    store.record_certification(goal_id, ok=True, axioms=["propext"], trust=None)
    store.close()
    with sqlite3.connect(tmp_path / "graph.db") as conn:
        assert conn.execute("SELECT trust FROM certifications").fetchone()[0] is None
    reopened = ProofGraphStore(tmp_path / "graph.db")
    assert reopened.latest_certification(goal_id).trust is None
