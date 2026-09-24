"""Splitting a candidate file into a SourceEnvelope: the rules, and the hash.

The hash matters most. EvoHarness reimplements the verifier's envelope hash
instead of depending on the verifier's package; one byte of difference in the
domain string, the key order or the separators, and every publication fails its
byte comparison. The verifier generates hash vectors from its own
implementation; `tests/fixtures/envelope_vectors.json` is a pinned copy,
checked against the original when `EVO_VERIFIER_CONTRACT` points at it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evoharness.proof.envelope import (
    EnvelopeError,
    SourceEnvelope,
    preamble_context,
    split_source,
)
from evoharness.proof.graph import Goal, GraphError
from evoharness.proof.run_solver import seed_text
from evoharness.proof.store import ProofGraphStore

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "envelope_vectors.json").read_text())
PREAMBLE = "import Mathlib\n\nopen Nat\nnamespace Foo"


def candidate(body="theorem target : 1 + 1 = 2 := by\n  rfl", preamble=PREAMBLE,
              tail="\n#print axioms target\n", between="\n\n"):
    head = f"{preamble}{between}" if preamble else ""
    return f"{head}-- EDIT-REGION-BEGIN\n{body}\n-- EDIT-REGION-END\n{tail}"


def split(text, **kw):
    kw.setdefault("preamble", PREAMBLE)
    kw.setdefault("root", "target")
    kw.setdefault("name_prefix", "Foo.")
    return split_source(text, **kw)


# -- the hash, against the verifier's own implementation ----------------------


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["envelope_hash"][:8])
def test_hash_matches_the_verifier_vectors(vector):
    fields = {k: v for k, v in vector.items() if k != "envelope_hash"}
    for key in ("imports", "expected_roots", "stripped_commands"):
        fields[key] = tuple(fields[key])
    assert SourceEnvelope(**fields).envelope_hash == vector["envelope_hash"]


def _upstream_contract() -> Path | None:
    """The verifier's contract file, when `EVO_VERIFIER_CONTRACT` points at one."""

    path = os.environ.get("EVO_VERIFIER_CONTRACT")
    return Path(path) if path and Path(path).is_file() else None


def test_pinned_vectors_still_equal_the_verifier():
    path = _upstream_contract()
    if path is None:
        pytest.skip("set EVO_VERIFIER_CONTRACT to the verifier's contract file")
    upstream = json.loads(path.read_text())["x-envelope-vectors"]
    assert VECTORS["vectors"] == upstream["vectors"]


# -- what a split produces -----------------------------------------------------


def test_split_of_a_seed_shaped_file():
    env = split(candidate())
    assert env.imports == ("import Mathlib",)
    assert env.ctx == {"raw": ["open Nat\nnamespace Foo"]}
    assert env.body == "theorem target : 1 + 1 = 2 := by\n  rfl"
    assert env.primary_root == "Foo.target" and env.expected_roots == ("Foo.target",)
    assert env.stripped_commands == ("#print axioms target",)


def test_the_real_seed_template_splits():
    """The file ApiRunSolver writes must pass its own split."""

    goal = Goal(id="g", identity="text:x", statement="theorem target : 1 + 1 = 2")
    env = split(seed_text(goal, PREAMBLE))
    assert env.body.startswith("theorem target : 1 + 1 = 2 := by")


def test_no_context_is_one_value_not_two():
    assert preamble_context("import Mathlib").ctx == {}
    assert preamble_context("").ctx == {}


def test_stripping_is_stable_and_recorded():
    """Same body, different trailing queries: the hash says so, the body does not."""

    a = split(candidate(tail="\n#print axioms target\n"))
    b = split(candidate(tail="\n#print axioms target\n#check target\n"))
    assert a.body == b.body
    assert b.stripped_commands == ("#print axioms target", "#check target")
    assert a.envelope_hash != b.envelope_hash


def test_key_order_does_not_move_the_hash():
    env = split(candidate())
    shuffled = json.loads(json.dumps(dict(reversed(list(env.to_dict().items())))))
    assert SourceEnvelope.from_dict(shuffled).envelope_hash == env.envelope_hash


# -- what a split refuses ----------------------------------------------------


def test_an_edited_preamble_is_refused():
    with pytest.raises(EnvelopeError, match="preamble"):
        split(candidate(preamble="import Mathlib\n\nopen Int\nnamespace Foo"))


def test_an_added_import_is_refused():
    """R1's stated cost: in this phase a candidate cannot add imports."""

    with pytest.raises(EnvelopeError, match="between the preamble"):
        split(candidate(between="\nimport Mathlib.Tactic\n"))


@pytest.mark.parametrize("command", ["#eval 1 + 1", "#exit", "theorem extra : True := trivial"])
def test_unlisted_trailing_commands_are_refused_not_stripped(command):
    with pytest.raises(EnvelopeError, match="only"):
        split(candidate(tail=f"\n{command}\n"))


def test_a_renamed_declaration_is_refused():
    with pytest.raises(EnvelopeError, match="own declaration"):
        split(candidate(body="theorem other : 1 + 1 = 2 := by\n  rfl"))


def test_missing_markers_are_refused():
    with pytest.raises(EnvelopeError, match="marker"):
        split(f"{PREAMBLE}\n\ntheorem target : True := trivial\n")


# -- persistence ---------------------------------------------------------------


def test_stored_envelope_reads_back_identical(tmp_path):
    store = ProofGraphStore(tmp_path / "graph.db")
    env = split(candidate())
    digest = store.record_envelope(env, candidate_path="runs/a/subgoal.lean")
    assert digest == env.envelope_hash
    assert store.record_envelope(env) == digest  # idempotent
    assert store.envelope(digest) == env


def test_a_tampered_stored_envelope_is_refused(tmp_path):
    store = ProofGraphStore(tmp_path / "graph.db")
    env = split(candidate())
    digest = store.record_envelope(env)
    store._conn.execute(
        "UPDATE envelopes SET envelope_json = replace(envelope_json, 'rfl', 'sorry')"
    )
    with pytest.raises((ValueError, GraphError)):
        store.envelope(digest)


def test_a_preamble_ending_in_a_dangling_in_is_refused():
    """A dangling `in` scopes nothing once split off: the context cannot be
    elaborated on its own, and every candidate would fail for it. Refused once,
    at the preamble, instead."""

    with pytest.raises(EnvelopeError, match="scopes only the next"):
        preamble_context("import Mathlib\nopen Nat in")
    # `in` closed by its own command, and words merely ending in "in", are fine.
    assert preamble_context("open Nat in\ntheorem h : True := trivial").ctx
    assert preamble_context("open Nat\n-- plugin").ctx == {"raw": ["open Nat\n-- plugin"]}
