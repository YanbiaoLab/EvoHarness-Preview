"""A graph has one scope; opening it under another is refused.

The hazard: a goal node carries its proof status, and memoization hands the node
back with that status. Before scopes, one forgotten `--lean-identity` or one
`open --preamble` with a different preamble was enough to let a PROVED earned
under one set of premises stand under another. These tests drive the CLI the way
a session does, and never need Lean.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields
from pathlib import Path

import pytest

from evoharness.proof import cli
from evoharness.proof.identity import LeanExprHasher
from evoharness.proof.scope import GraphScope, ScopeMismatch, preamble_sha256
from evoharness.proof.store import ProofGraphStore


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cli.WORK_ENV, raising=False)
    monkeypatch.delenv(cli.PROJECT_ENV, raising=False)


def run(capsys, work, *argv) -> tuple[int, dict]:
    code = cli.main(["--work", str(work), *argv])
    return code, json.loads(capsys.readouterr().out)


def goals_in(path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return [row[0] for row in conn.execute("SELECT statement FROM goals")]


# -- a new graph records its scope ------------------------------------------


def test_first_open_records_the_scope(tmp_path, capsys):
    code, _ = run(capsys, tmp_path, "open", "--statement", "theorem a : True")
    assert code == 0
    scope = ProofGraphStore.read_scope(tmp_path / "graph.db")
    assert scope.environment == "local:bare"
    assert scope.identity_hasher == "exact-text"
    assert scope.minimum_trust == "audited"
    assert scope.preamble_sha256 == preamble_sha256("")


# -- stated settings must match; unstated ones are inherited ------------------


def test_a_different_hasher_is_refused(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True")
    code, out = run(capsys, tmp_path, "--lean-identity", "status")
    assert code == 1
    assert "identity_hasher" in out["error"]


def test_an_unstated_hasher_is_inherited_not_defaulted(tmp_path):
    """Hashing with the default on a lean-expr graph is the mixing itself."""

    env = "local:bare"
    ProofGraphStore(
        tmp_path / "graph.db", GraphScope(env, identity_hasher="lean-expr")
    ).close()
    args = cli.build_parser().parse_args(["--work", str(tmp_path), "status"])
    cli._store(args).close()
    assert isinstance(cli._hasher(args), LeanExprHasher)


def test_trust_floor_is_inherited_and_reaches_the_grader(tmp_path, capsys):
    run(capsys, tmp_path, "--minimum-trust", "trusted",
        "open", "--statement", "theorem a : True")
    args = cli.build_parser().parse_args(["--work", str(tmp_path), "status"])
    cli._store(args).close()
    assert cli._policy(args).minimum_trust == "trusted"
    code, out = run(capsys, tmp_path, "--minimum-trust", "audited", "status")
    assert code == 1 and "minimum_trust" in out["error"]


def test_a_second_preamble_is_refused_and_the_first_is_kept(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True",
        "--preamble", "open Nat")
    code, out = run(capsys, tmp_path, "open", "--statement", "theorem b : True",
                    "--preamble", "open Int")
    assert code == 1 and "preamble_sha256" in out["error"]
    assert (tmp_path / "preamble.lean").read_text().strip() == "open Nat"
    assert goals_in(tmp_path / "graph.db") == ["theorem a : True"]


def test_a_hand_edited_preamble_is_refused(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True",
        "--preamble", "open Nat")
    (tmp_path / "preamble.lean").write_text("open Int\n")
    code, out = run(capsys, tmp_path, "status")
    assert code == 1 and "preamble_sha256" in out["error"]


def test_a_different_lean_is_refused(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True")
    project = tmp_path / "project"
    project.mkdir()
    (project / "lake-manifest.json").write_text("{}")
    code, out = run(capsys, tmp_path, "--lean-project", str(project), "status")
    assert code == 1 and "environment" in out["error"]


def test_moving_from_local_to_the_verifier_is_a_scope_change(tmp_path):
    """A PROVED judged by a local compile is not a verifier certification."""

    path = tmp_path / "graph.db"
    ProofGraphStore(path, GraphScope("local:bare")).close()
    with pytest.raises(ScopeMismatch) as caught:
        ProofGraphStore(path, GraphScope("verifier:0123abcd"))
    assert [d[0] for d in caught.value.diffs] == ["environment"]


def test_a_policy_upgrade_refuses_graphs_built_under_the_old_one(tmp_path):
    path = tmp_path / "graph.db"
    ProofGraphStore(path, GraphScope("local:bare", axiom_policy_version=0)).close()
    with pytest.raises(ScopeMismatch) as caught:
        ProofGraphStore(path, GraphScope("local:bare"))
    assert [d[0] for d in caught.value.diffs] == ["axiom_policy_version"]


def test_the_verifier_runtime_is_not_part_of_the_scope():
    """It changes on every worker rebuild; it is bound per certification."""

    assert "verifier_runtime_fingerprint" not in {f.name for f in fields(GraphScope)}


# -- graphs built before scopes ---------------------------------------------


def test_a_graph_without_scope_is_refused_until_adopted(tmp_path, capsys):
    legacy = ProofGraphStore(tmp_path / "graph.db")  # library mode: no scope
    legacy.upsert_goal("text:a", "theorem a : True")
    legacy.close()

    code, out = run(capsys, tmp_path, "status")
    assert code == 1 and "ScopeMissing" in out["error"]

    code, _ = run(capsys, tmp_path, "--adopt-scope", "status")
    assert code == 0
    assert ProofGraphStore.read_scope(tmp_path / "graph.db") is not None
    code, _ = run(capsys, tmp_path, "status")
    assert code == 0


def test_adopting_is_only_for_graphs_without_a_scope(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True")
    code, out = run(capsys, tmp_path, "--adopt-scope", "status")
    assert code == 1 and "adopt-scope" in out["error"]


# -- starting over keeps the old graph ----------------------------------------


def test_force_new_graph_moves_the_old_one_aside(tmp_path, capsys):
    run(capsys, tmp_path, "open", "--statement", "theorem a : True",
        "--preamble", "open Nat")
    code, out = run(capsys, tmp_path, "--force-new-graph",
                    "open", "--statement", "theorem b : True")
    assert code == 0

    retired = out["retired_graph"]
    old_db = next(p for p in retired if "/graph.db." in p and p.endswith(".bak"))
    old_preamble = next(p for p in retired if "preamble.lean." in p)
    # The old graph survives whole -- including whatever sat in the WAL.
    assert goals_in(old_db) == ["theorem a : True"]
    assert Path(old_preamble).read_text().strip() == "open Nat"
    assert goals_in(tmp_path / "graph.db") == ["theorem b : True"]
    assert ProofGraphStore.read_scope(tmp_path / "graph.db").preamble_sha256 == (
        preamble_sha256("")
    )


def test_retiring_twice_in_one_second_overwrites_nothing(tmp_path, capsys):
    for statement in ("theorem a : True", "theorem b : True", "theorem c : True"):
        run(capsys, tmp_path, "--force-new-graph", "open", "--statement", statement)
    backups = sorted(p.name for p in tmp_path.glob("graph.db.*.bak"))
    assert len(backups) == 2  # the first open had nothing to retire
    assert len(set(backups)) == 2


def test_a_preamble_the_verifier_cannot_use_never_becomes_a_graph(tmp_path, capsys):
    code, out = run(capsys, tmp_path, "open", "--statement", "theorem a : True",
                    "--preamble", "open Nat in")
    assert code == 1 and "scopes only the next" in out["error"]
    assert ProofGraphStore.read_scope(tmp_path / "graph.db") is None
