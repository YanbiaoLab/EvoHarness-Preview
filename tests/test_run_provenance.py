"""Run provenance: frozen identity before work, outcome appended at exit,
and declared-versus-observed feedback capability."""

import json
import subprocess

from evoharness.evocore.population import Candidate, EvalReport
from evoharness.evoguard import (
    code_provenance,
    finalize_manifest,
    start_manifest,
)
from evoharness.evoplus import CapabilityLedger
from experiments.run_evolution import main as run_evolution


# -- code provenance ------------------------------------------------------


def test_provenance_in_a_git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "x"],
        cwd=tmp_path, check=True,
    )
    clean = code_provenance(tmp_path)
    assert clean["commit"] and clean["dirty"] is False

    (tmp_path / "uncommitted.txt").write_text("x")
    assert code_provenance(tmp_path)["dirty"] is True


def test_provenance_outside_git_says_so_instead_of_guessing(tmp_path):
    assert code_provenance(tmp_path) == {
        "commit": None, "branch": None, "dirty": None,
    }


# -- manifest lifecycle ---------------------------------------------------


def test_a_crash_leaves_the_frozen_identity_behind(tmp_path):
    """The entire point: the runs that die are the runs whose identity a
    post-mortem needs, and they are exactly the runs the old end-of-run
    write recorded nothing about."""
    path = tmp_path / "manifest.json"
    start_manifest(path, recipe="e0", task="demo", models=["m1"])
    # ... the process dies here; finalize never runs ...
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "running"
    assert manifest["recipe"] == "e0"
    assert "code" in manifest      # commit / dirty / branch


def test_finalize_appends_without_touching_the_frozen_part(tmp_path):
    path = tmp_path / "manifest.json"
    start_manifest(path, recipe="e0", search={"seed": 7})
    finalize_manifest(path, report={"best_fitness": 1.0})

    manifest = json.loads(path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["recipe"] == "e0"
    assert manifest["search"] == {"seed": 7}
    assert manifest["report"] == {"best_fitness": 1.0}
    assert manifest["finished_at"] >= manifest["created_at"]


def test_a_resume_is_recorded_not_rewritten(tmp_path):
    """The code may have changed between the crash and the restart; that
    difference is what a post-mortem will want, so each resume carries its
    own provenance while the original identity stands."""
    path = tmp_path / "manifest.json"
    first = start_manifest(path, recipe="e0", models=["m1"])
    second = start_manifest(path, recipe="SHOULD-NOT-REPLACE")

    assert second["recipe"] == "e0"
    assert second["created_at"] == first["created_at"]
    assert len(second["resumes"]) == 1
    assert "code" in second["resumes"][0]


# -- capability ledger ----------------------------------------------------


def _graded(fitness=1.0, **report_kwargs):
    cand = Candidate(
        id="c1", code="x", generation=1, parent_id=None,
        island_idx=0, operator="revise",
        report=EvalReport(fitness=fitness, passed=True, **report_kwargs),
    )
    return cand


def test_ledger_counts_what_actually_flowed():
    ledger = CapabilityLedger()
    ledger.on_candidate_graded(
        _graded(structured_feedback={"items": []}, sem=0.1), store=None
    )
    ledger.on_candidate_graded(_graded(), store=None)

    summary = ledger.summary(declared=["scalar", "structured_feedback",
                                      "reflection"])
    assert summary["graded"] == 2
    assert summary["observed"]["scalar"] == 2
    assert summary["observed"]["structured_feedback"] == 1
    assert summary["observed"]["uncertainty"] == 1
    # Declared but never carried anything: the explicit-downgrade record.
    assert summary["silent"] == ["reflection"]


def test_ledger_state_roundtrip():
    ledger = CapabilityLedger()
    ledger.on_candidate_graded(_graded(sem=0.2), store=None)
    restored = CapabilityLedger()
    restored.set_state(ledger.state())
    assert restored.graded == 1
    assert restored.channels == ledger.channels


# -- the driver end to end ------------------------------------------------


def test_driver_freezes_identity_and_reports_capability(tmp_path):
    run_dir = tmp_path / "run"
    assert run_evolution([
        "--recipe", "e1",
        "--task", "demo_counter",
        "--run-dir", str(run_dir),
        "--set",
        "search.num_generations=2",
        "population.num_islands=1",
        'search.operators=["rewrite"]',
        "search.operator_probs=[1.0]",
    ]) == 0

    manifest = json.loads((run_dir / "manifest.json").read_text())
    # Frozen identity.
    assert manifest["recipe"] == "e1"
    assert manifest["code"]["commit"] is not None
    assert manifest["assembly"]["contributors"]
    assert "structured_feedback" in manifest["capabilities_declared"]
    # Outcome.
    assert manifest["status"] == "completed"
    assert manifest["report"]["generations_completed"] == 2
    # Observed capability: demo_counter reports structured feedback, so the
    # declared channel actually carried data and nothing is silent about it.
    observed = manifest["capabilities_observed"]
    assert observed["observed"]["scalar"] >= 2
    assert observed["observed"]["structured_feedback"] >= 2
    assert "structured_feedback" not in observed["silent"]
