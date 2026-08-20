"""Read-only views over a run directory."""

import json

import pytest

from evoharness.readout import (
    STALL_AFTER_S,
    ReadoutError,
    candidate_detail,
    list_runs,
    population,
    run_detail,
    run_status,
    trajectory,
)
from evoharness.core import Candidate, EvalReport, PopulationConfig, PopulationStore
from evoharness.core.workspace import FileWorkspace


def make_run(root, *, checkpoint=None, manifest=None, candidates=()):
    root.mkdir(parents=True, exist_ok=True)
    store = PopulationStore(PopulationConfig(), root / "run.db")
    for candidate in candidates:
        store.insert(candidate)
    store.close()
    if checkpoint is not None:
        (root / "checkpoint.json").write_text(
            json.dumps(checkpoint), encoding="utf-8"
        )
    if manifest is not None:
        (root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return root


def candidate(cid, generation, fitness, *, operator="revise", parent=None):
    cand = Candidate(
        id=cid,
        code=f"# {cid}\n",
        generation=generation,
        parent_id=parent,
        island_idx=0,
        operator=operator,
    )
    cand.workspace = FileWorkspace(f"# {cid}\n")
    cand.report = EvalReport(fitness=fitness, passed=True)
    return cand


def test_a_mid_flight_run_is_not_reported_as_finished(tmp_path):
    run = make_run(
        tmp_path / "r",
        # What SearchLoop actually writes while it is still going.
        checkpoint={"generation": 3, "run_report": {"stopped_reason": "running"}},
        manifest={"search": {"num_generations": 10}},
    )

    status = run_status(run)

    # "running" reads like a stop reason and is the opposite of one. Taking it
    # at face value makes a caller polling for completion stop at the first
    # checkpoint.
    assert status.state == "running"
    assert status.stopped_reason is None
    assert status.finished is False
    assert status.generation == 3


def test_the_finalized_report_wins_over_the_checkpoint(tmp_path):
    run = make_run(
        tmp_path / "r",
        # The checkpoint is rewritten every generation and still holds the
        # in-flight sentinel after the run ends.
        checkpoint={"generation": 2, "run_report": {"stopped_reason": "running"}},
        manifest={"report": {"stopped_reason": "completed", "evaluations": 4}},
    )

    status = run_status(run)

    assert status.finished is True
    assert status.state == "completed"
    assert status.evaluations == 4


def test_a_run_whose_checkpoint_stopped_moving_is_stalled(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {"stopped_reason": "running"}},
    )
    later = lambda: run.joinpath("checkpoint.json").stat().st_mtime + STALL_AFTER_S + 1

    status = run_status(run, now=later)

    # The loop cannot write "my process was killed", so nothing on disk
    # distinguishes a dead run from a slow one. Saying so is the whole point.
    assert status.state == "stalled"
    assert status.finished is False


def test_a_slow_run_is_not_called_stalled(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {"stopped_reason": "running"}},
    )
    soon = lambda: run.joinpath("checkpoint.json").stat().st_mtime + STALL_AFTER_S - 1

    assert run_status(run, now=soon).state == "running"


def test_best_fitness_falls_back_to_the_population(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {}},
        candidates=[candidate("a", 1, 0.25), candidate("b", 1, 0.75)],
    )

    # Older checkpoints carry no best_fitness. Reporting None when the answer
    # is sitting in run.db would look like a run that produced nothing.
    assert run_status(run).best_fitness == 0.75


def test_trajectory_groups_by_generation_and_carries_the_running_best(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 3, "run_report": {}},
        candidates=[
            candidate("s", 0, 0.1, operator="seed"),
            candidate("a", 1, 0.6, parent="s"),
            candidate("b", 2, 0.3, parent="a"),
            candidate("c", 2, 0.4, parent="a"),
        ],
    )

    generations = trajectory(run)

    assert [g.generation for g in generations] == [0, 1, 2]
    assert [g.best_fitness for g in generations] == [0.1, 0.6, 0.4]
    # A generation that tried two things and improved on neither is exactly
    # what an author revising a task needs to see; a flat candidate list shows
    # the winner and hides the plateau.
    assert [g.best_so_far for g in generations] == [0.1, 0.6, 0.6]
    assert len(generations[2].candidates) == 2


def test_candidate_detail_returns_program_text_not_the_genome(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {}},
        candidates=[candidate("a", 1, 0.5)],
    )

    detail = candidate_detail(run, "a")

    assert detail["code"] == "# a\n"
    assert detail["report"]["fitness"] == 0.5
    assert candidate_detail(run, "missing") is None


def test_population_exposes_the_session_reference(tmp_path):
    cand = candidate("a", 1, 0.5)
    cand.metadata = {"session_id": "proposal-123"}
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {}},
        candidates=[cand],
    )

    # An agentic candidate with no session reference cannot be traced back to
    # what the agent did; the audit checks for exactly this field.
    assert population(run)[0].session_id == "proposal-123"


def test_run_detail_carries_the_frozen_identity(tmp_path):
    run = make_run(
        tmp_path / "r",
        checkpoint={"generation": 1, "run_report": {}},
        manifest={
            "task": "dir:tasks/authored/x",
            "recipe": "e0",
            "spec_hashes": {"task": "abc"},
            "proposal": {"preflight_validators": ["imports"]},
        },
        candidates=[candidate("a", 1, 0.5)],
    )

    detail = run_detail(run)

    # A fitness read without the identity is a number with no experiment
    # attached to it.
    assert detail["identity"]["spec_hashes"] == {"task": "abc"}
    assert detail["identity"]["proposal"]["preflight_validators"] == ["imports"]
    assert len(detail["candidates"]) == 1


def test_reading_a_missing_directory_is_an_error_not_an_empty_run(tmp_path):
    with pytest.raises(ReadoutError):
        run_status(tmp_path / "nope")


def test_reading_never_creates_a_database(tmp_path):
    run = (tmp_path / "r")
    run.mkdir()
    (run / "checkpoint.json").write_text('{"generation": 0}', encoding="utf-8")

    run_status(run)

    # The writable store constructor runs schema DDL, so merely looking at a
    # run could create the file a typo'd path pointed at and then report it as
    # an empty run.
    assert not (run / "run.db").exists()


def test_a_half_written_checkpoint_reads_as_absent_not_as_a_crash(tmp_path):
    run = (tmp_path / "r")
    run.mkdir()
    (run / "checkpoint.json").write_text('{"generation": 3, "run_rep', encoding="utf-8")

    # Polling a live run will sometimes catch a checkpoint mid-write. Raising
    # would make every status poll a coin flip.
    assert run_status(run).generation == 0


def test_list_runs_finds_every_run_under_a_root(tmp_path):
    make_run(tmp_path / "b", checkpoint={"generation": 1, "run_report": {}})
    make_run(tmp_path / "a", checkpoint={"generation": 2, "run_report": {}})

    assert [status.name for status in list_runs(tmp_path)] == ["a", "b"]
