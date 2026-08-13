"""Viewer surfaces must be unable to write the population they display.

Every display surface used to instantiate the writable PopulationStore,
whose constructor runs schema DDL and creates the file when missing: merely
looking at a run could write to it, and a typo'd path manufactured an empty
database where the viewer then reported an empty run instead of an error.
"""

import pytest

from evoharness.core.config import PopulationConfig
from evoharness.core.population import Candidate, EvalReport, PopulationStore


def _populate(path):
    store = PopulationStore(PopulationConfig(num_islands=1), path)
    store.insert(
        Candidate(
            id="c1", code="x = 1", generation=1, parent_id=None,
            island_idx=0, operator="seed",
            report=EvalReport(fitness=0.5, passed=True),
        )
    )
    store.close()


def test_readonly_store_reads_what_the_writer_wrote(tmp_path):
    db = tmp_path / "run.db"
    _populate(db)
    viewer = PopulationStore.open_readonly(db)
    assert [c.id for c in viewer.all_candidates()] == ["c1"]
    assert viewer.best().fitness == 0.5
    viewer.close()


def test_readonly_store_refuses_every_mutation_fast(tmp_path):
    """PermissionError immediately -- NOT the transient-fault retry path.

    'attempt to write a readonly database' is also the signature of a
    storage blip, which the connection survives by backing off 10.5 seconds
    and reconnecting. On a read-only handle that signature is the contract
    working, and the reconnect would silently reopen WRITABLE. Both wrongs
    are prevented by refusing at the store boundary.
    """
    db = tmp_path / "run.db"
    _populate(db)
    viewer = PopulationStore.open_readonly(db)
    cand = Candidate(
        id="c2", code="y", generation=2, parent_id=None,
        island_idx=0, operator="revise",
    )
    for attempt in (
        lambda: viewer.insert(cand),
        lambda: viewer.note_attempt("c1"),
        lambda: viewer.mark_repair_attempted("c1"),
        lambda: viewer.update_candidate_flags(cand),
        lambda: viewer.refresh_archive(),
        lambda: viewer.maybe_migrate(10),
    ):
        with pytest.raises(PermissionError):
            attempt()
    viewer.close()


def test_readonly_open_of_a_missing_db_is_an_error_not_an_empty_run(tmp_path):
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        PopulationStore.open_readonly(tmp_path / "typo" / "run.db")
    assert not (tmp_path / "typo" / "run.db").exists()


def test_viewer_surfaces_leave_no_write_behind(tmp_path):
    """End to end through the actual display code paths."""
    from evoharness.evoweb.data import candidate_detail, run_detail
    from evoharness.evoviz.report import load_run

    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _populate(run_dir / "run.db")
    before = (run_dir / "run.db").read_bytes()

    detail = run_detail(run_dir)
    cand = candidate_detail(run_dir, "c1")
    loaded = load_run(run_dir)

    assert cand is not None
    assert detail["candidates"] or loaded["candidates"]
    assert (run_dir / "run.db").read_bytes() == before, (
        "a viewer mutated the database it was displaying"
    )
