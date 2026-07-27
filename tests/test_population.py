import sqlite3

import numpy as np
import pytest

from conftest import make_candidate
from evoharness.evocore import Candidate, EvalReport, PopulationConfig, PopulationStore
from evoharness.evocore.workspace import GitWorkspace


def test_insert_get_roundtrip_and_lineage():
    store = PopulationStore(PopulationConfig())
    parent = make_candidate("p", 1.0)
    store.insert(parent)
    child = make_candidate("c", 2.0, parent_id="p", generation=2)
    store.insert(child)
    got = store.get("c")
    assert got.parent_id == "p"
    assert got.report.fitness == 2.0
    # children_count is charged at PLAN time by note_attempt, not on insert:
    # a whole batch is planned before any of it is graded, so counting on
    # insert made the selector's 1/(1+children_count) penalty close only
    # once per generation. Run modmul_r7 drew all fifteen offspring from one
    # parent, which finished with children_count=15 that had never influenced
    # a single one of those choices.
    assert store.get("p").children_count == 0
    store.note_attempt("p")
    assert store.get("p").children_count == 1
    assert store.count() == 2


def test_island_assignment_inherits_parent():
    store = PopulationStore(PopulationConfig(num_islands=3))
    parent = make_candidate("p", 1.0, island=2)
    store.insert(parent)
    child = make_candidate("c", 1.5, parent_id="p")
    child.island_idx = -1  # request assignment
    store.insert(child)
    assert store.get("c").island_idx == 2


def test_seed_all_islands_copies():
    store = PopulationStore(PopulationConfig(num_islands=3))
    seed = make_candidate("s", 1.0, generation=0, island=0)
    copies = store.seed_all_islands(seed)
    assert len(copies) == 3
    assert store.count() == 3
    islands = {store.get(c.id).island_idx for c in copies}
    assert islands == {0, 1, 2}
    assert store.get(copies[1].id).metadata["seed_copy_of"] == "s"


def test_refresh_archive_keeps_top_n_passed():
    store = PopulationStore(PopulationConfig(archive_size=2))
    for i in range(4):
        store.insert(make_candidate(f"c{i}", float(i), in_archive=False))
    store.insert(make_candidate("bad", 99.0, passed=False, in_archive=False))
    store.refresh_archive()
    archived = {c.id for c in store.all_candidates() if c.in_archive}
    assert archived == {"c2", "c3"}  # top-2 by fitness, failed excluded


def test_best_ignores_failed():
    store = PopulationStore(PopulationConfig())
    store.insert(make_candidate("ok", 1.0))
    store.insert(make_candidate("bad", 9.0, passed=False))
    assert store.best().id == "ok"


def test_migration_moves_and_records_history():
    cfg = PopulationConfig(
        num_islands=2, migration_interval=5, migration_rate=0.5, island_elitism=True
    )
    store = PopulationStore(cfg)
    # island 0: seed (gen0, protected), best (protected by elitism), mover
    store.insert(make_candidate("seed", 0.5, generation=0, island=0))
    store.insert(make_candidate("best", 9.0, generation=1, island=0))
    store.insert(make_candidate("mover", 1.0, generation=2, island=0))
    store.insert(make_candidate("other", 1.0, generation=1, island=1))

    assert store.maybe_migrate(4, np.random.default_rng(0)) == 0  # off-interval
    moved = store.maybe_migrate(5, np.random.default_rng(0))
    assert moved >= 1
    mover = store.get("mover")
    assert mover.island_idx == 1  # only possible target
    assert mover.metadata["migration_history"][0] == {
        "generation": 5, "from": 0, "to": 1,
    }
    assert store.get("best").island_idx == 0  # elitism protected
    assert store.get("seed").island_idx == 0  # gen-0 protected


def test_migration_disabled_by_default():
    store = PopulationStore(PopulationConfig(num_islands=2))
    store.insert(make_candidate("a", 1.0, generation=1, island=0))
    store.insert(make_candidate("b", 1.0, generation=1, island=0))
    assert store.maybe_migrate(10) == 0  # migration_rate=0.0 upstream default


def test_eval_report_contract():
    with pytest.raises(ValueError, match="missing required"):
        EvalReport.from_json({"fitness": 1.0})
    nan = EvalReport.from_json({"fitness": float("nan"), "passed": True})
    assert not nan.passed and "non-finite" in nan.fault
    rt = EvalReport.from_json(
        EvalReport(fitness=1.5, passed=True, notes="n").to_json()
    )
    assert rt.fitness == 1.5 and rt.notes == "n"


def test_latest_failed_and_repair_mark():
    store = PopulationStore(PopulationConfig())
    store.insert(make_candidate("ok", 1.0))
    bad = make_candidate("bad", 0.0, passed=False, generation=3)
    store.insert(bad)
    assert store.latest_failed().id == "bad"
    store.mark_repair_attempted("bad")
    assert store.latest_failed() is None


def test_signatures_query():
    store = PopulationStore(PopulationConfig())
    a = make_candidate("a", 1.0, island=0)
    a.behavior_signature = "sig-a"
    b = make_candidate("b", 1.0, island=1)
    b.behavior_signature = "sig-b"
    store.insert(a)
    store.insert(b)
    assert set(store.all_signatures()) == {"sig-a", "sig-b"}
    assert store.all_signatures(island_idx=0) == ["sig-a"]


# -- WS-3 M1-2: workspace_kind column ------------------------------------------

def test_old_db_gains_workspace_kind(tmp_path):
    # Simulate a pre-M1 run.db: candidates table WITHOUT the new column.
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE candidates (
            id TEXT PRIMARY KEY, code TEXT NOT NULL, generation INTEGER NOT NULL,
            parent_id TEXT, island_idx INTEGER NOT NULL, operator TEXT NOT NULL,
            change_title TEXT, change_summary TEXT, model_name TEXT,
            inspiration_ids TEXT, report TEXT, embedding TEXT,
            behavior_signature TEXT, behavior_duplicate INTEGER DEFAULT 0,
            children_count INTEGER DEFAULT 0, in_archive INTEGER DEFAULT 0,
            metadata TEXT, timestamp REAL
        );
        CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO candidates (id, code, generation, island_idx, operator, timestamp)
        VALUES ('legacy1', 'x = 1', 0, 0, 'seed', 1.0);
        """
    )
    conn.commit()
    conn.close()

    store = PopulationStore(PopulationConfig(), db)  # migration happens here
    got = store.get("legacy1")
    assert got.workspace_kind == "file"  # historical rows ARE single-file genomes
    assert got.workspace.main_text() == "x = 1"
    store.insert(make_candidate("new1", 1.0))  # 19-column insert works post-ALTER
    assert store.get("new1").workspace_kind == "file"


def test_git_workspace_kind_roundtrip(tmp_path):
    ws = GitWorkspace(base_files={"main.py": "x = 1\n", "util.py": "y = 2\n"})
    cand = Candidate(
        id="g1",
        code=ws.serialize(),
        generation=1,
        parent_id=None,
        island_idx=0,
        operator="seed",
        workspace_kind="git",
        report=EvalReport(fitness=1.0, passed=True),
    )
    store = PopulationStore(PopulationConfig(), tmp_path / "run.db")
    store.insert(cand)
    got = store.get("g1")
    assert got.workspace_kind == "git"
    assert isinstance(got.workspace, GitWorkspace)
    assert got.workspace.main_text() == "x = 1\n"


def test_eval_report_artifacts_ref_roundtrip_and_backcompat():
    # New field survives a to_json/from_json round trip.
    report = EvalReport(fitness=0.5, passed=True, artifacts_ref="cand-123")
    assert EvalReport.from_json(report.to_json()).artifacts_ref == "cand-123"

    # Defaults to None when absent.
    assert EvalReport(fitness=0.5, passed=True).artifacts_ref is None

    # Legacy rows written before the field existed still load (defaults None).
    legacy = {"fitness": 1.0, "passed": True}
    assert EvalReport.from_json(legacy).artifacts_ref is None


def test_store_reads_survive_a_worker_thread():
    """Agent tools read candidates from the runtime's worker threads, and
    sqlite refuses a connection outside its creating thread — the first
    live session's inspect_candidate call died on exactly that."""
    import threading

    store = PopulationStore(PopulationConfig())
    store.insert(make_candidate("c1", 0.5))

    results, errors = [], []

    def read():
        try:
            results.append(store.get("c1"))
            results.append(len(store.all_candidates()))
        except Exception as exc:  # noqa: BLE001 — the point of the test
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r is not None for r in results)


def test_lcb_is_the_point_estimate_when_the_domain_reports_no_precision():
    """A domain that stays silent must keep today's behaviour exactly —
    otherwise adding the field silently reranks every existing task."""
    cand = make_candidate("c1", 0.5)
    assert cand.report.sem == 0.0 and cand.report.n_units == 0
    assert cand.fitness_lcb() == cand.fitness


def test_lcb_discounts_the_less_precise_candidate():
    loose = make_candidate("loose", 0.60)
    loose.report.sem, loose.report.n_units = 0.15, 12
    tight = make_candidate("tight", 0.55)
    tight.report.sem, tight.report.n_units = 0.02, 400

    # Point estimate prefers `loose`; commitment should prefer `tight`.
    assert max((loose, tight), key=lambda c: c.fitness) is loose
    assert max((loose, tight), key=lambda c: c.fitness_lcb()) is tight


def test_precision_fields_survive_a_json_round_trip():
    report = EvalReport(fitness=0.5, passed=True, n_units=12, sem=0.14)
    back = EvalReport.from_json(report.to_json())
    assert (back.n_units, back.sem) == (12, 0.14)
    # Reports written before the fields existed must still load.
    assert EvalReport.from_json({"fitness": 1.0, "passed": True}).sem == 0.0
