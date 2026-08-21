"""The narrow view a candidate gets of another candidate.

What is under test is mostly an absence. `hidden_metrics` is what a task
author withholds from the optimizer on purpose, and the raw grader logs are
the likeliest place for a holdout set to leak; neither may reach a candidate
through this path. Asserting they are gone is worth little on its own — a
filter passes that assertion until the day someone adds a field — so the
tests also pin the positive shape, which is what makes the view a fixed set
of keys rather than a dump with things taken out.
"""

import pytest

from evoharness.core.config import PopulationConfig
from evoharness.core.population import Candidate, EvalReport, PopulationStore
from evoharness.core.workspace import GitWorkspace
from evoharness.readout import ReadoutError, UnknownCandidate, peer_view

SECRET = "the-holdout-answer"


@pytest.fixture
def run_dir(tmp_path):
    store = PopulationStore(PopulationConfig(num_islands=1), tmp_path / "run.db")
    store.insert(
        Candidate(
            id="c1",
            code=GitWorkspace(
                base_files={"main.py": "x = 1\n", "helper.py": "y = 2\n"},
                main_file="main.py",
            ).serialize(),
            generation=1,
            parent_id=None,
            island_idx=0,
            operator="revise",
            workspace_kind="git",
            change_title="a focused change",
            report=EvalReport(
                fitness=0.75,
                passed=True,
                visible_metrics={"solved": 3},
                hidden_metrics={"holdout": SECRET},
                stdout_log=f"grader said {SECRET}",
                stderr_log=f"trace of {SECRET}",
                notes=f"note mentioning {SECRET}",
            ),
        )
    )
    store.close()
    return tmp_path


def test_the_inventory_is_exactly_these_keys(run_dir):
    view = peer_view(run_dir, "c1")
    assert set(view) == {
        "ok",
        "candidate_id",
        "fitness",
        "change_title",
        "files",
    }
    assert view["fitness"] == 0.75
    assert view["change_title"] == "a focused change"
    assert view["files"] == {"helper.py": 1, "main.py": 1}


def test_a_file_is_exactly_these_keys(run_dir):
    view = peer_view(run_dir, "c1", "helper.py")
    assert set(view) == {"ok", "candidate_id", "path", "content", "truncated"}
    assert view["content"] == "y = 2\n"
    assert view["truncated"] is False


def test_nothing_withheld_from_the_optimizer_survives_either_call(run_dir):
    """The whole reason this module exists rather than reusing candidate_detail."""

    for view in (peer_view(run_dir, "c1"), peer_view(run_dir, "c1", "main.py")):
        blob = repr(view)
        assert SECRET not in blob
        assert "hidden_metrics" not in blob
        assert "stdout_log" not in blob
        assert "stderr_log" not in blob


def test_the_full_view_does_carry_it(run_dir):
    """The control. Without this the test above passes on an empty report and
    proves nothing about the narrowing."""

    from evoharness.readout import candidate_detail

    assert SECRET in repr(candidate_detail(run_dir, "c1"))


def test_a_long_file_is_truncated_and_says_so(run_dir):
    view = peer_view(run_dir, "c1", "main.py", max_chars=3)
    assert view["content"] == "x ="
    assert view["truncated"] is True


def test_an_unknown_id_is_refused_with_a_usable_message(run_dir):
    with pytest.raises(UnknownCandidate) as caught:
        peer_view(run_dir, "list")
    # The first live in-process session called the equivalent tool with
    # candidate_id="list", reading the description as a command word.
    assert "opaque" in str(caught.value)


def test_an_unknown_path_names_what_is_available(run_dir):
    with pytest.raises(UnknownCandidate) as caught:
        peer_view(run_dir, "c1", "nope.py")
    assert "helper.py" in str(caught.value)
    assert "main.py" in str(caught.value)


def test_an_empty_id_is_refused_before_the_store_is_opened(tmp_path):
    with pytest.raises(UnknownCandidate):
        peer_view(tmp_path, "  ")


def test_a_directory_without_a_run_is_an_error_not_an_empty_answer(tmp_path):
    with pytest.raises(ReadoutError):
        peer_view(tmp_path, "c1")


def test_reading_a_peer_leaves_the_database_untouched(run_dir):
    before = (run_dir / "run.db").read_bytes()
    peer_view(run_dir, "c1")
    peer_view(run_dir, "c1", "main.py")
    assert (run_dir / "run.db").read_bytes() == before
