"""Where one goal's attempts are kept, and why they cannot share a directory.

Two attempts landing in one directory is not merely untidy. `api.run` refuses
a directory holding another run's checkpoint, so the second attempt fails as
infrastructure rather than saying anything about the goal; and the first
attempt's artifacts are overwritten while the store still points at them, so
an audit reads one goal's evidence under another goal's name.
"""

from __future__ import annotations

from pathlib import Path

from evoharness.proof.run_solver import claim_attempt_dir


def test_a_goals_attempts_are_numbered_within_its_own_directory(tmp_path: Path):
    first = claim_attempt_dir(tmp_path, "goal_aaaa")
    second = claim_attempt_dir(tmp_path, "goal_aaaa")

    assert first.parent == tmp_path / "goal_aaaa"
    assert [first.name, second.name] == ["attempt_0000", "attempt_0001"]


def test_two_goals_never_share_a_directory(tmp_path: Path):
    """The case a per-solver counter gets wrong.

    An attack is one process, so the counter restarts for the next goal and
    both take `attempt_0000`.
    """
    first = claim_attempt_dir(tmp_path, "goal_aaaa")
    second = claim_attempt_dir(tmp_path, "goal_bbbb")

    assert first != second
    assert first.name == second.name == "attempt_0000"


def test_numbering_resumes_from_what_is_already_on_disk(tmp_path: Path):
    """A fresh process must not reuse a directory an earlier one left."""
    claim_attempt_dir(tmp_path, "goal_aaaa")
    claim_attempt_dir(tmp_path, "goal_aaaa")

    # Nothing carried over in memory; only the filesystem says what exists.
    assert claim_attempt_dir(tmp_path, "goal_aaaa").name == "attempt_0002"


def test_an_unrelated_neighbour_does_not_shift_the_numbering(tmp_path: Path):
    (tmp_path / "goal_aaaa").mkdir(parents=True)
    (tmp_path / "goal_aaaa" / "notes").mkdir()
    (tmp_path / "goal_aaaa" / "attempt_x").mkdir()

    assert claim_attempt_dir(tmp_path, "goal_aaaa").name == "attempt_0000"


def test_the_directory_is_claimed_by_creating_it(tmp_path: Path):
    """Returned empty and already created, so a caller cannot be handed a name
    a concurrent attack is about to take."""
    claimed = claim_attempt_dir(tmp_path, "goal_aaaa")

    assert claimed.is_dir()
    assert list(claimed.iterdir()) == []


def test_an_index_taken_since_the_scan_is_skipped(tmp_path: Path):
    """The gap a scan alone leaves: two attacks read the same highest index."""
    (tmp_path / "goal_aaaa").mkdir(parents=True)
    (tmp_path / "goal_aaaa" / "attempt_0000").mkdir()

    assert claim_attempt_dir(tmp_path, "goal_aaaa").name == "attempt_0001"
