"""Starting a detached run."""

import json
import os
import sys
import time

import pytest

from evoharness.launch import JobSpec, StartError, StartedRun, start_run


def python_argv(*statements):
    return [sys.executable, "-c", "\n".join(statements)]


WRITE_MANIFEST = 'open("manifest.json", "w").write(\'{"report": {}}\')'
SLEEP = "import time; time.sleep(30)"


def reap(started):
    try:
        os.kill(started.pid, 9)
    except ProcessLookupError:
        pass


def test_a_started_run_is_confirmed_alive(tmp_path):
    started = start_run(
        tmp_path / "r", python_argv(WRITE_MANIFEST, SLEEP), cwd=tmp_path / "r"
    )
    try:
        assert isinstance(started, StartedRun)
        assert started.identified is True
        assert started.pid > 0
    finally:
        reap(started)


def test_the_result_survives_json_serialization(tmp_path):
    started = start_run(
        tmp_path / "r", python_argv(WRITE_MANIFEST, SLEEP), cwd=tmp_path / "r"
    )
    try:
        # This crosses a tool boundary as JSON. Returning the Paths unchanged
        # fails at the caller's json.dumps rather than here, where the cause
        # would be obvious.
        payload = json.loads(json.dumps(started.to_json()))
        assert payload["run_dir"].endswith("r")
        assert payload["identified"] is True
    finally:
        reap(started)


def test_a_run_that_dies_instantly_is_an_error_not_a_run_directory(tmp_path):
    # Returning a directory here would report a configuration error as a
    # successful start, and the caller would poll a run that never existed.
    with pytest.raises(StartError, match="boom"):
        start_run(
            tmp_path / "r",
            python_argv('import sys; sys.stderr.write("boom\\n"); sys.exit(3)'),
            cwd=tmp_path / "r",
        )


def test_the_failure_carries_the_output_that_would_otherwise_be_lost(tmp_path):
    with pytest.raises(StartError, match="ZeroDivisionError"):
        start_run(tmp_path / "r", python_argv("1 / 0"), cwd=tmp_path / "r")

    # A detached child's traceback has nowhere to go. Without the log
    # redirection an unstartable run is just an empty directory.
    assert (tmp_path / "r" / "run.log").exists()


def test_a_slow_start_is_reported_as_unidentified_not_as_a_failure(tmp_path):
    started = start_run(
        tmp_path / "r", python_argv(SLEEP), cwd=tmp_path / "r", handshake_s=0.3
    )
    try:
        # Alive but slow. Killing it or claiming the identity is on disk would
        # both be worse than saying which of the two we actually know.
        assert started.identified is False
    finally:
        reap(started)


def test_the_invocation_is_recorded_in_the_run_directory(tmp_path):
    argv = python_argv(WRITE_MANIFEST, SLEEP)
    started = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    try:
        job = JobSpec.read(tmp_path / "r")

        # A detached run has no caller left to ask; without this the only
        # record of what produced the directory is somebody's shell history.
        assert job is not None
        assert list(job.argv) == argv
    finally:
        reap(started)


def test_starting_over_a_live_run_is_refused(tmp_path):
    argv = python_argv(WRITE_MANIFEST, SLEEP)
    started = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    try:
        # Two processes writing one checkpoint corrupt it.
        with pytest.raises(StartError, match="has not finished"):
            start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    finally:
        reap(started)


def test_a_resume_that_dies_on_the_way_in_is_not_reported_as_started(tmp_path):
    """A resume has no fresh manifest to wait for.

    The handshake waits for the child to write one, but on a resume it is
    already there from the original start — so the wait succeeded instantly
    however the child fared. A real resume against a changed cordis config
    was reported as `identified: true` and had already died on the
    checkpoint's configuration fingerprint.
    """

    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"report": {"stopped_reason": "running"}}), encoding="utf-8"
    )
    (run / "job.json").write_text(
        json.dumps({
            "argv": ["x"], "cwd": str(run), "created_at": 1.0, "pid": None,
        }),
        encoding="utf-8",
    )

    with pytest.raises(StartError, match="before writing its manifest|exited"):
        start_run(
            run,
            python_argv("import sys; sys.exit(3)"),
            cwd=run,
            force=True,
        )


def test_a_resume_that_survives_is_reported_as_started(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"report": {"stopped_reason": "running"}}), encoding="utf-8"
    )

    started = start_run(run, python_argv(SLEEP), cwd=run, force=True)
    try:
        assert started.identified is True
    finally:
        reap(started)


def test_a_crashed_run_can_be_resumed_without_force(tmp_path):
    """The recovery path must not need the flag that also corrupts a live run.

    A run killed mid-generation leaves a checkpoint saying `running` forever,
    so recovering from a crash used to need the same `force` that lets a
    second writer loose on a working run. A flag whose safe use and dangerous
    use look identical becomes habitual.
    """

    argv = python_argv(WRITE_MANIFEST, SLEEP)
    first = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    reap(first)
    time.sleep(0.2)

    second = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    try:
        assert second.pid != first.pid
    finally:
        reap(second)


def test_the_run_directory_records_which_process_owns_it(tmp_path):
    argv = python_argv(WRITE_MANIFEST, SLEEP)
    started = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    try:
        assert JobSpec.read(tmp_path / "r").pid == started.pid
    finally:
        reap(started)


def test_a_directory_from_before_pids_were_recorded_still_refuses(tmp_path):
    """Absent evidence is not evidence of absence. An older run directory
    cannot say whether its process is gone, so it keeps the old answer."""

    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"report": {"stopped_reason": "running"}}), encoding="utf-8"
    )
    (run / "job.json").write_text(
        json.dumps({"argv": ["x"], "cwd": str(run), "created_at": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(StartError, match="has not finished"):
        start_run(run, python_argv(WRITE_MANIFEST, SLEEP), cwd=run)


def test_force_overrides_the_refusal(tmp_path):
    argv = python_argv(WRITE_MANIFEST, SLEEP)
    first = start_run(tmp_path / "r", argv, cwd=tmp_path / "r")
    reap(first)
    time.sleep(0.2)

    # The escape hatch for the case the directory cannot express: the previous
    # process is gone but its unfinished manifest remains.
    second = start_run(tmp_path / "r", argv, cwd=tmp_path / "r", force=True)
    try:
        assert second.pid != first.pid
    finally:
        reap(second)


def test_a_finished_run_can_be_started_again(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"report": {"stopped_reason": "completed"}}), encoding="utf-8"
    )

    started = start_run(run, python_argv(WRITE_MANIFEST, SLEEP), cwd=run)
    try:
        assert started.identified is True
    finally:
        reap(started)


def test_the_log_is_appended_across_attempts(tmp_path):
    run = tmp_path / "r"
    with pytest.raises(StartError):
        start_run(run, python_argv('print("first")', "raise SystemExit(1)"), cwd=run)
    with pytest.raises(StartError):
        start_run(run, python_argv('print("second")', "raise SystemExit(1)"), cwd=run)

    # A resume writes into the same directory; truncating loses the traceback
    # that explains why the resume was needed.
    log = (run / "run.log").read_text()
    assert "first" in log and "second" in log


def test_the_child_does_not_inherit_stdin(tmp_path):
    # The parent's stdin has to actually carry something, or the test cannot
    # tell DEVNULL from inheritance: under pytest fd 0 is already an empty
    # stream, so a child that inherits it reads "" just the same.
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"DATA-FROM-PARENT")
    os.close(write_fd)
    saved = os.dup(0)
    try:
        os.dup2(read_fd, 0)
        started = start_run(
            tmp_path / "r",
            python_argv(
                "import sys",
                'open("stdin.txt", "w").write(sys.stdin.read())',
                WRITE_MANIFEST,
                SLEEP,
            ),
            cwd=tmp_path / "r",
        )
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(read_fd)

    try:
        # A detached run that inherits stdin blocks forever the first time
        # anything reads from it, and reads the caller's data if there is any.
        assert (tmp_path / "r" / "stdin.txt").read_text() == ""
    finally:
        reap(started)


def test_the_child_is_in_its_own_session(tmp_path):
    started = start_run(
        tmp_path / "r", python_argv(WRITE_MANIFEST, SLEEP), cwd=tmp_path / "r"
    )
    try:
        # Its own session id means the parent exiting, the terminal closing or
        # a SIGHUP cannot take the run with it. Without this the failure looks
        # like a run that silently stopped, with nothing written down.
        assert os.getsid(started.pid) == started.pid
        assert os.getsid(started.pid) != os.getsid(os.getpid())
    finally:
        reap(started)
