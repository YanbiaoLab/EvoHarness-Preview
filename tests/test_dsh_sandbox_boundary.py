"""Where a candidate's writes can reach, and where a run must not sit.

`workspace-write` is the mode the candidate's cordis config selects, and dsh
documents its meaning as "the workspace root plus the platform temp areas" —
`writableRoots` in `@deepseek-ai/dsh-sandbox` returns the workspace, `/tmp`,
and `os.tmpdir()`. That is not a defect on their side; it is the mode working
as written.

It becomes a defect on ours the moment a run directory sits under one of
those. Every run in this project's history has been `/tmp/evo_*`, which put
`run.db`, `evidence.jsonl`, `checkpoint.json` and `manifest.json` — the
scores, the evidence, the run state and the frozen identity — inside the
grant. A candidate reading the harness is a research-integrity problem; a
candidate able to write to it is not a problem with the results, it is the
absence of results.

Measured against a real deployment on 2026-08-21 before the check existed:

    /private/tmp/evo_dsp2/run.db          writable
    /private/tmp/evo_dsp2/evidence.jsonl  writable
    /private/tmp/evo_dsp2/checkpoint.json writable
    /private/tmp/evo_dsp2/manifest.json   writable
"""

import sys
import tempfile
from pathlib import Path

import pytest

from evoharness.core.agent import dsh_backend as dsh_backend_module
from evoharness.core.agent import DshRuntimeSpec
from evoharness.core.agent.dsh_backend import (
    DshBackendError,
    check_run_dir_outside_candidate_writes,
)


@pytest.fixture(autouse=True)
def _stub_sdk(monkeypatch):
    module = type("_Module", (), {"DeepSeekHarness": object})
    monkeypatch.setitem(sys.modules, "deepseek_harness", module)


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "candidate.cordis.yml"
    path.write_text("- id: stub\n", encoding="utf-8")
    return path


def spec(config, **changes):
    values = {"config_path": config, "runtime_argv": ("node", "bin.js")}
    values.update(changes)
    return DshRuntimeSpec(**values)


@pytest.mark.parametrize(
    "root", [Path("/tmp"), Path(tempfile.gettempdir())]
)
def test_a_run_directory_inside_a_writable_root_is_refused(root):
    if not root.exists():
        pytest.skip(f"{root} does not exist on this platform")
    with pytest.raises(DshBackendError, match="workspace-write"):
        check_run_dir_outside_candidate_writes(root / "evo_some_run")


def test_the_refusal_says_what_is_at_stake_and_what_to_do():
    """A refusal nobody can act on gets bypassed rather than fixed."""

    with pytest.raises(DshBackendError) as caught:
        check_run_dir_outside_candidate_writes(Path("/tmp/evo_x"))
    message = str(caught.value)
    assert "run.db" in message
    assert "outside the temp areas" in message


def test_a_run_directory_outside_them_passes():
    """The liveness control: without it, a check that refused everything would
    satisfy every case above.

    A path under the home directory, not pytest's `tmp_path` — that one is
    inside a temp root by construction, which is the very reason a real launch
    refuses it. The directory need not exist; only its location decides.
    """

    check_run_dir_outside_candidate_writes(Path.home() / "evoharness-runs" / "r1")


def test_the_spec_refuses_at_construction_not_at_the_first_session(config):
    """Same discipline as the missing-SDK check: a run that cannot be
    trustworthy should not start, rather than produce candidates nobody can
    rely on and discover it afterwards."""

    with pytest.raises(DshBackendError, match="workspace-write"):
        spec(config, run_dir=Path("/tmp/evo_would_be_writable"))


def test_the_escape_hatch_exists_and_marks_the_run(config):
    """Tests need it — their run directories are temporary by construction.

    It has to cost something, or it becomes how real runs get started. It
    enters the fingerprint, so a run that used it is a different experiment
    and cannot be quietly compared with one that did not.
    """

    permitted = spec(
        config,
        run_dir=Path("/tmp/evo_permitted"),
        allow_writable_run_dir=True,
    )
    assert permitted.fingerprint()["allow_writable_run_dir"] is True

    strict = spec(config, allow_writable_run_dir=False)
    assert strict.fingerprint()["allow_writable_run_dir"] is False
    assert permitted.fingerprint() != strict.fingerprint()


def test_the_mirrored_root_list_still_matches_what_the_mode_grants(tmp_path):
    """This list is a second copy of dsh's `writableRoots`, in another
    language, so it can drift. Pinning it here at least makes a drift a
    failing test rather than a silently narrower check."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    roots = dsh_backend_module._granted_write_roots(workspace)  # noqa: SLF001

    assert workspace.resolve() in roots
    assert Path("/tmp").resolve() in roots
    assert Path(tempfile.gettempdir()).resolve() in roots
