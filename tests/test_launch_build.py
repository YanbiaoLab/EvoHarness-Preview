"""What the assembly freezes into a run's identity.

The claim under test is decision four of the dsh integration plan: the
deployment a candidate runs inside is part of the experiment, so changing it
must make the run a different one. Until this held, swapping
`candidate.cordis.yml` and resuming the same run directory was accepted, and
two runs under different tool sets compared as one.
"""

import pytest

from evoharness.launch.build import build
from evoharness.launch.config import LaunchConfig


@pytest.fixture
def runtime_tree(tmp_path):
    """A cordis file and a runtime entry deep enough for `parents[3]`."""

    config = tmp_path / "candidate.cordis.yml"
    config.write_text("- id: stub\n", encoding="utf-8")
    entry = tmp_path / "packages" / "examples" / "demo" / "bin.ts"
    entry.parent.mkdir(parents=True)
    entry.write_text("run()\n", encoding="utf-8")
    return config, entry


def make_config(tmp_path, config=None, entry=None, **changes):
    return LaunchConfig(
        recipe="e0",
        run_dir=tmp_path / "run",
        task="demo_counter",
        dsh_config=config,
        dsh_runtime=entry,
        overrides=("search.num_generations=1",),
        **changes,
    )


def test_the_cordis_deployment_reaches_the_run_hash(tmp_path, runtime_tree):
    config, entry = runtime_tree
    before = build(make_config(tmp_path, config, entry))

    config.write_text("- id: stub\n- id: extra-plugin\n", encoding="utf-8")
    after = build(make_config(tmp_path, config, entry))

    # Same command, same task, same seed — only the candidate's tool set moved.
    assert before.frozen_hashes["run_hash"] != after.frozen_hashes["run_hash"]
    assert before.frozen_hashes["task_hash"] == after.frozen_hashes["task_hash"]
    assert before.frozen_hashes["search_hash"] == after.frozen_hashes["search_hash"]


def test_the_runtime_entry_reaches_the_run_hash(tmp_path, runtime_tree):
    config, entry = runtime_tree
    before = build(make_config(tmp_path, config, entry))

    entry.write_text("run(); somethingElse()\n", encoding="utf-8")
    after = build(make_config(tmp_path, config, entry))

    assert before.frozen_hashes["run_hash"] != after.frozen_hashes["run_hash"]


def test_an_in_process_run_and_a_dsh_run_are_not_the_same_experiment(
    tmp_path, runtime_tree
):
    config, entry = runtime_tree
    plain = build(make_config(tmp_path))
    dsh = build(make_config(tmp_path, config, entry))

    assert plain.frozen_hashes["run_hash"] != dsh.frozen_hashes["run_hash"]
    assert plain.run_spec.proposer_backend.role == "proposer_backend"
    assert dsh.run_spec.proposer_backend.version == "dsh-v1"


def test_the_backend_spec_names_the_deployment_it_launched(
    tmp_path, runtime_tree
):
    config, entry = runtime_tree
    built = build(make_config(tmp_path, config, entry))
    recorded = built.run_spec.proposer_backend.to_payload()["config"]

    assert recorded["kind"] == "dsh-sdk"
    assert recorded["config_hash"].startswith("sha256:")
    # Not a path: the same checkout at another location is the same
    # experiment, so nothing location-dependent may enter the hash.
    assert not any(
        isinstance(value, str) and value.startswith(str(tmp_path))
        for value in recorded.values()
    )


def test_moving_the_checkout_does_not_change_the_run_hash(
    tmp_path, runtime_tree
):
    config, entry = runtime_tree
    before = build(make_config(tmp_path, config, entry))

    moved = tmp_path / "elsewhere"
    moved_entry = moved / "packages" / "examples" / "demo" / "bin.ts"
    moved_entry.parent.mkdir(parents=True)
    moved_entry.write_bytes(entry.read_bytes())
    moved_config = moved / "candidate.cordis.yml"
    moved_config.write_bytes(config.read_bytes())

    after = build(make_config(tmp_path, moved_config, moved_entry))

    assert before.frozen_hashes["run_hash"] == after.frozen_hashes["run_hash"]
