"""What the assembly freezes into a run's identity.

The claim under test is decision four of the dsh integration plan: the
deployment a candidate runs inside is part of the experiment, so changing it
must make the run a different one. Until this held, swapping
`candidate.cordis.yml` and resuming the same run directory was accepted, and
two runs under different tool sets compared as one.
"""

import pytest

from evoharness.launch.build import build
from evoharness.launch.config import LaunchConfig, LaunchConfigError


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
    overrides = ["search.num_generations=1"]
    # The default lane is single_shot, which never opens a session and so
    # never reaches an agent backend. A dsh config only means anything to a
    # lane that runs one.
    if config is not None:
        overrides.append("proposal.mode=agentic")
    return LaunchConfig(
        recipe="e0",
        run_dir=tmp_path / "run",
        task="demo_counter",
        dsh_config=config,
        dsh_runtime=entry,
        overrides=tuple(overrides),
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


def test_the_prompt_is_told_which_peer_tool_the_runtime_mounts(
    tmp_path, runtime_tree
):
    """End to end: the declaration reaches the prompt, and the run records it.

    `recipes/common.py` cannot see the runtime's tool table, so the name has
    to travel from the launch config through the backend spec. Nothing between
    them may quietly substitute the in-process name.
    """

    config, entry = runtime_tree
    dsh = build(make_config(tmp_path, config, entry))
    assert dsh.ctx.extras["prompt_tools"] == {
        "peer_fetch": "evo_inspect_candidate",
        # An external runtime's file tools have its own names, so ours is not
        # offered — the prompt describes the workspace without naming one.
        "workspace_read": None,
    }
    assert dsh.ctx.extras["proposal_manifest"]["tools"] == []

    plain = build(make_config(tmp_path))
    assert plain.ctx.extras["prompt_tools"]["peer_fetch"] != (
        "evo_inspect_candidate"
    )


def test_a_lane_that_never_opens_a_session_refuses_a_runtime(
    tmp_path, runtime_tree
):
    """Otherwise the run's identity names a runtime no candidate ran inside.

    The single-shot lane returns before it looks at the backend, so the pair
    would be recorded and hashed while every proposal went through the
    in-process transport — the same silent substitution that giving
    --dsh-config without --dsh-runtime is refused for.
    """

    config, entry = runtime_tree
    cfg = LaunchConfig(
        recipe="e0",
        run_dir=tmp_path / "run",
        task="demo_counter",
        dsh_config=config,
        dsh_runtime=entry,
        overrides=("search.num_generations=1", "proposal.mode=single_shot"),
    )
    with pytest.raises(LaunchConfigError, match="single_shot"):
        build(cfg)


def test_the_declared_peer_tool_is_part_of_the_experiment(tmp_path, runtime_tree):
    """Two deployments that differ in what the candidate can reach are two
    experiments, so the declaration has to reach the run hash."""

    config, entry = runtime_tree
    with_tool = build(make_config(tmp_path, config, entry))
    recorded = with_tool.run_spec.proposer_backend.to_payload()["config"]
    assert recorded["peer_fetch_tool"] == "evo_inspect_candidate"


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
