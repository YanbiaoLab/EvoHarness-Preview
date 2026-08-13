"""The domain half of a state merge: blending checkpoints, and the two ways
a genome-identical child can corrupt a weight cache."""

import json

import pytest

from evoharness.serve import GradeContext

torch = pytest.importorskip("torch")

from experiments.modmul import grade as grade_module  # noqa: E402


def _checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head.weight": torch.full((1, 4), float(value)),
            "steps": torch.tensor(int(value * 100)),
        },
        path,
    )


def _donor_dirs(tmp_path, base_value, donor_value):
    lineage = tmp_path / "lineage"
    _checkpoint(lineage / "base" / "weights.pt", base_value)
    _checkpoint(lineage / "donor" / "weights.pt", donor_value)
    return lineage


def _candidate(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "arch.py").write_text("D_MODEL = 64\n")
    (candidate / "train.py").write_text("def train(d):\n    pass\n")
    return candidate


def test_weights_are_blended_at_the_requested_ratio(tmp_path):
    lineage = _donor_dirs(tmp_path, base_value=1.0, donor_value=5.0)
    candidate = _candidate(tmp_path)
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.25},
            {"id": "donor", "weight": 0.75},
        ),
    )

    result = grade_module._merge_state_donors(
        candidate, tmp_path / "merge_runner.py", ctx
    )

    assert result["state_merge"] == "applied"
    blended = torch.load(candidate / "weights.pt", map_location="cpu")
    # 0.25 * 1.0 + 0.75 * 5.0
    assert blended["head.weight"].flatten()[0].item() == pytest.approx(4.0)


def test_integer_buffers_take_the_heaviest_source_rather_than_averaging(tmp_path):
    lineage = _donor_dirs(tmp_path, base_value=1.0, donor_value=5.0)
    candidate = _candidate(tmp_path)
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.25},
            {"id": "donor", "weight": 0.75},
        ),
    )

    grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)

    blended = torch.load(candidate / "weights.pt", map_location="cpu")
    assert blended["steps"].item() == 500      # the donor's, not an average


def test_optimizer_moments_are_dropped(tmp_path):
    """They describe a trajectory that ended at one source, not at the blend.

    Carrying them in is the stale-moment family this project already lost
    three generations of candidates to.
    """
    lineage = _donor_dirs(tmp_path, 1.0, 5.0)
    candidate = _candidate(tmp_path)
    (candidate / "optimizer.pt").write_bytes(b"inherited-moments")
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.5},
            {"id": "donor", "weight": 0.5},
        ),
    )

    grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)
    assert not (candidate / "optimizer.pt").exists()


def test_a_missing_donor_cancels_the_merge_rather_than_substituting(tmp_path):
    """Falling back to a different blend would report a merge that never
    happened, and the fitness would be attributed to it."""
    lineage = tmp_path / "lineage"
    _checkpoint(lineage / "base" / "weights.pt", 1.0)
    candidate = _candidate(tmp_path)
    (candidate / "weights.pt").write_bytes(b"inherited")
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.5},
            {"id": "ghost", "weight": 0.5},
        ),
    )

    result = grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)

    assert result["state_merge"] == "skipped"
    assert "ghost" in result["state_merge_why"]
    assert (candidate / "weights.pt").read_bytes() == b"inherited"


def test_incompatible_checkpoints_are_refused_not_approximated(tmp_path):
    lineage = tmp_path / "lineage"
    _checkpoint(lineage / "base" / "weights.pt", 1.0)
    (lineage / "donor").mkdir(parents=True)
    torch.save({"head.weight": torch.zeros(1, 9)}, lineage / "donor" / "weights.pt")
    candidate = _candidate(tmp_path)
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.5},
            {"id": "donor", "weight": 0.5},
        ),
    )

    result = grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)
    assert result["state_merge"] == "failed"


def test_an_ordinary_candidate_is_untouched(tmp_path):
    candidate = _candidate(tmp_path)
    assert grade_module._merge_state_donors(
        candidate, tmp_path / "r.py", GradeContext("c", tmp_path)
    ) == {}


def test_merged_weights_never_enter_the_shared_recipe_cache(tmp_path):
    """A merge child's genome is byte-identical to its base's, so it hashes
    to the SAME recipe. Publishing a blend there would hand these weights to
    every future candidate sharing the recipe, none of which asked for one.
    """
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    candidate = _candidate(tmp_path)
    (candidate / "weights.pt").write_bytes(b"blended")
    (candidate / "train_state.json").write_text('{"steps": 1}')
    ctx = GradeContext("merged-child", tmp_path, lineage_dir=lineage)

    grade_module._publish_to_lineage(candidate, ctx, 5400.0, merged=True)

    own = lineage / "merged-child" / "weights.pt"
    shared = grade_module._pretrained_dir(
        lineage, grade_module._training_digest(candidate)
    ) / "weights.pt"
    assert own.exists(), "the merge's own children must still inherit it"
    assert not shared.exists(), "the shared recipe cache was poisoned"


def test_an_ordinary_candidate_still_fills_the_shared_cache(tmp_path):
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    candidate = _candidate(tmp_path)
    (candidate / "weights.pt").write_bytes(b"trained")
    ctx = GradeContext("normal-child", tmp_path, lineage_dir=lineage)

    grade_module._publish_to_lineage(candidate, ctx, 5400.0, merged=False)

    shared = grade_module._pretrained_dir(
        lineage, grade_module._training_digest(candidate)
    ) / "weights.pt"
    assert shared.exists()


def test_the_blend_survives_inheritance_running_first(tmp_path):
    """Inheritance copies the BASE's weights in, and the merge must win.

    The recipe digest of a merge child equals its base's, so the
    content-addressed cache hands back exactly the weights the merge is
    supposed to replace. Order is the only thing that settles it.
    """
    lineage = _donor_dirs(tmp_path, base_value=1.0, donor_value=5.0)
    (lineage / "base" / "train_state.json").write_text('{"steps": 900}')
    candidate = _candidate(tmp_path)
    ctx = GradeContext(
        "child",
        tmp_path,
        parent_id="base",
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.25},
            {"id": "donor", "weight": 0.75},
        ),
    )

    inherited = grade_module._inherit_from_parent(candidate, ctx)
    merged = grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)

    assert inherited["inherited_steps"] == 900     # lineage still supplies this
    assert merged["state_merge"] == "applied"
    blended = torch.load(candidate / "weights.pt", map_location="cpu")
    assert blended["head.weight"].flatten()[0].item() == pytest.approx(4.0)


def test_merge_metrics_are_reported_for_reading_back(tmp_path):
    lineage = _donor_dirs(tmp_path, 1.0, 5.0)
    candidate = _candidate(tmp_path)
    ctx = GradeContext(
        "child",
        tmp_path,
        lineage_dir=lineage,
        state_donors=(
            {"id": "base", "weight": 0.25},
            {"id": "donor", "weight": 0.75},
        ),
    )
    result = grade_module._merge_state_donors(candidate, tmp_path / "r.py", ctx)
    assert result["state_merge_sources"] == 2
    assert "@0.25" in result["state_merge_why"]
    assert "@0.75" in result["state_merge_why"]
