"""Contract tests for the three modmul seed genomes: shape, legality and the
resumable-training contract only. Real accuracy comes from GPU training, not
from these seconds-long smoke runs.

Each seed is a three-file workspace (model.py / arch.py / train.py); the tests
copy it to a temp dir and import it the way the official loader does (seed dir
on sys.path), so `import arch` resolves exactly as it will in production.
"""

import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from modmul.grade import BENCH_DIR, check_submission

SEEDS = Path(__file__).resolve().parents[1] / "experiments" / "modmul" / "seeds"
SEED_NAMES = ("limb_horner", "horner_cell", "serial_ar")
GENOME_FILES = ("model.py", "arch.py", "train.py")


@pytest.fixture
def seed_dir(request, tmp_path):
    """Materialize one seed and put it on sys.path, isolated per test."""
    name = request.param
    destination = tmp_path / name
    destination.mkdir()
    for filename in GENOME_FILES:
        shutil.copy(SEEDS / name / filename, destination / filename)
    sys.path.insert(0, str(destination))
    try:
        yield destination
    finally:
        sys.path.remove(str(destination))
        for module in ("model", "arch", "train"):
            sys.modules.pop(module, None)


def shrink(seed_dir: Path, name: str) -> None:
    """Make the seed tiny enough for a seconds-long smoke run. Asserts on the
    replaced strings so a drifting seed fails loudly instead of silently
    training at full size inside the test suite."""
    edits = {
        "limb_horner": [("arch.py", "D_MODEL = 64", "D_MODEL = 16"),
                        ("arch.py", "HIDDEN = 128", "HIDDEN = 32")],
        "horner_cell": [("arch.py", "HIDDEN = 512", "HIDDEN = 32"),
                        ("train.py", "BATCH = 4096", "BATCH = 64")],
        "serial_ar": [("arch.py", "D_MODEL = 128", "D_MODEL = 32"),
                      ("arch.py", "LAYERS = 4", "LAYERS = 1"),
                      ("train.py", "BATCH = 256", "BATCH = 16")],
    }
    for filename, old, new in edits[name]:
        path = seed_dir / filename
        text = path.read_text()
        assert old in text, f"{name}/{filename} drifted: {old!r} not found"
        path.write_text(text.replace(old, new))


@pytest.mark.parametrize("name", SEED_NAMES)
def test_seed_passes_official_adjudication(name):
    assert check_submission(SEEDS / name) == []


@pytest.mark.parametrize("seed_dir", SEED_NAMES, indirect=True)
def test_seed_contract_end_to_end(seed_dir, monkeypatch):
    name = seed_dir.name
    shrink(seed_dir, name)
    monkeypatch.setenv("MODMUL_TRAIN_SECONDS", "3")

    train = importlib.import_module("train")
    train.train(str(seed_dir))
    assert (seed_dir / "weights.pt").exists()

    model_module = importlib.import_module("model")
    manifest = model_module.MANIFEST
    assert manifest["entry_class"].endswith("EvolvedModel")
    assert manifest["model_description"] and manifest["training_description"]

    model = model_module.EvolvedModel()
    model.load(str(seed_dir))
    case = json.loads((BENCH_DIR / "tier_1.jsonl").read_text().splitlines()[0])
    digits = model.predict_digits(
        model.preprocess_a(case["a"]),
        model.preprocess_b(case["b"]),
        model.preprocess_p(case["p"]),
    )
    base = manifest["output_base"]
    limit = int(case["p"]) if base == "p" else base
    assert isinstance(digits, list) and digits
    assert all(isinstance(d, int) and 0 <= d < limit for d in digits)


@pytest.mark.parametrize("seed_dir", SEED_NAMES, indirect=True)
def test_training_resumes_instead_of_restarting(seed_dir, monkeypatch):
    """ASHA hands the same model_dir back for the next rung. A seed that
    restarts from scratch throws away the previous rung's compute."""
    shrink(seed_dir, seed_dir.name)
    monkeypatch.setenv("MODMUL_TRAIN_SECONDS", "3")

    train = importlib.import_module("train")
    train.train(str(seed_dir))
    first = json.loads((seed_dir / "train_state.json").read_text())["steps"]
    weights = torch.load(seed_dir / "weights.pt", map_location="cpu")

    train.train(str(seed_dir))
    second = json.loads((seed_dir / "train_state.json").read_text())["steps"]
    assert second > first, "second rung restarted instead of resuming"

    resumed = torch.load(seed_dir / "weights.pt", map_location="cpu")
    assert set(resumed) == set(weights)
    assert any(
        not torch.equal(resumed[key], weights[key]) for key in weights
    ), "weights did not move — resume ran but learned nothing"


@pytest.mark.parametrize("seed_dir", ["limb_horner"], indirect=True)
def test_limb_horner_is_width_generic(seed_dir):
    """The whole bet of this family: one set of weights, any state width.
    A width-specific parameter would raise here."""
    arch = importlib.import_module("arch")
    cell = arch.HornerCell()
    for width in (2, 7, 16, 64):
        state = torch.zeros(3, width)
        digit = torch.zeros(3, arch.RADIX_BITS)
        assert cell(state, state, state, digit).shape == (3, width)
