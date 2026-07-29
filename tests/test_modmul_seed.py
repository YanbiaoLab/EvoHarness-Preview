import re
import ast
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


@pytest.mark.parametrize("seed", ["limb_horner", "horner_cell", "serial_ar"])
def test_training_survives_having_nothing_left_to_do(seed):
    """A run that executes zero steps must not crash on the way out.

    The loop is `while step < TOTAL_STEPS and time.monotonic() < deadline`, so
    it is skipped entirely once the cap is reached or the budget is spent, and
    the state written afterwards refers to `loss`. Run modmul_r11's island-1
    seed hit this: its cached weights stood at exactly TOTAL_STEPS, the body
    never ran, and UnboundLocalError came back as `train-failed` -- which
    reads as the candidate's own code being broken, and cost that island its
    architecture for the whole run.

    limb_horner and serial_ar already initialised loss; horner_cell did not.
    """
    src = (SEEDS / seed / "train.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.While, ast.For))]
    assigned_in_loop = {
        t.id
        for loop in loops
        for n in ast.walk(loop)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    used_after = {
        n.id for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    bound_before = set()
    for node in fn.body:
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        bound_before.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        bound_before.update(
                            e.id for e in t.elts if isinstance(e, ast.Name)
                        )
    risky = (assigned_in_loop & used_after) - bound_before
    assert not risky, (
        f"{seed}/train.py uses {sorted(risky)} outside the training loop "
        "without binding them first; a zero-step run raises UnboundLocalError"
    )


@pytest.mark.parametrize("seed", ["limb_horner", "horner_cell", "serial_ar"])
def test_the_step_cap_is_not_already_reached(seed):
    """The cap must not bind before the wall clock does.

    "upper bound; the time budget is what binds" was written in all three and
    was false in two of them. In limb_horner the lineage silently stopped
    improving, which run modmul_r8 found and fixed; in horner_cell the cache
    sat exactly ON the cap, so training became a no-op and then a crash.
    """
    src = (SEEDS / seed / "train.py").read_text()
    cap = int(re.search(r"TOTAL_STEPS\s*=\s*([\d_]+)", src).group(1).replace("_", ""))
    assert cap >= 1_000_000, (
        f"{seed} caps training at {cap:,} steps; lineages here already "
        "accumulate several hundred thousand"
    )


@pytest.mark.parametrize("seed", ["limb_horner", "horner_cell", "serial_ar"])
def test_a_reshaping_mutation_can_train_twice(seed, tmp_path):
    """RADIX_BITS reshapes one matrix, and that used to kill training at the
    SECOND rung, every time.

    Adam's exp_avg / exp_avg_sq are keyed by parameter position and carry the
    parameter's shape. load_state_dict copies them without checking, so a
    checkpoint that predates the reshape loads cleanly and then dies at the
    first step inside _multi_tensor_adam:

        RuntimeError: The size of tensor a (7) must match the size of
        tensor b (16) at non-singleton dimension 1

    7 and 16 are in_features for RADIX_BITS 1 and 4. limb_horner tried to
    guard this by rebinding a local path, which left the stale file on disk:
    the first rung skipped it and saved elsewhere, and the second rung -- by
    which point the weights matched and the guard no longer fired -- loaded it
    and crashed. horner_cell and serial_ar had no guard at all, and their
    `except: pass` around the load catches nothing, because nothing is raised
    there.

    Every RADIX_BITS candidate in runs r7, r10 and r11 died this way at R1,
    which is why the highest-value lever in this task had never once been
    evaluated end to end.
    """
    import torch

    src = (SEEDS / seed / "train.py").read_text()

    # A stale optimizer file must be REMOVED, not merely skipped, or the next
    # rung finds it again.
    assert "unlink" in src, (
        f"{seed}/train.py never removes an optimizer state that no longer fits"
    )
    assert "opt.state.clear()" in src

    # And the mismatch has to be detected after load_state_dict, since that
    # call does not raise.
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    checks_shape = any(
        isinstance(n, ast.Attribute) and n.attr == "shape"
        for h in handlers for n in ast.walk(h)
    )
    assert checks_shape, (
        f"{seed}/train.py loads optimizer state without comparing shapes; "
        "load_state_dict does not raise, the first opt.step() does"
    )


def test_the_optimizer_guard_actually_catches_a_reshape(tmp_path):
    """The mechanism itself, on real tensors rather than on source text."""
    import torch

    small = torch.nn.Linear(7, 4)
    opt = torch.optim.AdamW(small.parameters(), lr=1e-3)
    small(torch.zeros(2, 7)).sum().backward()
    opt.step()
    path = tmp_path / "optimizer.pt"
    torch.save(opt.state_dict(), path)

    wide = torch.nn.Linear(16, 4)                 # the RADIX_BITS 1 -> 4 shape
    opt2 = torch.optim.AdamW(wide.parameters(), lr=1e-3)
    opt2.load_state_dict(torch.load(path))        # loads without complaint

    mismatch = [
        value.shape
        for param, entry in opt2.state.items()
        for value in entry.values()
        if torch.is_tensor(value) and value.dim() and value.shape != param.shape
    ]
    assert mismatch, "the shape check would not have noticed"

    wide(torch.zeros(2, 16)).sum().backward()
    with pytest.raises(RuntimeError):
        opt2.step()                                # this is what killed r11
