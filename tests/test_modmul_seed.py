"""Contract tests for the serial_ar seed: shape and legality only —
correctness comes from real GPU training, not the 8-step smoke run."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from modmul.grade import BENCH_DIR, MalformedOutput, check_source, decode_answer

SEED_PATH = Path(__file__).resolve().parent.parent / "modmul" / "seeds" / "serial_ar.py"


def smoke_code() -> str:
    code = SEED_PATH.read_text()
    for old, new in (("TRAIN_STEPS = 3000", "TRAIN_STEPS = 8"),
                     ("D_MODEL = 128", "D_MODEL = 32"),
                     ("LAYERS = 4", "LAYERS = 2"),
                     ("BATCH = 256", "BATCH = 32")):
        assert old in code, f"seed drifted: {old!r} not found"
        code = code.replace(old, new)
    return code


def test_seed_passes_official_adjudication():
    assert check_source(SEED_PATH.read_text(), "model.py") == []


def test_smoke_variant_still_clean():
    """The string-surgery smoke preset must not accidentally trip adjudication."""
    assert check_source(smoke_code(), "model.py") == []


def test_seed_contract_end_to_end(tmp_path):
    code = smoke_code()
    (tmp_path / "model.py").write_text(code)
    spec = importlib.util.spec_from_file_location("seed_smoke", tmp_path / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MANIFEST["entry_class"] == "model.EvolvedModel"

    module.train(str(tmp_path))                       # v0.5 hook
    assert (tmp_path / "weights.pt").exists()

    model = module.EvolvedModel()
    model.load(str(tmp_path))
    case = json.loads((BENCH_DIR / "tier_1.jsonl").read_text().splitlines()[0])
    digits = model.predict_digits(
        model.preprocess_a(case["a"]),
        model.preprocess_b(case["b"]),
        model.preprocess_p(case["p"]),
    )
    assert isinstance(digits, list) and len(digits) == 5
    assert all(isinstance(d, int) and 0 <= d < 10 for d in digits)
    try:                                              # shape contract only:
        decode_answer(digits, base=10, prime=int(case["p"]))
    except MalformedOutput:
        pass                                          # value >= p legal here


HORNER_PATH = SEED_PATH.parent / "horner_cell.py"


def horner_smoke_code() -> str:
    code = HORNER_PATH.read_text()
    for old, new in (("TRAIN_STEPS = 3000", "TRAIN_STEPS = 30"),
                     ("HIDDEN = 512", "HIDDEN = 64"),
                     ("BATCH = 4096", "BATCH = 256")):
        assert old in code, f"horner seed drifted: {old!r} not found"
        code = code.replace(old, new)
    return code


def test_horner_seed_passes_official_adjudication():
    assert check_source(HORNER_PATH.read_text(), "model.py") == []
    assert check_source(horner_smoke_code(), "model.py") == []


def test_horner_seed_contract_end_to_end(tmp_path):
    (tmp_path / "model.py").write_text(horner_smoke_code())
    spec = importlib.util.spec_from_file_location(
        "horner_smoke", tmp_path / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.train(str(tmp_path))
    assert (tmp_path / "weights.pt").exists()
    model = module.EvolvedModel()
    model.load(str(tmp_path))
    case = json.loads((BENCH_DIR / "tier_1.jsonl").read_text().splitlines()[0])
    digits = model.predict_digits(
        model.preprocess_a(case["a"]),
        model.preprocess_b(case["b"]),
        model.preprocess_p(case["p"]),
    )
    assert isinstance(digits, list) and len(digits) >= 1
    assert all(isinstance(d, int) and 0 <= d < 10 for d in digits)
