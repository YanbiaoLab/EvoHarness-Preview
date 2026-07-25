"""Tests for the modmul grader (v1: H90-isomorphic fitness, ASHA, time budget,
weight-perturbation gate).

The load-bearing test is `test_reproduces_official_baseline_numbers`: the
grader is only trustworthy if it reproduces the accuracies the organizers
published for their own reference models. Everything else here checks a
specific failure mode gets its own verdict instead of crashing the run.
"""

import json
import shutil
from pathlib import Path

import pytest

from evoharness.evoserve import GradeContext
from modmul import grade as grade_module
from modmul.grade import grade_workspace

EXAMPLES = grade_module._REPO / "examples"

ALWAYS_ZERO = '''\
from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {"entry_class": "model.EvolvedModel", "output_base": 10,
            "model_description": "emits [0]", "training_description": "none"}


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]
'''

CHEATER = ALWAYS_ZERO.replace(
    "        return [0]",
    "        v = int(a_enc) * int(b_enc) % int(p_enc)\n"
    "        return [ord(c) - 48 for c in str(v)]",
)

# Hand-coded arithmetic that the AST scanner does NOT fingerprint: the product
# and the reduction are split across helpers. This is what the L3 behavioural
# gate exists for — it has trained parameters that do nothing.
CIRCUIT = '''\
import torch
from torch import nn

from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {"entry_class": "model.EvolvedModel", "output_base": 10,
            "model_description": "net + arithmetic", "training_description": "trained"}


def train(model_dir):
    torch.manual_seed(0)
    torch.save(nn.Linear(4, 4).state_dict(), model_dir + "/weights.pt")


def _reduce(value, modulus):
    return value - (value // modulus) * modulus


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir):
        self.net = nn.Linear(4, 4)
        self.net.load_state_dict(torch.load(model_dir + "/weights.pt"))

    def predict_digits(self, a_enc, b_enc, p_enc):
        self.net(torch.zeros(1, 4))
        product = int(a_enc) * int(b_enc)
        value = _reduce(product, int(p_enc))
        return [ord(c) - 48 for c in str(value)]
'''

SLOW = ALWAYS_ZERO.replace(
    "        return [0]",
    "        import time\n        time.sleep(0.5)\n        return [0]",
)

TRAINER = '''\
from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {"entry_class": "model.EvolvedModel", "output_base": 10,
            "model_description": "reads its own weights",
            "training_description": "writes a digit to disk"}


def train(model_dir):
    with open(model_dir + "/weights.txt", "w") as handle:
        handle.write("0")


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir):
        with open(model_dir + "/weights.txt") as handle:
            self.digit = 0 if "0" in handle.read() else 1

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [self.digit]
'''

HANGER = TRAINER.replace(
    '        handle.write("0")',
    '        handle.write("0")\n    while True:\n        pass',
)


@pytest.fixture(autouse=True)
def quick_rungs(monkeypatch):
    """Seconds-long rungs; the real ones are 8/30/90 minutes."""
    monkeypatch.setenv("MODMUL_QUICK", "1")


def candidate_dir(tmp_path: Path, source: str) -> Path:
    directory = tmp_path / "candidate"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(source)
    return directory


def ctx(tmp_path: Path) -> GradeContext:
    return GradeContext(candidate_id="cand-test", workdir=tmp_path)


def grade(tmp_path: Path, source: str):
    return grade_workspace(candidate_dir(tmp_path, source), ctx(tmp_path))


# -- fitness is the leaderboard key ----------------------------------------


def test_fitness_is_the_official_ranking_key():
    """(h90, overall) ordering must survive the collapse into one float."""
    strong_tier4 = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95}
    perfect_below = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.89, 5: 0.5}
    assert grade_module._h90(strong_tier4) == 4
    assert grade_module._h90(perfect_below) == 3
    # One more tier at >=90% outranks any amount of accuracy below it.
    assert grade_module._fitness(strong_tier4) > grade_module._fitness(perfect_below)
    # Ties on h90 are broken by overall accuracy, as the official rules say.
    same_h90_more_accurate = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.89, 5: 0.7}
    assert grade_module._h90(same_h90_more_accurate) == 3
    assert (grade_module._fitness(same_h90_more_accurate)
            > grade_module._fitness(perfect_below))
    assert 0.0 <= grade_module._fitness({}) <= 1.0


def test_unevaluated_tiers_count_as_zero():
    """Official policy: an incomplete tier scores 0%, so a candidate cannot
    look good merely by never reaching the hard tiers."""
    assert grade_module._overall({1: 1.0}) == pytest.approx(0.1)


# -- calibration against the organizers' own published numbers --------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,expected_overall,expected_h90",
    [("always_zero", 0.079, 0),
     ("digit_transformer", 0.121, 1),
     ("dlp_grokking", 0.127, 1)],
)
def test_reproduces_official_baseline_numbers(
    tmp_path, monkeypatch, name, expected_overall, expected_h90
):
    """rules/evaluation.md publishes overall_accuracy for these reference
    models on the full public benchmark. If our grader disagrees, it is not
    measuring the same thing the leaderboard measures."""
    pytest.importorskip("torch")
    monkeypatch.delenv("MODMUL_QUICK", raising=False)
    monkeypatch.setenv("MODMUL_FORCE_ALL_RUNGS", "1")
    # Single rung, full 100 cases per tier, no training: the official setup.
    monkeypatch.setattr(
        grade_module, "RUNGS",
        (grade_module.Rung("R2", 1.0, grade_module.SCORED_TIERS, 100),),
    )
    source = EXAMPLES / name
    if not source.exists():
        pytest.skip(f"reference model {name} not vendored")
    destination = tmp_path / "candidate"
    destination.mkdir()
    # Only the submission itself: the examples ship a train.py / eval_tiers.py
    # that are not part of what the organizers scored.
    for item in source.iterdir():
        if item.name in ("model.py", "manifest.json") or item.suffix == ".pt":
            shutil.copy(item, destination / item.name)

    report = grade_workspace(destination, ctx(tmp_path))
    overall = report.visible_metrics["overall_accuracy"]
    assert report.visible_metrics["h90"] == expected_h90
    assert overall == pytest.approx(expected_overall, abs=0.01)


# -- per-failure-mode verdicts ---------------------------------------------


def test_static_scanner_catches_the_modmul_shortcut(tmp_path):
    report = grade(tmp_path, CHEATER)
    assert report.passed is False
    assert report.fault == "adjudication: modmul-shortcut"
    assert report.stage_reached == 0


def test_scanner_reads_side_files_not_just_model_py(tmp_path):
    """A multi-file genome can hide the cheat in arch.py."""
    directory = candidate_dir(tmp_path, ALWAYS_ZERO)
    (directory / "arch.py").write_text(
        "def f(a, b, p):\n    return int(a) * int(b) % int(p)\n"
    )
    report = grade_workspace(directory, ctx(tmp_path))
    assert report.passed is False
    assert "modmul-shortcut" in report.fault


def test_weight_perturbation_gate_kills_a_hand_coded_circuit(tmp_path):
    """The AST scanner cannot see this one: the multiply and the reduction are
    split across helpers. Randomizing the weights leaves it perfect, which is
    exactly the signal the official L3 layer looks for."""
    pytest.importorskip("torch")
    report = grade(tmp_path, CIRCUIT)
    assert report.passed is False
    assert report.fault.startswith("perturbation-insensitive")
    assert report.visible_metrics["perturbation_random_acc"] > 0.5


def test_weak_candidate_skips_the_perturbation_gate(tmp_path):
    """Nothing to learn from perturbing a model that is already at chance —
    spending the time would just slow every generation down."""
    report = grade(tmp_path, ALWAYS_ZERO)
    assert report.passed
    assert report.visible_metrics["perturbation"] == "skipped-weak-candidate"


def test_slow_model_loses_the_tier_and_everything_above(tmp_path):
    """Official timeout policy: the running tier scores 0 and so does every
    tier above it. Without this a candidate can win locally and time out in
    the real evaluation."""
    report = grade(tmp_path, SLOW)
    assert report.passed                       # a verdict, not a crash
    categories = {
        item["error_category"] for item in report.structured_feedback["items"]
    }
    assert any("budget-exceeded" in c or "not-reached" in c for c in categories)
    # Whatever it managed before the budget ran out, the tiers it never
    # finished are zeroed — so it cannot out-score a model that is merely fast.
    zeroed = [t for t in (2, 3, 4) if report.visible_metrics.get(f"acc_tier_{t}", 0.0) == 0.0]
    assert zeroed, "tiers after the timeout should score 0"


def test_malformed_digits_degrade_and_do_not_crash(tmp_path):
    report = grade(tmp_path, ALWAYS_ZERO.replace("return [0]", "return [999]"))
    assert report.passed                       # ran to completion
    assert report.fitness == 0.0
    categories = {
        item["error_category"] for item in report.structured_feedback["items"]
    }
    assert any(c.startswith("malformed-output") for c in categories)


def test_train_artifact_reaches_predict(tmp_path):
    report = grade(tmp_path, TRAINER)
    assert report.passed
    assert report.fitness > 0                  # digit 0 from trained weights
    assert (tmp_path / "candidate" / "weights.txt").exists()


def test_runaway_training_is_killed_and_judged(tmp_path, monkeypatch):
    monkeypatch.setattr(grade_module, "TRAIN_SLACK_S", 2.0)
    report = grade(tmp_path, HANGER)
    assert report.passed is False
    assert report.fault.startswith("train-timeout")
    assert report.stage_reached == 1


def test_relative_workdir_is_resolved(tmp_path, monkeypatch):
    """SearchLoop may hand a RELATIVE gen_dir, and Sandbox chdirs its
    subprocess — a relative path would be re-resolved from the wrong cwd."""
    monkeypatch.chdir(tmp_path)
    rundir = Path("rundir")
    rundir.mkdir()
    directory = rundir / "candidate"
    directory.mkdir()
    (directory / "model.py").write_text(TRAINER)
    report = grade_workspace(
        directory, GradeContext(candidate_id="rel", workdir=rundir)
    )
    assert report.passed
    assert report.fitness > 0


def test_feedback_carries_per_tier_accuracy_and_metrics(tmp_path):
    report = grade(tmp_path, ALWAYS_ZERO)
    assert "h90" in report.visible_metrics
    assert "overall_accuracy" in report.visible_metrics
    assert report.visible_metrics["acc_tier_1"] > 0      # a=0 / b=0 edge cases
    assert "leaderboard key" in report.structured_feedback["summary"]
    assert json.dumps(report.structured_feedback)        # JSON-serializable
