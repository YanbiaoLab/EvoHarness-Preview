from evoharness.evoserve import GradeContext
from modmul import grade as grade_module
from modmul.grade import grade_fn

SEED = '''\
from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {"entry_class": "model.EvolvedModel", "output_base": 10,
            "model_description": "seed", "training_description": "none"}

# EDIT-REGION-BEGIN
class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir):
        pass

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [0]
# EDIT-REGION-END
'''

CHEATER = SEED.replace(
    "        return [0]",
    "        v = int(a_enc) * int(b_enc) % int(p_enc)\n"
    "        return [ord(c) - 48 for c in str(v)]",
)

TRAINER = '''\
from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {"entry_class": "model.EvolvedModel", "output_base": 10,
            "model_description": "trainer", "training_description": "writes weights"}

def train(model_dir):
    with open(model_dir + "/weights.txt", "w") as f:
        f.write("0")

class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir):
        with open(model_dir + "/weights.txt") as f:
            data = f.read().strip()
        self.digit = 0 if "0" in data else 1

    def predict_digits(self, a_enc, b_enc, p_enc):
        return [self.digit]
'''

HANGER = TRAINER.replace('    f.write("0")',
                         '    f.write("0")\n    while True:\n        pass')


def ctx(tmp_path):
    return GradeContext(candidate_id="cand-test", workdir=tmp_path)


def test_seed_baseline_scores_nonzero(tmp_path):
    grade = grade_fn(SEED, ctx(tmp_path))
    assert grade.passed
    assert 0.0 < grade.fitness < 0.5          # edge cases with answer "0"
    assert grade.visible_metrics["acc_tier_1"] > 0
    assert len(grade.structured_feedback["items"]) == 3 * grade_module.CASES_PER_TIER


def test_official_scanner_catches_modmul_shortcut(tmp_path):
    grade = grade_fn(CHEATER, ctx(tmp_path))
    assert grade.passed is False
    assert grade.fault == "adjudication: modmul-shortcut"
    assert grade.stage_reached == 0


def test_malformed_digits_degrade_not_crash(tmp_path):
    bad = SEED.replace("return [0]", "return [999]")   # digit out of base-10
    grade = grade_fn(bad, ctx(tmp_path))
    assert grade.passed                                # ran to completion
    assert grade.fitness == 0.0
    cats = {i["error_category"] for i in grade.structured_feedback["items"]}
    assert any(c.startswith("malformed-output") for c in cats)


def test_train_artifact_reaches_predict(tmp_path):
    grade = grade_fn(TRAINER, ctx(tmp_path))
    assert grade.passed
    assert grade.fitness > 0                  # digit 0 from trained weights
    assert (tmp_path / "submission" / "weights.txt").exists()


def test_relative_workdir_is_resolved(tmp_path, monkeypatch):
    """Regression: SearchLoop may hand a RELATIVE gen_dir; Sandbox chdirs the
    train subprocess, which double-resolved relative paths (first real run
    failed exactly here)."""
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    rel = Path("rundir")
    rel.mkdir()
    grade = grade_fn(TRAINER, GradeContext(candidate_id="rel", workdir=rel))
    assert grade.passed
    assert grade.fitness > 0


def test_runaway_train_is_killed_and_judged(tmp_path, monkeypatch):
    monkeypatch.setattr(grade_module, "TRAIN_TIMEOUT_S", 2.0)
    grade = grade_fn(HANGER, ctx(tmp_path))
    assert grade.passed is False
    assert grade.fault.startswith("train-timeout")
    assert grade.stage_reached == 1
