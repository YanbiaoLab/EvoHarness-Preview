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


def test_leaderboard_key_reproduces_the_official_ordering():
    """Ranking and reporting use (h90, overall) — the official key."""
    strong_tier4 = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95}
    perfect_below = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.89, 5: 0.5}
    # One more tier at >=90% outranks any amount of accuracy below it.
    assert grade_module.leaderboard_key(strong_tier4) > grade_module.leaderboard_key(
        perfect_below
    )
    # Ties on h90 are broken by overall accuracy, as the official rules say.
    same_h90_more_accurate = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.89, 5: 0.7}
    assert grade_module.leaderboard_key(
        same_h90_more_accurate
    ) > grade_module.leaderboard_key(perfect_below)


def test_fitness_rises_with_every_tier_so_the_search_has_a_slope():
    """Fitness is the SEARCH signal, not the ranking key, and the two want
    opposite things: ranking wants the discrete fact of crossing 90%,
    search wants to know how far from it you are.

    Collapsing both into (h90 + overall)/11 made a tier accuracy doubling
    worth +0.0018 and a threshold crossing worth +0.0918 — a 50x cliff on
    otherwise flat ground, which is close to the worst possible terrain for
    a search whose only move is a small edit."""
    for tier in (2, 5, 10):
        base = {1: 1.0}
        rising = [
            grade_module._fitness({**base, tier: value})
            for value in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
        ]
        assert rising == sorted(rising) and rising[0] < rising[-1], (
            f"tier {tier} gives the search no slope"
        )

    # Crossing 90% must still pay more than ordinary progress — just not 50x.
    doubling = grade_module._fitness({1: 1.0, 2: 0.4}) - grade_module._fitness(
        {1: 1.0, 2: 0.2}
    )
    crossing = grade_module._fitness({1: 1.0, 2: 0.95}) - grade_module._fitness(
        {1: 1.0, 2: 0.85}
    )
    assert 1.5 < crossing / doubling < 5.0
    assert 0.0 <= grade_module._fitness({}) <= 1.0


def test_leaderboard_fitness_stays_available_for_the_ab(monkeypatch):
    """The old definition is the control arm; it must still be one flag away."""
    monkeypatch.setenv("MODMUL_FITNESS", "leaderboard")
    accuracy = {1: 1.0, 2: 1.0, 3: 0.95}
    assert grade_module._fitness(accuracy) == pytest.approx(
        (3 + grade_module._overall(accuracy)) / 11
    )


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


# -- lineage: trained weights are not in the genome -------------------------


def test_weights_are_inherited_when_the_architecture_is_unchanged(tmp_path):
    """The genome is three text files, so without a lineage channel every
    candidate trains from random init and a 120-generation run discards its
    whole compute budget, each generation throwing away the last."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    parent = lineage / "p1"
    parent.mkdir(parents=True)
    (parent / "weights.pt").write_bytes(b"trained-weights")
    (parent / "optimizer.pt").write_bytes(b"opt")
    (parent / "train_state.json").write_text('{"steps": 12345}')

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("D_MODEL = 64\n")
    (parent / "arch.sha256").write_text(
        grade_module._arch_digest(child) + "\n"
    )

    result = grade_module._inherit_from_parent(
        child,
        GradeContext("c1", tmp_path, parent_id="p1", lineage_dir=lineage),
    )

    assert result == {"warm_start": "full", "inherited_steps": 12345}
    assert (child / "weights.pt").read_bytes() == b"trained-weights"
    assert (child / "optimizer.pt").exists()


def test_a_changed_architecture_cold_starts_rather_than_loading_junk(tmp_path):
    """Tensor shapes follow arch.py, so inheriting across a changed
    architecture would load weights that do not fit the model."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    parent = lineage / "p1"
    parent.mkdir(parents=True)
    (parent / "weights.pt").write_bytes(b"trained-weights")
    (parent / "arch.sha256").write_text("a" * 64 + "\n")

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("D_MODEL = 128   # architecture mutated\n")

    result = grade_module._inherit_from_parent(
        child,
        GradeContext("c1", tmp_path, parent_id="p1", lineage_dir=lineage),
    )

    assert result["warm_start"] == "cold-arch-changed"
    assert not (child / "weights.pt").exists()


def test_no_lineage_channel_means_todays_behaviour(tmp_path):
    """Every task that has not opted in must keep cold-starting."""
    from evoharness.evoserve import GradeContext

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("D_MODEL = 64\n")
    result = grade_module._inherit_from_parent(
        child, GradeContext("c1", tmp_path)
    )
    assert result == {"warm_start": "cold", "inherited_steps": 0}


def test_publishing_makes_a_candidate_inheritable_by_its_children(tmp_path):
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "arch.py").write_text("D_MODEL = 64\n")
    (cand / "weights.pt").write_bytes(b"w")
    (cand / "train_state.json").write_text('{"steps": 7}')

    ctx = GradeContext("c1", tmp_path, lineage_dir=lineage)
    grade_module._publish_to_lineage(cand, ctx)

    published = lineage / "c1"
    assert (published / "weights.pt").read_bytes() == b"w"
    assert (published / "arch.sha256").read_text().strip() == (
        grade_module._arch_digest(cand)
    )
    # A child with the same arch.py must now inherit it.
    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("D_MODEL = 64\n")
    assert grade_module._inherit_from_parent(
        child, GradeContext("c2", tmp_path, parent_id="c1", lineage_dir=lineage)
    ) == {"warm_start": "full", "inherited_steps": 7}


def test_cost_projection_recovers_the_measured_speedup_requirement():
    """The time budget is what actually caps this domain, and it only reveals
    itself at R2. The real baseline run: tiers 1-8 all at 100%, tier 9 at 92%,
    tier 10 at 0 without a single case run, because tier 9 alone ate 83% of
    the budget. `infer_s_tier_10` was simply absent from the report.

    A projection from R0 must recover that, and must not exaggerate it: an
    earlier version compared the top tiers against a pro-rata slice of the
    budget and reported 13x over when the truth was 2.9x. The budget is one
    shared pool — cheap low tiers subsidise expensive high ones.
    """
    real = {1: 0.185, 2: 0.691, 3: 1.569, 4: 1.419, 5: 3.523,
            6: 10.815, 7: 12.063, 8: 34.342, 9: 117.644}
    scored = grade_module.RUNGS[-1].cases
    truth_total = sum(real.values()) + real[9] * 2      # tier 10 ~ 2x tier 9

    # What R0 sees (tiers 1-3 at 30 cases) plus a 3-case probe of tiers 9-10.
    projected = {t: real[t] / 50 * scored for t in (1, 2, 3)}
    projected[9] = real[9] / 50 * scored
    projected[10] = real[9] / 50 * 2 * scored
    known = sorted(projected)
    for low, high in zip(known, known[1:]):
        gap = [t for t in range(low + 1, high) if t in grade_module.SCORED_TIERS]
        if not gap:
            continue
        ratio = (projected[high] / projected[low]) ** (1 / (high - low))
        for step, tier in enumerate(gap, start=1):
            projected[tier] = projected[low] * ratio ** step

    allowed = grade_module.SECONDS_PER_PROBLEM * (
        scored * len(grade_module.SCORED_TIERS) + 20
    )
    required = sum(projected.values()) / allowed
    truth_required = truth_total / allowed

    assert truth_required > 1.0, "the real run was over budget; the test data is wrong"
    # Within 25% of the truth, and pessimistic rather than optimistic — a gate
    # that under-reports the wall is worse than one that over-reports it.
    assert truth_required <= required <= truth_required * 1.25
