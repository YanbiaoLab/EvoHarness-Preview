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
    (parent / "arch.sha256").write_text(grade_module._arch_digest(child) + "\n")

    result = grade_module._inherit_from_parent(
        child, GradeContext("c1", tmp_path, parent_id="p1", lineage_dir=lineage)
    )

    assert result["warm_start"] == "parent-full"
    assert result["inherited_steps"] == 12345
    assert (child / "weights.pt").read_bytes() == b"trained-weights"


def test_a_changed_architecture_still_inherits_what_fits(tmp_path):
    """Raising RADIX_BITS reshapes exactly one matrix — 896 of 91,841
    parameters. Rejecting the whole checkpoint over 1% made the highest-value
    mutation the most expensive one to try, and it had to compete against
    siblings carrying ninety minutes of inherited training. The loader decides
    per tensor; the harness just supplies the file."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    parent = lineage / "p1"
    parent.mkdir(parents=True)
    (parent / "weights.pt").write_bytes(b"trained-weights")
    (parent / "arch.sha256").write_text("a" * 64 + "\n")

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("RADIX_BITS = 2   # architecture mutated\n")

    result = grade_module._inherit_from_parent(
        child, GradeContext("c1", tmp_path, parent_id="p1", lineage_dir=lineage)
    )

    assert result["warm_start"] == "parent-partial"
    assert (child / "weights.pt").exists(), "a 1% reshape must not cost 100%"


def test_no_lineage_channel_means_todays_behaviour(tmp_path):
    """Every task that has not opted in must keep cold-starting."""
    from evoharness.evoserve import GradeContext

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("D_MODEL = 64\n")
    result = grade_module._inherit_from_parent(child, GradeContext("c1", tmp_path))
    assert result["warm_start"] == "cold"
    assert result["inherited_steps"] == 0
    # Why it went cold is recorded too: run modmul_r1's best-reasoned
    # offspring collapsed from 0.846 to 0.213 purely from starting cold, and
    # nothing said which branch had sent it there.
    assert result["warm_start_why"] == "no-lineage-dir"


def test_a_parentless_candidate_reuses_weights_trained_for_the_same_recipe(tmp_path):
    """Seeds have no parent, so without a content-addressed store every run
    re-derives weights already measured — ninety minutes to reproduce a file
    on disk. Keyed by what produced the weights, not by who."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "arch.py").write_text("RADIX_BITS = 1\n")
    (seed / "train.py").write_text("LR = 1e-3\n")
    (seed / "weights.pt").write_bytes(b"w")
    (seed / "train_state.json").write_text('{"steps": 5577}')

    ctx = GradeContext("s1", tmp_path, lineage_dir=lineage)
    grade_module._publish_to_lineage(seed, ctx, 5400.0)

    # A second parentless candidate with the same arch+train hits the store.
    twin = tmp_path / "twin"
    twin.mkdir()
    (twin / "arch.py").write_text("RADIX_BITS = 1\n")
    (twin / "train.py").write_text("LR = 1e-3\n")
    twin_ctx = GradeContext("s2", tmp_path, lineage_dir=lineage)

    result = grade_module._inherit_from_parent(twin, twin_ctx)
    assert result["warm_start"] == "pretrained-full"
    assert result["inherited_steps"] == 5577
    # ...and training would only re-derive what it just loaded.
    assert grade_module._may_skip_training(twin, twin_ctx, 5400.0) is True


def test_changing_the_recipe_does_not_skip_training(tmp_path):
    """The fast path is only sound while training would be byte-identical.
    train.py decides the data distribution, so touching it must retrain."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "arch.py").write_text("RADIX_BITS = 1\n")
    (seed / "train.py").write_text("LR = 1e-3\n")
    (seed / "weights.pt").write_bytes(b"w")
    grade_module._publish_to_lineage(
        seed, GradeContext("s1", tmp_path, lineage_dir=lineage), 5400.0
    )

    changed = tmp_path / "changed"
    changed.mkdir()
    (changed / "arch.py").write_text("RADIX_BITS = 1\n")
    (changed / "train.py").write_text("LR = 3e-4   # recipe mutated\n")
    ctx = GradeContext("c1", tmp_path, lineage_dir=lineage)
    assert grade_module._may_skip_training(changed, ctx, 5400.0) is False


def test_an_inference_only_mutation_skips_training(tmp_path):
    """train.py imports from arch.py and never from model.py, so a mutation
    confined to the inference contract trains to byte-identical weights.
    This is the axis the largest known win in this domain sits on."""
    from evoharness.evoserve import GradeContext

    lineage = tmp_path / "lineage"
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "arch.py").write_text("RADIX_BITS = 1\n")
    (seed / "train.py").write_text("LR = 1e-3\n")
    (seed / "model.py").write_text("WIDTH_MARGIN = 0\n")
    (seed / "weights.pt").write_bytes(b"w")
    grade_module._publish_to_lineage(
        seed, GradeContext("s1", tmp_path, lineage_dir=lineage), 5400.0
    )

    child = tmp_path / "child"
    child.mkdir()
    (child / "arch.py").write_text("RADIX_BITS = 1\n")
    (child / "train.py").write_text("LR = 1e-3\n")
    (child / "model.py").write_text("WIDTH_MARGIN = 32   # inference policy\n")
    ctx = GradeContext("c1", tmp_path, lineage_dir=lineage)
    assert grade_module._may_skip_training(child, ctx, 5400.0) is True
    # ...but training beyond what the stored weights cover still has to be paid.
    assert grade_module._may_skip_training(child, ctx, 9000.0) is False


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
    # `real` was measured at 50 cases per tier; both the truth and the
    # projection have to be expressed at the rung's case count or the test
    # compares two different scales (it did, once R2 went 50 -> 100).
    measured_cases = 50
    truth_total = (
        (sum(real.values()) + real[9] * 2)              # tier 10 ~ 2x tier 9
        / measured_cases * scored
    )

    # What R0 sees (tiers 1-3 at 30 cases) plus a 3-case probe of tiers 9-10.
    projected = {t: real[t] / measured_cases * scored for t in (1, 2, 3)}
    projected[9] = real[9] / measured_cases * scored
    projected[10] = real[9] / measured_cases * 2 * scored
    known = sorted(projected)
    for low, high in zip(known, known[1:]):
        gap = [t for t in range(low + 1, high) if t in grade_module.SCORED_TIERS]
        if not gap:
            continue
        ratio = (projected[high] / projected[low]) ** (1 / (high - low))
        for step, tier in enumerate(gap, start=1):
            projected[tier] = projected[low] * ratio ** step

    allowed = grade_module.SECONDS_PER_PROBLEM * (
        scored * (len(grade_module.SCORED_TIERS) + 1)   # +1: tier-0 diagnostic
    )
    required = sum(projected.values()) / allowed
    truth_required = truth_total / allowed

    assert truth_required > 1.0, "the real run was over budget; the test data is wrong"
    # Within 25% of the truth, and pessimistic rather than optimistic — a gate
    # that under-reports the wall is worse than one that over-reports it.
    assert truth_required <= required <= truth_required * 1.25


def test_the_cost_probe_survives_the_later_rungs(tmp_path, monkeypatch):
    """The probe runs once, at the first promotion, and its results have to
    reach the report. They did not: `metrics` is reassigned wholesale at the
    top of every rung, so everything written into it at the end of one rung
    was wiped by the next, and budget_headroom silently came back empty from
    a run that had measured it."""
    calls = []
    real = grade_module._cost_projection

    def spy(runner, model_dir, workdir, measured, measured_cases):
        calls.append(measured_cases)
        return {"cost_probe": "checked", "budget_headroom": 0.31,
                "projected_infer_s_all_tiers": 464.0}

    monkeypatch.setattr(grade_module, "_cost_projection", spy)
    monkeypatch.setenv("MODMUL_FORCE_ALL_RUNGS", "1")
    report = grade(tmp_path, TRAINER)

    assert calls, "the probe never ran"
    assert len(calls) == 1, "the probe must run once, not per rung"
    assert report.visible_metrics.get("budget_headroom") == 0.31
    assert report.visible_metrics.get("cost_probe") == "checked"
    assert real is not None


def test_speed_is_a_gradient_not_a_cliff():
    """Wall clock had selection pressure only as a cliff, and cliffs are what
    _fitness exists to soften.

    A candidate that fits every tier inside the budget scores tier 10 and
    gains a whole h90 level, so the pressure was there -- but nothing paid
    for approaching it. Run modmul_r8's candidate 8ab3347e was 38% slower
    than its parent at byte-identical accuracy and scored byte-identically
    for it, so selection saw two equals on the one axis that gates tier 10.
    """
    fitness = grade_module._fitness
    acc = {**{t: 1.0 for t in range(1, 9)}, 9: 0.98, 10: 0.0}
    base = dict(zip(range(1, 10),
                    [0.26, 0.82, 1.6, 1.39, 3.58, 10.8, 11.9, 45.2, 175.8]))
    budget = 300.0
    at = lambda k: {t: s * 2 / k for t, s in base.items()}   # noqa: E731

    slower = fitness(acc, at(1 / 1.38), budget)
    same = fitness(acc, at(1.0), budget)
    faster = fitness(acc, at(1.25), budget)
    fastest = fitness(acc, at(1.5), budget)
    assert slower < same < faster < fastest, (slower, same, faster, fastest)
    # and the steps are large enough for a weighted selector to see
    assert same - slower > 0.005


def test_being_fast_is_worth_nothing_without_accuracy():
    """The obvious way to game a speed term is to answer nothing very fast.

    _fitness's contract is that an all-zero candidate scores exactly 0; an
    ungated speed bonus paid such a candidate 0.048 and a real test caught
    it.
    """
    fitness = grade_module._fitness
    nothing = {t: 0.0 for t in range(1, 11)}
    instant = {1: 0.01, 2: 0.01, 3: 0.01}
    assert fitness(nothing, instant, 300.0) == 0.0
    assert fitness(nothing, None, None) == 0.0


def test_the_top_rung_is_the_official_problem_set():
    """h90 should mean the official h90, not a scaled proxy for it.

    The budget was scaled by problem count but the BATCH count was not, and
    the official timer is checked between batches -- so which tiers get
    scored depended on a granularity our calibration did not reproduce.
    """
    top = grade_module.RUNGS[-1]
    assert top.cases == 100
    assert top.diagnostic, "tier 0 is one of the official eleven tiers"
    problems = top.cases * (len(grade_module.SCORED_TIERS) + 1)
    assert problems == grade_module.OFFICIAL_TOTAL_PROBLEMS
    budget = grade_module.SECONDS_PER_PROBLEM * problems
    assert budget == grade_module.OFFICIAL_INFERENCE_BUDGET_S


def test_speed_still_matters_after_every_tier_has_run():
    """The pressure must not switch off at the moment it has the most to do.

    An earlier version went neutral as soon as every scored tier had run, on
    the reasoning that there was nothing left to unlock. Measurement killed
    it: the official timer is checked only between batches, so a model that
    batches a whole tier at once is checked once per tier, at its start.
    Tier 10 begins at a clock of ~190s, is never checked again, and runs 788
    seconds. Every tier runs, the total is 978s against a 300s budget, and
    going neutral there would have scored 978s and 400s identically.
    """
    fitness = grade_module._fitness
    acc = {**{t: 1.0 for t in range(1, 9)}, 9: 0.99, 10: 0.96}
    measured = dict(zip(range(1, 11),
                        [0.26, 0.82, 1.6, 1.39, 3.58, 10.8, 11.9, 45.2,
                         114.8, 788.0]))
    budget = 300.0
    assert grade_module._time_factor(measured, budget) < 1.0

    halved = {**measured, 10: 394.0}
    inside = {t: s * budget / sum(measured.values()) for t, s in measured.items()}
    assert (fitness(acc, measured, budget)
            < fitness(acc, halved, budget)
            < fitness(acc, inside, budget))
    # ...and crossing into tier 10 still beats being fast without it, because
    # the leaderboard pays for the crossing.
    without = {**{t: 1.0 for t in range(1, 9)}, 9: 0.99, 10: 0.0}
    quick = {t: s for t, s in measured.items() if t != 10}
    assert fitness(without, quick, budget) < fitness(acc, measured, budget)


def test_the_feedback_names_the_tier_that_spent_the_budget():
    """The largest cost in the run was the one item the search could not see.

    Tier 0 scores nothing -- "diagnostic only and not counted toward either
    metric" -- but it runs first and it is on the same clock, and the summary
    only ever received the scored tiers. Measured on the seed at the official
    calibration it took 243.9s of the 300s budget, which is why tiers 9 and 10
    never ran despite answering at 99% and 96% when given the time.

    The old summary also described fitness as "(H90 + overall)/11", which
    stopped being true when _fitness was softened: the search was reading the
    wrong objective off its own feedback.
    """
    acc = {**{t: 1.0 for t in range(1, 9)}, 9: 0.0, 10: 0.0}
    scored = dict(zip(range(1, 9),
                      [0.187, 0.657, 1.5, 1.405, 3.61, 10.749, 13.02, 58.666]))
    text = grade_module._summary(
        acc, grade_module.RUNGS[-1], scored, 300.0, 243.9
    )
    assert "243.9" in text, "tier 0's cost is missing"
    assert "tier was 0" in text, "tier 0 is not named as the most expensive"
    assert "333.7" in text and "300" in text
    assert "[9, 10] never ran" in text
    assert "(H90 + overall)/11" not in text, "stale fitness definition"


def test_inheritance_walks_past_an_ancestor_that_published_nothing(tmp_path):
    """A failed candidate publishes nothing; its children must not start over.

    Run modmul_r9 lost 323,546 training steps in three generations: a
    candidate faulted, its repair child found no weights under the parent's
    id and cold-started, and that cold start outscored the fully-trained seed
    -- which was scoring 0 for an unrelated reason -- so the entire island
    then descended from noise.
    """
    import torch

    lineage = tmp_path / "lineage"
    grandparent = lineage / "gp"
    grandparent.mkdir(parents=True)
    torch.save({"w": torch.zeros(2)}, grandparent / "weights.pt")
    (grandparent / "arch.sha256").write_text("deadbeef\n")
    (lineage / "failed-parent").mkdir()          # exists, but has no weights

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in ("arch.py", "train.py", "model.py"):
        (candidate / name).write_text(f"# {name}\n")

    ctx = grade_module.GradeContext(
        candidate_id="child",
        workdir=tmp_path / "work",
        parent_id="failed-parent",
        ancestor_ids=("gp",),
        lineage_dir=lineage,
    )
    out = grade_module._inherit_from_parent(candidate, ctx)
    assert (candidate / "weights.pt").exists(), "did not inherit"
    assert out["warm_start"].startswith("ancestor1"), out
    assert "ancestor-weights(+1)" in out["warm_start_why"]

    # And with no ancestry recorded it still reports the cold start honestly.
    cold_ctx = grade_module.GradeContext(
        candidate_id="child2",
        workdir=tmp_path / "work2",
        parent_id="failed-parent",
        lineage_dir=lineage,
    )
    cold = grade_module._inherit_from_parent(tmp_path / "candidate2", cold_ctx)
    assert cold["warm_start"] == "cold"


def test_a_starved_lineage_gets_a_slope_not_a_plateau():
    """All-zero was a plateau, and the search walked off it the wrong way.

    When the diagnostic tier exhausts the budget, every scored tier is skipped
    and every candidate in the lineage scores exactly 0 -- however near it is
    to fixing that. Run modmul_r9 spent eight generations there and a
    candidate that discarded 323,546 training steps outscored the
    fully-trained seed, 0.10 against 0.00.

    Honest about what this does not do: the seed reaches 0.0418 against that
    cold start's 0.1043, so the slope does not rescue a starved lineage from a
    rival that actually scores. Raising the weight until it did would make
    "almost ran" worth as much as crossing a tier. The plateau's cause is
    fixed in the model, not here.
    """
    fitness = grade_module._fitness
    nothing_ran = {t: 0.0 for t in range(1, 11)}
    scores = [
        fitness(nothing_ran, {}, 300.0, clock_s=float(c))
        for c in (600, 400, 340, 310, 301)
    ]
    assert scores == sorted(scores), scores
    assert scores[0] > 0.0, "still a plateau"
    assert scores[-1] > scores[0] * 1.5


def test_a_tier_that_ran_and_scored_zero_earns_no_speed_credit():
    """The gaming vector: answer nothing, very fast.

    Those tiers RAN -- the model was asked and got it wrong -- which is not
    the same as never being asked, and only the second earns credit.
    """
    fitness = grade_module._fitness
    nothing = {t: 0.0 for t in range(1, 11)}
    instant = {t: 0.01 for t in range(1, 11)}
    assert fitness(nothing, instant, 300.0, clock_s=0.1) == 0.0


def test_the_clock_includes_the_unscored_diagnostic_tier():
    """`seconds` holds only the scored tiers, and on this task the unscored
    one is the most expensive item in the run: reading the clock off `seconds`
    under-counted it by 78%."""
    factor = grade_module._time_factor
    scored_only = dict(zip(range(1, 10), [0.2, 0.7, 1.5, 1.4, 3.6, 10.8,
                                          13.0, 58.7, 114.8]))
    assert factor(scored_only, 300.0) == 1.0            # 205s, inside budget
    assert factor(scored_only, 300.0, clock_s=205.0 + 244.0) < 1.0


def test_the_timed_inference_takes_the_card_alone(tmp_path):
    """Sixteen candidates share one GPU, and the clock is what we score by.

    Measured on identical weights and identical problems: the diagnostic tier
    took 61.4s alone and 146.3s under run modmul_r10's load, tier 9 took
    114.8s alone and 235.0s. Roughly 2x, varying with however many siblings
    happen to be running -- while the official evaluation runs one model on
    its own. A wall clock that moves with the neighbours means selection is
    partly selecting on noise, and rejecting candidates the official harness
    would pass.
    """
    import threading
    import time as _time

    lock = tmp_path / "timed.lock"
    order: list[str] = []
    started = threading.Event()

    def hold():
        with grade_module._exclusive_gpu(lock):
            order.append("first-in")
            started.set()
            _time.sleep(0.3)
            order.append("first-out")

    def contend():
        started.wait(2.0)
        with grade_module._exclusive_gpu(lock):
            order.append("second-in")

    threads = [threading.Thread(target=hold), threading.Thread(target=contend)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)

    assert order == ["first-in", "first-out", "second-in"], order

    # And with no lock path it is a no-op, so tasks that never opted in are
    # unaffected.
    with grade_module._exclusive_gpu(None):
        pass


def test_an_architecture_change_gets_time_to_re_adapt(tmp_path, monkeypatch):
    """The first gate was judging a network that had not finished adapting.

    Run modmul_r10's candidate 2f46bb2f is one line -- ROUNDS 3 -> 1, every
    weight inherited whole, because ROUNDS reshapes nothing. Retrained and
    probed:

        480s   t1=100%  t2=53%  t3=10%   <- gated out here
       1080s   t1=100%  t2=50%  t3=13%
       1680s   t1=100%  t2=67%  t3=30%   <- clears BOTH gate conditions

    Thirteen of thirteen architecture-changing candidates died at that gate,
    four of them having found RADIX_BITS. ROUNDS 3 -> 1 cuts per-step cost
    threefold, which is the whole margin between this seed and tier 10.
    """
    seen: list[float] = []

    def fake_training(runner, candidate_dir, seconds):
        seen.append(seconds)
        return "stop-here"          # a fault ends grading after the first rung

    monkeypatch.setattr(grade_module, "_run_training", fake_training)
    monkeypatch.setattr(grade_module, "_may_skip_training",
                        lambda *a, **k: False)

    def grade_with(warm: str) -> float:
        seen.clear()
        monkeypatch.setattr(
            grade_module, "_inherit_from_parent",
            lambda *a, **k: {"warm_start": warm, "warm_start_why": "test",
                             "inherited_steps": 1},
        )
        candidate = tmp_path / warm
        candidate.mkdir(exist_ok=True)
        for name in ("arch.py", "train.py"):
            (candidate / name).write_text("x = 1\n")
        (candidate / "model.py").write_text(
            "MANIFEST = {}\n\n\nclass EvolvedModel:\n    pass\n"
        )
        ctx = grade_module.GradeContext(
            candidate_id=warm, workdir=tmp_path / f"w-{warm}",
            lineage_dir=tmp_path / "lin",
        )
        grade_module.grade_workspace(candidate, ctx)
        return seen[0] if seen else 0.0

    rungs = grade_module._rungs()
    unchanged = grade_with("parent-full")
    changed = grade_with("parent-partial")
    assert unchanged == rungs[0].train_seconds
    assert changed == rungs[0].train_seconds + rungs[1].train_seconds
    assert changed > unchanged
