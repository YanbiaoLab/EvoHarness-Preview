# modmul/grade.py — SAIR modmul 评估侧 grade_fn(v0.5)
#
# ⚠️ 骨架:preamble 已按本包位置写好,函数体由你手码
#   (完整实现见教程对话"给我完整 grade_fn"那一轮,函数体一字不改;
#    与那版唯一的差异就是下面 _REPO 的 parents[1])。
#
# 评估逻辑版本 = third_party 克隆的 commit(当前 fb558ce);
# 克隆更新后 serve.sh 的 --task-version 必须跟着换(协议 §6)。

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from evoharness.evoguard import Sandbox
from evoharness.evoserve import Grade, GradeContext

_REPO = Path(__file__).resolve().parents[1] / "third_party" / "modular-arithmetic-challenge"
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modchallenge.evaluation.decoder import MalformedOutput, decode_answer  # noqa: E402
from modchallenge.security.static_check import check_source  # noqa: E402

BENCH_DIR = _REPO / "public_benchmark"
V0_TIERS = (1, 2, 3)
# Run-2 scoring (task_version modmul-fb558ce-v1): the competition is about
# SCALABILITY, so higher tiers weigh more — equal weights let candidates farm
# cheap tier-1 gains (observed in gpu_run1: champion had t3=0 while the
# structural family sat on t3=6.7%).
TIER_WEIGHTS = {1: 1.0, 2: 2.0, 3: 4.0}
CASES_PER_TIER = 50            # n=30 -> 50: smaller measurement noise
TRAIN_TIMEOUT_S = 600.0

# ------------------------------------------------------------------
#_TRAIN_RUNNER 常量、_empty_feedback、_load_cases、
# _run_training、_load_candidate、grade_fn
# ------------------------------------------------------------------


# Runs inside the sandbox subprocess: import the candidate file, call its
# train() if present. Kept tiny so its own failures read trivially.
_TRAIN_RUNNER = """\
import importlib.util, sys

spec = importlib.util.spec_from_file_location("candidate", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if hasattr(module, "train"):
    module.train(sys.argv[2])
"""

def _empty_feedback(summary: str) -> dict:
    return {
        "schema_version": 1,
        "items": [],
        "summary": summary,
    }

def _load_cases(tier: int) -> list[dict]:
    """Load the public benchmark cases for a given tier."""
    lines = (BENCH_DIR / f"tier_{tier}.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines[:CASES_PER_TIER]]

def _run_training(model_dir: Path):
    runner = model_dir / "train_runner.py"
    runner.write_text(_TRAIN_RUNNER)
    # Run the training in a sandboxed subprocess, with a hard timeout.
    result = Sandbox().run(
        [sys.executable, str(runner), str(model_dir / "model.py"), str(model_dir)],
        workdir=model_dir,
        timeout_s=TRAIN_TIMEOUT_S,
        env={"PYTHONPATH": str(_SRC)},
    )

    if result.timed_out:
        return f"train-timeout: exceeded {TRAIN_TIMEOUT_S:.0f}s hard budget"
    
    if result.return_code != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        return f"train-failed (exit {result.return_code}): {tail[:400]}"
    return None

def _load_candidate(ctx: GradeContext, model_dir: Path):
    model_name = f"candidate_{ctx.candidate_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(model_name, model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = module

    try:
        spec.loader.exec_module(module)
        manifest = getattr(module, "MANIFEST", None)
        if (not isinstance(manifest, dict) or "entry_class" not in manifest
                or "output_base" not in manifest):
            raise ValueError(
                "candidate must define MANIFEST with entry_class and output_base"
            )
        
        (model_dir / "manifest.json").write_text(json.dumps(manifest))

        cls_name = str(manifest["entry_class"]).rsplit(".", 1)[-1]
        model = getattr(module, cls_name)()
        model.load(str(model_dir))
    except Exception as e:
        sys.modules.pop(model_name, None)
        raise RuntimeError(f"failed to load candidate: {e}") from e
    return model, manifest["output_base"], model_name


def grade_fn(code: str, ctx: GradeContext) -> Grade:
    # 1. Official adjudication BEFORE any execution of candidate code —
    #    import runs code, so order is load-bearing.
    findings = check_source(code, "model.py")
    if findings:
        first = findings[0]
        return Grade(
            fitness=0.0, passed=False, stage_reached=0,
            fault=f"adjudication: {first.rule}",
            stderr_log="\n".join(f.format() for f in findings[:10]),
            structured_feedback=_empty_feedback(
                f"static check rejected: {first.rule}"
            ),
        )

    # Resolve to absolute FIRST: Sandbox chdirs its subprocess into model_dir,
    # so any relative path here would be re-resolved from the wrong cwd.
    model_dir = Path(ctx.workdir).resolve() / "submission"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.py").write_text(code)

    # 2. Optional bi-level inner loop: killable, hard-budgeted (v0.5).
    fault = _run_training(model_dir)
    if fault:
        return Grade(fitness=0.0, passed=False, stage_reached=1, fault=fault,
                     structured_feedback=_empty_feedback("training phase failed"))

    # 3. Load — candidate bugs here are verdicts, not infra.
    try:
        model, output_base, mod_name = _load_candidate(ctx, model_dir)
    except Exception as exc:
        return Grade(
            fitness=0.0, passed=False, stage_reached=1,
            fault=f"load-error: {type(exc).__name__}: {exc}",
            structured_feedback=_empty_feedback("candidate failed to load"),
        )

    # 4. Tier loop with the official decoder; every failure mode gets its
    #    own error_category — C1/C2 feed on these.
    items, tier_acc = [], {}
    try:
        for tier in V0_TIERS:
            cases = _load_cases(tier)
            correct = 0
            for i, case in enumerate(cases):
                category, predicted = "", "?"
                try:
                    digits = model.predict_digits(
                        model.preprocess_a(case["a"]),
                        model.preprocess_b(case["b"]),
                        model.preprocess_p(case["p"]),
                    )
                    value = decode_answer(
                        digits, base=output_base, prime=int(case["p"])
                    )
                    predicted = str(value)
                    if predicted == case["expected"]:
                        correct += 1
                    else:
                        category = f"wrong-answer-t{tier}"
                except MalformedOutput:
                    category = f"malformed-output-t{tier}"
                except Exception as exc:
                    category = f"predict-{type(exc).__name__}"
                items.append({
                    "item_id": f"t{tier}#{i}",
                    "passed": category == "",
                    "predicted": predicted[:40],
                    "expected": case["expected"][:40],
                    "error_category": category,
                })
            tier_acc[tier] = correct / len(cases)
    finally:
        sys.modules.pop(mod_name, None)

    total_w = sum(TIER_WEIGHTS[t] for t in tier_acc)
    fitness = sum(TIER_WEIGHTS[t] * a for t, a in tier_acc.items()) / total_w
    return Grade(
        fitness=fitness,
        visible_metrics={f"acc_tier_{t}": a for t, a in tier_acc.items()},
        structured_feedback={
            "schema_version": 1,
            "items": items,
            "summary": (
                f"weighted tiers {V0_TIERS} (w={[TIER_WEIGHTS[t] for t in V0_TIERS]},"
                " higher tiers count more): "
                + ", ".join(f"t{t}={a:.0%}" for t, a in tier_acc.items())
            ),
        },
    )
    



