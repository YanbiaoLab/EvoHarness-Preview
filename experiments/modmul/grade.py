# modmul/grade.py — SAIR modmul 评估侧 grade 函数 (v1.0, Round-1)
#
# 与 v0.5 的四处结构性差异（每一处都对应"H90 封顶在 tier 3"的一个原因）：
#
#   1. fitness 与官方排行榜键同构：(h90 + overall_accuracy) / 11。
#      官方排序是 (H90 降序, overall 降序)；整数 h90 主导、overall 破平手，
#      归一到 [0,1] 后就是同一个偏序。v0.5 的人为加权均值和它不同构。
#   2. ASHA 三级晋级 + 可续训：弱候选 ~9 分钟出局，强候选拿满 90 分钟；
#      同样机时覆盖 3-5 倍候选数。
#   3. 推理墙钟预算，官方同款失败语义（超预算的那一层及其上全部记 0）。
#      不测这个，进化会挑出"本地分高、官方 5 分钟超时"的候选。
#   4. 权重扰动闸（官方 L3 行为信号的自测版）：扰动权重后精度必须塌，
#      否则答案不来自训练参数 —— 静态检查抓不到，但官方人工复核会毙掉。
#
# 候选是**多文件 workspace**（model.py 推理契约 / arch.py 架构 / train.py 配方）。
# 训练与推理都在**子进程**里跑：既隔离崩溃与显存，也避免并发候选之间
# `import arch` 的模块名撞车（eval_batch_size>1 时这是真 bug）。
#
# 评估逻辑版本 = third_party 克隆的 commit；克隆更新后 serve.sh 的
# --task-version 必须跟着换（协议 §6）。

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from evoharness.evoguard import AntiHackScanner, Sandbox
from evoharness.evoserve import Grade, GradeContext, InfraError

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1] / "third_party" / "modular-arithmetic-challenge"
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modchallenge.security.static_check import (  # noqa: E402
    check_source,
    check_submission,
)

BENCH_DIR = _REPO / "public_benchmark"
HOLDOUT_DIR = _HERE / ".holdout"          # 由 modmul.holdout 一次性生成（可选）

SCORED_TIERS = tuple(range(1, 11))        # 官方计分层；tier 0 只做诊断
DIAGNOSTIC_TIER = 0

# 官方预算：1100 题 5 分钟。按题数等比折算到我们的评测集大小。
OFFICIAL_TOTAL_PROBLEMS = 1100
OFFICIAL_INFERENCE_BUDGET_S = 300.0
SECONDS_PER_PROBLEM = OFFICIAL_INFERENCE_BUDGET_S / OFFICIAL_TOTAL_PROBLEMS

LOAD_ALLOWANCE_S = 240.0                  # load() 官方单独计量，这里给足余量

# Ceiling on the eval subprocess, as a multiple of the inference budget. This
# is a pathology stop, NOT the budget -- the budget is enforced inside the
# runner, exactly as the official pipeline enforces it.
#
# It used to be budget + LOAD_ALLOWANCE_S, which made us STRICTER than the
# official harness and cost run modmul_r9 its seed. The official timer is
# checked only between batches, so a model that batches a whole tier at once
# gets one check per tier, at its start: a tier beginning under budget then
# runs to completion however long it takes, and is scored. Measured on the
# seed at the official calibration -- tier 10 starts at a clock of ~190s and
# runs 788 seconds, for 978 seconds total against a 300-second budget. Killing
# at 540 reported a fault for a run the official pipeline would have scored
# h90=10.
#
# 6x clears the measured 3.3x with room for a candidate several times slower,
# and still stops a runaway at half an hour.
INFERENCE_KILL_FACTOR = 6.0
TRAIN_SLACK_S = 240.0                     # 候选无视自身预算时的硬 kill 余量
MAX_FEEDBACK_ITEMS_PER_TIER = 20          # 逐题反馈体积上限（报告要进 run.db）

# 扰动闸：只对"强到值得怀疑"的候选跑。弱候选塌不塌没有信息量,而且 tier 1 的
# 4 个固定素数 + 每层前几题是 a=0/b=0 边界,恒零模型也能蒙到分 —— 所以触发条件
# 看的是 tier>=2 的实力,不是任意一层的分数。
PERTURB_PROBE_CASES = 10
PERTURB_MIN_TIER_ACC = 0.30        # 哪些层拿来做探针
PERTURB_TRIGGER_TIER_ACC = 0.50    # tier>=2 到这个水平才值得查
PERTURB_MAX_SURVIVING_RATIO = 0.20


@dataclass(frozen=True)
class Rung:
    """一级 ASHA 晋级档。train_seconds 是**追加**训练时间，不是重训。"""

    name: str
    train_seconds: float
    tiers: tuple[int, ...]
    cases: int
    diagnostic: bool = False              # 是否带 tier 0 诊断
    perturbation: bool = False            # 是否跑权重扰动闸
    holdout: bool = False                 # 是否复测私有种子集


# 晋级阈值：gen-0 基线校准前的初值（计划 S5 会按三个种子的实测重新固化）。
RUNGS: tuple[Rung, ...] = (
    Rung("R0", 480.0, (1, 2, 3), 30),
    Rung("R1", 1320.0, (1, 2, 3, 4, 5, 6), 40),
    # 100 cases, not 50: at 50 the budget was scaled proportionally (141.8s)
    # but the BATCH COUNT was not, and the official timer is checked between
    # batches. The public benchmark holds exactly 100 per tier across 11
    # tiers, so this rung is now the official 1100 problems and its budget
    # falls on 300.0s exactly -- our h90 becomes the official h90 instead of
    # a scaled proxy for it.
    Rung("R2", 3600.0, SCORED_TIERS, 100,
         diagnostic=True, perturbation=True, holdout=True),
)

QUICK_RUNGS: tuple[Rung, ...] = (
    Rung("R0", 20.0, (1, 2, 3), 5),
    Rung("R1", 20.0, (1, 2, 3, 4), 5, perturbation=True),
)


def _rungs() -> tuple[Rung, ...]:
    return QUICK_RUNGS if os.environ.get("MODMUL_QUICK") == "1" else RUNGS


# Promotion is by RANK within a stratum, against the distribution earlier
# generations produced -- not against a fixed accuracy.
#
# Fixed thresholds cannot work here and three attempts proved it. A candidate
# that changed arch.py scores low at the first rung because it has not
# re-adapted yet, not because it is bad; a candidate that inherited its
# parent's weights whole scores high because it carries 323,546 steps of
# training. Measured on run modmul_r10's second generation, at the first rung:
#
#     inherited whole   0.8 and above     all promoted
#     arch.py changed   0.222 - 0.544     none promoted, 13 of 13
#
# One of those thirteen was a one-line ROUNDS 3 -> 1 that recovers to
# t2=100% t3=90% given the training, and cuts per-step cost threefold -- the
# margin that decides tier 10. Measuring both groups with one line is a
# standing decision never to accept an architectural change.
#
# Failing this gate is not elimination, which is what makes it expensive: the
# loop breaks, the unevaluated tiers are scored 0 by _overall, and the
# candidate is filed permanently as unable to do them. A candidate stopped at
# R0 cannot exceed fitness 0.2863 however good it is. The record is false and
# selection reads it forever.
_PROMOTION_QUANTILE = 0.5          # successive halving: keep the better half
_PROMOTION_MIN_HISTORY = 6         # below this the distribution is a guess
_PROMOTION_WINDOW = 60             # recent generations only; the bar moves


def _rung_score(acc: dict[int, float], rung: Rung) -> float:
    """Mean accuracy over the tiers THIS rung evaluated.

    Not fitness: fitness scores unevaluated tiers as 0, so it is not
    comparable between rungs. This is only ever compared with scores from the
    same rung.
    """
    if not rung.tiers:
        return 0.0
    return sum(acc.get(t, 0.0) for t in rung.tiers) / len(rung.tiers)


def _promotion_log(lineage_dir: Path, stratum: str, rung: Rung) -> Path:
    # The rung's shape is in the name: change what a rung evaluates and the
    # old distribution stops being comparable, so it starts a fresh file
    # instead of silently poisoning the bar.
    shape = f"{rung.cases}c{len(rung.tiers)}t"
    return Path(lineage_dir) / "promotion" / f"{stratum}-{rung.name}-{shape}.jsonl"


def _record_rung_score(
    lineage_dir: Path | None,
    stratum: str,
    rung: Rung,
    generation: int | None,
    score: float,
) -> None:
    """Append one line. Generation 0 is NOT recorded: a seed's score is not a
    sample from the distribution its offspring are drawn from."""
    if not lineage_dir or not generation:
        return
    path = _promotion_log(lineage_dir, stratum, rung)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"generation": int(generation),
                       "score": round(float(score), 6)}) + "\n"
    with _exclusive_gpu(None):          # no GPU needed; kept for symmetry
        with open(path, "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(line)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _promotion_bar(
    lineage_dir: Path | None,
    stratum: str,
    rung: Rung,
    generation: int | None,
) -> tuple[float | None, int]:
    """The score to beat, and how many samples it came from.

    Reads only generations STRICTLY EARLIER than this one. Sixteen candidates
    are graded concurrently and finish in an order that varies run to run, so
    ranking within the current generation would need every sibling to finish
    first -- the slowest one would hold up the rest, and a resumed run would
    not reproduce its own decisions. Earlier generations are finished and
    read-only, so every candidate sees the same bar whenever it happens to
    arrive.
    """
    if not lineage_dir or generation is None:
        return None, 0
    path = _promotion_log(lineage_dir, stratum, rung)
    if not path.exists():
        return None, 0
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue                    # a torn line is not worth a crash
        if int(row.get("generation", 0)) < int(generation):
            rows.append((int(row["generation"]), float(row["score"])))
    if len(rows) < _PROMOTION_MIN_HISTORY:
        return None, len(rows)
    rows.sort(reverse=True)             # newest generations first
    scores = sorted(s for _g, s in rows[:_PROMOTION_WINDOW])
    index = min(
        len(scores) - 1, int(len(scores) * _PROMOTION_QUANTILE)
    )
    return scores[index], len(scores)


def _promotes(
    rung_index: int,
    acc: dict[int, float],
    rung: Rung,
    stratum: str,
    lineage_dir: Path | None,
    generation: int | None,
) -> tuple[bool, dict]:
    """Returns (promote, metrics-about-the-decision).

    The metrics are reported so the scheme can be audited from a run's data
    rather than argued about: if the number reaching the top rung drifts away
    from batch/4, the quantile is wrong and the run will say so.
    """
    score = _rung_score(acc, rung)
    info = {"promotion_stratum": stratum, "promotion_score": round(score, 4)}
    if os.environ.get("MODMUL_FORCE_ALL_RUNGS") == "1":
        return True, {**info, "promotion_reason": "forced"}
    if rung_index >= len(_rungs()) - 1:
        return False, {**info, "promotion_reason": "top-rung"}
    bar, n = _promotion_bar(lineage_dir, stratum, rung, generation)
    info["promotion_history_n"] = n
    if bar is None:
        # Nothing to compare against yet. Be generous: the cost is one
        # generation of candidates running further than they deserve, and the
        # alternative is a hand-picked number, which is what this replaces.
        return True, {**info, "promotion_reason": "no-history"}
    info["promotion_bar"] = round(bar, 4)
    return score >= bar, {**info, "promotion_reason": "quantile"}


# ---------------------------------------------------------------------------
# 子进程脚本：训练 / 评测。都不放进候选目录（否则会被下一轮静态检查扫到，
# 也会污染 workspace 的 diff），只写到 ctx.workdir。
# ---------------------------------------------------------------------------

_TRAIN_RUNNER = '''\
"""Harness-side train driver: import the candidate's recipe and run it.

Contract: train() must be RESUMABLE (continue from weights already in
model_dir) and must respect MODMUL_TRAIN_SECONDS as a wall-clock budget.
"""
import importlib.util
import os
import sys

model_dir = sys.argv[1]
sys.path.insert(0, model_dir)          # so `import arch` resolves


def _load(filename):
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        return None
    name = filename[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = _load("train.py")
if trainer is None or not hasattr(trainer, "train"):
    trainer = _load("model.py")
if trainer is None:
    raise SystemExit("candidate has neither train.py nor model.py")
if hasattr(trainer, "train"):
    trainer.train(model_dir)
'''


_EVAL_RUNNER = '''\
"""Harness-side inference driver (mirrors modchallenge pipeline.run_inference).

Ground truth NEVER enters this process: the plan carries inputs only and the
parent scores the decoded predictions. Decoding happens here, inside the timed
section, exactly as the official timer policy specifies.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

model_dir, plan_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, model_dir)

plan = json.loads(open(plan_path).read())
out = {"tiers": {}, "error": None, "error_kind": "", "params": 0,
       "load_seconds": 0.0, "randomized_tensors": None}


def finish():
    with open(out_path, "w") as handle:
        json.dump(out, handle)
    raise SystemExit(0)


def fail(kind, message):
    out["error_kind"] = kind
    out["error"] = str(message)[:600]
    finish()


try:
    from modchallenge.evaluation.decoder import MalformedOutput, decode_answer
except Exception as exc:                                   # harness-side
    fail("harness-import", exc)


def randomize(model):
    """L3 self-test: replace every trained tensor the loaded model holds with
    random values of the same shape and scale. Done IN MEMORY, after load(),
    so it works whatever file format the weights were stored in.

    Returns the number of tensors replaced; 0 means the model carries no
    trained parameters at all.
    """
    import torch
    from torch import nn

    generator = torch.Generator().manual_seed(20260725)
    touched = 0

    def noise_like(tensor):
        scale = 1.0
        if tensor.numel() > 1:
            value = float(tensor.detach().float().std())
            if value == value and value > 0:
                scale = value
        sample = torch.randn(
            tensor.shape, generator=generator, dtype=torch.float32
        )
        return (sample * scale).to(tensor.dtype).to(tensor.device)

    def walk(obj, depth=0):
        nonlocal touched
        if depth > 4:
            return
        if isinstance(obj, nn.Module):
            for tensor in list(obj.parameters()) + list(obj.buffers()):
                if tensor.is_floating_point():
                    with torch.no_grad():
                        tensor.copy_(noise_like(tensor))
                    touched += 1
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_floating_point():
                with torch.no_grad():
                    obj.copy_(noise_like(obj))
                touched += 1
            return
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for value in obj:
                walk(value, depth + 1)

    walk(vars(model), 1)
    return touched


spec = importlib.util.spec_from_file_location(
    "candidate_model", os.path.join(model_dir, "model.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules["candidate_model"] = module
try:
    spec.loader.exec_module(module)
except Exception as exc:
    fail("import-error", "%s: %s" % (type(exc).__name__, exc))

# Evolved genomes declare MANIFEST in model.py; an official submission
# directory declares it in manifest.json. Accept either, so this grader can
# score a packaged submission (and the organizers' own examples) unchanged.
manifest = getattr(module, "MANIFEST", None)
manifest_path = os.path.join(model_dir, "manifest.json")
if not isinstance(manifest, dict) and os.path.exists(manifest_path):
    try:
        manifest = json.loads(open(manifest_path).read())
    except Exception as exc:
        fail("manifest", "unreadable manifest.json: %s" % exc)
if not isinstance(manifest, dict) or "entry_class" not in manifest \\
        or "output_base" not in manifest:
    fail("manifest", "declare MANIFEST in model.py (or ship a manifest.json) "
                     "with entry_class and output_base")
out["output_base"] = manifest["output_base"]

# The materialized candidate dir must also be a valid official submission.
with open(manifest_path, "w") as handle:
    json.dump(manifest, handle)

class_name = str(manifest["entry_class"]).rsplit(".", 1)[-1]
try:
    model = getattr(module, class_name)()
except Exception as exc:
    fail("entry-class", "%s: %s" % (type(exc).__name__, exc))

started = time.monotonic()
try:
    model.load(model_dir)
except Exception as exc:
    fail("load-error", "%s: %s" % (type(exc).__name__, exc))
out["load_seconds"] = time.monotonic() - started

if plan["mode"] == "perturb":
    try:
        out["randomized_tensors"] = randomize(model)
    except Exception as exc:
        fail("perturb-setup", "%s: %s" % (type(exc).__name__, exc))

try:
    import torch
    from torch import nn

    total = 0
    for value in vars(model).values():
        if isinstance(value, nn.Module):
            total += sum(p.numel() for p in value.parameters())
    out["params"] = int(total)
except Exception:
    pass

try:
    batch_size = max(1, int(model.max_batch_size()))
except Exception:
    batch_size = 1

budget = float(plan["budget_s"])
clock = time.monotonic()
for tier_key in plan["order"]:
    if time.monotonic() - clock >= budget:
        break                                  # this tier and all above: 0
    cases = plan["cases"][tier_key]
    is_tier_zero = tier_key == "0"
    tier_started = time.monotonic()
    predictions = []
    complete = True
    note = ""
    for start in range(0, len(cases), batch_size):
        if time.monotonic() - clock >= budget:
            complete = False
            note = "inference-budget-exceeded"
            break
        chunk = cases[start:start + batch_size]
        try:
            encoded = [
                (
                    model.preprocess_a(case["a"]),
                    model.preprocess_b(case["b"]),
                    model.preprocess_p(case["p"]),
                )
                for case in chunk
            ]
            digit_lists = model.predict_digits_batch(encoded)
            if len(digit_lists) != len(chunk):
                complete = False
                note = "batch-contract-violation"
                break
            for digits, case in zip(digit_lists, chunk):
                try:
                    value = decode_answer(
                        digits,
                        base=manifest["output_base"],
                        prime=int(case["p"]),
                        is_tier_zero=is_tier_zero,
                    )
                    predictions.append(str(value))
                except MalformedOutput:
                    predictions.append("!malformed")
        except Exception as exc:
            predictions.extend(
                ["!%s" % type(exc).__name__] * len(chunk)
            )
    out["tiers"][tier_key] = {
        "predictions": predictions,
        "seconds": time.monotonic() - tier_started,
        "complete": complete,
        "note": note,
    }

finish()
'''


# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------


def _load_cases(tier: int, count: int, directory: Path = BENCH_DIR) -> list[dict]:
    """Read a tier's problems. The benchmark is HARNESS state, not candidate
    state, so a failure to read it is never the candidate's fault.

    Raising InfraError instead of letting the exception escape is what routes
    it to the framework's existing path: the candidate is dropped rather than
    inserted, the consecutive-failure breaker counts it, and a run stops
    instead of grinding on. Without this it lands in task.py's generic
    handler, which files the candidate as passed=False with "uncaught
    exception in grade_func" -- a permanent record blaming a candidate for a
    disk.

    Seen once, on 2026-07-28: OSError(EIO) on tier_1.jsonl during a storage
    fault that was simultaneously closing ssh sessions. The run kept going and
    seeded an island with a failure that had nothing to do with its seed. Zero
    such failures in the 322 candidates of runs r7 through r10, so this is
    rare -- but the breaker built for exactly this case could not see it.
    """
    path = directory / f"tier_{tier}.jsonl"
    try:
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise InfraError(f"cannot read the benchmark at {path}: {exc}") from exc
    return [json.loads(line) for line in lines[:count]]


def _inputs_only(cases: list[dict]) -> list[dict]:
    """Ground truth stays in the parent; the candidate process never sees it."""
    return [{"a": c["a"], "b": c["b"], "p": c["p"]} for c in cases]


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Sandbox builds a minimal env; torch on the L40S image needs the HPCX
    library paths (l40s_env.sh) or `import torch` dies on libucc."""
    env = {"PYTHONPATH": os.pathsep.join(
        p for p in (str(_SRC), os.environ.get("PYTHONPATH", "")) if p
    )}
    for name in (
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
        "CUDA_MODULE_LOADING", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME",
        "TORCH_HOME", "OMP_NUM_THREADS", "MPS_VISIBLE_DEVICES",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    env.update(extra or {})
    return env


# ---------------------------------------------------------------------------
# 阶段
# ---------------------------------------------------------------------------


def _adjudicate(candidate_dir: Path) -> str | None:
    """官方 AST 检查（整个候选目录）+ 框架 L0 静态扫描。返回第一条违规。"""
    findings = check_submission(candidate_dir)
    if findings:
        return f"adjudication: {findings[0].rule}"

    texts = {}
    for path in sorted(candidate_dir.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        texts[str(path.relative_to(candidate_dir))] = path.read_text()
    guard = AntiHackScanner().scan_files(texts)
    if guard:
        first = guard[0]
        return f"L0 {first.rule}@{first.path}:{first.lineno}: {first.detail}"
    return None


def _run_training(runner: Path, model_dir: Path, seconds: float) -> str | None:
    result = Sandbox().run(
        [sys.executable, str(runner), str(model_dir)],
        workdir=model_dir,
        timeout_s=seconds + TRAIN_SLACK_S,
        env=_subprocess_env({"MODMUL_TRAIN_SECONDS": str(int(seconds))}),
    )
    if result.timed_out:
        return (
            f"train-timeout: ignored the {seconds:.0f}s budget and was killed "
            f"at {seconds + TRAIN_SLACK_S:.0f}s"
        )
    if result.return_code != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-6:])
        return f"train-failed (exit {result.return_code}): {tail[:500]}"
    return None


@contextlib.contextmanager
def _exclusive_gpu(lock_path: Path | None):
    """Hold the GPU alone for a TIMED inference.

    eval_batch_size is 16, so sixteen candidates train and evaluate on one
    card at once, and the seconds this harness records are the seconds under
    that load -- while the official evaluation runs one model alone. Measured
    on the same weights and the same problems: the diagnostic tier took 61.4s
    alone and 146.3s under r10's load, tier 9 took 114.8s alone and 235.0s.
    Roughly 2x, varying with however many siblings happen to be running.

    That is not a small inaccuracy. The wall clock is the only thing standing
    between this task and its top tier, _fitness now prices it, and a number
    that moves with the neighbours means selection is partly selecting on
    noise -- and systematically rejecting candidates the official harness
    would have passed.

    Only the timed section is serialised. Training stays concurrent, which is
    where the throughput actually comes from, and the lower rungs stay
    concurrent too: their inference is seconds, and their timings feed nothing
    but the promotion gate.
    """
    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _run_eval(
    runner: Path,
    model_dir: Path,
    plan: dict,
    workdir: Path,
    tag: str,
    gpu_lock: Path | None = None,
) -> tuple[dict | None, str]:
    plan_path = workdir / f"plan_{tag}.json"
    out_path = workdir / f"eval_{tag}.json"
    plan_path.write_text(json.dumps(plan))
    with _exclusive_gpu(gpu_lock):
        result = Sandbox().run(
            [sys.executable, str(runner), str(model_dir), str(plan_path),
             str(out_path)],
            workdir=model_dir,
            timeout_s=(
                plan["budget_s"] * INFERENCE_KILL_FACTOR + LOAD_ALLOWANCE_S
            ),
            env=_subprocess_env(),
        )
    if out_path.exists():
        try:
            return json.loads(out_path.read_text()), ""
        except json.JSONDecodeError:
            pass
    if result.timed_out:
        return None, "eval-timeout: model exceeded the inference wall clock"
    tail = "\n".join(result.stderr.strip().splitlines()[-6:])
    return None, f"eval-crashed (exit {result.return_code}): {tail[:500]}"


def _score_tiers(
    result: dict,
    truth: dict[int, list[dict]],
    tiers: tuple[int, ...],
) -> tuple[dict[int, float], list[dict], dict[int, float]]:
    """Compare decoded predictions against ground truth (parent side only)."""
    accuracy: dict[int, float] = {}
    seconds: dict[int, float] = {}
    items: list[dict] = []
    for tier in tiers:
        cases = truth[tier]
        payload = result["tiers"].get(str(tier))
        if payload is None:
            accuracy[tier] = 0.0
            items.append({
                "item_id": f"t{tier}#skipped",
                "passed": False,
                "predicted": "",
                "expected": "",
                "error_category": f"tier-not-reached-t{tier}",
            })
            continue
        seconds[tier] = round(float(payload.get("seconds", 0.0)), 3)
        predictions = payload.get("predictions", [])
        if not payload.get("complete", False):
            # Official policy: an incomplete tier scores 0, partials discarded.
            accuracy[tier] = 0.0
            items.append({
                "item_id": f"t{tier}#incomplete",
                "passed": False,
                "predicted": "",
                "expected": "",
                "error_category": (
                    f"{payload.get('note') or 'tier-incomplete'}-t{tier}"
                ),
            })
            continue
        correct = 0
        for index, case in enumerate(cases):
            predicted = predictions[index] if index < len(predictions) else ""
            if predicted == case["expected"]:
                correct += 1
                category = ""
            elif predicted == "!malformed":
                category = f"malformed-output-t{tier}"
            elif predicted.startswith("!"):
                category = f"predict-{predicted[1:]}"
            elif not predicted:
                category = f"missing-output-t{tier}"
            else:
                category = f"wrong-answer-t{tier}"
            if index < MAX_FEEDBACK_ITEMS_PER_TIER:
                items.append({
                    "item_id": f"t{tier}#{index}",
                    "passed": category == "",
                    "predicted": predicted[:40],
                    "expected": case["expected"][:40],
                    "error_category": category,
                })
        accuracy[tier] = correct / len(cases) if cases else 0.0
    return accuracy, items, seconds


def _h90(accuracy: dict[int, float]) -> int:
    hits = [t for t in SCORED_TIERS if accuracy.get(t, 0.0) >= 0.90]
    return max(hits) if hits else 0


def _overall(accuracy: dict[int, float]) -> float:
    """官方 overall_accuracy：tier 1-10 等权；未评/超时的层记 0。"""
    return sum(accuracy.get(t, 0.0) for t in SCORED_TIERS) / len(SCORED_TIERS)


def leaderboard_key(accuracy: dict[int, float]) -> tuple[int, float]:
    """官方排序键 (H90, overall_accuracy)。**报告和终选用这个**。

    与 fitness 分开:排名要的是"跨没跨过阈值"这个离散事实,搜索要的是
    "离阈值还有多远"这个连续量。把两者压进同一个 float,得到的是排名
    正确、但搜索无坡可爬的地形。"""
    return (_h90(accuracy), _overall(accuracy))


# 阈值软化尺度。0.15 意味着精度 0.75 已能拿到约 16% 的跨越奖励,0.9 拿一半
# —— 足够远地伸出去,让"还差得远"的层也有方向,而不是只在阈值边缘有梯度。
_THRESHOLD_TAU = 0.15
# 跨越奖励相对于连续精度的权重。总权重 0.3 x 10 层 = 3,连续项权重 1,
# 所以跨阈值仍然更值钱,但只是 ~2 倍,不是原来的 50 倍。
_THRESHOLD_WEIGHT = 0.3


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _soft_h90(accuracy: dict[int, float]) -> float:
    """连续版的"有多少层跨过了 90%"。"""
    return sum(
        _sigmoid((accuracy.get(tier, 0.0) - 0.90) / _THRESHOLD_TAU)
        for tier in SCORED_TIERS
    )


# sigmoid 在 0 处不为 0,所以"什么都不会"的候选会拿到一个正的软奖励。减去
# 这个地板并按跨度归一,让 fitness 在全 0 时确实是 0、全对时确实是 1 ——
# 否则一个毫无长进的候选看起来比一个崩溃的候选(硬记 0)更值钱。
_SOFT_FLOOR = len(SCORED_TIERS) * _sigmoid(-0.90 / _THRESHOLD_TAU)
_SOFT_CEIL = len(SCORED_TIERS) * _sigmoid(0.10 / _THRESHOLD_TAU)


_TIME_WEIGHT = 0.15

# Credit for how close the clock is when tiers were SKIPPED for want of
# budget. Without it the all-zero case is a plateau: when the diagnostic tier
# exhausts the budget every candidate in the lineage scores exactly 0, however
# near it is to fixing that, and the search has no direction at all. Run
# modmul_r9 spent eight generations on that plateau and walked off it in the
# wrong direction -- a candidate that discarded 323,546 training steps scored
# 0.10 against the fully-trained seed's 0.00.
#
# This does NOT rescue such a lineage from a rival that actually scores:
# measured on r9's own numbers, the seed reaches 0.0418 against the cold
# start's 0.1043. Raising the weight until it did would make "almost ran"
# worth as much as crossing a tier, which is worse. It turns a plateau into a
# slope; the plateau's cause is fixed elsewhere.
_STARVED_WEIGHT = 0.15


def _starved_reach(
    accuracy: dict[int, float],
    seconds: dict[int, float] | None,
    budget: float | None,
    rung_tiers: tuple[int, ...] = SCORED_TIERS,
    clock_s: float | None = None,
) -> float:
    """How near the clock came to letting the skipped tiers run. 0 if none
    were skipped, or if they were skipped for any reason but time.

    A tier that RAN and scored zero earns nothing here -- that is a model
    answering wrongly, and paying it for being quick is the gaming vector an
    existing test already guards. This pays only for tiers never asked.
    """
    if not budget or budget <= 0:
        return 0.0
    spent = (
        clock_s if clock_s is not None else sum((seconds or {}).values())
    )
    skipped = [t for t in rung_tiers if t not in (seconds or {})]
    if not skipped or spent < budget:
        return 0.0
    return (budget / spent) * (len(skipped) / len(rung_tiers))


def _time_factor(
    seconds: dict[int, float] | None,
    budget: float | None,
    clock_s: float | None = None,
) -> float:
    """How close the clock is to letting the NEXT tier run. 1.0 = no obstacle.

    The wall clock is not invisible to selection -- a candidate that fits
    tiers 1-10 inside the budget scores tier 10 and jumps a whole h90 level.
    The problem is that this is a CLIFF, and cliffs are the exact terrain
    _fitness was written to fix on the accuracy axis: nearly flat, with one
    step nobody can climb by small mutations. Nothing was ever done for the
    time axis, so "38% slower at identical accuracy" cost run modmul_r8's
    candidate 8ab3347e precisely nothing, and "12% faster" would have earned
    nothing either.

    Measured, not projected. budget_headroom from the cost probe would have
    been the natural input and is not usable: it read 0.047 for the seed --
    a 21x speedup demanded -- where the seed's own R2 timings put the true
    figure near 3.5x, and it read 53.0 for a tier-3-capped seed that is
    "fast" only because it never reaches anything expensive. Only 8 of 67
    candidates in modmul_r8 had the probe run at all.

    sum(seconds) is the clock at the moment the first unrun tier would have
    started, so budget/that is exactly the fraction of the way there, and its
    reciprocal is the speedup still required. Above 1.0 -- a lower ASHA rung
    that simply does not score the upper tiers -- it clamps to neutral and
    leaves promotion undistorted.

    An earlier version returned neutral as soon as every scored tier had run,
    on the reasoning that there was then nothing left to unlock. Measurement
    killed that: the official timer is checked only BETWEEN batches, and a
    model batching a whole tier at once gets exactly one check per tier, at
    its start. So tier 10 begins at a clock of ~190s, is never checked again,
    and runs 788 seconds to completion -- every tier runs, and the run takes
    978 seconds against a 300-second budget. Going neutral there would switch
    the pressure off at precisely the point it has the most work to do, and
    would score 978s and 400s identically.
    """
    if not budget or budget <= 0:
        return 1.0
    # The FULL clock, which includes the unscored diagnostic tier. `seconds`
    # only ever holds the scored tiers, and on this task the diagnostic one is
    # the most expensive item in the run -- reading the clock off `seconds`
    # under-counted it by 78%.
    spent = clock_s if clock_s is not None else sum((seconds or {}).values())
    if not spent or spent <= 0:
        return 1.0
    return min(1.0, budget / spent)


def _fitness(
    accuracy: dict[int, float],
    seconds: dict[int, float] | None = None,
    budget: float | None = None,
    rung_tiers: tuple[int, ...] | None = None,
    clock_s: float | None = None,
) -> float:
    """**搜索信号**,不是排名键 —— 严格随每一层精度递增。

    原来的 `(h90 + overall) / 11` 与官方排序同构,但作为演化搜索的地形是
    几乎最差的形状。实测(QUICK,limb_horner 种子):

        某层精度翻倍 0.2 -> 0.4   fitness +0.0018
        跨过某层 0.90 阈值         fitness +0.091      (50 倍)

    即几乎全平 + 偶尔悬崖。变异能提供的是小步改进,而小步改进在这个地形
    上几乎不产生选择压力,于是搜索在平地上随机游走,直到偶然撞上悬崖。

    这里换成:连续的 overall 打底,加一个**软化**的跨阈值奖励。跨阈值仍然
    比堆低层精度值钱(约 2 倍),但每一层精度的每一点提升都有回报。

    MODMUL_FITNESS=leaderboard 恢复旧定义,用于 A/B。
    """
    if os.environ.get("MODMUL_FITNESS") == "leaderboard":
        return (_h90(accuracy) + _overall(accuracy)) / (len(SCORED_TIERS) + 1)
    bonus = _THRESHOLD_WEIGHT * (_soft_h90(accuracy) - _SOFT_FLOOR)
    # Weighted at half the threshold bonus on purpose: crossing a tier's 90%
    # line is what the leaderboard actually pays for, so speed must never
    # outrank it. This term only has to make the approach to the cliff
    # climbable.
    # Scaled by overall accuracy, or a candidate that answers NOTHING collects
    # the full speed bonus for being trivially fast — which is how the first
    # version of this scored 0.048 on a model emitting malformed digits, and
    # broke the invariant that an all-zero candidate is worth exactly 0.
    # Speed is only worth anything if there is accuracy to convert with it.
    speed = _TIME_WEIGHT * _time_factor(
        seconds, budget, clock_s
    ) * _overall(accuracy)
    reach = _STARVED_WEIGHT * _starved_reach(
        accuracy, seconds, budget, rung_tiers or SCORED_TIERS, clock_s
    )
    span = (
        1.0
        + _THRESHOLD_WEIGHT * (_SOFT_CEIL - _SOFT_FLOOR)
        + _TIME_WEIGHT
        + _STARVED_WEIGHT
    )
    return max(0.0, (_overall(accuracy) + bonus + speed + reach) / span)


def _empty_feedback(summary: str) -> dict:
    return {"schema_version": 1, "items": [], "summary": summary}


def _perturbation_verdict(
    runner: Path,
    model_dir: Path,
    workdir: Path,
    accuracy: dict[int, float],
) -> tuple[str | None, dict]:
    """官方 L3 的自测版：随机化权重后精度必须塌。

    只对"强到值得怀疑"的候选跑 —— 弱候选塌不塌没有信息量，白花时间。
    返回 (fault | None, metrics)。"""
    strong = _h90(accuracy) >= 1 or any(
        tier >= 2 and value >= PERTURB_TRIGGER_TIER_ACC
        for tier, value in accuracy.items()
    )
    probes = sorted(
        (t for t in accuracy if accuracy[t] >= PERTURB_MIN_TIER_ACC),
        key=lambda t: -t,
    )[:3]
    if not strong or not probes:
        return None, {"perturbation": "skipped-weak-candidate"}

    truth = {t: _load_cases(t, PERTURB_PROBE_CASES) for t in probes}
    plan = {
        "mode": "perturb",
        "budget_s": SECONDS_PER_PROBLEM * PERTURB_PROBE_CASES * len(probes) * 4,
        "order": [str(t) for t in sorted(probes)],
        "cases": {str(t): _inputs_only(truth[t]) for t in probes},
    }
    result, fault = _run_eval(runner, model_dir, plan, workdir, "perturb")
    if result is None or result.get("error"):
        # A crash under randomized weights is itself evidence of dependence.
        return None, {"perturbation": "collapsed-to-error",
                      "perturbation_detail": (fault or result.get("error"))[:200]}

    randomized = result.get("randomized_tensors")
    if not randomized:
        return (
            "perturbation-insensitive (L3): no trained tensors found — a "
            "submission with no trained parameters is a circuit, not a model",
            {"perturbation": "no-trained-parameters"},
        )

    perturbed, _, _ = _score_tiers(result, truth, tuple(sorted(probes)))
    original_mean = sum(accuracy[t] for t in probes) / len(probes)
    perturbed_mean = sum(perturbed.get(t, 0.0) for t in probes) / len(probes)
    metrics = {
        "perturbation": "checked",
        "perturbation_tensors": randomized,
        "perturbation_original_acc": round(original_mean, 4),
        "perturbation_random_acc": round(perturbed_mean, 4),
    }
    if perturbed_mean > PERTURB_MAX_SURVIVING_RATIO * original_mean:
        return (
            "perturbation-insensitive (L3): randomizing the weights left "
            f"accuracy at {perturbed_mean:.2f} vs {original_mean:.2f} — the "
            "answer is not coming from trained parameters",
            metrics,
        )
    return None, metrics


COST_PROBE_TIERS = (9, 10)
COST_PROBE_CASES = 3


def _cost_projection(
    runner: Path,
    model_dir: Path,
    workdir: Path,
    measured: dict[int, float],
    measured_cases: int,
) -> dict:
    """Project whether the top tiers can be answered inside the time budget.

    The budget is the binding constraint at tiers 9-10 and it is invisible
    until R2, ninety minutes in. Measured on the limb_horner seed: tiers 1-8
    were all at 100% and tier 9 at 92%, yet tier 10 scored 0 without running
    a single case, because tier 9 alone consumed 83% of the whole budget.
    Nothing in the feedback said so — `infer_s_tier_10` was simply absent.

    Timing does not depend on accuracy, so a handful of cases answers it. Run
    once at the first promotion, this turns a ninety-minute surprise into a
    dense signal available in the first few minutes, on the axis that
    actually decides how far a candidate can reach.
    """
    truth = {t: _load_cases(t, COST_PROBE_CASES) for t in COST_PROBE_TIERS}
    truth = {t: cases for t, cases in truth.items() if cases}
    if not truth:
        return {}
    tiers = tuple(sorted(truth))
    total = sum(len(c) for c in truth.values())
    plan = {
        # Generous: this measures cost, so it must not be cut off by the very
        # budget it is measuring.
        "mode": "normal",
        "budget_s": SECONDS_PER_PROBLEM * total * 100,
        "order": [str(t) for t in tiers],
        "cases": {str(t): _inputs_only(truth[t]) for t in tiers},
    }
    result, fault = _run_eval(runner, model_dir, plan, workdir, "costprobe")
    if result is None or result.get("error"):
        return {"cost_probe": "unavailable"}
    out: dict = {"cost_probe": "checked"}
    scored_cases = RUNGS[-1].cases
    # The budget is ONE shared pool for the whole set, so the cheap low tiers
    # subsidise the expensive high ones. Comparing the top tiers against a
    # pro-rata slice of the budget said 13x over when the truth was 2.9x — a
    # 4x exaggeration on the one signal that matters. Project the whole set:
    # the tiers already measured this rung, rescaled to the final case count,
    # plus the probed cost of the top tiers.
    projected = {
        tier: seconds / measured_cases * scored_cases
        for tier, seconds in measured.items()
        # A tier that timed as zero carries no rate to extrapolate from, and
        # dividing by it is how the first version of this crashed.
        if measured_cases > 0 and seconds > 0
    }
    for tier in tiers:
        payload = result["tiers"].get(str(tier))
        if not payload:
            continue
        per_case = float(payload.get("seconds", 0.0)) / len(truth[tier])
        out[f"probe_s_per_case_tier_{tier}"] = round(per_case, 4)
        out[f"projected_infer_s_tier_{tier}"] = round(per_case * scored_cases, 1)
        if per_case > 0:
            projected[tier] = per_case * scored_cases
    # Tiers between the measured ones and the probed ones are unknown here;
    # cost grows with operand width, so interpolate each gap geometrically
    # rather than pretending the gap is free.
    known = sorted(projected)
    for low, high in zip(known, known[1:]):
        gap = [t for t in range(low + 1, high) if t in SCORED_TIERS]
        if not gap or projected[low] <= 0:
            continue
        ratio = (projected[high] / projected[low]) ** (1 / (high - low))
        for step, tier in enumerate(gap, start=1):
            projected[tier] = projected[low] * ratio ** step
    total = sum(projected.values())
    if total <= 0:
        # Nothing measurable — say so rather than reporting a fabricated
        # headroom, which is exactly what a mutation would learn to game.
        return {"cost_probe": "unmeasurable"}
    out["projected_infer_s_all_tiers"] = round(total, 1)
    allowed = SECONDS_PER_PROBLEM * (
        scored_cases * (len(SCORED_TIERS) + 1)      # +1: the tier-0 diagnostic
    )
    if total > 0:
        # Below 1.0: the top tiers cannot be answered in time however accurate
        # the model becomes. 1/headroom is the speedup required.
        out["budget_headroom"] = round(allowed / total, 3)
    return out


def _holdout_metrics(
    runner: Path,
    model_dir: Path,
    workdir: Path,
    cases_per_tier: int,
) -> dict:
    """私有种子集复测（可选）：检测"只在公开集上涨分"的过拟合。

    集合由 `python -m modmul.holdout` 一次性生成；没生成就跳过。"""
    truth = {t: _load_cases(t, cases_per_tier, HOLDOUT_DIR) for t in SCORED_TIERS}
    truth = {t: cases for t, cases in truth.items() if cases}
    if not truth:
        return {"holdout": "unavailable"}
    tiers = tuple(sorted(truth))
    total = sum(len(c) for c in truth.values())
    plan = {
        "mode": "normal",
        "budget_s": SECONDS_PER_PROBLEM * total,
        "order": [str(t) for t in tiers],
        "cases": {str(t): _inputs_only(truth[t]) for t in tiers},
    }
    result, fault = _run_eval(runner, model_dir, plan, workdir, "holdout")
    if result is None or result.get("error"):
        return {"holdout": f"failed: {(fault or result.get('error'))[:120]}"}
    accuracy, _, _ = _score_tiers(result, truth, tiers)
    return {
        "holdout": "checked",
        "holdout_h90": _h90(accuracy),
        "holdout_overall": round(_overall(accuracy), 4),
        **{f"holdout_acc_tier_{t}": round(a, 4) for t, a in accuracy.items()},
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


_SEED_ROOT = _HERE / "seeds"


def _refuse_to_grade_the_seeds(candidate_dir: Path) -> None:
    """Grading trains IN PLACE, so the seeds must never be graded directly.

    train() is resumable by contract: it continues from whatever weights sit
    in model_dir and writes back there. Point this function at a seed and it
    quietly rewrites the genome every future run starts from — observed
    2026-07-26, when a baseline script pushed limb_horner from 5,577 steps
    to 37,088 and left the repo dirty. Nothing failed, and the measurement
    silently stopped being a baseline. Callers must grade a copy.
    """
    try:
        candidate_dir.relative_to(_SEED_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(
        f"refusing to grade {candidate_dir}: it is inside {_SEED_ROOT}, and "
        "grading trains in place. Copy the seed to a temporary directory "
        "and grade the copy."
    )


_INHERITED = ("weights.pt", "optimizer.pt", "train_state.json")


def _digest(candidate_dir: Path, *names: str) -> str:
    """Hash of the named genome files, in order."""
    sha = hashlib.sha256()
    for name in names:
        path = candidate_dir / name
        sha.update(path.read_bytes() if path.exists() else b"")
    return sha.hexdigest()


def _arch_digest(candidate_dir: Path) -> str:
    """What decides whether the parent's weights can be loaded at all."""
    return _digest(candidate_dir, "arch.py")


def _training_digest(candidate_dir: Path) -> str:
    """What decides whether training would do anything new.

    `train.py` imports only from `arch.py`, never from `model.py`, so a
    mutation that touches only the inference contract produces byte-identical
    training. Paying ninety minutes to re-derive weights we already hold is
    pure waste — and the largest known win in this domain is an inference-time
    setting with no retraining at all, which is exactly the axis that waste
    falls on.
    """
    return _digest(candidate_dir, "arch.py", "train.py")


def _pretrained_dir(lineage_dir: Path, training_digest: str) -> Path:
    """Content-addressed store of already-trained weights.

    Keyed by what produced them rather than by who produced them, so a seed
    with no parent, a child that changed nothing about training, and a second
    island that converged on the same recipe all hit the same entry.
    """
    return Path(lineage_dir) / "_pretrained" / training_digest


def _inherit_from_parent(candidate_dir: Path, ctx: GradeContext) -> dict:
    """Carry trained weights along the lineage when the architecture is
    unchanged.

    Weights are not in the genome — the genome is three text files — so
    without this every candidate trains from random initialisation and the
    run discards its entire compute budget: 120 generations x 90 minutes of
    training, each throwing away the last. The public h90=10 submission got
    there the other way, warm-starting a trained model and annealing it
    repeatedly, and this seed already needs more training than one rung can
    buy (per-step error 1.6e-5 at tier 9, where tier 10 wants ~1e-5).

    Weights are copied in whenever a source exists; whether they are USABLE
    is settled per tensor by the candidate's own loader, which keeps every
    tensor whose name and shape still match. That matters because changing
    RADIX_BITS reshapes exactly one matrix — 896 of 91,841 parameters — and
    an all-or-nothing rule would reject the whole checkpoint over 1%, making
    the highest-value mutation the most expensive one to try.

    Two sources, in order: the parent, and failing that a content-addressed
    store keyed by what produced the weights rather than by who. The second
    is what lets a seed skip re-deriving weights already measured, and lets
    two islands that converge on the same recipe share training.

    Reported so it can be read back later, because inheritance makes fitness
    path-dependent — a lineage's score then reflects accumulated compute and
    not only its genome.
    """
    if not ctx.lineage_dir:
        return {"warm_start": "cold", "warm_start_why": "no-lineage-dir",
                "inherited_steps": 0}
    arch = _arch_digest(candidate_dir)
    source = None
    origin = "cold"
    # Why a candidate started cold matters as much as that it did: the
    # best-reasoned offspring of run modmul_r1 collapsed from 0.846 to 0.213
    # purely because it started cold, and nothing recorded which of these
    # branches sent it there.
    # Nearest first: the parent, then ITS parent, and so on. A candidate that
    # failed evaluation publishes nothing, so without the walk-up its children
    # start from random initialisation however much the lineage had banked.
    # Run modmul_r9 lost 323,546 steps in three generations exactly that way --
    # one candidate faulted, its repair child cold-started at h90=1, and that
    # cold start OUTSCORED the fully-trained seed, which was scoring 0 for an
    # unrelated reason. From there the whole island descended from noise.
    chain = [ctx.parent_id, *ctx.ancestor_ids] if ctx.parent_id else list(
        ctx.ancestor_ids
    )
    seen: set[str] = set()
    chain = [a for a in chain if a and not (a in seen or seen.add(a))]
    if not chain:
        why = "no-parent"
    else:
        why = "parent-published-no-weights"
        for depth, ancestor_id in enumerate(chain):
            candidate_source = Path(ctx.lineage_dir) / ancestor_id
            if (candidate_source / "weights.pt").exists():
                source = candidate_source
                origin = "parent" if depth == 0 else f"ancestor{depth}"
                why = (
                    "parent-weights" if depth == 0
                    else f"ancestor-weights(+{depth})"
                )
                break
    if source is None:
        cached = _pretrained_dir(ctx.lineage_dir, _training_digest(candidate_dir))
        if (cached / "weights.pt").exists():
            source, origin, why = cached, "pretrained", "pretrained-cache"
        else:
            why += "+no-cache-for-this-recipe"
    if source is None:
        return {"warm_start": "cold", "warm_start_why": why,
                "inherited_steps": 0}

    for name in _INHERITED:
        path = source / name
        if path.exists():
            shutil.copy2(path, candidate_dir / name)
    steps = 0
    state = candidate_dir / "train_state.json"
    if state.exists():
        try:
            steps = int(json.loads(state.read_text()).get("steps", 0))
        except (ValueError, TypeError):
            steps = 0
    recorded = (source / "arch.sha256")
    same_arch = recorded.exists() and recorded.read_text().strip() == arch
    return {
        # `partial` is not a failure: the loader keeps what fits, and the
        # candidate only has to relearn the tensors the mutation reshaped.
        "warm_start": f"{origin}-full" if same_arch else f"{origin}-partial",
        "warm_start_why": why,
        "inherited_steps": steps,
    }


def _cached_train_seconds(candidate_dir: Path, ctx: GradeContext) -> float:
    """How much training the reusable weights for this recipe represent.

    Training is cumulative, so "same recipe" is not enough to skip it — the
    same recipe run longer gives a better model. Without this the first
    candidate to die at R0 would fill the store with 480 seconds of training,
    and every later candidate sharing that recipe would silently inherit an
    undertrained model and look bad for reasons having nothing to do with its
    own mutation.
    """
    if not ctx.lineage_dir:
        return 0.0
    cached = _pretrained_dir(ctx.lineage_dir, _training_digest(candidate_dir))
    marker = cached / "train_seconds"
    if not (cached / "weights.pt").exists() or not marker.exists():
        return 0.0
    try:
        return float(marker.read_text().strip())
    except ValueError:
        return 0.0


def _may_skip_training(
    candidate_dir: Path, ctx: GradeContext, through_seconds: float
) -> bool:
    """True when training to `through_seconds` would only reproduce weights
    already held.

    `train.py` imports from `arch.py` and never from `model.py`, so a
    mutation confined to the inference contract trains to byte-identical
    weights. Ninety minutes to re-derive them is the single largest waste in
    the loop — and it falls on the axis where the largest known win lives,
    an inference-time setting requiring no retraining at all.
    """
    return _cached_train_seconds(candidate_dir, ctx) >= through_seconds


def _publish_to_lineage(
    candidate_dir: Path, ctx: GradeContext, train_seconds: float
) -> None:
    """Hand these trained weights to children, and to anything that would
    otherwise re-derive them."""
    if not ctx.lineage_dir:
        return
    targets = [Path(ctx.lineage_dir) / ctx.candidate_id]
    cached = _pretrained_dir(ctx.lineage_dir, _training_digest(candidate_dir))
    # Fill the shared entry only when this run trained the recipe FURTHER than
    # whatever is there. Otherwise a candidate cut off at the first rung would
    # overwrite a fully trained entry with its own undertrained weights.
    if train_seconds > _cached_train_seconds(candidate_dir, ctx):
        targets.append(cached)
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for name in _INHERITED:
            source = candidate_dir / name
            if source.exists():
                shutil.copy2(source, target / name)
        (target / "arch.sha256").write_text(_arch_digest(candidate_dir) + "\n")
        (target / "train_seconds").write_text(f"{train_seconds:.1f}\n")


def grade_workspace(candidate_dir: Path, ctx: GradeContext) -> Grade:
    started = time.monotonic()
    candidate_dir = Path(candidate_dir).resolve()
    _refuse_to_grade_the_seeds(candidate_dir)
    workdir = Path(ctx.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. 裁定先于任何执行 —— import 就是执行代码，顺序是承重的。
    fault = _adjudicate(candidate_dir)
    if fault:
        return Grade(
            fitness=0.0, passed=False, stage_reached=0, fault=fault,
            structured_feedback=_empty_feedback(fault),
        )
    if not (candidate_dir / "model.py").exists():
        return Grade(
            fitness=0.0, passed=False, stage_reached=0,
            fault="missing model.py (the inference contract file)",
            structured_feedback=_empty_feedback("candidate has no model.py"),
        )

    # Runner 脚本写在候选目录**外**：否则它们会被下一轮静态检查扫到，
    # 也会污染 workspace 的 diff。
    train_runner = workdir / "train_runner.py"
    eval_runner = workdir / "eval_runner.py"
    train_runner.write_text(_TRAIN_RUNNER)
    eval_runner.write_text(_EVAL_RUNNER)

    # After adjudication (importing candidate code is execution, and the
    # order is load-bearing), before training: inherit the parent's weights
    # so training continues rather than restarting.
    lineage = _inherit_from_parent(candidate_dir, ctx)
    lineage["training_skipped"] = 0

    accuracy: dict[int, float] = {}
    items: list[dict] = []
    seconds: dict[int, float] = {}
    metrics: dict = {}
    # NOT `metrics`: that one is reassigned wholesale at the top of every
    # rung, so anything written into it at the end of a rung is wiped by the
    # next. The cost probe ran once and its results silently vanished.
    projection: dict = {}
    diagnostic: dict = {}
    budget_s: float | None = None      # the last rung's clock, for _fitness
    reached = _rungs()[0]
    train_spent = 0.0
    for index, rung in enumerate(_rungs()):
        reached = rung
        # Per rung, not once: the stored weights cover a specific amount of
        # training, so a candidate reaching further than they go still has to
        # pay for the difference.
        if _may_skip_training(
            candidate_dir, ctx, train_spent + rung.train_seconds
        ):
            train_spent += rung.train_seconds
            lineage["training_skipped"] += 1
        else:
            fault = _run_training(train_runner, candidate_dir, rung.train_seconds)
            if fault:
                return Grade(
                    fitness=_fitness(accuracy), passed=False, stage_reached=1,
                    fault=fault,
                    visible_metrics={"rung": rung.name, **_metric_block(accuracy)},
                    structured_feedback=_empty_feedback(
                        f"training failed at {rung.name}: {fault[:160]}"
                    ),
                    execution_time=time.monotonic() - started,
                )
            train_spent += rung.train_seconds

        tiers = rung.tiers
        truth = {t: _load_cases(t, rung.cases) for t in tiers}
        order = [str(t) for t in tiers]
        cases = {str(t): _inputs_only(truth[t]) for t in tiers}
        total = sum(len(c) for c in truth.values())
        if rung.diagnostic:
            # 官方顺序：tier 0 先跑，且计入同一个墙钟预算。
            # rung.cases, not a hardcoded 20: tier 0 is one of the official
            # eleven tiers and carries the same 100 problems as the others.
            diagnostic_cases = _load_cases(DIAGNOSTIC_TIER, rung.cases)
            if diagnostic_cases:
                order = ["0", *order]
                cases["0"] = _inputs_only(diagnostic_cases)
                total += len(diagnostic_cases)

        budget_s = SECONDS_PER_PROBLEM * total
        plan = {
            "mode": "normal",
            "budget_s": budget_s,
            "order": order,
            "cases": cases,
        }
        # The top rung's timings are what _fitness prices and what decides
        # which tiers get to run at all, so that one measurement takes the
        # card alone. The cheap rungs stay concurrent -- their inference is
        # seconds and feeds only the promotion gate.
        result, fault = _run_eval(
            eval_runner, candidate_dir, plan, workdir, rung.name.lower(),
            gpu_lock=(
                Path(ctx.lineage_dir) / "timed-inference.lock"
                if rung is _rungs()[-1] and ctx.lineage_dir
                else None
            ),
        )
        if result is None:
            return Grade(
                fitness=0.0, passed=False, stage_reached=2, fault=fault,
                visible_metrics={"rung": rung.name},
                structured_feedback=_empty_feedback(fault),
                execution_time=time.monotonic() - started,
            )
        if result.get("error"):
            fault = f"{result['error_kind']}: {result['error']}"
            return Grade(
                fitness=0.0, passed=False, stage_reached=2, fault=fault,
                visible_metrics={"rung": rung.name},
                structured_feedback=_empty_feedback(fault),
                execution_time=time.monotonic() - started,
            )

        accuracy, items, seconds = _score_tiers(result, truth, tiers)
        metrics = {
            "rung": rung.name,
            "params": result.get("params", 0),
            "load_seconds": round(float(result.get("load_seconds", 0.0)), 2),
            "train_seconds": int(train_spent),
            "inference_budget_s": round(plan["budget_s"], 1),
        }
        if rung.diagnostic and "0" in result["tiers"]:
            zero_cases = _load_cases(DIAGNOSTIC_TIER, rung.cases)
            zero_acc, _, _ = _score_tiers(
                {"tiers": {"0": result["tiers"]["0"]}},
                {0: zero_cases},
                (0,),
            )
            # Tier 0's SECONDS, not just its accuracy. It scores nothing --
            # "diagnostic only and not counted toward either metric" -- but it
            # runs FIRST and it is on the same clock, and _score_tiers is only
            # ever handed the scored tiers, so its cost was reported nowhere.
            # Measured on the seed at the official calibration: 243.9s of a
            # 300s budget, 81% of the whole allowance, spent before tier 1
            # begins. That is why tier 9 and tier 10 do not run. The single
            # largest cost in the run was the one item the search could not
            # see.
            metrics["infer_s_tier_0"] = round(
                float(result["tiers"]["0"].get("seconds", 0.0)), 3
            )
            diagnostic = {"diag_tier0_acc": round(zero_acc.get(0, 0.0), 4)}

        if rung.perturbation:
            verdict, perturb_metrics = _perturbation_verdict(
                eval_runner, candidate_dir, workdir, accuracy
            )
            metrics.update(perturb_metrics)
            if verdict:
                return Grade(
                    fitness=0.0, passed=False, stage_reached=3, fault=verdict,
                    visible_metrics={**metrics, **_metric_block(accuracy)},
                    hidden_metrics=diagnostic,
                    structured_feedback=_empty_feedback(verdict),
                    execution_time=time.monotonic() - started,
                )
        if rung.holdout:
            diagnostic.update(
                _holdout_metrics(eval_runner, candidate_dir, workdir, 20)
            )

        stratum = (
            "stable" if "full" in str(lineage.get("warm_start", "")) else "arch"
        )
        promoted, promo_info = _promotes(
            index, accuracy, rung, stratum, ctx.lineage_dir, ctx.generation
        )
        metrics.update(promo_info)
        _record_rung_score(
            ctx.lineage_dir, stratum, rung, ctx.generation,
            promo_info["promotion_score"],
        )
        if not promoted:
            break
        if index == 0 and not projection:
            # At the first promotion: cheap enough to be worth it only for a
            # candidate that survived R0, early enough to matter.
            projection = _cost_projection(
                eval_runner, candidate_dir, workdir, seconds, rung.cases
            )
            if projection.get("budget_headroom", 1.0) < 1.0:
                diagnostic["budget_verdict"] = (
                    "top tiers project over the time budget; accuracy there "
                    "cannot be scored until inference gets faster"
                )

    # Hand the trained weights to this candidate's children. Only done on the
    # success path: a candidate that faulted has nothing worth inheriting.
    _publish_to_lineage(candidate_dir, ctx, train_spent)

    visible = {**metrics, **projection, **lineage, **_metric_block(accuracy),
               **{f"infer_s_tier_{t}": s for t, s in seconds.items()}}
    return Grade(
        fitness=_fitness(
            accuracy, seconds, budget_s, reached.tiers,
            # The clock the official timer would read: every tier that ran,
            # scored or not.
            clock_s=sum(seconds.values()) + metrics.get("infer_s_tier_0", 0.0),
        ),
        visible_metrics=visible,
        hidden_metrics=diagnostic,
        structured_feedback={
            "schema_version": 1,
            "items": items,
            "summary": _summary(
                accuracy, reached, seconds, budget_s,
                metrics.get("infer_s_tier_0"),
            ),
        },
        execution_time=time.monotonic() - started,
    )


def _metric_block(accuracy: dict[int, float]) -> dict:
    return {
        "h90": _h90(accuracy),
        "overall_accuracy": round(_overall(accuracy), 4),
        **{f"acc_tier_{t}": round(a, 4) for t, a in sorted(accuracy.items())},
    }


def _summary(
    accuracy: dict[int, float],
    rung: Rung,
    seconds: dict,
    budget_s: float | None = None,
    tier0_s: float | None = None,
) -> str:
    """The feedback the next mutation actually reads.

    Two things it used to get wrong. It described fitness as
    "(H90 + overall)/11", which stopped being true when _fitness was softened
    and is now doubly untrue with a wall-clock term -- the search was told the
    wrong objective. And it reported the slowest SCORED tier, while the
    largest cost in the run belongs to tier 0, which is scored by nobody and
    was in no dictionary the summary could see: 243.9s of a 300s budget on the
    seed, spent before tier 1 begins.
    """
    per_tier = ", ".join(
        f"t{t}={accuracy[t]:.0%}" for t in sorted(accuracy)
    )
    clock = dict(seconds)
    if tier0_s:
        clock[0] = tier0_s
    cost = ""
    if clock:
        spent = sum(clock.values())
        slowest = max(clock, key=lambda t: clock[t])
        cost = (
            f" Inference clock: {spent:.1f}s"
            + (f" of a {budget_s:.0f}s budget" if budget_s else "")
            + f"; the most expensive tier was {slowest} at {clock[slowest]:.1f}s"
        )
        unrun = [t for t in SCORED_TIERS if t not in seconds]
        if unrun and budget_s:
            cost += (
                f". Tiers {unrun} never ran: the budget is ONE shared pool "
                "spent in tier order, and it was gone before they started"
            )
        cost += "."
    return (
        f"[{rung.name}] leaderboard key = (H90={_h90(accuracy)}, "
        f"overall={_overall(accuracy):.3f}); {per_tier}.{cost}"
        " Crossing one more tier's 90% line is worth more than any accuracy"
        " gain below it, and a tier that never runs scores 0 however"
        " accurate the model is."
    )


def grade_fn(code: str, ctx: GradeContext) -> Grade:
    """单文件兼容入口（旧测试 / evoserve 远程路径，协议 v1 只传 main_text）。"""
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
    model_dir = Path(ctx.workdir).resolve() / "submission"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.py").write_text(code)
    return grade_workspace(model_dir, ctx)


__all__ = [
    "BENCH_DIR",
    "RUNGS",
    "SCORED_TIERS",
    "Grade",
    "grade_fn",
    "grade_workspace",
]
