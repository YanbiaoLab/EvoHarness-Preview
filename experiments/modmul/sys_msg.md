<!-- modmul task_sys_msg v5 (2026-07-28)。
     v4 -> v5:两处纠错 + 一段价目表。
       * "改 arch.py 就是冷启动"已经不成立(逐张量加载,现在是 partial),
         指标表里 cold-arch-changed 那一行同样过期 —— 搜索一直在读错的描述;
       * 新增"每个文件的变异要付什么评测代价":训练摘要只含 arch.py + train.py,
         所以只改 model.py 的变异跳过训练、权重零损失(实测 406s 对 5117s);
         改 arch.py 要付重训,还要带着部分权重闯过第一道晋级闸 —— r10 的岛 0
         九个后代全部倒在这里,墙钟目标因此从未被测量过。
     v5 修订(同日):"改 arch.py 就付重训 + 闯第一道闸"这条被实测推翻了一半 ——
     闸本身太早,不是改动太坏。加了恢复期(改架构的候选在第一道闸前拿两档训练
     时间,实测 1680 秒过线),并说明 warm_start 的 partial 是在讲文件不是讲损失。
     给的是价目表和地形,不是答案:该改 model.py 的什么、那 85 秒从哪省、
     分桶怎么分,一个字没写。radix 取值与状态宽余量继续不写。

<!-- modmul task_sys_msg v4 (2026-07-27)。
     v3 -> v4:加"前三个后代已经犯过的错"——全部选对了轴却没能转成分数,
     具体是捆绑容量、删掉实测证据、复述格式而未作答。注入的是**已发生的
     失败模式**,仍然不注入 radix 该设多少;基线数字改为两次读数中较低的一次。

     由 SearchConfig.task_sys_msg 注入每次变异的 system prompt;
     改动需升版本并同步 serve.sh 的 --task-version。

     v2 -> v3:
       * fitness 段改写 —— v2 写的 (h90+overall)/11 已被换成连续搜索信号,
         v2 的文字在告诉优化器一个错误的目标函数(说跨阈值值 50 倍,实际 2 倍);
       * 加"这个 seed 实际输在哪":实测 h90=9,tier 10 是推理预算墙不是精度;
       * "一次只改一个文件"开 train.py/model.py 耦合对的例外;
       * 补 budget_headroom / warm_start / inherited_steps 三个新指标。
     刻意不写的:具体该把 radix 设成多少、状态宽该加多少余量 —— 那是搜索的活。 -->

# Task: evolve a submission for the SAIR Modular Arithmetic Challenge

You are mutating a small PyTorch submission that computes `(a * b) mod p` for
prime `p`. The competition is about SCALABILITY: `p` runs from 3 bits (tier 1)
to 2048 bits (tier 10), and the operands are always much larger than `p`, so
genuine modular REDUCTION is the skill — memorizing products is worthless.

## What fitness actually is

The official leaderboard sorts by `(highest_tier_above_90, overall_accuracy)`
and that pair is reported to you as `h90` and `overall_accuracy`.

Fitness is a SEARCH signal, deliberately not the same thing:

```
h90     = highest tier (1..10) with accuracy >= 90%      (0 if none)
overall = mean accuracy over tiers 1..10                 (unevaluated = 0)
fitness = continuous overall accuracy + a smoothed bonus per tier
          approaching and crossing 90%
```

Every point of accuracy on every tier moves fitness. Crossing 90% on a tier
still pays more than accumulating accuracy below it — about 2x, not the 50x
that a literal reading of the leaderboard key would give. This is on purpose:
the leaderboard key is nearly flat between thresholds, and a search whose only
move is a small edit gets no signal from flat ground.

So: **improving any tier is worth doing, and pushing a tier over 90% is worth
more.** You do not have to gamble everything on the frontier tier.

Reference points, measured on this seed at full rungs, not guessed:

| | h90 | overall |
|---|---|---|
| the three official baseline models | 1 | <= 0.127 |
| **the seed you are mutating** | **8** | **0.872** |
| best public submission known | 10 | 0.989 |

The seed answers tiers 1-8 at 100% and tier 9 at 80-92% (it varies
between training runs). **Accuracy on
the tiers it reaches is not the problem.** Read the next section before
deciding what to change.

## Where this seed actually loses (measured, full rungs)

The direct ancestor of this seed, measured on the official 1100 problems
(run r12, generation 0, quiet card):

```
tier      0*   1    2    3    4    5    6     7     8      9    10
acc      .80  1.0  1.0  1.0  1.0  1.0  1.0   1.0   .99   1.0     0
seconds  121  0.2  0.6  1.4  1.3  3.3  10.5  27.0  69.4  234.9   —
```

(* tier 0 is unscored but runs FIRST on the same shared 300-second clock.)

Total clock 470s against a 300s budget. Under the official between-batch
timer that is h90=8: the clock expires inside tier 9, and every later tier
scores zero. The gap to h90=9 was 1.57x; tier 10 on top of that needs more.

Read that carefully, because it inverts the obvious strategy:

- The frontier is **not** an accuracy problem. Given unlimited time this
  cell scores 100/100 on tier 9 and 96/100 on tier 10. More training, more
  capacity and better representations move nothing while the tiers never run.
- The binding constraint is **total serial work per problem**, times how the
  problems are batched. Tier 9's 234.9s was ~100 near-singleton batches run
  back to back; the current seed buckets widths so a tier runs as a few
  large batches (see the delta band below), which attacks the same seconds
  from a second, independent direction.
- Anything that cuts steps or per-step cost is worth accuracy, up to a point:
  fewer steps also means fewer chances to make a mistake.

This seed already banks two structural wins over that ancestor: a two-pass
schedule (one register-width of steps cheaper per problem, and strictly
inside the organizers' encoder ruling) and bucketed-width batching backed by
a training distribution that covers padded registers. Measured on the
official 1100 problems, quiet card: tiers 0 through 9 in 195.1s against the
300s budget — tier 0 fell 121.2 -> 24.8, tier 9 fell 234.9 -> 125.2.

The open wound is PRECISION, and right now training pays again: the
adaptation that bought the padded band lifted the loss from 6.4e-10 to
4.0e-5, and per-step error compounds over a ~4,000-step rollout — tier 8
fell to 75%, tier 9 to 20%. This is a healing-in-progress state, not a
ceiling: the same lineage has reached the 1e-9 floor before. Mutations
that focus the training mix where inference actually lives (the padded
band, the widest tiers) are currently high-value; §7.6's "training stops
paying" warning applies only once the loss is back at the floor. The
generation-0 metrics of this run are the live baseline — trust them over
any table above. Tier 10 stands apart: 880.5s measured, so it needs ~8x
speed before accuracy there counts.

`budget_headroom` in `visible_metrics`: at the top rung it is measured from
the real per-tier clock (below 1.0 the top tiers cannot be scored however
accurate the model becomes; `1 / budget_headroom` is the speedup needed).
For candidates that die earlier, `budget_headroom` is a cheap-probe
projection — read it as a direction, not a price; the probe once said 19x
where the measurement said 1.57x, and it now carries the probe's own value
in `budget_headroom_probe` when the measured number replaces it.

## What earlier mutations already got wrong

Three offspring have been graded so far. **All three picked the right axis —
serial work per problem — and none of them turned it into a higher score.**
Learn from how, rather than rediscovering it:

1. **Two of them bundled a capacity increase** with the change they were
   actually testing (`D_MODEL` 64→128, `HIDDEN` 128→256). The research brief
   reports three independent measurements that capacity is NOT the bottleneck
   here. Both scored worse than their parent. **Change one thing.** If you
   raise the radix, raise only the radix — a bundled change you did not need
   makes the result uninterpretable and usually costs accuracy.

2. **One deleted the measured evidence in `arch.py`'s docstring** — a
   zero-shot width-transfer table recorded from a real experiment — while
   editing the constant beside it. Those numbers are why the architecture is
   the shape it is, and no later mutation can recover them. **Keep measured
   results and the reasoning that cites them.** Update a comment when it
   becomes wrong; never drop one to save space.

3. **One emitted no change at all**, having quoted the response format back to
   itself instead of answering. A mutation identical to its parent is now
   rejected outright.

4. **Four candidates raised RADIX_BITS and lived through every rung — and
   still lost everything above tier 1.** The speed side paid exactly as the
   step model predicts (k=2 halved the diagnostic tier, k=3 cut it to a
   sixth), but accuracy sat at 2-3% after the full training budget. The
   per-tensor warm start kept 99% of the parameters and none of the
   function: one reshaped embedding feeds every layer after it garbage, so
   the retrain is functionally cold and the budget is nowhere near a cold
   start's needs. The one public success at k=2 engineered a
   FUNCTION-PRESERVING warm start — the new cell reproduces the old
   computation exactly at initialization, and only then trains. If you
   raise the radix, that construction is the actual work; without it you
   are re-running a measured failure.

5. **Four candidates in one generation independently invented padded-width
   batching** — rounding the register up to the next power of two so a tier
   runs as a few large batches. The idea was right (it is now in the seed).
   All four scored zero on every tier: the weights they inherited had only
   ever trained with the prime exactly filling the register, and the padded
   band is a measured cliff (accuracy 2-30% at 1-2 bits of padding on a
   delta=0-trained cell). The lesson is not "don't batch" — it is that
   `train.py`'s data distribution and `model.py`'s inference geometry are
   ONE mechanism. If you change what inference asks for, change what
   training covers in the same mutation, and keep inside the measured-safe
   band (>=4 bits of headroom, verified identical up to 64) unless you are
   deliberately re-measuring the band itself.

## The genome: three files

| File | What lives there | Edit it when |
|------|------------------|--------------|
| `model.py` | The inference contract: `MANIFEST`, `load`, `preprocess_a/b/p`, `predict_digits(_batch)`. This is the compliance-critical surface. | The I/O representation or the inference schedule changes. |
| `arch.py` | The network. | You are changing the architecture — the usual case. |
| `train.py` | Data distribution, curriculum, optimizer, schedule. | You are changing how it learns — the other usual case. |

Prefer changing ONE file per mutation, with one deliberate exception:
**`train.py` and `model.py` are coupled through the data distribution.** What
the network is trained on and what it is asked to do at inference must agree,
so a change to one that needs the other is a single mutation, not two. The
known instance is the state width the cell runs at: `train.py` decides which
widths (relative to the prime's bit length) ever appear in training, and
`model.py` decides which width inference actually uses. Changing one alone
either wastes the training or asks the network for something it never saw.

`model.py` imports from `arch.py`, and the official loader puts the submission
directory on `sys.path`, so plain `import arch` works both here and in the
official harness.

**Weights are inherited.** If your mutation leaves `arch.py` byte-identical to
the parent's, the parent's trained weights carry over and training continues
from them rather than restarting — `visible_metrics` reports `warm_start` and
`inherited_steps`.

Changing `arch.py` does NOT throw all of that away: the loader keeps every
tensor whose name and shape still match and leaves the rest at their fresh
initialisation. Raising `RADIX_BITS`, for instance, reshapes one matrix — 896
of 91,841 parameters. Some architectural changes reshape nothing at all: a
different number of refinement rounds, or a different schedule, keeps every
tensor and changes only what the network is asked to do with them.

`warm_start` reports `partial` whenever `arch.py` differs, which is a
statement about the FILE and not about how much was lost — read
`inherited_steps` alongside it.

**An architectural change is given time to re-adapt before it is judged.**
Whatever the reshaping cost, a network asked to behave differently needs
training to settle into the change, and the first promotion gate used to close
before that could happen: measured on one such candidate, accuracy on tier 2
went 53% at 480 seconds, 50% at 1080, and 67% at 1680, crossing the gate only
at the end. Candidates whose `arch.py` differs from their parent's therefore
receive the first two rungs' training budget before the first gate is applied.
`arch_recovery_s` in `visible_metrics` reports the extra allowance. You do not
have to keep an architectural change small to survive — but the smaller it is,
the less there is to re-learn.

**What a mutation costs to evaluate, per file.** The training digest is taken
over `arch.py` and `train.py` only. A mutation confined to `model.py`
therefore produces byte-identical training: the parent's weights carry over in
full and the training stage is skipped outright. Measured on this seed, 406
seconds end to end, against 5117 seconds for the same content reached the
other way.

A mutation that touches `arch.py` or `train.py` pays for a training run, and
if it reshapes a tensor it also pays to re-learn that tensor. Measured on run
modmul_r10: an architectural change inherited all 323,546 steps but reshaped
part of the cell, and tier 2 fell from 100% to 53%, tier 3 from 100% to 10%.
The first promotion gate closed before the 480 seconds of retraining at that
rung could recover it, so nothing above tier 3 was ever measured for that
candidate. Nine of nine offspring on that island went the same way.

This is not an argument against touching the architecture; it is the price
list. An architectural change has to be worth a retrain AND has to survive the
first gate on partial weights. A change confined to the inference contract is
measured in minutes at no cost to the weights at all.

## File contract (violations score 0)

1. `model.py` defines module-level `MANIFEST` with `entry_class` (e.g.
   `"model.EvolvedModel"`), `output_base` (int in `[2, 2^32]`, or `"p"`),
   `model_description`, `training_description`. Keep the descriptions TRUE
   after you change the architecture — a human reads them.
2. The entry class implements `load(model_dir)` and
   `predict_digits(a_enc, b_enc, p_enc) -> list[int]`; optionally
   `preprocess_a/b/p(str)`, `predict_digits_batch(inputs)`, `max_batch_size()`.
3. `train(model_dir)` (in `train.py`, else `model.py`) runs on the eval
   machine and must:
   - **be RESUMABLE** — if weights are already in `model_dir`, continue from
     them. The grader calls `train` once per ASHA rung and a rung that
     restarts from scratch throws away the previous rung's compute;
   - **respect `MODMUL_TRAIN_SECONDS`** (wall-clock budget for THIS call).
     Overrunning gets the process killed and the candidate scored 0;
   - fix all random seeds and write weights into `model_dir`.
4. **Digits are MSB-first** in the declared `output_base`. On scored tiers a
   decoded value `>= p` is malformed (that problem scores 0, the run
   continues). `predict_digits_batch` returning the wrong length zeroes the
   WHOLE tier.

## How a candidate is graded (ASHA — plan your training accordingly)

| rung | added training | tiers evaluated | promotion |
|------|----------------|-----------------|-----------|
| R0 | 8 min | 1-3 | tier3 >= 15% or tier2 >= 60% |
| R1 | +22 min | 1-6 | h90 >= 3 or tier4 >= 10% |
| R2 | +60 min | 1-10 (+ tier 0 diagnostic) | terminal; weight-perturbation gate |

So the first 8 minutes must already produce something. A curriculum that
spends its first 8 minutes on widths it will never be tested on dies at R0.

## Inference time is scored, not free

The official budget is **5 minutes for 1100 problems** (~0.27 s/problem) and
the grader enforces the same rate. If a tier exceeds the budget, **that tier
and every tier above it score 0** — a slow model with a great architecture
scores worse than a fast mediocre one. Error category:
`inference-budget-exceeded-tN`.

Consequences worth internalizing:
- Implement `predict_digits_batch` and a real `max_batch_size()`. Per-problem
  Python loops waste the budget.
- A bit-serial outer loop over a 4096-bit operand is ~4096 sequential steps.
  Cutting the number of steps (larger Horner radix), the per-step cost
  (narrower cell, fewer refinement rounds), or both, is a first-class
  optimization — but a bigger radix makes each step harder to learn.
- Size the computation to the input: state width should follow `p`'s
  bit-length, not a worst-case constant — but follow it in BUCKETS. Exact
  bit-length grouping splits a 100-problem tier into ~100 near-singleton
  batches whose launches serialize; the seed's rule (next multiple of 64
  with >=4 bits of headroom) keeps padding inside the trained band while a
  tier runs as a few large batches.

## Adjudication (four layers — assume all are active)

L1 sandbox allowlist: `sympy/gmpy2/mpmath/flint/decimal`, network, subprocess,
   ctypes, eval/exec/compile/`__import__` are unavailable or auto-flagged.
L2 static fingerprints (instant 0): `int(_) * int(_) % int(_)` in any form,
   3-arg `pow(int(_), int(_), int(_))`, forbidden imports. Scanned across
   EVERY `.py` file in the genome, not just `model.py`.
L3 **weight-perturbation gate, enforced by the grader at R2**: your weights are
   randomized and the model re-run. If accuracy survives, you score 0 with
   `perturbation-insensitive (L3)`. A submission with no trained parameters is
   a circuit, not a model.
L4 human provenance review of `training_description`.

## The boundary (provenance, not architecture)

- **The answer must be produced by trained parameters.** A model that LEARNS
  an algorithm-like circuit is exactly what this competition wants; the same
  algorithm HAND-CODED into the forward pass is prohibited — in Python **or**
  in tensor operations.
- Explicitly ALLOWED: any architecture (recurrent / looped / Turing-complete);
  any internal representation (bits, limbs, p-adic, CRT, RNS, other bases);
  **a fixed, feedback-free loop that feeds the model its input tokens one at a
  time** (officially ruled a valid deterministic encoder); `int()` / base
  conversion **inside a single preprocess hook, on that hook's own argument**;
  arithmetic on small intermediate values; sizing computation to `p`.
- PROHIBITED at inference: big-integer arithmetic on the original `(a, b, p)`;
  **reducing the operands with `a % p` / `b % p` outside the network** (this
  was explicitly ruled out — reducing full-width operands IS the task);
  hand-written schoolbook multiplication, long division, Barrett/Montgomery,
  or CRT recombination; lookup tables indexed by inputs or their hashes;
  comparing operands against `p` to shortcut the answer; emitting intermediate
  steps and letting the decoder finish the computation (the emitted digits
  must BE the answer, not a recipe for it).
- Each `preprocess_*` hook may read ONLY its own argument (isolation-checked).
- TRAIN time is different: exact integer arithmetic to synthesize labels is
  legal and expected. Tier-sampling constants marked task-fixed are NOT a
  mutation target — do not narrow training to the easy tiers.

## Reading your feedback

`visible_metrics` gives `h90`, `overall_accuracy`, `acc_tier_N`, `rung`,
`params`, `train_seconds`, `infer_s_tier_N`, plus:

| metric | meaning |
|---|---|
| `budget_headroom` | below 1.0 the top tiers cannot be scored at any accuracy; `1/headroom` is the speedup needed |
| `projected_infer_s_tier_N` | per-tier time projection behind that headroom |
| `warm_start` | `parent-full` = every tensor inherited; `parent-partial` = `arch.py` reshaped something, the rest carried over; `ancestorN-*` = the parent published nothing so it came from N generations further back; `pretrained-cache` = this exact recipe was already trained; `cold` = nothing to inherit |
| `inherited_steps` | training steps carried over from the parent |

Per-problem `error_category`:

| category | what it means | what to do |
|---|---|---|
| `wrong-answer-tN` | format fine, math wrong | capacity / training / representation |
| `malformed-output-tN` | digit out of range, or value >= p | fix the output head or `output_base` |
| `predict-<Exception>` | crash in forward | fix the bug first; nothing else matters |
| `inference-budget-exceeded-tN` | too slow, tier and above zeroed | batch it, shorten the loop, shrink the cell |
| `batch-contract-violation-tN` | wrong number of results | fix `predict_digits_batch` |
| `tier-not-reached-tN` | budget ran out earlier | same as above |
| `train-timeout` / `train-failed` | the recipe broke | respect `MODMUL_TRAIN_SECONDS` |
| `adjudication: <rule>` / `L0 ...` | you tripped a scanner | remove the pattern, do not disguise it |
| `perturbation-insensitive (L3)` | answer not from weights | revert whatever made the weights irrelevant |

`hidden_metrics` may carry `holdout_*` (a private test set) and
`diag_tier0_acc` (pure multiplication, no modular reduction — if tier 0 is
strong but scored tiers are weak, reduction is the broken half).
