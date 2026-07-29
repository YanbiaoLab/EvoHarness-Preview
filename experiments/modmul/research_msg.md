<!-- modmul research_msg v5 (v4 -> v5, 2026-07-28):加 §8.4(tier 0 不计分
     但第一个跑且占同一时钟;实测吃掉 81% 预算;它是唯一横跨全宽度范围的层)。
     同样只给现象与坐标轴。原 v4 头注如下。 -->
<!-- modmul research_msg v4 (v3 -> v4, 2026-07-27):加 §8(照官方 harness 读出的
     计费方式:按批不按题、超时后续层全 0、批宽是未测轴、每步成本第二项无人攻)。
     仍然只注入现象与坐标轴,不注入设定值 —— 该设多大的批、该怎么改扫描,留给搜索答。
     原 v3 头注如下。 -->
<!-- modmul research_msg v3 (冻结领域简报,tier-1 knowledge)。
     v2 -> v3 (2026-07-26):加 §7(当日实测:前沿是时间预算不是精度上限、
     步数同时买时间与可靠性、状态宽余量是未测轴、已证否清单、model soup、
     仪器极限)。**刻意只注入现象与死路,不注入具体设置** —— 该设多少 radix、
     加多少位余量留给搜索答,否则"框架能否自己发现"这个论证就作废了。
     经 TaskBundle.research_brief -> StaticBriefContributor 注入,所有实验组共享,
     不是消融变量。来源:官方 rules/literature.md + NeuralHorner 公开仓库 +
     组织者 Zulip 裁决链(2026-07 合并的 commit 82510bb)+ 本项目 2026-06/07
     手工实验与调研报告《神经 Horner 线》(2026-07-25)。改动需升版本并记 run manifest。 -->

# Research brief: what is known about neural modular multiplication

## 1. The terrain

- Ten scored tiers: `p` from 3 bits to 2048 bits, operands from 32 to 4096
  bits. Operands are always >> `p`, so genuine modular REDUCTION is the skill.
  Scalability is the whole game — the ranking key is the highest tier at 90%.
- Official baselines top out at `h90 = 1`, best `overall_accuracy` 0.127.
- This project's own history, in one line each:
  - direct tier-3 classifier, 37M params, 185k steps, 530M samples → 0.001.
    WALL CONFIRMED. Scaling capacity did not move it.
  - RNS / parallel-head decomposition → answer digits never left random.
  - Cross-diagnosis of both failures: **there was no serial computation path**
    from operands through intermediates to the answer. Parallel heads cannot
    condition the answer on an intermediate quotient.
  - Evolution run 1: the top six champions were all the SAME species (serial
    autoregressive with a raw→quotient→answer scaffold); every parallel and
    every scaffold-free control died.

## 2. The publicly known strong solution shape (NeuralHorner)

One modulus-conditioned recurrent cell (~471K-param GRU) driven by a FIXED
bit-serial Horner schedule; the cell learns the step `s' = (2s + d*x) mod p`:

- Three passes with the SAME weights: reduce `a` to `a mod p`, reduce `b`,
  then multiply the two residues (the computed residue becomes the control
  input). Nothing is reduced outside the network.
- `p` conditions the cell → it transfers to unseen primes, where monolithic
  seq2seq shows near-zero cross-prime transfer.
- Inference sizes the state to each prime's bit-length (dynamic width) —
  correctness-preserving (padded high bits are zero) and what keeps the
  10-tier run inside the time budget (383s → ~170s).
- Training: AdamW warmup+cosine, warm-started across widths, plus an
  on-policy DAgger pass to fix long-chain drift.
- Passes the weight-perturbation test; the forward path has no big-int ops,
  no lookup, no compare-against-p.
- **Known hole**: power-of-two-adjacent (Fermat-like) operands at the top of
  the trained width range. Lesson: put such operands in the training data.
- Legality pattern to imitate: *the arithmetic is learned, the schedule is not.*

A second contestant reached tier-3 ≈ 0.69 with the same two-phase cell where
this project's direct approach sat at ≈ 0.02.

## 3. The rule clarifications that reshaped the design space (2026-07)

The organizers ruled on the exact boundary this family lives on:

- **ALLOWED**: "a loop that feeds the model its input tokens one at a time is a
  valid deterministic encoder, so long as the encoder receives **no feedback**
  from the model to help it choose the next token." (merged into
  `rules/evaluation.md`) — the whole neural-Horner line is officially blessed.
- **NOT ALLOWED**: preprocessing the inputs into `(a % p, b % p, p)`. The model
  must eat the raw `(a, b, p)`; reducing full-width operands is itself a core
  part of the task, and moving it outside the network is offloading the work.
  (This invalidated an earlier, widely-used "legal trick" — including one of
  this project's own submissions.)
- **NOT ALLOWED**: an autoregressive chain-of-thought decoder where the model
  emits intermediate steps and a deterministic decoder finishes the job.
  Principle: *"We want the model to output a result, not a set of rules for
  calculating it."* Intermediates computed INSIDE the model are fine; what
  leaves the model must be the answer.
- Two decision principles now in the rules: (1) the emitted digits must
  materially determine the answer; (2) the capability must reside in the
  trained parameters (randomizing weights must collapse accuracy).
- The shipped `digit_transformer` example still reduces `mod p` in
  preprocessing and contradicts the clarification; the organizers have
  acknowledged the gap and said to follow the latest clarification. **Do not
  cite that example as precedent.**

## 4. Capacity is not the bottleneck (three independent measurements)

- This project: scaling to 37M params still hit the tier-3 wall.
- Contestant A: a single frozen step overfits to 1.00 held-out in EVERY family
  including Fermat — representational capacity is sufficient; the failure is
  coverage/drift over a ~2048-step rollout.
- Contestant B: doubling model size made held-out accuracy WORSE, and the
  in-distribution/held-out gap widened with scale.

**Consensus: the lever is exposure bias — on-policy / DAgger-style training on
states the model actually reaches — not parameter count.** Long rollouts need
per-step exactness around 1 - 1e-5; teacher-forced training on the true state
distribution does not deliver that on its own.

Five fixes have been tried against the Fermat/sparse residual by others:
boundary curriculum, on-policy DAgger, all-family DAgger, subtract-count
sweeps, PCGrad. Each either failed to close it or broke another family.
EWC / L2-SP anti-forgetting has since been tried too and also failed: no
usable lambda window was found — 10, 30 and 100 all merely moved the
interference between width bands.
**Detecting long zero runs with an if/else and branching to a shortcut is
explicitly prohibited — that is hand-coded arithmetic.**

## 5. Literature distillation (official bibliography)

- **Grokking family** (Power et al.; Gromov; Doshi/He/Das/Gromov): delayed
  generalization on SMALL moduli; weight decay is load-bearing; networks
  converge to Fourier/rotation-style circuits (Zhong et al. clock/pizza; Li et
  al. Fourier circuits; McCracken et al. universal modular-addition
  algorithm). CAVEAT from the organizers: small-modulus grokking may not
  transfer to the medium moduli of this competition.
- **Hardness evidence** (Lauter et al., ML for Modular Multiplication; SALSA
  lineage): direct seq2seq modular multiplication largely fails and does not
  transfer across primes — motivates decomposition + modulus conditioning.
- **Data distribution as a lever** (Saxena et al.): custom training
  distributions + loss regularization turn unlearnable modular tasks
  learnable. Treat the training distribution as a first-class mutation axis.
- **Repetition helps emergence** (Charton & Kempe): repeated examples beat
  always-fresh sampling at a fixed budget — the repetition schedule matters.
- **Position robustness** (Yudin 2026): digit models fail under position
  shift; position curriculum + template diversity mitigate — directly relevant
  to generalizing across operand widths.
- **Related op** (Africa et al. 2025): modular exponentiation learned with
  serialized intermediate steps — same serial-decomposition moral.

## 6. Design axioms for mutations (distilled, all contract-legal)

1. SERIALIZE: expose intermediates (bit-serial state, quotient, residue) as
   supervised steps; never predict the answer in one parallel shot.
2. DECOMPOSE + CONDITION: a small shared cell on a fixed schedule, conditioned
   on `p`, beats a monolithic learner; each step should be a small EXACT
   classification, not a regression.
3. SIZE TO THE INPUT: scale loop length and state width to bit-length at
   inference — exactness and the time budget both depend on it. **But sizing
   the state to EXACTLY the prime's bit length is a choice, not a
   requirement**, and it is one nobody here has tested against. See §7.
4. MAKE THE STATE'S INFORMATION FLOW BOTH WAYS. Carries travel from low bits
   to high bits, but the mod-`p` reduction decision is determined by the high
   bits and must reach the low ones. A mechanism that only propagates one way
   plateaus far below exactness (measured in this project: bit-accuracy 0.80,
   exact 0.21, flat for 20k steps — adding the reverse direction reached
   exact 1.00 on the same budget).
5. CURRICULUM + WARM-START across widths; expect late generalization, so train
   long enough before judging a lineage dead.
6. DATA IS A LEVER: stratify by bit-length, include edge cases (0, 1) and
   power-of-two-adjacent operands, consider repetition schedules, and move
   toward on-policy sampling once teacher-forced accuracy saturates.
7. VERIFY LIKE THE JUDGE: accuracy must collapse under weight randomization.
   If a change makes accuracy insensitive to the weights, you built a circuit,
   not a model — revert it.

## 7. What was measured on 2026-07-26 (v3)

### 7.1 The frontier is a time budget, not an accuracy ceiling

The seed lineage was graded at full rungs for the first time:

```
tier   1   2   3   4   5   6   7   8    9    10
acc  100 100 100 100 100 100 100 100   92     0
```

Tier 10 scored 0 **without a single case running** — the inference budget was
already spent. Tier 9 alone took 83% of it, at 8.6x the per-problem rate the
official evaluator allows; the full set projects to roughly 3x over budget.

Two consequences that overturn the obvious strategy:

- **More training, more capacity and better representations cannot move the
  frontier while the frontier never runs.** §4 said capacity is not the
  bottleneck; this says accuracy is not either, at the top.
- **Total serial work per problem is the quantity that decides how far this
  family reaches.** Steps per problem and cost per step are the two terms.

### 7.2 Fewer steps buys accuracy as well as time

Per-step reliability implied by the measurements: tier 9 at 92% over its step
count works out to a per-step error of about 1.6e-5. Because a tier is exact
only if every step is, **halving the steps roughly doubles the per-step error
a tier can tolerate**. Step count therefore appears twice — once in the time
budget, once in the reliability budget — and both in the same direction.

Against this: a coarser schedule makes each step harder to learn, because the
intermediate the cell must reduce spans a wider range. One team measured a
coarser radix on a GRU cell as ~2x faster and slightly less accurate. **That
was a different cell and the trade-off is OPEN for the scan-based cell here.**
Whether the reliability gained from fewer steps outweighs the reliability lost
per step is unmeasured, and is the single highest-value question open.

### 7.3 The state-width margin: now measured, banded, and load-bearing

The width the cell runs at need not equal the prime's bit length; the extra
high bits are mathematically inert. The community measurement (576 problems
per band, on a public fork of this exact family) is now precise: with
delta = register width − prime bits, accuracy collapses to 2-30% at delta in
{1, 2}, is marginal at 3 (degrades with rollout depth), is safe at 4 and
above, and the error sets are **identical** across delta 16..64. On the
original v8 weights the curve is flat at every delta — the difference is the
training distribution: robustness belongs to whatever margins training
covered, not to the loop.

This axis was confirmed here the expensive way. In run r12 generation 1,
four candidates independently invented pad-to-power-of-two batching — the
right idea, on weights whose training only ever saw delta = 0. All four
scored zero on every tier while paying 463-477s of clock. The same padding
on the current weights reproduces the collapse exactly (55/55 correct at
exact width, 0 at padded width, same problems, same weights).

The seed now covers the axis end to end: `train.py` draws half its samples
with the prime 4-67 bits narrower than the register (`DELTA_SHARE`), and
`model.py` buckets inference widths to the next multiple of 64 with at
least 4 bits of headroom, so a 100-problem tier runs as a few large batches
instead of ~100 near-singleton ones. Do not undo either side alone: the two
files are one mechanism, and the delta {1, 2, 3} region is a measured
cliff, not a style preference.

### 7.4 Known dead ends — do not spend mutations re-testing

| tried | outcome |
|---|---|
| more capacity | three independent measurements: not the bottleneck (§4) |
| L2-SP / EWC anti-forgetting | no usable lambda; moves interference, does not remove it |
| ensembling (width committees, cross-checkpoint voting) | both forms failed |
| re-annealing with the same recipe, alone | no gain |
| small-prime fine-tuning without an anchor | fixes small primes, costs large-scale accuracy |
| padded-width batching on delta=0-trained weights | four independent r12 candidates, all zero on every tier (§7.3); the padding is right, the training coverage is the missing half |
| a third pass streaming digits sliced from the model's own output | works numerically, but sits in the gray zone of the encoder ruling (§3) and costs one register-width of steps; the two-pass schedule is equivalent, cheaper, and strictly inside the ruling |

### 7.5 One cheap thing that does work

Averaging the weights of an annealed checkpoint with a second checkpoint
re-annealed from it beat both endpoints on paired evaluation (one-sided
McNemar p = 0.02, n = 1008). Cheap, and orthogonal to architecture.

### 7.6 Why more training stops paying

The same team measured their single-step verification floor at about 8e-6
while their effective per-step error was already 1.8e-6. **The objective and
the checkpoint-selection signal had both dropped below the noise of the
measurement** — past that point additional training is a random walk. If your
per-step error approaches your ability to measure it, the remaining gains are
in coverage, not optimization.

## 8. How the budget is actually charged (2026-07-27, read off the official harness)

§7 established that the frontier is a time budget. This section says what that
budget is charged AGAINST, because the answer moves the target.

### 8.1 The billing unit is the batch, not the problem

The official timer is cooperative and is checked **between batches**, never
per problem (`rules/evaluation.md`, "Wall-clock measurement"; enforced in
`evaluation/pipeline.py::run_inference`). There is no per-problem limit at
all — the ~273 ms figure in the rules is labelled a **soft** target and is
just 300 s divided by 1100 problems. The 5-minute total is likewise printed
under a column headed "Reference value" with the note "May be tuned before
the official runs", and is a CLI option rather than a constant.

What IS hard: on timeout the tier in flight scores 0% and **every subsequent
tier scores 0%**, while completed tiers keep their scores. Tiers run in
index order. So the budget is a single shared pool spent from tier 1 upward,
and running out is not a partial loss.

Consequences worth reasoning about:

- Cost per problem is the wrong quantity to optimise. Cost per BATCH, times
  the number of batches, is the quantity that is charged.
- A tier's problem count divided by the model's batch width decides how many
  batches that tier costs. A batch that is not full still pays whatever a
  batch costs.
- `max_batch_size()` is therefore a load-bearing inference-time parameter and
  it lives in the compliance-critical file, not the architecture file.

### 8.2 The open question underneath it

Whether a batch's wall clock is roughly FLAT in batch size or roughly LINEAR
in it decides which of two very different strategies pays:

- **Flat** (the cost is the sequential depth of the schedule — for a Horner
  family, `operand_bits/RADIX_BITS` steps times the per-step scan depth,
  none of which depends on how many problems ride along) → widening the batch
  is close to free throughput, and the number of batches per tier is the
  lever.
- **Linear** (the cost is arithmetic throughput) → batch width buys nothing,
  and the only way down is to do less work per step: a scan with better work
  complexity, fewer refinement rounds, or a coarser radix.

**This is unmeasured here.** It is cheap to measure — time one tier at two
batch widths — and it is worth measuring before spending mutations on either
branch, because the two branches recommend opposite things.

### 8.3 The second factor of per-step cost has had no attention

Total serial work per problem factorises into two terms:

    (number of outer steps) x (cost of one step)

Every mutation observed so far has attacked the first term. The second term —
how much sequential work one step performs, and with what work complexity —
has not been touched by any candidate. A step whose internal propagation has
depth `d` and work `w` pays both; a scan that is depth-optimal is not
automatically work-optimal, and on this hardware the two are not
interchangeable. Whether the second term has slack is open, and it multiplies
with the first rather than competing with it.

### 8.4 An unscored tier spends the same clock (measured 2026-07-28)

Tier 0 is "diagnostic only and is not counted toward either metric"
(`rules/evaluation.md`). It also runs FIRST, and it is on the same shared
clock as everything else: `Tiers run in order: Tier 0 first, then Tiers 1, 2,
..., 10`.

Its geometry is unlike any scored tier. The official generator declares it as
`TierConfig(tier_id=0, min_bits=1, max_bits=4096, is_multiplication_only=True)`
-- one tier spanning the entire range, where every scored tier occupies a
narrow band (tier 9's primes are 1020-1024 bits, tier 10's are 2047-2048).
In the public benchmark its hundred problems carry primes from 8 bits to 8192
and operands up to 4096.

Measured on this project's seed at the official calibration: tier 0 took
**243.9 seconds of a 300-second budget** -- 81% of the whole allowance, spent
before tier 1 begins -- and contributed nothing to either metric. Tiers 1
through 8 then brought the clock to 333.7s, so tiers 9 and 10 never started
and scored 0, although the same weights answer tier 9 at 99% and tier 10 at
96% when given the time.

Two consequences:

- The largest single cost in a run can sit in a tier that pays nothing. Any
  reasoning about "where the budget goes" that only looks at scored tiers is
  looking at 27% of it.
- Cost per tier is not a function of tier difficulty alone. A tier's internal
  spread of widths is a separate property, and the scored tiers are all narrow
  while the unscored one is maximally wide. Whether a model's cost is
  sensitive to that spread, and whether it has to be, is unmeasured here.
