<!-- modmul task_sys_msg v2 (Round-1, multi-file genome + ASHA + H90 fitness).
     由 SearchConfig.task_sys_msg 注入每次变异的 system prompt;改动需升版本
     并同步 serve.sh 的 --task-version。 -->

# Task: evolve a submission for the SAIR Modular Arithmetic Challenge

You are mutating a small PyTorch submission that computes `(a * b) mod p` for
prime `p`. The competition is about SCALABILITY: `p` runs from 3 bits (tier 1)
to 2048 bits (tier 10), and the operands are always much larger than `p`, so
genuine modular REDUCTION is the skill — memorizing products is worthless.

## What fitness actually is

The official leaderboard sorts by `(highest_tier_above_90, overall_accuracy)`.
Fitness reproduces that key exactly:

```
h90     = highest tier (1..10) with accuracy >= 90%      (0 if none)
overall = mean accuracy over tiers 1..10                 (unevaluated = 0)
fitness = (h90 + overall) / 11
```

**One more tier crossing 90% is worth more than every accuracy gain below it.**
Pushing tier 3 from 80% to 89% is worth 0.008; getting tier 4 from 0% to 91%
is worth more than 0.09. Aim at the frontier tier, not at polishing.

Reference points: the three official baseline models all have `h90 = 1` and
`overall <= 0.127`. Anything with `h90 >= 4` is past the published state of
this project's own work.

## The genome: three files

| File | What lives there | Edit it when |
|------|------------------|--------------|
| `model.py` | The inference contract: `MANIFEST`, `load`, `preprocess_a/b/p`, `predict_digits(_batch)`. This is the compliance-critical surface. | The I/O representation or the inference schedule changes. |
| `arch.py` | The network. | You are changing the architecture — the usual case. |
| `train.py` | Data distribution, curriculum, optimizer, schedule. | You are changing how it learns — the other usual case. |

Prefer changing ONE file per mutation. `model.py` imports from `arch.py`, and
the official loader puts the submission directory on `sys.path`, so plain
`import arch` works both here and in the official harness.

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
  bit-length, not a worst-case constant.

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
`params`, `train_seconds`, `infer_s_tier_N`. Per-problem `error_category`:

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
