<!-- modmul research_msg v2 (冻结领域简报,tier-1 knowledge)。
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
sweeps, PCGrad. Each either failed to close it or broke another family;
EWC / L2-SP style anti-forgetting regularization is the untried suggestion.
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
   inference — exactness and the time budget both depend on it.
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
