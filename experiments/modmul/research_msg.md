<!-- modmul research_msg v1(冻结领域简报,tier-1 knowledge)。
     经 TaskBundle.research_brief -> StaticBriefContributor 注入,所有实验组共享,
     不是消融变量。来源:官方 rules/literature.md + NeuralHorner 公开仓库 +
     本项目 2026-06/07 手工实验。改动需升版本并记 run manifest。 -->

# Research brief: what is known about neural modular multiplication

## 1. The terrain

- Tiers 1–3 (current fitness): p up to 16 bits, operands up to 64 bits.
  Operands >> p, so genuine modular REDUCTION is the skill, not memorization.
  Full challenge extends to p ~ 2048 bits: scalability is the whole game.
- Measured baseline (this project, serial AR transformer, 500 steps):
  fitness 0.256 — t1 63%, t2 10%, t3 3%, zero malformed. The output contract
  is easy to learn; the accuracy cliff between tiers is the actual problem.
- Our 2026-07-03 experiments: parallel prediction heads and single-shot
  tier-3 classification BOTH fail the same way — there must be a serial
  computation path from operands through intermediates (raw product,
  quotient) to the answer. Scratchpad-internalization results agree.

## 2. A publicly known strong solution shape (NeuralHorner, 100% automated score)

One modulus-conditioned recurrent cell (~471K-param GRU) driven by a FIXED
bit-serial Horner schedule; the cell learns the step `s' = (2s + d*x) mod p`:

- Three passes with the SAME weights: reduce `a` to `a mod p`, reduce `b`,
  then multiply the two residues (computed residue becomes the control input).
- `p` conditions the cell (fed as 32-bit limbs) -> transfers to unseen primes,
  where monolithic seq2seq shows near-zero cross-prime transfer.
- Inference sizes state width to each prime's bit-length (dynamic-L) —
  correctness-preserving (padded high bits are zero) and what keeps the
  10-tier run inside the time budget (383s -> ~170s).
- Training: AdamW warmup+cosine, warm-started across widths, plus an
  on-policy DAgger pass to fix long-chain drift.
- Passes the weight-perturbation test (randomized weights -> 0 everywhere);
  forward path has no big-int ops, no lookup, no compare-against-p.
- Known hole: power-of-two-adjacent (Fermat-like) operands at the top of the
  trained width range. Lesson: include such operands in training data.
- Legality pattern to imitate: "the arithmetic is learned, the schedule is not."

## 3. Literature distillation (official bibliography)

- **Grokking family** (Power et al.; Gromov; Doshi/He/Das/Gromov): delayed
  generalization on SMALL moduli; weight decay is load-bearing; networks
  converge to Fourier/rotation-style circuits (Zhong et al. clock/pizza;
  Li et al. Fourier circuits; McCracken et al. universal modular-addition
  algorithm). CAVEAT from the organizers: small-modulus grokking may not
  transfer to the medium moduli of this competition.
- **Hardness evidence** (Lauter et al., ML for Modular Multiplication; SALSA
  lineage): direct seq2seq modular multiplication largely fails and does not
  transfer across primes — motivates decomposition + modulus conditioning.
- **Data distribution as a lever** (Saxena et al.): custom training
  distributions + loss regularization turn unlearnable modular tasks
  learnable. Treat the training distribution as a first-class mutation axis.
- **Repetition helps emergence** (Charton & Kempe): repeated examples beat
  always-fresh sampling at fixed budget — dataset repetition schedule matters.
- **Position robustness** (Yudin 2026): text/digit models fail under position
  shift; position curriculum + template diversity mitigate — relevant when
  generalizing across operand widths.
- **Related op** (Africa et al. 2025): modular exponentiation learned with
  serialized intermediate steps — same serial-decomposition moral.

## 4. Design axioms for mutations (distilled, all contract-legal)

1. SERIALIZE: expose intermediates (raw, quotient, or bit-serial state) as
   supervised steps; never predict the answer in one parallel shot.
2. DECOMPOSE + CONDITION: a small shared cell on a fixed schedule,
   conditioned on p, beats a monolithic learner; each step should be a small
   EXACT classification, not a regression.
3. SIZE TO THE INPUT: scale loop length / state width to bit-length at
   inference — exactness and time budget both depend on it.
4. CURRICULUM + WARM-START across widths; expect late generalization
   (train long enough before judging a lineage dead).
5. DATA IS A LEVER: stratify by bit-length, include edge cases (0, 1) and
   power-of-two-adjacent operands; consider repetition schedules.
6. VERIFY LIKE THE JUDGE: your accuracy must collapse under weight
   randomization; if a change makes accuracy insensitive to weights, you
   have built a circuit, not a model — revert it.
