# IMO Proof experiment

This directory is a self-contained EvoHarness experiment for evolving a
multi-file IMO proof-solving agent. Task evaluation and EvoHarness adaptation
are separate layers, joined by one `CandidateEvaluation` contract.

## Boundary

```text
imo_proof/
├── assets/             # frozen dataset, grader prompt, and attribution
├── seed_agent/         # candidate genome: solver.py, prompts.py, policy.py
├── evaluation/
│   ├── contract.py     # sole evaluation IR and backend protocol
│   ├── engine.py       # admission, execution, judge, aggregation
│   ├── worker.py       # independent evaluator and candidate process entry
│   └── service.py      # optional HTTP service and client backend
├── benchmark.v1.json   # models, budgets, splits, hashes, scoring
├── protocol.py         # typed protocol and integrity checks
├── grade.py            # CandidateEvaluation -> EvoHarness Grade only
├── evolution.py        # EvoHarness task assembly and search run
└── run.py              # verify, provider probe, and live run CLI
```

Only `seed_agent/solver.py`, `seed_agent/prompts.py`, and
`seed_agent/policy.py` are candidate-mutable. Evaluation assets never enter the
candidate workspace. The experiment does not import another experiment or a
competing agent framework.

`evaluation/contract.py` and `evaluation/service.py` do not import EvoHarness.
The local engine, independent worker, and HTTP client all return the same
`CandidateEvaluation`; HTTP does not introduce another result model.

## Frozen protocol

`benchmark.v1.json` fixes:

- 24 train, 12 validation, and 24 held-out test problems;
- optimizer, solver, and grader model settings;
- candidate, call, token, timeout, and optional cost limits;
- dataset and grader-prompt SHA-256 digests;
- the 0/1/6/7 IMO scoring map.

The solver may return up to 8,192 tokens per call, with a 32,768-token
per-problem completion budget across solve/review/revise calls. A response
ending in `max_tokens` is rejected rather than graded as a truncated proof.

`verify` checks the hashes, row count, unique IDs, and exact split coverage:

```bash
python -m experiments.imo_proof.run verify
```

## Run

Live execution uses one OpenAI-compatible endpoint:

```bash
export EVOHARNESS_API_BASE=https://example.com/v1
export EVOHARNESS_API_KEY=...

python -m experiments.imo_proof.run probe-live
python -m experiments.imo_proof.run run \
  --run-dir results/imo_proof_seed_0 \
  --seed 0
```

The experiment transport maps MiniMax's model-specific switch to
`thinking={"type":"disabled"}` when `enable_thinking` is false. As a safety
boundary, a leading `<think>...</think>` block is removed before task code sees
the response, and reasoning without a final answer is rejected as a provider
protocol error rather than graded as a proof. The seed reviewer uses an exact
`<verdict>PASS|REVISE</verdict>` contract.

The run writes `experiment_manifest.json`, the EvoHarness database and
transcripts, per-generation evaluations, final validation/test evidence, and
`summary.json` under the selected run directory.

## Independent worker

The worker is the process boundary used by evaluation services:

```bash
python -m experiments.imo_proof.evaluation.worker evaluate \
  --project-root "$PWD" \
  --spec experiments/imo_proof/benchmark.v1.json \
  --candidate-dir experiments/imo_proof/seed_agent \
  --candidate-id seed \
  --split train \
  --result-dir results/imo_worker_check
```

It writes `evaluation.json`. A nonzero worker exit is an infrastructure
failure; candidate rejection or runtime failure is represented inside the
normal `CandidateEvaluation` result.

## HTTP evaluation

Each service is pinned to exactly one profile. Start separate services (and,
in production, use separate credentials) for train, validation, and test:

```bash
export EVOHARNESS_EVAL_TOKEN=...

python -m experiments.imo_proof.evaluation.service \
  --project-root "$PWD" --profile train --port 8322
python -m experiments.imo_proof.evaluation.service \
  --project-root "$PWD" --profile validation --port 8323
python -m experiments.imo_proof.evaluation.service \
  --project-root "$PWD" --profile test --port 8324
```

Run evolution against those services:

```bash
python -m experiments.imo_proof.run run \
  --run-dir results/imo_proof_seed_0 --seed 0 \
  --train-eval-url http://127.0.0.1:8322 \
  --validation-eval-url http://127.0.0.1:8323 \
  --test-eval-url http://127.0.0.1:8324
```

Protocol v2 sends the complete UTF-8 workspace, hashes the canonical file
tree, combines that hash with the frozen protocol fingerprint and service
profile for idempotency, and distinguishes retryable `infra_error`, hard
`protocol_error`, and normal candidate evaluation results. Identical workspace
content reuses one service job even when the framework assigns a new candidate
ID; the client rebinds the cached result to that current ID.

## Tests

```bash
python -m pytest tests/test_imo_proof.py -q
```

The offline acceptance test executes the real EvoHarness search loop with fake
model transports, so it consumes no provider quota.
