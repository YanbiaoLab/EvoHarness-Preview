# HyperAgents × IMO: comparable experiment

This experiment is a controlled comparison with `results/imo_trace_run_v2`.
It freezes the common ProofBench corpus, the 12-item training split, model
identities/output ceilings, and the observed five-candidate budget (seed plus
four evaluated children).

Run the read-only admission check:

```bash
python -m experiments.hyperagents_imo.run verify
```

The check does **not** invoke a model or Docker.  A live run is intentionally
not enabled yet: native HyperAgents evolves its own `TaskAgent`/`MetaAgent`,
whereas the EvoHarness score evaluates the three-file `solver.py` workspace.
The required next implementation is an adapter that turns a HyperAgents patch
into an admitted EvoHarness candidate and evaluates it with the frozen
train/validation/test protocol.

Do not compare a native `generate_loop.py --domains imo_proof` score directly
to `imo_trace_run_v2`: without that adapter, it uses a different candidate
representation and evaluation execution path.

For a narrower, fair evaluator check, a native HyperAgents proof can be scored
beside the matching EvoHarness proof with the exact frozen grader:

```bash
python -m experiments.hyperagents_imo.run score-pair \
  --problem-id PB-Basic-001 \
  --hyperagents-csv /path/to/predictions.csv \
  --evoharness-item results/imo_trace_run_v2/artifacts/343c77f64b50/items/PB-Basic-001.json \
  --output /path/to/paired_score.json
```

Live scoring reads `EVOHARNESS_API_BASE` and `EVOHARNESS_API_KEY`. This is a
one-problem paired evaluator comparison, not a full outer-loop comparison.

To score a completed 60-problem prediction file with checkpointed per-item
results and frozen train/validation/test aggregation:

```bash
python -m experiments.hyperagents_imo.run score-benchmark \
  --hyperagents-csv /path/to/predictions.csv \
  --output-dir /path/to/frozen_grades \
  --workers 3
```

The scorer rejects missing, duplicate, or extra problem IDs. Existing item
grades are reused only when their proof hashes still match.

## Host-native architecture evolution

When Docker is unavailable, the real HyperAgents MetaAgent/TaskAgent archive
loop can run as a restricted operating-system user:

```bash
python -m experiments.hyperagents_imo.native_evolution \
  --public-run-dir /var/lib/hyperagent-evolution/imo_benchmark_v1_train12 \
  --private-run-dir /protected/hyperagent_native_train12 \
  --sandbox-user hyperagent \
  --agent-python /opt/hyperagent-runtime/venv/bin/python \
  evolve --candidates 15 --seed 0
```

The public directory contains candidate workspaces, patch lineage, sanitized
Train reports, and the final architecture. Reference solutions, grader
responses, and grader credentials remain in the private directory. Candidate
solver calls are capped externally by `BenchmarkSpec.solver_budget`.

After evolution completes, freeze the selected workspace and evaluate it
without further MetaAgent calls:

```bash
python -m experiments.hyperagents_imo.native_evolution \
  --public-run-dir /var/lib/hyperagent-evolution/imo_benchmark_v1_train12 \
  --private-run-dir /protected/hyperagent_native_train12 \
  --sandbox-user hyperagent \
  --agent-python /opt/hyperagent-runtime/venv/bin/python \
  evaluate --split validation
```

Run the same command with `--split test` for the final hidden-split result.
