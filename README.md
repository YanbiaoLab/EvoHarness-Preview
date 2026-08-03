# EvoHarness

[![CI](https://github.com/YanbiaoLab/EvoHarness-Preview/actions/workflows/ci.yml/badge.svg)](https://github.com/YanbiaoLab/EvoHarness-Preview/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

EvoHarness is a verifier-grounded evolutionary search framework for code and
agent workspaces. An agent proposes changes; preflight checks reject malformed
or unchanged proposals; a task-owned grader produces fitness; and the outer
search loop decides which candidates survive.

> [!IMPORTANT]
> EvoHarness is an alpha-stage research preview. Its local subprocess guardrails
> enforce time and resource limits, but network isolation is best-effort and is
> not a production security boundary. Run untrusted candidates inside a hardened
> container or virtual machine.

## What is implemented

- Multi-island population search with configurable parent selection, mutation
  operators, novelty checks, migration, checkpointing, and resume.
- Single-file and Git-backed multi-file candidate workspaces.
- Single-shot, conversational, and tool-using agent proposal modes with
  transcripts, budgets, preflight validation, and bounded tool execution.
- Typed local grading plus an optional remote evaluation service with version
  checks and auditable artifacts.
- Optional structured feedback, behavior diversity, and experience retrieval
  extensions assembled through experiment recipes.
- Run manifests, metrics, HTML reports, and a local web console.

These mechanisms are implemented and tested; that alone is not evidence that a
particular search recipe improves every task. Treat experiment results as
task-, model-, budget-, and seed-specific.

## Quick start

Requirements: Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YanbiaoLab/EvoHarness-Preview.git
cd EvoHarness-Preview
uv sync --locked --extra dev
```

Run the bundled offline demo. It uses a fake model transport, so it needs no API
key and makes no network request:

```bash
uv run python -m experiments.run_evolution \
  --recipe e0 \
  --task demo_counter \
  --run-dir results/demo \
  --set search.num_generations=3 \
        population.num_islands=1 \
        proposal.mode=agentic \
        'search.operators=["rewrite"]' \
        'search.operator_probs=[1.0]'
```

Generate a static report or open the local console:

```bash
uv run python -m evoharness.evoviz results/demo
uv run python -m evoharness.evoweb results
```

The console binds to `127.0.0.1:7861` by default.

## Using a real model

The experiment driver accepts any OpenAI-compatible endpoint:

```bash
export EVOHARNESS_API_BASE="https://your-endpoint.example/v1"
export EVOHARNESS_API_KEY="your-api-key"

uv run python -m experiments.run_evolution \
  --recipe e3r \
  --task demo_counter \
  --live \
  --budget-usd 5 \
  --run-dir results/live-demo
```

Never commit API keys. Use `--budget-usd` and task-specific evaluation limits
before starting a live run.

## Defining a task

The public entry point is `ScorableTask`. A task supplies a seed workspace and
a grading function; EvoHarness adapts them to the search engine:

```python
from pathlib import Path

from evoharness import ScorableTask


def grade(candidate_dir: Path, context):
    source = (candidate_dir / "main.py").read_text()
    solved = "return 42" in source
    return {
        "fitness": 1.0 if solved else 0.0,
        "passed": solved,
        "visible_metrics": {"solved": float(solved)},
    }


task = ScorableTask.from_directory(
    Path("path/to/seed"),
    grade,
    task_sys_msg="Make solve() return 42.",
)
```

See [`evoharness/task.py`](evoharness/task.py) for the complete task contract and
[`tasks/demo_counter.py`](tasks/demo_counter.py) for an offline example. More
specialized examples live under [`experiments/`](experiments/).

## Repository layout

```text
evoharness/
  evocore/    search loop, candidates, workspaces, proposals, selection
  evoguard/   budgets, subprocess limits, and anti-hack checks
  evoplus/    feedback, behavior, experience, and research extensions
  evoserve/   local/remote evaluation protocol and service
  evoviz/     static run reports
  evoweb/     local run console
experiments/  runnable research domains and experiment drivers
recipes/      composable search configurations and ablations
tasks/        small task registry and offline examples
tests/        unit, integration, and protocol tests
```

For the real execution flow, start with
[`experiments/run_evolution.py`](experiments/run_evolution.py), then follow the
selected module under [`recipes/`](recipes/) into `SearchLoop`. Check source,
recipe wiring, and tests before treating a design note as implemented behavior.

## Development

```bash
uv sync --locked --extra dev
uv run pytest -m "not slow"
uv build
```

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Security
issues should follow [`SECURITY.md`](SECURITY.md), not the public issue tracker.

## License

Licensed under the [Apache License 2.0](LICENSE). Third-party attribution and
provenance notes are recorded in [NOTICE](NOTICE) and in the relevant experiment
asset directories.
