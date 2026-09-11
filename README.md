# EvoHarness

[![CI](https://github.com/YanbiaoLab/EvoHarness-Preview/actions/workflows/ci.yml/badge.svg)](https://github.com/YanbiaoLab/EvoHarness-Preview/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

EvoHarness is a verifier-grounded evolutionary search framework for code and
agent workspaces. An agent proposes changes; preflight checks reject malformed
or unchanged proposals; a task-owned grader produces fitness; and the outer
search loop decides which candidates survive.

The same principle carries a second mode. In **proof mode** the verifier is
Lean 4 and the artifact is a proof graph: a model chooses which lemma to try
and how to decompose a goal, and nothing counts as proved until the assembled
article compiles with no `sorry` and no unexpected axiom. Both modes can be
driven from a terminal or mounted as tools inside a
[dsh](#running-inside-dsh) agent session.

> [!IMPORTANT]
> EvoHarness is an alpha-stage research preview. Its local subprocess guardrails
> enforce time and resource limits, but network isolation is best-effort and is
> not a production security boundary. Run untrusted candidates inside a hardened
> container or virtual machine.

## What is implemented

**Search.** Multi-island population search with configurable parent selection,
mutation operators, novelty checks, migration, checkpointing, and resume.
Single-file and Git-backed multi-file candidate workspaces. Single-shot,
conversational, and tool-using agent proposal modes with transcripts, budgets,
preflight validation, and bounded tool execution.

**Grading and evidence.** Typed local grading plus an optional remote
evaluation service with version checks and auditable artifacts. Evaluation
produces an evidence envelope with a fault taxonomy that keeps `timeout` and
`protocol_error` distinct from a low score, so a broken run is never read as a
capability result.

**Contracts.** Frozen `TaskSpec`, `RunSpec`, and `SearchProfile` identities
with stable hashes, so a resume can prove it is continuing the same experiment
rather than a similar one.

**Proof graph.** A persistent SQLite graph of goals, decompositions, attempts,
and certifications; leases so several workers can share one graph; Lean-judged
decompositions; and an assembler that recompiles the finished article and
records the axioms Lean says it depends on.

**Research layer.** Append-only experiment store, typed claims, a versioned
assessment guard that fails closed on anything it does not know, and a
governance queue whose decisions are signed by an operating-system identity
rather than one the caller supplies.

**Reading runs.** `evoharness.readout` prints run state, per-generation
trajectory, single candidates, and the governance record as JSON, so a caller
in another language never has to parse `run.db` itself.

**Agent integration.** A dsh plugin package and two agent presets, covered in
[Running inside dsh](#running-inside-dsh).

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

Run the bundled offline demo. It uses a fake model transport, so it needs no
API key and makes no network request. The run is started **detached** — the
command returns the run directory and its pid, and the search outlives it:

```bash
uv run python -m evoharness.launch.start \
  --recipe e0 \
  --task demo_counter \
  --run-dir results/demo \
  --set search.num_generations=3 \
        population.num_islands=1 \
        proposal.mode=agentic \
        'search.operators=["rewrite"]' \
        'search.operator_probs=[1.0]'
```

Read it back. Every subcommand prints JSON on stdout, failures included:

```bash
uv run python -m evoharness.readout status --run-dir results/demo
uv run python -m evoharness.readout trajectory --run-dir results/demo
uv run python -m evoharness.readout list --root results
```

`status` reports `state`, the generation reached, best fitness, and
`stopped_reason` once the run finishes. A run that is still working reports a
heartbeat age instead, which is how a stall is told apart from a long
generation.

Starting and running are two commands on purpose. `python -m evoharness.launch
--run-dir <dir>` **blocks** for as long as the search takes and is what the
detached process executes; it reads the invocation out of the run directory, so
a resume cannot accidentally carry a different configuration than the run it
claims to continue.

## Using a real model

The launcher defaults to the Aliyun Model Studio OpenAI-compatible endpoint.
Supply the credential through the environment; never put it in a tracked file:

```bash
export EVOHARNESS_API_KEY="your-api-key"

uv run python -m evoharness.launch.start \
  --recipe e3r \
  --task demo_counter \
  --live \
  --budget-usd 5 \
  --run-dir results/live-demo
```

Set `EVOHARNESS_API_BASE` to override the default endpoint. The two names go
together: a key is good at one gateway, so the credential is named after the
project rather than after a vendor — `ALIYUN_MAAS_API_KEY` is still read for
deployments configured before the endpoint moved, and nothing is named after
it any more.

Two more environment settings matter on a slow or unusual gateway.
`EVOHARNESS_LLM_PROTOCOL=responses` speaks the Responses API instead of
`/chat/completions`; it is explicit and never autodetected, because a gateway
that serves only one of the two rejects the other with the same status it uses
for a bad token. `EVOHARNESS_LLM_TIMEOUT_S` (default 400) is sized to measured
latency — a reasoning model can spend minutes before its first token, and a
timeout under that reads as a dead proposer.

Never commit API keys. Use `--budget-usd` and task-specific evaluation limits
before starting a live run.

## Running inside dsh

dsh (`deepseek-harness`) is an agent runtime. EvoHarness ships a plugin package
and two **agent presets** for it, so "proof mode" and "research mode" are modes
a person picks in a session rather than scripts somebody launches. Everything under [`integrations/dsh/`](integrations/dsh/) is the
source of truth for that integration; its own
[README](integrations/dsh/README.md) explains the mirror in detail.

| preset | what a session can do |
| --- | --- |
| 证明模式 (`proof`) | Open a Lean goal, propose a decomposition and let Lean judge it, hand a subgoal to the solver, read an attempt back, assemble and compile. |
| 研究模式 (`research`) | Read a run and its trajectory, inspect a candidate, check an authored task, read the decision queue and the audit record, and start a run a person approved. |

### Install

Requires a dsh checkout with the `packages/examples/evo-harness` package. Put
it beside EvoHarness, or point `DSH_ROOT` at it:

```bash
scripts/install_dsh_presets.sh
```

The installer copies the plugin sources into the dsh checkout, renders the
preset templates into `$DSH_HOME/.agent-presets/`, and prints what it resolved.
Re-run it after moving either checkout, changing interpreter, or editing a
plugin. It overwrites only the files it owns.

Three things are machine-specific and none of them can live in a checked-in
file, which is why this is a script rather than a directory you copy: the
interpreter EvoHarness is installed in, where runs and tasks live, and where
the plugin modules physically sit. The last one has to be inside the dsh
checkout — the modules import `@deepseek-ai/dsh-tools`, and Node resolves a
bare specifier by walking up from the file's real path, which from anywhere
under `$DSH_HOME` never reaches dsh's dependencies.

Settings the installer reads from the environment or `.env`, with defaults:
`EVO_PYTHON`, `EVO_LEAN_PROJECT`, `EVO_RUNS_ROOT`, `EVO_TASKS_ROOT`,
`EVO_RESEARCH_ROOT`, `EVO_DSH_CONFIG`, `EVO_DSH_RUNTIME`, `EVO_DSH_PROVIDER`,
`DSH_MODEL`. `EVO_LEAN_PROJECT` is optional for core-Lean goals and **required**
the moment a goal says `import Mathlib`, because Mathlib's olean search path
comes from the lake environment and nowhere else.

### What the installer does not configure

A **model route** and a **credential**, deliberately. A route belongs to the
deployment and a key belongs in dsh's credential store — put it in
`$DSH_HOME/.env` or through the Models page, never in a preset. A preset that
carried a route would silently override whatever the host was already using.

`DSH_MODEL` has no default for the same reason: the name has to be one the
candidate config's catalog actually offers, and a wrong one fails every request
and stops a run as `proposer_dead` — a verdict about a proposer that was
working. Left empty, `evo_start` reports what to set and everything read-only
keeps working.

### The proof tools

Start the session in the directory the proof belongs to: the graph is `.evo/`
under that workspace and it outlives the session.

| tool | cost | what it does |
| --- | --- | --- |
| `proof_open` | free | Put a Lean goal on the board and get its `goal_id`. |
| `proof_status` | free | Read the board: open, proved or exhausted goals, attempts, spend, and whether a goal is `certified`. |
| `proof_sketch` | one Lean compile | Propose a decomposition and let Lean judge it. Accepted only if the file compiles and `sorry` appears solely where the subgoals are. |
| `proof_attack` | solver budget, minutes | Hand one goal to EvoHarness to solve, at `L1` (one model call) or `L2` (an agent session that edits and compiles Lean). |
| `proof_attempt` | free | Read one finished attempt back: what the solver tried, what Lean said, what it spent. |
| `proof_assemble` | one Lean compile | Put the proved lemmas back together, compile the whole article, and record a certification. |

Two properties are worth stating because the rest of the design leans on them.

**The board is a record, not a verdict.** A proof session can reach
`.evo/graph.db` with a shell, and that is acceptable for exactly one reason: a
node marked proved by hand still has to survive `proof_assemble`, which
recompiles the finished article and reports the axioms Lean says it depends on,
so a forged `sorry` comes back as `sorryAx`. Do not extend that reasoning to
anything else a session can write.

**Only `task-failed` means the goal resisted.** `infra-failed`,
`budget-exhausted` and `interrupted` say nothing about the goal — they say the
run was stopped. `proof_attempt` reports this as `conclusive`, in the payload
rather than in documentation, because an interrupted solver is holding whatever
line it happened to be on, and that debris presented flatly beside a real
verdict is a story waiting to be built on nothing.

One ceiling governs an attack, not two. The tool derives the solver's own
`--attack-timeout` from the ceiling it will wait, keeping a margin back, so the
solver always stops first and produces `timeout` — an outcome about the goal —
rather than the caller stopping first and producing `interrupted`, which
records no verdict and no cost.

### The research tools

| tool | what it does |
| --- | --- |
| `evo_status` | How a run is doing: generation, best fitness, whether it is still moving, why it stopped. Omit `run` to list every run. Read-only. |
| `evo_trajectory` | What each generation tried and whether anything improved, with a count of each failure kind. Read-only. |
| `evo_decisions` | The queue of decision requests waiting for a person, or one card in full with its evidence references. Read-only: a session can draft a reply, it cannot sign one. |
| `evo_decided` | Cards already answered, newest first — the audit record, not a queue. Read-only. |
| `evo_task_check` | Load an authored task directory the way a run would and report what it found. Loading it is the only way to know the grade function imports and the declaration parses. |
| `evo_start` | Start an evolution run over an authored task. **Asks a person; it does not launch.** Returns once the run is confirmed alive; the run outlives the session. |
| `evo_inspect_candidate` | Read the source of a previously evaluated candidate, for prompts that list reference programs. |

Signing a governance card is deliberately not a tool. The write side lives in
`python -m evoharness.research answer`, which takes no `--actor`: the identity
comes from the operating-system user and is checked against a file beside the
ledger, because an identity the caller supplies at signing time is no identity
at all.

### Editing the plugins

Edit under [`integrations/dsh/package/`](integrations/dsh/package/), run the
installer, test in the dsh checkout. The copy in the checkout is **generated**,
and `package/tests/mirror.spec.ts` runs there and fails naming any file that
differs — editing the copy is the easy mistake, because that is where the test
terminal already is. The mirror check fails rather than skipping when it cannot
find EvoHarness, since a drift check that quietly passes is worse than none.

The package's own tests only run inside the dsh checkout: they resolve
`@deepseek-ai/*` and a shared `tsconfig.base.json` that exist only there.

## Proof mode from a terminal

The dsh tools are a thin wrapper over a CLI that works on its own:

```bash
export EVO_PROOF_WORK=.proof
export EVO_LEAN_PROJECT=tasks/lean_env   # required once a goal imports Mathlib

uv run python -m evoharness.proof.cli open \
  --statement "$(cat statement.txt)" --preamble "$(cat preamble.txt)"
uv run python -m evoharness.proof.cli status --goal <goal_id>
uv run python -m evoharness.proof.cli attack --goal <goal_id> --level L2
uv run python -m evoharness.proof.cli attempt --goal <goal_id>
uv run python -m evoharness.proof.cli assemble --goal <goal_id>
```

Output is JSON in every case. [`tasks/lean_env/`](tasks/lean_env/) carries the
toolchain, lakefile and manifest that reproduce the Lean environment — Lean and
Mathlib pinned to v4.27.0, because changing that version changes the task. The
several gigabytes of built oleans are not committed; `lake exe cache get` in
that directory fetches them once, before the first Mathlib goal compiles.

## Defining a task

The public task entry is `ResolvedTask`. It binds process-local services to an
immutable, serializable `TaskSpec`; runtime resources and search behavior live
in separate `RunSpec` and `SearchProfile` contracts:

```python
from pathlib import Path

from evoharness import ResolvedTask


def grade(candidate_dir: Path, context):
    source = (candidate_dir / "main.py").read_text()
    solved = "return 42" in source
    return {
        "fitness": 1.0 if solved else 0.0,
        "passed": solved,
        "visible_metrics": {"solved": float(solved)},
    }


task = ResolvedTask.from_directory(
    Path("path/to/seed"), grade,
    task_id="return-42",
    version="v1",
    domain_prompt="Make solve() return 42.",
)
```

A task can also be authored as a directory — seed files, prompt, knowledge
chunks, preflight checks and a `task.json` — and loaded by path. See
[`tasks/authored/`](tasks/authored/) for worked examples and check one with
`evo_task_check` or `python -m evoharness.authoring check` before spending a
run on it.

See [`evoharness/contracts/`](evoharness/contracts/) for the frozen public
contracts, [`evoharness/runtime/`](evoharness/runtime/) for runtime resolution,
and [`tasks/demo_counter.py`](tasks/demo_counter.py) for an offline example.

## Repository layout

```text
evoharness/
  contracts/   frozen TaskSpec, RunSpec, and SearchProfile identities
  runtime/     resolved services, graders, and contract compilation
  core/        search loop, candidates, workspaces, proposals, selection
  guard/       budgets, subprocess limits, and anti-hack checks
  evaluation/  evidence envelopes, fault taxonomy, score namespaces
  evoplus/     feedback, behavior, experience, and research extensions
  serve/       local/remote evaluation protocol and service
  proof/       the Lean proof graph, its solver, and its CLI
  research/    experiments, claims, governance ledger, and the signing CLI
  launch/      assembling a run and starting one that outlives its caller
  readout/     reading a run directory from outside Python
  authoring/   authored task directories and their checks
integrations/  the dsh plugin package and agent presets
recipes/       composable search configurations and ablations
tasks/         task registry, authored tasks, and the Lean environment
scripts/       installers and one-off measurement tools
tests/         unit, integration, and protocol tests
```

Two directories are **local-only and not published**: `experiments/` (research
domains, datasets, seeds, and certification bundles) and `docs/` (design
notes). Both are in [`.gitignore`](.gitignore) — a clone of this repository
does not have them, and the test modules bound to them cannot run there. See
[Development](#development).

For the real execution flow, start with
[`evoharness/launch/build.py`](evoharness/launch/build.py), then follow the
selected module under [`recipes/`](recipes/) into `SearchLoop`. Check source,
recipe wiring, and tests before treating a design note as implemented behavior.

## Development

```bash
uv sync --locked --extra dev
uv run pytest -m "not slow"
uv build
```

> [!NOTE]
> In a fresh clone, 22 test modules fail at collection or assertion because
> they import the experiment workspaces under `experiments/`, which is not
> published. The framework's own tests — roughly 1060 of them — pass. Add
> `--continue-on-collection-errors` to run them past the import failures.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Security
issues should follow [`SECURITY.md`](SECURITY.md), not the public issue tracker.

## License

Licensed under the [Apache License 2.0](LICENSE). Third-party attribution and
provenance notes are recorded in [NOTICE](NOTICE) and in the relevant experiment
asset directories.
