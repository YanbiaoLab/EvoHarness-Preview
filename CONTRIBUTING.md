# Contributing to EvoHarness

Thank you for helping improve EvoHarness. The project welcomes focused bug
fixes, tests, documentation improvements, task adapters, and evidence-backed
changes to the search system.

## Before you start

- Search existing issues and pull requests before opening a duplicate.
- Open an issue before a large architectural change so the scope and evaluation
  plan can be agreed on first.
- Do not include private datasets, model credentials, generated run artifacts,
  or third-party material without a compatible license.
- Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).

## Development setup

EvoHarness requires Python 3.10 or newer. The repository uses
[uv](https://docs.astral.sh/uv/) for reproducible environments.

```bash
git clone https://github.com/YanbiaoLab/EvoHarness-Preview.git
cd EvoHarness-Preview
uv sync --locked --extra dev
```

Run the standard checks:

```bash
uv run pytest -m "not slow"
uv build
```

Tests marked `slow` are calibration or benchmark runs and are not part of the
default pull-request gate. If your change affects one of those paths, run the
relevant slow test explicitly and report the environment and observed result.

## Making a change

1. Create a branch from `main`.
2. Keep the patch focused; avoid mixing refactors with behavior changes.
3. Add or update tests for observable behavior.
4. Update public documentation when a contract, command, or artifact changes.
5. Run the standard checks and any relevant experiment smoke test.

The dependency direction is intentional: consumer packages such as `tasks`,
`recipes`, and `experiments` may import `evoharness`, while the framework must
not import those consumers. `tests/test_layering.py` enforces this boundary.

## Experimental claims

Green tests establish implementation correctness, not algorithmic superiority.
When a pull request includes performance or search-quality claims, include:

- the exact task and dataset version;
- model and endpoint configuration;
- recipe, seed, and budget;
- baseline and treatment protocols;
- raw run artifacts or a reproducible summary; and
- limitations such as single-seed results or unequal search spaces.

State clearly whether a feature is implemented, partially wired, design-only,
or still unverified in a real run.

## Pull requests

A reviewable pull request should explain the problem, the chosen change, the
tests run, and any compatibility or security implications. Maintainers may ask
for a smaller patch or additional evidence before merging.

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0 and that you have the right to contribute it.

All contributors must follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
