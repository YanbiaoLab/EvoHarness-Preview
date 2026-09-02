# integrations/dsh

Everything EvoHarness runs **inside a dsh process**, and the source of truth
for all of it.

```
package/    a byte-for-byte mirror of the dsh checkout's
            packages/examples/evo-harness — plugins, their specs, fixtures,
            package.json, tsconfig.json
presets/    agent-preset templates, rendered into $DSH_HOME/.agent-presets/
*.py        entry points that drive a dsh session from Python
```

`scripts/install_dsh_presets.sh` copies `package/` into the dsh checkout and
renders `presets/`. Edit here; run the installer; test there.

## Why the mirror exists

Two facts decide this, and neither is about taste.

**The dsh checkout is not a place to keep anything.** `packages/examples/` does
not exist in its `origin/master`: the whole `evo-harness` package is a local
addition on a branch with no upstream. Anything kept only there exists on one
machine and in no history. EvoHarness has a remote, which is the whole argument
for the direction.

**A copy is unavoidable, because of how Node resolves imports.** Every plugin
imports `@deepseek-ai/dsh-tools` and friends, and Node resolves a bare
specifier by walking up from the file's real path. From anywhere in EvoHarness
that walk never reaches dsh's dependencies, and a symlink does not help: it is
resolved to its target before the module's own imports are. So the files have
to physically be inside the dsh package to compile, to typecheck, and to run
under its vitest.

Hence: authored here, copied there, and the copy is generated.

## What stops the copy from drifting

`package/tests/mirror.spec.ts` — it runs in the dsh checkout, which is where
the temptation is, and fails naming any file that differs. Editing the copy is
the easy mistake: the tests run there, so that is where a person already has a
terminal open, and nothing about the file says it is generated.

The check locates EvoHarness through `EVO_HARNESS_ROOT`, then beside the dsh
checkout. Finding neither, it fails rather than skipping — a mirror check that
quietly passes when it cannot see the original is worse than none, because the
suite then reports green for exactly the case it exists to catch.

## What is NOT mirrored

`presets/` and the Python entry points, because nothing in the dsh package
consumes them: a preset is rendered into `$DSH_HOME`, and `proof_repl.py` runs
as EvoHarness. They have one copy each and live only here.

`node_modules/` and `lib/` are dsh's to produce.
