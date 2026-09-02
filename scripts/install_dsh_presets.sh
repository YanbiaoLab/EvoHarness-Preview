#!/usr/bin/env bash
# Install EvoHarness's dsh agent presets.
#
#     scripts/install_dsh_presets.sh
#
# After this, 「证明模式」and「研究模式」appear in dsh's preset roster and a
# session that picks one has the matching tools — with no launcher in the loop.
#
#   proof     open a Lean goal, sketch a decomposition, dispatch a subgoal,
#             assemble and compile. Lean decides; the model chooses routes.
#   research  read a run and its trajectory, check a task, read the decision
#             queue and the audit record, and start a run a person approved.
#
# This replaces `scripts/dsh_demo.sh`, which mounted the research tools by
# rendering a patch file, exporting nine variables and moving the dsh
# checkout's own `.env` aside so dsh would boot. What that script configured by
# exporting, a preset declares on its plugin rows.
#
# Three things are machine-specific and none can live in a checked-in file,
# which is why this is a script and not a directory you copy:
#
#   - the interpreter EvoHarness is installed in, and this checkout's path;
#   - where runs, tasks and the research ledger live;
#   - where the plugin modules really are, which has to be inside the dsh
#     checkout: they import `@deepseek-ai/dsh-tools`, and Node resolves that by
#     walking up from the file, which never reaches the harness from $DSH_HOME.
#
# What this does NOT install is a model route or a credential. A route belongs
# to the deployment, and a key belongs in dsh's credential store — put it in
# `$DSH_HOME/.env` or through the Models page, never in a preset.
#
# Re-run it after moving either checkout, changing interpreter, or editing the
# plugins; it overwrites the files it owns and touches nothing else.

set -euo pipefail

fail() { printf '%s\n' "$1" >&2; exit 1; }

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DSH_CANDIDATE="${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}"
[ -d "$DSH_CANDIDATE" ] || fail \
  "deepseek-harness checkout not found: set DSH_ROOT or place it beside EvoHarness"
DSH_ROOT="$(cd "$DSH_CANDIDATE" && pwd)"
DSH_EVO_ROOT="$DSH_ROOT/packages/examples/evo-harness"
[ -d "$DSH_EVO_ROOT/src" ] || fail \
  "not a dsh checkout with the evo-harness example package: $DSH_EVO_ROOT/src"

SOURCE_ROOT="$HARNESS_ROOT/integrations/dsh/presets"
[ -d "$SOURCE_ROOT" ] || fail "preset templates missing: $SOURCE_ROOT"
PACKAGE_MIRROR="$HARNESS_ROOT/integrations/dsh/package"
[ -d "$PACKAGE_MIRROR/src" ] || fail "plugin sources missing: $PACKAGE_MIRROR/src"

# Read the same `.env` the launchers read, so a machine already configured for
# `dsh_proof.sh` needs nothing extra here.
if [ -f "$HARNESS_ROOT/.env" ]; then
  set -a; . "$HARNESS_ROOT/.env"; set +a
fi

if [ -z "${EVO_PYTHON:-}" ]; then
  if [ -x "$HARNESS_ROOT/.venv/bin/python" ]; then
    EVO_PYTHON="$HARNESS_ROOT/.venv/bin/python"
  else
    EVO_PYTHON="$(command -v python3 || true)"
  fi
fi
[ -n "${EVO_PYTHON:-}" ] || fail \
  "no Python interpreter found: set EVO_PYTHON or create $HARNESS_ROOT/.venv"
(cd "$HARNESS_ROOT" && "$EVO_PYTHON" -c 'import evoharness') 2>/dev/null || fail \
  "$EVO_PYTHON cannot import evoharness — wrong interpreter?"

# Omitted for core-Lean goals; required the moment a goal says `import Mathlib`,
# because Mathlib's olean search path comes from the lake environment and
# nowhere else. Empty reaches a plugin as unset.
if [ -z "${EVO_LEAN_PROJECT:-}" ] && [ -d "$HARNESS_ROOT/tasks/lean_env/.lake" ]; then
  EVO_LEAN_PROJECT="$HARNESS_ROOT/tasks/lean_env"
fi

# Run directories deliberately NOT under a temp root: a workspace-write
# candidate may write there, and a run directory is where its own score is
# read from.
EVO_RUNS_ROOT="${EVO_RUNS_ROOT:-$HOME/evoharness-runs}"
EVO_TASKS_ROOT="${EVO_TASKS_ROOT:-$HARNESS_ROOT/tasks/authored}"
EVO_RESEARCH_ROOT="${EVO_RESEARCH_ROOT:-$HOME/evoharness-research}"
EVO_DSH_CONFIG="${EVO_DSH_CONFIG:-$DSH_EVO_ROOT/fixtures/candidate.cordis.yml}"
EVO_DSH_RUNTIME="${EVO_DSH_RUNTIME:-$DSH_ROOT/packages/examples/jsonrpc-demo/src/bin.ts}"
EVO_DSH_PROVIDER="${EVO_DSH_PROVIDER:-evo-gateway}"

# No default, deliberately. The name has to be one the candidate config's
# catalog actually offers, and no value here can know that — a wrong one
# 404s every request and stops the run as `proposer_dead`, a verdict about a
# proposer that was working. Empty leaves `evo_start` reporting what to set
# while everything read-only keeps working.
DSH_MODEL="${DSH_MODEL:-}"

EVO_LEAN_PROJECT="${EVO_LEAN_PROJECT:-}"

# The template quotes each value in single quotes, so one inside a path would
# end the string and leave a composition that parses as something else.
for value in "$EVO_PYTHON" "$HARNESS_ROOT" "$EVO_LEAN_PROJECT" "$EVO_RUNS_ROOT" \
             "$EVO_TASKS_ROOT" "$EVO_RESEARCH_ROOT" "$EVO_DSH_CONFIG" \
             "$EVO_DSH_RUNTIME" "$EVO_DSH_PROVIDER" "$DSH_MODEL"; do
  case "$value" in
    *"'"*) fail "values containing a single quote cannot be rendered into a preset: $value" ;;
  esac
done

# EvoHarness authors the whole dsh example package; the dsh checkout holds the
# copy that can compile. Copying every authored file on every install is what
# keeps the two from disagreeing, and `package/tests/mirror.spec.ts` fails in
# the dsh suite when they do.
#
# Only files the mirror carries are written. `node_modules/` and `lib/` are the
# dsh toolchain's to produce, and deleting anything not in the mirror would
# take a half-finished local experiment with it — the drift check reports an
# extra file rather than this removing it.
synced=0
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  mkdir -p "$DSH_EVO_ROOT/$(dirname "$rel")"
  cp "$PACKAGE_MIRROR/$rel" "$DSH_EVO_ROOT/$rel"
  synced=$((synced + 1))
done <<EOF
$(cd "$PACKAGE_MIRROR" && find . -type f | sed 's|^\./||')
EOF
[ "$synced" -gt 0 ] || fail "the package mirror is empty: $PACKAGE_MIRROR"

mkdir -p "$EVO_RUNS_ROOT"

# Exported because the renderer reads tokens by name from the environment:
# a template that grows a setting is then one this script already supplies,
# rather than one that also needs its argument list edited.
export EVO_PYTHON EVO_LEAN_PROJECT EVO_RUNS_ROOT EVO_TASKS_ROOT \
       EVO_RESEARCH_ROOT EVO_DSH_CONFIG EVO_DSH_RUNTIME EVO_DSH_PROVIDER \
       DSH_MODEL
export EVO_HARNESS_ROOT="$HARNESS_ROOT"

DSH_HOME="${DSH_HOME:-$HOME/.dsh}"
PRESET_ROOT="$DSH_HOME/.agent-presets"

for source in "$SOURCE_ROOT"/*/; do
  id="$(basename "$source")"
  target="$PRESET_ROOT/$id"
  mkdir -p "$target"

  "$EVO_PYTHON" - "$source/agent.cordis.yml" "$target/agent.cordis.yml" <<'PY'
import os
import pathlib
import re
import sys

source, target = sys.argv[1:3]
text = pathlib.Path(source).read_text(encoding="utf-8")

# Every token the template uses must be one this installer resolved. An
# unknown one is a template that grew a setting nobody supplies, and leaving
# it verbatim produces a config whose first tool call reports a path literally
# called `@EVO_SOMETHING@`.
declared = set(re.findall(r"@([A-Z_]+)@", text))
unknown = sorted(name for name in declared if name not in os.environ)
if unknown:
    raise SystemExit(f"{source}: no value for {', '.join(unknown)}")
for name in declared:
    text = text.replace(f"@{name}@", os.environ[name])

pathlib.Path(target).write_text(text, encoding="utf-8")
PY

  if [ -f "$source/preset.yml" ]; then
    cp "$source/preset.yml" "$target/preset.yml"
  fi

  # Links rather than copies: Node resolves the real path before it resolves a
  # module's own imports, so the plugin's `@deepseek-ai/*` rows come from the
  # dsh checkout while the composition beside it stays free of absolute paths.
  # Derived from the composition rather than listed here, so a preset that
  # mounts another plugin does not also need this script edited.
  #
  # `.ts` is deliberate — dsh boots through tsx from a source checkout.
  # Against an installed dsh with no tsx, point the rows at `lib/types/*.js`.
  linked=""
  while read -r module; do
    [ -n "$module" ] || continue
    [ -f "$DSH_EVO_ROOT/src/$module" ] || fail \
      "$id names './$module', which is not in $DSH_EVO_ROOT/src"
    ln -sfn "$DSH_EVO_ROOT/src/$module" "$target/$module"
    linked="$linked $module"
  done <<EOF
$(sed -n "s/^[[:space:]]*name: '\.\/\([A-Za-z0-9_-]*\.ts\)'[[:space:]]*$/\1/p" \
   "$source/agent.cordis.yml")
EOF

  printf '%-10s %s  (%s )\n' "$id" "$target" "$linked"
done

printf '\npackage   %s files -> %s\n' "$synced" "$DSH_EVO_ROOT"
printf 'python    %s\nharness   %s\nruns      %s\ntasks     %s\nresearch  %s\nmathlib   %s\nmodel     %s\n' \
  "$EVO_PYTHON" "$HARNESS_ROOT" "$EVO_RUNS_ROOT" "$EVO_TASKS_ROOT" \
  "$EVO_RESEARCH_ROOT" "${EVO_LEAN_PROJECT:-unset — goals importing Mathlib will not compile}" \
  "${DSH_MODEL:-unset — evo_start will report it; reading runs is unaffected}"

printf '\nPick a mode in a dsh session. For 证明模式, start the session in the\n'
printf 'directory the proof belongs to: the graph is `.evo/` under that\n'
printf 'workspace and it outlives the session.\n'

if [ -z "${EVOHARNESS_API_KEY:-}" ] && [ ! -f "$DSH_HOME/.env" ]; then
  printf '\nNo EVOHARNESS_API_KEY in this shell and no %s/.env.\n' "$DSH_HOME"
  printf 'That credential is read by the dsh PROCESS, not by this script:\n'
  printf 'proof_attack and evo_start need it, while reading runs, opening goals\n'
  printf 'and sketching keep working without it.\n'
fi
