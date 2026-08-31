#!/usr/bin/env bash
# Launch a dsh session with the proof tools mounted.
#
# Modelled on `dsh_demo.sh`, and the same reasoning applies: the session's
# tools shell out to Python, so they have to be told which interpreter and
# where the checkout is, and a patch file's relative module paths have to be
# rendered to checkout-local absolute ones before boot. Assembling that by hand
# is how a session ends up half-configured and fails at the first tool call
# with a message about an environment variable.
#
#     scripts/dsh_proof.sh
#
# The session's MODEL comes from your own dsh profile; this patch adds tools
# and nothing else. Refuses to start rather than launching a session whose
# tools cannot work.

set -euo pipefail

fail() { printf '%s\n' "$1" >&2; exit 1; }

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSH_CANDIDATE="${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}"
[ -d "$DSH_CANDIDATE" ] || fail \
  "deepseek-harness checkout not found: set DSH_ROOT or place it beside EvoHarness"
DSH_ROOT="$(cd "$DSH_CANDIDATE" && pwd)"
export DSH_ROOT
DSH_EVO_ROOT="$DSH_ROOT/packages/examples/evo-harness"

if [ -f "$HARNESS_ROOT/.env" ]; then
  set -a; . "$HARNESS_ROOT/.env"; set +a
fi

export EVO_HARNESS_ROOT="$HARNESS_ROOT"
# NOT under /tmp. `workspace-write` grants a candidate the workspace root plus
# the platform temp dir, so a graph there would put the proof state where a
# program being scored can rewrite it.
export EVO_PROOF_WORK="${EVO_PROOF_WORK:-$HOME/evoharness-proof}"
export EVO_LEAN_PROJECT="${EVO_LEAN_PROJECT:-$HARNESS_ROOT/tasks/lean_env}"

if [ -z "${EVO_PYTHON:-}" ]; then
  if [ -x "$HARNESS_ROOT/.venv/bin/python" ]; then
    EVO_PYTHON="$HARNESS_ROOT/.venv/bin/python"
  else
    EVO_PYTHON="$(command -v python3 || true)"
  fi
fi
[ -n "${EVO_PYTHON:-}" ] || fail \
  "no Python interpreter found: set EVO_PYTHON or create $HARNESS_ROOT/.venv"
export EVO_PYTHON
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DSH_ROOT/python/sdk/src"

mkdir -p "$EVO_PROOF_WORK"

"$EVO_PYTHON" -c 'import evoharness' 2>/dev/null || fail \
  "$EVO_PYTHON cannot import evoharness — wrong interpreter?"
command -v lean >/dev/null || fail "no \`lean\` on PATH: every verdict here comes from it"
[ -f "$DSH_EVO_ROOT/src/proof.ts" ] || fail \
  "the plugin is not in the dsh checkout: copy integrations/dsh/proof.ts to $DSH_EVO_ROOT/src/"
[ -f "$DSH_EVO_ROOT/fixtures/host.proof.cordis.yml" ] || fail \
  "the patch is not in the dsh checkout: copy integrations/dsh/host.proof.cordis.yml to $DSH_EVO_ROOT/fixtures/"
[ -f "$DSH_EVO_ROOT/scripts/render-config.mjs" ] || fail \
  "cordis renderer missing: $DSH_EVO_ROOT/scripts/render-config.mjs"

# Warn rather than refuse: opening a goal, reading the board and checking a
# decomposition all work without a model endpoint. Only `proof_attack` spends.
if [ -z "${EVOHARNESS_API_KEY:-}" ] || [ -z "${EVOHARNESS_API_BASE:-}" ]; then
  printf 'attack    EVOHARNESS_API_BASE/KEY unset — proof_attack will refuse\n' >&2
fi

PATCH="$(mktemp "${TMPDIR:-/tmp}/evoharness-dsh-proof.XXXXXX")"
cleanup() { rm -f -- "$PATCH"; }
trap cleanup EXIT
node "$DSH_EVO_ROOT/scripts/render-config.mjs" \
  "$DSH_EVO_ROOT/fixtures/host.proof.cordis.yml" "$PATCH"

printf 'harness   %s\ndsh       %s\nwork      %s\nlean      %s\n\n' \
  "$HARNESS_ROOT" "$DSH_ROOT" "$EVO_PROOF_WORK" "${EVO_LEAN_PROJECT:-bare lean}"

cd "$DSH_ROOT"
node --import tsx/esm apps/cli/src/bin.ts web --patch "$PATCH" "$@"
