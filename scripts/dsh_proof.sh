#!/usr/bin/env bash
# Launch a dsh session that can drive the proof graph.
#
#     scripts/dsh_proof.sh
#
# The session's five tools shell out to `python -m evoharness.proof.cli`, so
# they need to be told which interpreter and which checkout, and -- for goals
# that import Mathlib -- the lake project that provides it. Assembling that by
# hand is how a session ends up half-configured and fails at the first tool
# call with a message about an environment variable, so this refuses to start
# instead.
#
# Where the graph goes is NOT set here: it is `.evo/` under the session's own
# workspace. Launch this from the directory the proof belongs to.
#
# `host.proof.cordis.yml` is a PATCH: it adds the tools to whatever profile
# your dsh already runs, and takes the model from there. Patch files resolve
# modules against the selected profile rather than against their own directory,
# which is why it is rendered to absolute paths first rather than passed
# straight to `--patch`.

set -euo pipefail

fail() { printf '%s\n' "$1" >&2; exit 1; }

# Captured FIRST, because this script later cd's into the dsh checkout and the
# directory you launched from is the one the proof belongs to.
EVO_PROOF_PROJECT="${EVO_PROOF_PROJECT:-$PWD}"
export EVO_PROOF_PROJECT

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSH_CANDIDATE="${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}"
[ -d "$DSH_CANDIDATE" ] || fail \
  "deepseek-harness checkout not found: set DSH_ROOT or place it beside EvoHarness"
DSH_ROOT="$(cd "$DSH_CANDIDATE" && pwd)"
export DSH_ROOT
DSH_EVO_ROOT="$DSH_ROOT/packages/examples/evo-harness"

export EVO_HARNESS_ROOT="$HARNESS_ROOT"

# NOT exported as a work directory any more. The graph, the evidence and every
# run directory live in `.evo/` under the SESSION's own workspace, which the
# tools read from the session header on each call -- a web server process
# serves many sessions with different workspaces, so no process-level variable
# can say where they go. `EVO_PROOF_PROJECT` above only picks the workspace for
# the `--repl` path, where this script is the one starting the session.
#
# One consequence to know: the graph now sits inside the directory a
# `workspace-write` session may write to, so the agent's own shell can reach
# `graph.db`. Reading it is the point; nothing stops it writing.

# Omit for core-Lean goals; required the moment a goal says `import Mathlib`,
# because Mathlib's olean search path comes from the lake environment and
# nowhere else.
if [ -z "${EVO_LEAN_PROJECT:-}" ] && [ -d "$HARNESS_ROOT/tasks/lean_env/.lake" ]; then
  export EVO_LEAN_PROJECT="$HARNESS_ROOT/tasks/lean_env"
fi

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
export EVO_PYTHON

# One name for one credential: `EVOHARNESS_API_KEY`, which is what every
# cordis config, the proof CLI and the launch builder now read. A machine
# configured before the endpoint moved off Aliyun still supplies the old name,
# so it is adopted here — in this direction only.
#
# The reverse copy used to be here too, and its own comment forbade it: a key
# adopted INTO `ALIYUN_MAAS_API_KEY` is a credential relabelled as belonging to
# a vendor it does not belong to, and the config reading that name then says
# Aliyun while spending against somewhere else.
if [ -z "${EVOHARNESS_API_KEY:-}" ] && [ -n "${ALIYUN_MAAS_API_KEY:-}" ]; then
  export EVOHARNESS_API_KEY="$ALIYUN_MAAS_API_KEY"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DSH_ROOT/python/sdk/src"

[ -f "$DSH_EVO_ROOT/fixtures/host.proof.cordis.yml" ] || fail \
  "proof patch missing: $DSH_EVO_ROOT/fixtures/host.proof.cordis.yml
run scripts/install_dsh_presets.sh, which copies the authored package there"
[ -f "$DSH_EVO_ROOT/src/proof.ts" ] || fail \
  "proof plugin missing: $DSH_EVO_ROOT/src/proof.ts — run scripts/install_dsh_presets.sh"
[ -f "$DSH_EVO_ROOT/scripts/render-config.mjs" ] || fail \
  "cordis renderer missing: $DSH_EVO_ROOT/scripts/render-config.mjs"
"$EVO_PYTHON" -c 'import evoharness' 2>/dev/null || fail \
  "$EVO_PYTHON cannot import evoharness — wrong interpreter?"

# Warn rather than refuse. Opening goals, reading the board and validating a
# sketch all work without a model endpoint; only `proof_attack` needs one.
if [ -z "${EVOHARNESS_API_KEY:-}" ]; then
  printf 'model     NO CREDENTIAL — proof_attack will fail; the rest works\n' >&2
fi
if [ -z "${EVO_LEAN_PROJECT:-}" ]; then
  printf 'mathlib   not configured — goals importing Mathlib will not compile\n' >&2
fi

PATCH="$(mktemp "${TMPDIR:-/tmp}/evoharness-dsh-proof.XXXXXX")"
cleanup() { rm -f -- "$PATCH"; }
trap cleanup EXIT
node "$DSH_EVO_ROOT/scripts/render-config.mjs" \
  "$DSH_EVO_ROOT/fixtures/host.proof.cordis.yml" "$PATCH"

# `work` is per session, so this reports what the --repl path would use and
# says so; a web session's workspace is chosen in the browser, not here.
printf 'harness   %s\ndsh       %s\nwork      %s\nmathlib   %s\nmodel     %s\nprofile   %s\n\n' \
  "$HARNESS_ROOT" "$DSH_ROOT" "$EVO_PROOF_PROJECT/.evo (--repl; a web session uses its own workspace)" \
  "${EVO_LEAN_PROJECT:-unset}" "${DSH_MODEL:-profile default}" \
  "${DSH_PROFILE:-web}"

# Which profile to boot. `web` serves a browser session and needs that
# profile's client bundles built; `tui` is the same conversation in a terminal;
# `headless` answers one task and exits, which is what a smoke wants.
DSH_PROFILE="${DSH_PROFILE:-web}"

# `--repl` skips the profile entirely: one runtime, the same five tools, and
# the conversation on stdin. It needs no client bundles, which is what makes it
# usable in a checkout whose web profile has not been built.
if [ "${1:-}" = "--repl" ]; then
  shift
  exec "$EVO_PYTHON" "$HARNESS_ROOT/integrations/dsh/proof_repl.py" "$@"
fi

cd "$DSH_ROOT"
node --import tsx/esm apps/cli/src/bin.ts \
  --profile "$DSH_PROFILE" --patch "$PATCH" "$@"
