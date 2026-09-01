#!/usr/bin/env bash
# Launch a dsh session wired to this checkout.
#
# The session's tools shell out to Python, so they need to be told which
# interpreter and where the checkout is; and they resolve run and task NAMES
# under configured roots rather than accepting paths, so those roots have to
# exist before the session does. Assembling that by hand is how a session ends
# up half-configured and failing at the first tool call with a message about
# an environment variable.
#
#     scripts/dsh_demo.sh
#
# Refuses to start rather than launching a session whose tools cannot work.

set -euo pipefail

fail() { printf '%s\n' "$1" >&2; exit 1; }

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DSH_CANDIDATE="${DSH_ROOT:-$HARNESS_ROOT/../deepseek-harness}"
[ -d "$DSH_CANDIDATE" ] || fail \
  "deepseek-harness checkout not found: set DSH_ROOT or place it beside EvoHarness"
DSH_ROOT="$(cd "$DSH_CANDIDATE" && pwd)"
export DSH_ROOT
DSH_EVO_ROOT="$DSH_ROOT/packages/examples/evo-harness"

# Run directories must not sit under a temp area: `workspace-write` grants a
# candidate the workspace root plus /tmp and the platform temp dir, so a run
# directory there would put run.db, the evidence ledger and the checkpoint
# where the program being scored can rewrite them.
export EVO_RUNS_ROOT="${EVO_RUNS_ROOT:-$HOME/evoharness-runs}"
export EVO_TASKS_ROOT="${EVO_TASKS_ROOT:-$HARNESS_ROOT/tasks/authored}"
export EVO_RESEARCH_ROOT="${EVO_RESEARCH_ROOT:-$HOME/evoharness-research}"
export EVO_HARNESS_ROOT="$HARNESS_ROOT"

export EVO_DSH_CONFIG="${EVO_DSH_CONFIG:-$DSH_EVO_ROOT/fixtures/candidate.cordis.yml}"
export EVO_DSH_RUNTIME="${EVO_DSH_RUNTIME:-$DSH_ROOT/packages/examples/jsonrpc-demo/src/bin.ts}"
export EVO_DSH_PROVIDER="${EVO_DSH_PROVIDER:-evo-gateway}"

# The candidate runtime is a subprocess of the run, which is a subprocess of
# nothing this script starts — but both inherit from here, so the model
# endpoint, the judge and the dsh SDK path all have to be set here or nowhere.
if [ -f "$HARNESS_ROOT/.env" ]; then
  set -a; . "$HARNESS_ROOT/.env"; set +a
fi
# dsh refuses to boot when its own .env declares a bootstrap-only name, so the
# old DEEPSEEK_* values live beside it and are exported from here instead.
if [ -f "$DSH_ROOT/.env.moved-by-demo.bak" ]; then
  set -a; . "$DSH_ROOT/.env.moved-by-demo.bak"; set +a
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

export EVOHARNESS_API_BASE="${EVOHARNESS_API_BASE:-https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
# The DSH catalog names the Aliyun variable. Expose that same credential under
# the old EvoHarness name for its in-process transport, but never reinterpret
# an existing non-Aliyun credential as an Aliyun one.
if [ -z "${EVOHARNESS_API_KEY:-}" ] && [ -n "${ALIYUN_MAAS_API_KEY:-}" ]; then
  export EVOHARNESS_API_KEY="$ALIYUN_MAAS_API_KEY"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$DSH_ROOT/python/sdk/src"
export DSH_MODEL="${DSH_MODEL:-deepseek-v4-pro}"

mkdir -p "$EVO_RUNS_ROOT"

[ -n "${ALIYUN_MAAS_API_KEY:-}" ] || fail \
  "ALIYUN_MAAS_API_KEY is not set: export it in the launching environment"
[ -d "$EVO_TASKS_ROOT" ] || fail "EVO_TASKS_ROOT does not exist: $EVO_TASKS_ROOT"
[ -f "$EVO_DSH_CONFIG" ] || fail "candidate config missing: $EVO_DSH_CONFIG"
[ -f "$EVO_DSH_RUNTIME" ] || fail "dsh runtime entry missing: $EVO_DSH_RUNTIME"
[ -f "$DSH_EVO_ROOT/fixtures/demo.cordis.yml" ] || fail \
  "demo config missing: $DSH_EVO_ROOT/fixtures/demo.cordis.yml"
[ -f "$DSH_EVO_ROOT/scripts/render-config.mjs" ] || fail \
  "cordis renderer missing: $DSH_EVO_ROOT/scripts/render-config.mjs"
"$EVO_PYTHON" -c 'import evoharness' 2>/dev/null || fail \
  "$EVO_PYTHON cannot import evoharness — wrong interpreter?"
"$EVO_PYTHON" -c 'import deepseek_harness' 2>/dev/null || fail \
  "$EVO_PYTHON cannot import deepseek_harness — check PYTHONPATH"

# The Lean judge is a separate service and the ETP task has no offline mode:
# a run without it would report numbers that look like results. Warn rather
# than refuse, because reading runs and checking tasks work without it.
if [ -n "${ETP_JUDGE_URL:-}" ] \
   && curl -sf --max-time 5 "$ETP_JUDGE_URL/health" >/dev/null 2>&1; then
  printf 'judge     %s (up)\n' "$ETP_JUDGE_URL"
else
  printf 'judge     %s — NOT ANSWERING; ETP runs will stop rather than score\n' \
    "${ETP_JUDGE_URL:-unset}" >&2
fi

DEMO_PATCH="$(mktemp "${TMPDIR:-/tmp}/evoharness-dsh-demo.XXXXXX")"
cleanup() { rm -f -- "$DEMO_PATCH"; }
trap cleanup EXIT
node "$DSH_EVO_ROOT/scripts/render-config.mjs" \
  "$DSH_EVO_ROOT/fixtures/demo.cordis.yml" "$DEMO_PATCH"

printf 'harness   %s\ndsh       %s\nruns      %s\ntasks     %s\nmodel     %s\nendpoint  %s\n\n' \
  "$HARNESS_ROOT" "$DSH_ROOT" "$EVO_RUNS_ROOT" "$EVO_TASKS_ROOT" \
  "$DSH_MODEL" "$EVOHARNESS_API_BASE"

cd "$DSH_ROOT"
node --import tsx/esm apps/cli/src/bin.ts web --patch "$DEMO_PATCH" "$@"
