#!/usr/bin/env python3
"""Drive one dsh session through the proof tools, and read the session log.

    export EVO_PYTHON=$(cd /path/to/EvoHarness && uv run which python)
    export EVO_HARNESS_ROOT=/path/to/EvoHarness
    export EVO_PROOF_WORK=/tmp/dsh_proof
    export EVO_LEAN_PROJECT=/path/to/EvoHarness/tasks/lean_env
    export EVOHARNESS_API_BASE=... EVOHARNESS_API_KEY=... DSH_MODEL=gpt-5.6-sol
    uv run python integrations/dsh/session_smoke.py

What it checks is not "did the agent say it worked". It reads `tool/call` and
`tool/result` out of the session log, for the reason the existing smoke gives:
**a session's own account of what a tool returned is not evidence.** The agent
could describe a decomposition Lean rejected, or claim a lemma is proved that
the graph never recorded; the log and the graph are what settle it.

`proof_attack` is deliberately NOT asked for here. It is the call that spends
real money and minutes, and this smoke exists to prove the wiring. Attacking is
one more tool call once the first four answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DSH_ROOT = Path(
    os.environ.get("DSH_ROOT", Path.home() / "Documents/Projects/deepseek-harness")
).expanduser().resolve()
CONFIG = DSH_ROOT / "packages/examples/evo-harness/fixtures/smoke.proof.cordis.yml"
RUNTIME = DSH_ROOT / "packages/examples/jsonrpc-demo/src/bin.ts"

GOAL = (
    "theorem dsh_demo (a b c : Nat) : "
    "(a + b) * c = a * c + b * c ∧ a * 0 = 0"
)

PROMPT = f"""\
We are working on one Lean goal together. Do these in order and stop.

1. Call proof_open with this statement, exactly as written:

{GOAL}

2. Call proof_status to read the board back.

3. Propose a decomposition with proof_sketch that DELIBERATELY leaves `sorry`
   in the parent body, so we can see the rejection. Report the reason verbatim.

4. Propose a real decomposition with proof_sketch: two lemmas, one for each
   conjunct, and a parent body that closes the goal from them.

5. Call proof_status once more and tell me which subgoals are now open.

Do not call proof_attack. Report each tool's raw json.
"""

REQUIRED = (
    "EVO_PYTHON", "EVO_HARNESS_ROOT", "EVO_PROOF_WORK",
    "EVOHARNESS_API_BASE", "EVOHARNESS_API_KEY",
)


def tool_traffic(events: list[dict]) -> list[dict]:
    """Every proof_* call and the result that came back, from the log itself."""

    calls: dict[str, str] = {}
    traffic: list[dict] = []
    for event in events:
        data = event.get("data", {})
        if event.get("type") == "tool/call":
            name = str(data.get("name", ""))
            if name.startswith("proof_"):
                calls[str(data.get("callId"))] = name
            continue
        if event.get("type") != "tool/result":
            continue
        message = data.get("message", {})
        call_id = str(message.get("source", {}).get("callId"))
        if call_id not in calls:
            continue
        text = ""
        for block in message.get("content", []):
            for inner in block.get("content", []):
                if inner.get("type") == "text":
                    text = inner["text"]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"unparsed": text[:300]}
        traffic.append({"tool": calls[call_id], "result": payload})
    return traffic


def graph_state() -> dict:
    """Ask the CLI directly. The graph is the fact; the transcript is a story."""

    done = subprocess.run(
        [os.environ["EVO_PYTHON"], "-m", "evoharness.proof.cli", "status"],
        cwd=os.environ["EVO_HARNESS_ROOT"], capture_output=True, text=True,
        env=os.environ,
    )
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"error": done.stdout[:200] + done.stderr[-400:]}


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"not configured: {', '.join(missing)}")
    for label, path in (("cordis config", CONFIG), ("runtime entry", RUNTIME)):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    from deepseek_harness import DeepSeekHarness

    workspace = Path(tempfile.mkdtemp(prefix="dsh-proof-"))
    sessions = Path(tempfile.mkdtemp(prefix="dsh-proof-sessions-"))
    started = time.monotonic()
    with DeepSeekHarness(
        provider="evo-gateway",
        model=os.environ.get("DSH_MODEL", "gpt-5.6-sol"),
        cwd=str(workspace),
        runtime_cwd=str(DSH_ROOT),
        session_root=str(sessions),
        cordis=str(CONFIG),
        launch_args_override=("node", "--import", "tsx/esm", str(RUNTIME)),
        env={
            key: os.environ[key]
            for key in (*REQUIRED, "EVO_LEAN_PROJECT", "DSH_MODEL")
            if os.environ.get(key)
        },
        request_timeout_seconds=900,
    ) as harness:
        booted = time.monotonic()
        result = harness.run(PROMPT, session_id="proof-smoke")
    finished = time.monotonic()

    traffic = tool_traffic(result.events)
    print(json.dumps({
        "boot_seconds": round(booted - started, 1),
        "turn_seconds": round(finished - booted, 1),
        "finish_reason": str(result.finish_reason),
        "proof_tools_called": [item["tool"] for item in traffic],
    }, ensure_ascii=False, indent=2))

    print("\n--- what each tool actually returned (from the log) ---")
    for item in traffic:
        summary = {
            k: v for k, v in item["result"].items()
            if k in ("accepted", "stage", "reason", "error")
        }
        print(f"  {item['tool']:16s} {json.dumps(summary, ensure_ascii=False)[:180]}")

    print("\n--- the graph, asked directly ---")
    print(json.dumps(graph_state(), ensure_ascii=False, indent=2)[:1500])

    print("\n--- final response ---")
    print(result.final_response[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
