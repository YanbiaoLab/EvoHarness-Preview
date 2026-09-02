#!/usr/bin/env python3
"""Minimal SDK smoke for the `DshAgentBackend` architecture.

One dsh runtime is started per candidate, the candidate edits a file in its own
workspace, and the run's events come back to Python. It answers the four facts
the backend contract needs before it can be written:

  1. does `DSH_CORDIS_CONFIG` mount the guard plugin;
  2. what delegation depth does the candidate sit at (the guard's predicate
     reads exactly this number);
  3. is per-request token usage recoverable from `RunResult.events`;
  4. what does one runtime cost in wall-clock time.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from deepseek_harness import DeepSeekHarness

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG = Path(__file__).with_name("candidate.cordis.yml")
RUNTIME_ENTRY = REPO_ROOT / "packages/examples/jsonrpc-demo/src/bin.ts"

PROMPT = (
    "Do exactly two things, in order.\n"
    "1. Write the single line CANDIDATE_WAS_HERE into a file named proof.txt "
    "in the current directory.\n"
    "2. Call the evo_spike_tools tool once and report its raw JSON output verbatim.\n"
    "Then stop."
)


def token_usage(events: list[dict]) -> dict[str, int]:
    """Sum the per-step usage the session log carries on `assistant/message`."""
    totals: dict[str, int] = {}
    for event in events:
        if event.get("type") != "assistant/message":
            continue
        usage = event.get("data", {}).get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def spike_report(events: list[dict]) -> dict | None:
    """Recover the probe's payload from the session log rather than from prose.

    `tool/result` carries no tool name — the name lives on the paired
    `tool/call`, and the two are correlated by call id (`data.callId` on the
    call, `data.message.source.callId` on the result). Reading the log instead
    of the assistant's summary is the point: a candidate's own account of what a
    tool returned is not evidence.
    """
    wanted: set[str] = set()
    for event in events:
        data = event.get("data", {})
        if event.get("type") == "tool/call" and data.get("name") == "evo_spike_tools":
            wanted.add(str(data.get("callId")))
        if event.get("type") != "tool/result":
            continue
        message = data.get("message", {})
        if str(message.get("source", {}).get("callId")) not in wanted:
            continue
        for block in message.get("content", []):
            for inner in block.get("content", []):
                if inner.get("type") != "text":
                    continue
                try:
                    return json.loads(inner["text"])
                except json.JSONDecodeError:
                    return {"unparsed": inner["text"][:400]}
    return None


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="evo-candidate-"))
    session_root = Path(tempfile.mkdtemp(prefix="evo-sessions-"))
    started = time.monotonic()
    try:
        with DeepSeekHarness(
            provider="deepseek-official",
            model=os.environ.get("DSH_MODEL", "deepseek-v4-flash"),
            cwd=str(workspace),
            runtime_cwd=str(REPO_ROOT),
            session_root=str(session_root),
            cordis=str(CONFIG),
            launch_args_override=("node", "--import", "tsx/esm", str(RUNTIME_ENTRY)),
            env={"EVO_SPIKE_REPORT": "1"},
            request_timeout_seconds=300,
        ) as harness:
            booted = time.monotonic()
            result = harness.run(PROMPT, session_id="evo-smoke")
        finished = time.monotonic()

        proof = workspace / "proof.txt"
        print(json.dumps({
            "boot_seconds": round(booted - started, 2),
            "turn_seconds": round(finished - booted, 2),
            "finish_reason": result.finish_reason,
            "workspace_changed": proof.exists(),
            "proof_text": proof.read_text().strip() if proof.exists() else None,
            "event_types": sorted({event.get("type", "?") for event in result.events}),
            "token_usage": token_usage(result.events),
            "spike_report": spike_report(result.events),
        }, ensure_ascii=False, indent=2))
        print("\n--- final response ---")
        print(result.final_response[:1200])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(session_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
