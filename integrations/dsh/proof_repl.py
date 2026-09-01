#!/usr/bin/env python3
"""Talk to a dsh session that can drive the proof graph, from a terminal.

    scripts/dsh_proof.sh --repl        # or set the env yourself and run this

The same runtime and the same five tools the web session would mount; the only
difference is that the conversation arrives on stdin instead of over HTTP. That
makes it usable before the web profile's client bundles are built, and it makes
what happened readable afterwards without a browser.

The session is persistent: every turn reuses one `session_id`, so the agent
remembers the goal you opened three messages ago. The graph outlives even that
-- it is on disk in `.evo/` under the project directory, so `/graph` still
answers after this process is gone, and a long-running controller can pick up
where you left off.

**The project directory is the argument, defaulting to the one you ran from.**
It is the session's workspace, which is where the tools read it from; a
throwaway temporary directory would put the graph somewhere that disappears
with it.

Commands, which never reach the model:

    /graph     the board as the CLI reports it, not as the agent describes it
    /tools     which proof tools this runtime actually mounted
    /quit
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DSH_ROOT = Path(
    os.environ.get("DSH_ROOT", Path.home() / "Documents/Projects/deepseek-harness")
).expanduser().resolve()
CONFIG = Path(os.environ.get(
    "EVO_PROOF_CORDIS",
    DSH_ROOT / "packages/examples/evo-harness/fixtures/smoke.proof.cordis.yml",
)).expanduser().resolve()
RUNTIME = DSH_ROOT / "packages/examples/jsonrpc-demo/src/bin.ts"

REQUIRED = ("EVO_PYTHON", "EVO_HARNESS_ROOT")

#: Mirrors `WORK_DIR` in `proof.ts`. The tools derive this from the session's
#: own workspace; `/graph` has to name it explicitly to read the same board.
WORK_DIR = ".evo"

SYSTEM = """\
You help a mathematician prove one Lean goal, using the proof_* tools.

You decide which way is worth trying. You never decide whether something is
correct: proof_sketch returns Lean's verdict on a decomposition, proof_attack
returns an outcome derived from the run itself, and proof_assemble returns the
compiler's answer about the finished proof. Report what they said rather than
your reading of it, and quote a rejection reason verbatim.

Work in small steps and say what you are about to do before an expensive call.
proof_attack costs real money and minutes; proof_sketch costs one Lean compile
and no model budget, so check a route before spending on it.
"""


def graph(work: Path) -> str:
    """The board as the CLI reports it, read from the SAME directory the tools
    write to. Passed explicitly rather than left to a default: a `/graph` that
    silently read a different graph would answer `no goals` about work that did
    happen, which is worse than not answering."""

    done = subprocess.run(
        [os.environ["EVO_PYTHON"], "-m", "evoharness.proof.cli",
         "--work", str(work), "status"],
        cwd=os.environ["EVO_HARNESS_ROOT"], capture_output=True, text=True,
    )
    if done.returncode != 0:
        return (done.stderr or done.stdout)[-800:]
    try:
        return json.dumps(json.loads(done.stdout), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return done.stdout


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            f"not configured: {', '.join(missing)} — run via scripts/dsh_proof.sh"
        )
    for label, path in (("cordis config", CONFIG), ("runtime entry", RUNTIME)):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    from deepseek_harness import DeepSeekHarness

    # The session's workspace, and therefore where its tools put the graph.
    workspace = Path(
        sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("EVO_PROOF_PROJECT") or os.getcwd()
    ).expanduser().resolve()
    if not workspace.is_dir():
        raise SystemExit(f"not a directory: {workspace}")
    work = workspace / WORK_DIR
    sessions = work / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)

    print(f"project  {workspace}")
    print(f"work     {work}")
    print(f"mathlib  {os.environ.get('EVO_LEAN_PROJECT', 'unset')}")
    print(f"model    {os.environ.get('DSH_MODEL', 'gpt-5.6-sol')}")
    print("booting the runtime...\n")

    with DeepSeekHarness(
        provider=os.environ.get("EVO_DSH_PROVIDER", "evo-gateway"),
        model=os.environ.get("DSH_MODEL", "gpt-5.6-sol"),
        cwd=str(workspace),
        runtime_cwd=str(DSH_ROOT),
        session_root=str(sessions),
        cordis=str(CONFIG),
        launch_args_override=("node", "--import", "tsx/esm", str(RUNTIME)),
        env={
            key: value for key, value in os.environ.items()
            if key.startswith(("EVO_", "EVOHARNESS_", "DSH_", "ALIYUN_"))
        },
        request_timeout_seconds=1800,
    ) as harness:
        print("ready. /graph reads the board, /quit leaves.\n")
        turn = 0
        while True:
            try:
                line = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/graph":
                print(graph(work), "\n")
                continue
            if line == "/tools":
                # Asked of the runtime, not remembered from the config: a row
                # in a yaml file is not evidence that a plugin mounted.
                print("(ask the agent: 'list your tools')\n")
                continue

            turn += 1
            prompt = f"{SYSTEM}\n\n{line}" if turn == 1 else line
            result = harness.run(prompt, session_id="proof-repl")
            called = [
                str(event.get("data", {}).get("name"))
                for event in result.events
                if event.get("type") == "tool/call"
                and str(event.get("data", {}).get("name", "")).startswith("proof_")
            ]
            print(f"\ndsh> {result.final_response}\n")
            if called:
                print(f"     [tools: {', '.join(called)}]\n")

    print("\n--- the graph, on disk, after the session ---")
    print(graph(work))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
