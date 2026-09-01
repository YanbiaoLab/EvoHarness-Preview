import json
import os
import shutil
import tempfile
from pathlib import Path

from evoharness.core import AgentSessionRequest, Candidate, AgentSessionLimits, ProposalPreflight, PreflightPipeline
from evoharness.core.agent import DshRuntimeSpec, DshAgentBackend

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DSH_ROOT = Path(
    os.environ.get("DSH_ROOT", HARNESS_ROOT.parent / "deepseek-harness")
).expanduser().resolve()
CONFIG = Path(
    os.environ.get(
        "EVO_DSH_CONFIG",
        DSH_ROOT / "packages/examples/evo-harness/fixtures/candidate.cordis.yml",
    )
).expanduser().resolve()
RUNTIME = Path(
    os.environ.get(
        "EVO_DSH_RUNTIME",
        DSH_ROOT / "packages/examples/jsonrpc-demo/src/bin.ts",
    )
).expanduser().resolve()

SEED = "def solve(n): \n   return n \n"


PROMPT = (
    "Do exactly two things, in order.\n"
    "1. Edit solution.py in the current directory so that solve(n) returns n * 2.\n"
    "2. Call the evo_spike_tools tool once and report its raw JSON output verbatim.\n"
    "Then stop."
)

class PrintingSink:
    def emit(self, event):
        print(f"  [Event {event}] {event.kind.value:16} {event.tool_name or ""}")

def main() -> None:
    for label, path in (
        ("deepseek-harness checkout", DSH_ROOT),
        ("candidate config", CONFIG),
        ("runtime entry", RUNTIME),
    ):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")

    workdir = Path(tempfile.mkdtemp(prefix= "dsh-backend-try-"))
    sessions = Path(tempfile.mkdtemp(prefix= "dsh-backend-session-"))
    (workdir / "solution.py").write_text(SEED, encoding="utf-8")

    spec = DshRuntimeSpec(
        config_path=CONFIG,
        runtime_argv=("node", "--import", "tsx/esm", str(RUNTIME)),
        runtime_cwd=DSH_ROOT,
        session_root=sessions,
        model=os.environ.get("DSH_MODEL", "deepseek-v4-pro"),

    )

    backend = DshAgentBackend(spec)

    request = AgentSessionRequest(
        system="You are a Python engineer. make the smallest change that works.",
        user = PROMPT,
        parent= Candidate(
            id="parent",
            code="SEED",
            generation=0,
            parent_id="None",
            island_idx=0,
            operator="seed"
        ),
        operator="rewrite",
        workdir=workdir,
        limits=AgentSessionLimits(max_turns=12, max_tool_calls=20, timeout_s=300),
        preflight=ProposalPreflight(PreflightPipeline()),
        event_sink=PrintingSink(),
    )

    print("--- events as they arrive ---")
    try:
        result = backend.run(request)
    finally:
        pass

    print("\n--- result ---")
    print(json.dumps({
        "termination": result.termination.value,
        "session_id": result.session_id,
        "model": result.model,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "elapsed_s": round(result.elapsed_s, 2),
        "untranslated": result.events[-1].data.get("untranslated_event_types"),
    }, ensure_ascii=False, indent=2))

    print("\n--- workspace on disk ---")
    print((workdir / "solution.py").read_text(encoding="utf-8"))

    print("--- final message ---")
    print(result.final_message[:1500])

    backend.release(result.session_id)
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.rmtree(sessions, ignore_errors=True)


if __name__ == "__main__":
    main()
