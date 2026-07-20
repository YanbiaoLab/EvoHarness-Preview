"""Deterministic S8 structural-DOA replay across proposal-generation arms.

This is an offline regression benchmark, not a claim about production-model
quality. It drives the real proposer/preflight/transcript implementations with
controlled model responses so framework behavior and accounting are stable.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from evoharness.evocore import (
    AgentSessionLimits,
    AgentSessionProposer,
    Candidate,
    ConversationalAgentBackend,
    JsonlEventSinkFactory,
    LLMClient,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    NativeToolAgentBackend,
    PreflightContext,
    PreflightIssue,
    PreflightPipeline,
    PreflightResult,
    ProposalPreflight,
    SingleShotProposer,
    StaticRouter,
    atomic_write_json,
)
from evoharness.evocore.agent import (
    AgentToolRegistry,
    WorkspaceWriteTool,
)
from evoharness.evoguard import Sandbox


MODEL = "s8-replay-model"
SEED_CODE = """# EDIT-REGION-BEGIN
def transform(values):
    return list(values)
# EDIT-REGION-END
"""
FIXED_CODE = """# EDIT-REGION-BEGIN
def transform(values):
    return [value + 1 for value in values]
# EDIT-REGION-END
"""


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    faulty_code: str
    expected_issue: str


CASES = (
    ReplayCase(
        "syntax",
        """# EDIT-REGION-BEGIN
def transform(values):
    return [value + 1 for value in values
# EDIT-REGION-END
""",
        "syntax-error",
    ),
    ReplayCase(
        "interface",
        """# EDIT-REGION-BEGIN
def wrong_name(values):
    return [value + 1 for value in values]
# EDIT-REGION-END
""",
        "interface-error",
    ),
    ReplayCase(
        "shape",
        """# EDIT-REGION-BEGIN
def transform(values):
    return [values[0] + 1]
# EDIT-REGION-END
""",
        "shape-error",
    ),
    ReplayCase("no-op", SEED_CODE, "no-changes"),
    ReplayCase(
        "timeout",
        """# EDIT-REGION-BEGIN
def transform(values):
    while True:
        pass
# EDIT-REGION-END
""",
        "execution-timeout",
    ),
)


_VERIFY_SCRIPT = """
import runpy
import sys

namespace = runpy.run_path("main.py")
function = namespace.get("transform")
if not callable(function):
    print("interface-error", file=sys.stderr)
    raise SystemExit(2)
result = function([1, 2, 3])
if result != [2, 3, 4]:
    print(f"shape-error: {result!r}", file=sys.stderr)
    raise SystemExit(3)
"""


class ReplayContractValidator:
    name = "replay-contract"

    def __init__(self, timeout_s: float = 0.25):
        self.runner = Sandbox(allow_network=False)
        self.timeout_s = timeout_s

    def validate(self, ctx: PreflightContext) -> PreflightResult:
        command = (sys.executable, "-I", "-c", _VERIFY_SCRIPT)
        result = self.runner.run(
            list(command),
            workdir=ctx.workdir,
            timeout_s=self.timeout_s,
        )
        if result.ok:
            return PreflightResult(self.name)
        if result.timed_out:
            code = "execution-timeout"
        elif "SyntaxError" in result.stderr:
            code = "syntax-error"
        elif result.return_code == 2:
            code = "interface-error"
        elif result.return_code == 3:
            code = "shape-error"
        else:
            code = "execution-error"
        return PreflightResult(
            self.name,
            (
                PreflightIssue(
                    validator=self.name,
                    code=code,
                    message=f"Replay contract failed: {code}",
                    command=command,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ),
            ),
        )


class _ReplayTransport:
    def __init__(self, mode: str, case: ReplayCase):
        self.mode = mode
        self.case = case
        self.calls = 0

    @staticmethod
    def _completion(code: str) -> str:
        return (
            "TITLE: repair structural candidate\n"
            "SUMMARY: satisfy the replay contract\n"
            f"### FILE: main.py\n```python\n{code}```"
        )

    def __call__(self, *, messages, model, tools, **kwargs):
        self.calls += 1
        code = self.case.faulty_code if self.calls <= 2 else FIXED_CODE

        if self.mode == "single_shot":
            code = self.case.faulty_code
        elif self.mode == "conversational":
            code = self.case.faulty_code if self.calls == 1 else FIXED_CODE

        if tools:
            if messages[-1].role != "tool":
                return LLMResponse(
                    text="",
                    model=model,
                    cost=0.01,
                    prompt_tokens=10,
                    completion_tokens=4,
                    stop_reason=LLMStopReason.TOOL_CALLS,
                    tool_calls=(
                        LLMToolCall(
                            call_id=f"{self.case.case_id}-{self.calls}",
                            name="workspace_write",
                            arguments={"path": "main.py", "content": code},
                        ),
                    ),
                )
            return LLMResponse(
                text=(
                    "TITLE: repair structural candidate\n"
                    "SUMMARY: finish the current replay round"
                ),
                model=model,
                cost=0.01,
                prompt_tokens=8,
                completion_tokens=3,
            )

        return LLMResponse(
            text=self._completion(code),
            model=model,
            cost=0.01,
            prompt_tokens=10,
            completion_tokens=6,
        )


@dataclass(frozen=True)
class ReplayResult:
    arm: str
    case_id: str
    expected_issue: str
    observed_initial_issue: str | None
    proposal_ok: bool
    final_contract_passed: bool
    structural_doa: bool
    preflight_intercepted: bool
    self_repaired: bool
    attempts: int
    repair_rounds: int
    tool_calls: int
    cost_usd: float
    tool_pairing_complete: bool
    trace_path: str | None


def _parent() -> Candidate:
    return Candidate(
        id="s8-parent",
        code=SEED_CODE,
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def _read_preflight(trace_path: str | None) -> tuple[bool, str | None]:
    if trace_path is None:
        return False, None
    path = Path(trace_path) / "preflight.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    for record in records:
        report = record["report"]
        if report["ok"]:
            continue
        issue = next(
            issue
            for result in report["results"]
            for issue in result["issues"]
        )
        return True, issue["code"]
    return False, None


def _tool_pairing_complete(trace_path: str | None) -> tuple[bool, int]:
    if trace_path is None:
        return True, 0
    path = Path(trace_path) / "events.jsonl"
    events = [
        json.loads(line)["event"]
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    calls = Counter(
        event["call_id"]
        for event in events
        if event["kind"] == "tool_call"
    )
    results = Counter(
        event["call_id"]
        for event in events
        if event["kind"] == "tool_result"
    )
    return calls == results, sum(calls.values())


def _build_proposer(
    arm: str,
    case: ReplayCase,
    run_dir: Path,
    preflight: ProposalPreflight,
):
    transport = _ReplayTransport(arm, case)
    client = LLMClient(transport=transport, sleep=lambda _: None)
    if arm == "single_shot":
        return SingleShotProposer(
            llm=client,
            model_router=StaticRouter([MODEL]),
            max_resamples=1,
        )

    tools = () if arm == "conversational" else (WorkspaceWriteTool(),)
    backend = NativeToolAgentBackend(
        client=client,
        model=MODEL,
        registry=AgentToolRegistry(tools),
        max_input_tokens=8_192,
        token_estimator=lambda messages, definitions: 100,
        session_id_factory=lambda: f"{arm}-{case.case_id}-session",
    )
    if arm == "conversational":
        backend = ConversationalAgentBackend(backend)
    return AgentSessionProposer(
        backend=backend,
        preflight=preflight,
        limits=AgentSessionLimits(
            max_turns=8,
            max_tool_calls=4,
            timeout_s=10,
        ),
        event_sink_factory=JsonlEventSinkFactory(run_dir),
        max_repair_rounds=1,
        work_root=run_dir / ".work",
        proposal_id_factory=lambda: f"{arm}-{case.case_id}",
    )


def _assess_final(
    parent: Candidate,
    result,
    preflight: ProposalPreflight,
) -> bool:
    if result.proposal is None:
        return False
    workspace = result.proposal.workspace or parent.workspace.with_main_text(
        result.proposal.code
    )
    with tempfile.TemporaryDirectory(prefix="s8-assess-") as directory:
        workdir = workspace.materialize(Path(directory))
        checked = preflight.check(
            PreflightContext(
                parent=parent,
                operator="rewrite",
                workdir=workdir,
            )
        )
    return checked.ok


def replay_case(arm: str, case: ReplayCase, run_dir: Path) -> ReplayResult:
    parent = _parent()
    preflight = ProposalPreflight(
        PreflightPipeline((ReplayContractValidator(),))
    )
    proposer = _build_proposer(arm, case, run_dir, preflight)
    result = proposer.propose(
        "rewrite",
        parent,
        "You repair Python candidates.",
        "Make transform([1, 2, 3]) return [2, 3, 4].",
    )
    final_passed = _assess_final(parent, result, preflight)
    intercepted, observed_issue = _read_preflight(result.trace_path)
    pairing_complete, transcript_tool_calls = _tool_pairing_complete(
        result.trace_path
    )
    metadata = {} if result.proposal is None else result.proposal.metadata
    repair_rounds = int(metadata.get("repair_rounds", 0))
    tool_calls = int(metadata.get("tool_calls", transcript_tool_calls))
    return ReplayResult(
        arm=arm,
        case_id=case.case_id,
        expected_issue=case.expected_issue,
        observed_initial_issue=observed_issue,
        proposal_ok=result.ok,
        final_contract_passed=final_passed,
        structural_doa=result.ok and not final_passed,
        preflight_intercepted=intercepted,
        self_repaired=final_passed and result.attempts > 1,
        attempts=result.attempts,
        repair_rounds=repair_rounds,
        tool_calls=tool_calls,
        cost_usd=result.llm_cost,
        tool_pairing_complete=pairing_complete,
        trace_path=result.trace_path,
    )


def _summarize(arm: str, results: list[ReplayResult]) -> dict[str, object]:
    count = len(results)
    valid = sum(item.final_contract_passed for item in results)
    doa = sum(item.structural_doa for item in results)
    intercepted = sum(item.preflight_intercepted for item in results)
    repaired = sum(item.self_repaired for item in results)
    total_cost = sum(item.cost_usd for item in results)
    return {
        "arm": arm,
        "cases": count,
        "valid_candidates": valid,
        "structural_doa": doa,
        "grader_structural_doa_rate": doa / count if count else 0.0,
        "preflight_interceptions": intercepted,
        "preflight_interception_rate": intercepted / count if count else 0.0,
        "self_repairs": repaired,
        "self_repair_rate": (
            repaired / intercepted if intercepted else None
        ),
        "average_repair_rounds": sum(
            item.repair_rounds for item in results
        ) / count,
        "average_tool_calls": sum(
            item.tool_calls for item in results
        ) / count,
        "total_llm_cost_usd": total_cost,
        "cost_per_valid_candidate_usd": (
            total_cost / valid if valid else None
        ),
        "tool_pairing_complete": all(
            item.tool_pairing_complete for item in results
        ),
    }


def run_replay(run_dir: Path) -> dict[str, object]:
    run_dir = Path(run_dir)
    arms = ("single_shot", "conversational", "agentic")
    all_results: list[ReplayResult] = []
    for arm in arms:
        for case in CASES:
            all_results.append(
                replay_case(arm, case, run_dir / arm / case.case_id)
            )

    report = {
        "schema_version": 1,
        "benchmark": "controlled-offline-structural-doa-replay",
        "cases": [asdict(case) for case in CASES],
        "arms": [
            _summarize(
                arm,
                [item for item in all_results if item.arm == arm],
            )
            for arm in arms
        ],
        "results": [asdict(item) for item in all_results],
        "interpretation": {
            "controlled_fixture": True,
            "production_model_quality_claim": False,
            "unmeasured": [
                "safe concurrency wall-clock savings",
                "real-provider cost overshoot distribution",
                "production-task structural DOA rate",
            ],
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_replay(args.run_dir)
    print(json.dumps(report["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
