"""Durability and schema tests for agent proposal transcripts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from evoharness.evocore import (
    AgentEvent,
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionProposer,
    Candidate,
    JsonlEventSinkFactory,
    LLMClient,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    LLMToolDefinition,
    ManagedEventSink,
    NativeToolAgentBackend,
    PreflightIssue,
    PreflightPipeline,
    PreflightReport,
    PreflightResult,
    PreflightTraceRecord,
    ProposalPreflight,
    ProposalTraceSummary,
    TRANSCRIPT_SCHEMA_VERSION,
    render_workspace_patch,
)
from evoharness.evocore.agent import AgentToolRegistry, make_tool_result
from evoharness.evocore.workspace import FileWorkspace


class QueueTransport:
    def __init__(self, *responses):
        self.responses = list(responses)

    def __call__(self, **kwargs):
        return self.responses.pop(0)


class TranscriptAwareWriteTool:
    definition = LLMToolDefinition(
        name="write_main",
        description="Replace the candidate main file.",
        input_schema={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
            "additionalProperties": False,
        },
    )

    def __init__(self, events_path):
        self.events_path = events_path

    def is_concurrency_safe(self, call, ctx):
        return False

    def invoke(self, call, ctx):
        persisted = read_jsonl(self.events_path)
        assert persisted[-1]["event"]["kind"] == "tool_call"
        assert persisted[-1]["event"]["call_id"] == call.call_id
        (ctx.workdir / "main.py").write_text(call.arguments["content"])
        return make_tool_result(call.call_id, {"ok": True})


def event(sequence, *, kind=AgentEventKind.MODEL_RESPONSE, round_index=0):
    return AgentEvent(
        session_id="session-1",
        round_index=round_index,
        sequence=sequence,
        kind=kind,
        turn=sequence,
        elapsed_s=sequence / 10,
        content=f"event {sequence}",
        data={"sequence_copy": sequence},
    )


def summary(*, success=True, failure_reason=None):
    return ProposalTraceSummary(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
        success=success,
        session_id="session-1",
        attempts=2,
        repair_rounds=1,
        turns=3,
        tool_calls=4,
        cost_usd=0.25,
        prompt_tokens=100,
        completion_tokens=40,
        elapsed_s=2.5,
        termination="completed",
        failure_reason=failure_reason,
        model="fake-model",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_sink_writes_versioned_events_preflight_summary_and_patch(tmp_path):
    factory = JsonlEventSinkFactory(tmp_path / "run")
    sink = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    assert isinstance(sink, ManagedEventSink)

    sink.emit(event(0))
    sink.emit(
        event(
            1,
            kind=AgentEventKind.CONTEXT_COMPACT,
            round_index=1,
        )
    )
    issue = PreflightIssue(
        validator="compile",
        code="syntax-error",
        message="invalid syntax",
        path="main.py",
        line=2,
    )
    sink.record_preflight(
        PreflightTraceRecord(
            round_index=0,
            session_id="session-1",
            report=PreflightReport(
                (PreflightResult("compile", (issue,), 0.1),)
            ),
        )
    )
    sink.finalize(summary(), "--- a/main.py\n+++ b/main.py\n")
    sink.close()

    directory = tmp_path / "run" / "agent_sessions" / "proposal-1"
    events = read_jsonl(directory / "events.jsonl")
    assert len(events) == 2
    assert events[0]["schema_version"] == TRANSCRIPT_SCHEMA_VERSION
    assert events[0]["proposal_id"] == "proposal-1"
    assert events[0]["session_id"] == "session-1"
    assert events[0]["round"] == 0
    assert events[0]["event"]["kind"] == "model_response"
    assert events[0]["event"]["content"] == "event 0"
    assert events[1]["event"]["kind"] == "context_compact"
    assert events[0]["event"]["content"] == "event 0"

    preflight = read_jsonl(directory / "preflight.jsonl")
    assert preflight[0]["report"]["failed_stage"] == "compile"
    assert (
        preflight[0]["report"]["results"][0]["issues"][0]["code"]
        == "syntax-error"
    )

    persisted_summary = json.loads((directory / "summary.json").read_text())
    assert persisted_summary["schema_version"] == TRANSCRIPT_SCHEMA_VERSION
    assert persisted_summary["success"] is True
    assert persisted_summary["final_patch_present"] is True
    assert (directory / "final.patch").read_text().startswith("---")


def test_concurrent_emit_produces_complete_uncorrupted_lines(tmp_path):
    sink = JsonlEventSinkFactory(tmp_path).open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: sink.emit(event(index)), range(100)))
    sink.close()

    events = read_jsonl(
        tmp_path / "agent_sessions" / "proposal-1" / "events.jsonl"
    )
    assert len(events) == 100
    assert {row["event"]["sequence"] for row in events} == set(range(100))


def test_reopen_appends_instead_of_overwriting_history(tmp_path):
    factory = JsonlEventSinkFactory(tmp_path)
    first = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    first.emit(event(0))
    first.close()

    resumed = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    resumed.emit(event(1, round_index=1))
    resumed.finalize(summary(), "patch\n")
    resumed.close()

    events = read_jsonl(
        tmp_path / "agent_sessions" / "proposal-1" / "events.jsonl"
    )
    assert [row["event"]["sequence"] for row in events] == [0, 1]


def test_reopen_rejects_mixed_proposal_identity(tmp_path):
    factory = JsonlEventSinkFactory(tmp_path)
    first = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    first.emit(event(0))
    first.close()

    with pytest.raises(ValueError, match="identity"):
        factory.open(
            proposal_id="proposal-1",
            parent_id="different-parent",
            operator="rewrite",
        )


def test_non_json_event_data_fails_without_partial_line(tmp_path):
    sink = JsonlEventSinkFactory(tmp_path).open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    invalid = AgentEvent(
        session_id="session-1",
        round_index=0,
        sequence=0,
        kind=AgentEventKind.MODEL_RESPONSE,
        turn=0,
        elapsed_s=0,
        data={"bad": object()},
    )

    with pytest.raises(TypeError):
        sink.emit(invalid)
    sink.close()

    path = tmp_path / "agent_sessions" / "proposal-1" / "events.jsonl"
    assert path.read_text() == ""


def test_file_workspace_patch_is_deterministic():
    patch = render_workspace_patch(
        FileWorkspace("x = 1\n"),
        FileWorkspace("x = 2\n"),
    )

    assert patch == (
        "--- a/main.py\n"
        "+++ b/main.py\n"
        "@@ -1 +1 @@\n"
        "-x = 1\n"
        "+x = 2\n"
    )


def test_file_workspace_patch_marks_missing_final_newline():
    patch = render_workspace_patch(
        FileWorkspace("x = 1"),
        FileWorkspace("x = 2"),
    )

    assert patch.count("\\ No newline at end of file") == 2
    assert "-x = 1\n\\ No newline at end of file\n" in patch
    assert "+x = 2\n\\ No newline at end of file\n" in patch


def test_reopen_rejects_corrupt_tail_instead_of_appending(tmp_path):
    factory = JsonlEventSinkFactory(tmp_path)
    sink = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    sink.emit(event(0))
    sink.close()

    path = tmp_path / "agent_sessions" / "proposal-1" / "events.jsonl"
    with path.open("a") as handle:
        handle.write('{"schema_version":1')

    with pytest.raises(ValueError, match="unreadable"):
        factory.open(
            proposal_id="proposal-1",
            parent_id="parent-1",
            operator="rewrite",
        )


def test_reopen_rejects_finalized_proposal(tmp_path):
    factory = JsonlEventSinkFactory(tmp_path)
    sink = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    sink.finalize(summary(), "patch\n")
    sink.close()

    with pytest.raises(ValueError, match="already finalized"):
        factory.open(
            proposal_id="proposal-1",
            parent_id="parent-1",
            operator="rewrite",
        )


def test_event_rejects_non_finite_json_number_without_partial_line(tmp_path):
    sink = JsonlEventSinkFactory(tmp_path).open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )
    invalid = event(0)
    object.__setattr__(invalid, "data", {"bad": float("nan")})

    with pytest.raises(ValueError, match="JSON compliant"):
        sink.emit(invalid)
    sink.close()

    path = tmp_path / "agent_sessions" / "proposal-1" / "events.jsonl"
    assert path.read_text() == ""


def test_factory_rejects_symlinked_artifact_file(tmp_path):
    directory = tmp_path / "agent_sessions" / "proposal-1"
    directory.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("do not modify\n")
    (directory / "events.jsonl").symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        JsonlEventSinkFactory(tmp_path).open(
            proposal_id="proposal-1",
            parent_id="parent-1",
            operator="rewrite",
        )

    assert outside.read_text() == "do not modify\n"


def test_native_session_transcript_reconstructs_tool_and_workspace(tmp_path):
    run_dir = tmp_path / "run"
    events_path = (
        run_dir
        / "agent_sessions"
        / "proposal-1"
        / "events.jsonl"
    )
    call = LLMToolCall(
        "call-1",
        "write_main",
        {"content": "x = 2\n"},
    )
    transport = QueueTransport(
        LLMResponse(
            "",
            "fake-model",
            tool_calls=(call,),
            stop_reason=LLMStopReason.TOOL_CALLS,
            cost=0.1,
            prompt_tokens=10,
            completion_tokens=3,
        ),
        LLMResponse(
            "TITLE: improve value\nSUMMARY: write through tool",
            "fake-model",
            cost=0.05,
            prompt_tokens=8,
            completion_tokens=4,
        ),
    )
    backend = NativeToolAgentBackend(
        client=LLMClient(transport=transport, sleep=lambda _: None),
        model="fake-model",
        registry=AgentToolRegistry(
            (TranscriptAwareWriteTool(events_path),)
        ),
        max_input_tokens=1_000,
        token_estimator=lambda messages, tools: 10,
        session_id_factory=lambda: "session-1",
    )
    proposer = AgentSessionProposer(
        backend=backend,
        preflight=ProposalPreflight(PreflightPipeline()),
        limits=AgentSessionLimits(timeout_s=30),
        event_sink_factory=JsonlEventSinkFactory(run_dir),
        work_root=tmp_path / "work",
        proposal_id_factory=lambda: "proposal-1",
    )
    parent = Candidate(
        id="parent-1",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )

    result = proposer.propose(
        "rewrite",
        parent,
        "You are a coding agent.",
        "Improve the candidate.",
    )

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.code == "x = 2\n"
    events = read_jsonl(events_path)
    assert [row["event"]["kind"] for row in events] == [
        "session_start",
        "model_response",
        "tool_call",
        "tool_result",
        "model_response",
        "termination",
    ]
    tool_call = events[2]["event"]
    tool_result = events[3]["event"]
    assert tool_call["data"]["arguments"] == {"content": "x = 2\n"}
    assert tool_call["data"]["admitted"] is True
    assert tool_result["call_id"] == tool_call["call_id"]
    assert tool_result["data"]["executed"] is True
    summary_path = events_path.with_name("summary.json")
    persisted_summary = json.loads(summary_path.read_text())
    assert persisted_summary["turns"] == 2
    assert persisted_summary["tool_calls"] == 1
    assert persisted_summary["cost_usd"] == pytest.approx(0.15)
    patch = events_path.with_name("final.patch").read_text()
    assert "-x = 1" in patch
    assert "+x = 2" in patch


@pytest.mark.parametrize(
    ("success", "failure_reason"),
    [(True, "failed"), (False, None)],
)
def test_summary_requires_failure_reason_exactly_on_failure(
    success,
    failure_reason,
):
    with pytest.raises(ValueError, match="failure_reason"):
        summary(success=success, failure_reason=failure_reason)
