"""Provider-neutral contracts for AgentSessionProposer (WS-3 M3)."""

from dataclasses import FrozenInstanceError

import pytest

from evoharness.core import (
    AgentBackend,
    AgentEvent,
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionRequest,
    AgentSessionResult,
    AgentTermination,
    Candidate,
    EventSinkFactory,
    ManagedEventSink,
    PreflightContext,
    PreflightIssue,
    PreflightPipeline,
    PreflightResult,
    PreflightValidator,
    ProposalPreflight,
)


class FakeAgentBackend:
    def __init__(self):
        self.requests = []
        self.released = []

    def run(self, request):
        self.requests.append(request)
        (request.workdir / "main.py").write_text("x = 2\n")
        return AgentSessionResult(
            termination=AgentTermination.COMPLETED,
            session_id="fake-session",
            cost_usd=0.01,
            turns=2,
            tool_calls=1,
            events=(
                AgentEvent(
                    session_id="fake-session",
                    round_index=0,
                    sequence=0,
                    kind=AgentEventKind.TOOL_CALL,
                    turn=1,
                    elapsed_s=0.1,
                    tool_name="workspace_edit",
                    content="updated main.py",
                ),
            ),
        )

    def release(self, session_id):
        self.released.append(session_id)


class StaticValidator:
    name = "static"

    def validate(self, ctx):
        assert ctx.workdir.is_dir()
        return PreflightResult(stage=self.name)


class MemoryEventSink:
    trace_path = None

    def __init__(self):
        self.events = []
        self.preflight_records = []
        self.finalizations = []
        self.flushed = False
        self.closed = False

    def emit(self, event):
        self.events.append(event)

    def record_preflight(self, record):
        self.preflight_records.append(record)

    def finalize(self, summary, final_patch):
        self.finalizations.append((summary, final_patch))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


class MemoryEventSinkFactory:
    def __init__(self):
        self.calls = []

    def open(self, *, proposal_id, parent_id, operator):
        self.calls.append(
            {
                "proposal_id": proposal_id,
                "parent_id": parent_id,
                "operator": operator,
            }
        )
        return MemoryEventSink()


def make_parent():
    return Candidate(
        id="p",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def make_preflight():
    return ProposalPreflight(PreflightPipeline())


def test_fake_backend_receives_structured_ir_and_edits_workspace(tmp_path):
    backend = FakeAgentBackend()
    assert isinstance(backend, AgentBackend)

    issue = PreflightIssue(
        validator="shape-contract",
        code="shape-mismatch",
        message="expected [B, N], got [B]",
        path="model.py",
        line=42,
    )
    request = AgentSessionRequest(
        system="system",
        user="improve it",
        parent=make_parent(),
        operator="rewrite",
        workdir=tmp_path,
        limits=AgentSessionLimits(max_turns=3, max_tool_calls=4),
        preflight=make_preflight(),
        feedback=(issue,),
    )

    result = backend.run(request)

    assert result.completed and result.session_id == "fake-session"
    assert result.cost_usd == 0.01 and result.tool_calls == 1
    assert backend.requests[0].feedback == (issue,)
    assert (tmp_path / "main.py").read_text() == "x = 2\n"


def test_preflight_validator_protocol_and_result_semantics(tmp_path):
    validator = StaticValidator()
    assert isinstance(validator, PreflightValidator)
    ctx = PreflightContext(make_parent(), "rewrite", tmp_path)
    assert validator.validate(ctx).ok

    repairable = PreflightResult(
        stage="shape",
        issues=(PreflightIssue("shape", "mismatch", "wrong shape"),),
    )
    unsafe = PreflightResult(
        stage="workspace",
        issues=(
            PreflightIssue(
                "workspace", "unsafe-path", "path escaped", repairable=False
            ),
        ),
    )
    assert repairable.repairable
    assert not unsafe.repairable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_turns": 0},
        {"max_turns": True},
        {"max_tool_calls": True},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"max_cost_usd": -0.01},
        {"max_cost_usd": float("nan")},
    ],
)
def test_session_limits_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AgentSessionLimits(**kwargs)


def test_session_limits_allow_zero_tool_budget():
    limits = AgentSessionLimits(max_tool_calls=0)

    assert limits.max_tool_calls == 0


def test_result_and_events_reject_invalid_accounting():
    with pytest.raises(ValueError):
        AgentSessionResult(AgentTermination.BACKEND_ERROR, turns=-1)
    with pytest.raises(ValueError):
        AgentEvent(
            session_id="session-1",
            round_index=0,
            sequence=-1,
            kind=AgentEventKind.MODEL_RESPONSE,
            turn=0,
            elapsed_s=0,
        )
    with pytest.raises(ValueError):
        PreflightResult("compile", elapsed_s=-1)


def test_result_is_an_immutable_value_object():
    result = AgentSessionResult(AgentTermination.COMPLETED)
    with pytest.raises(FrozenInstanceError):
        result.turns = 2


def test_agent_backend_supports_explicit_session_release():
    backend = FakeAgentBackend()

    backend.release("session-1")

    assert backend.released == ["session-1"]


def test_managed_event_sink_and_factory_contracts():
    sink = MemoryEventSink()
    factory = MemoryEventSinkFactory()

    assert isinstance(sink, ManagedEventSink)
    assert isinstance(factory, EventSinkFactory)

    opened = factory.open(
        proposal_id="proposal-1",
        parent_id="parent-1",
        operator="rewrite",
    )

    opened.flush()
    opened.close()

    assert opened.flushed
    assert opened.closed
    assert factory.calls == [
        {
            "proposal_id": "proposal-1",
            "parent_id": "parent-1",
            "operator": "rewrite",
        }
    ]
