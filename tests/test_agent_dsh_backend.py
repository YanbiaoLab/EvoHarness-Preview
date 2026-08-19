"""Contract tests for DshAgentBackend against a stubbed dsh SDK.

The dsh runtime is replaced by a fake harness so these run without a model, a
key, or a built TypeScript tree. What they pin is the translation layer, which
is where a backend lies: token accounting, termination mapping, event
translation, and runtime lifetime.
"""

import json

import pytest

from evoharness.core import (
    AgentBackend,
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionRequest,
    AgentTermination,
    Candidate,
    PreflightIssue,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.core.agent import (
    UNSUPPORTED_LIMITS,
    DshAgentBackend,
    DshRuntimeSpec,
)
from evoharness.core.agent import dsh_backend as dsh_backend_module


class FakeRunResult:
    def __init__(self, events, final_response="done", finish_reason=None):
        self.events = events
        self.final_response = final_response
        self.finish_reason = finish_reason
        self.session_id = "ignored-by-backend"


class FakeHarness:
    """Stands in for deepseek_harness.DeepSeekHarness."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = []
        self.started = False
        self.closed = False
        self.results = []
        self.raise_on_run = None
        FakeHarness.instances.append(self)

    def start(self):
        self.started = True

    def run(self, prompt, session_id=None):
        self.prompts.append((prompt, session_id))
        if self.raise_on_run is not None:
            raise self.raise_on_run
        if self.results:
            return self.results.pop(0)
        return FakeRunResult([])

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_sdk(monkeypatch):
    FakeHarness.instances = []
    module = type("_Module", (), {"DeepSeekHarness": FakeHarness})
    monkeypatch.setitem(
        __import__("sys").modules, "deepseek_harness", module
    )
    yield
    FakeHarness.instances = []


@pytest.fixture
def spec(tmp_path):
    config = tmp_path / "candidate.cordis.yml"
    config.write_text("- id: stub\n", encoding="utf-8")
    return DshRuntimeSpec(
        config_path=config,
        runtime_argv=("node", "bin.js"),
        model="deepseek-v4-flash",
    )


def make_parent():
    return Candidate(
        id="parent",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def make_request(tmp_path, **changes):
    values = {
        "system": "Follow the repository rules.",
        "user": "Improve the candidate.",
        "parent": make_parent(),
        "operator": "rewrite",
        "workdir": tmp_path,
        "limits": AgentSessionLimits(timeout_s=30),
        "preflight": ProposalPreflight(PreflightPipeline()),
    }
    values.update(changes)
    return AgentSessionRequest(**values)


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def assistant(turn, step, text, usage=None):
    data = {"turn": turn, "step": step, "message": {
        "content": [{"type": "text", "text": text}]}}
    if usage is not None:
        data["usage"] = usage
    return {"type": "assistant/message", "data": data}


def tool_call(turn, call_id, name, arguments="{}"):
    return {"type": "tool/call", "data": {
        "turn": turn, "step": 0, "callId": call_id,
        "name": name, "arguments": arguments}}


def tool_result(turn, call_id, text):
    return {"type": "tool/result", "data": {"turn": turn, "step": 0, "message": {
        "source": {"callId": call_id},
        "content": [{"content": [{"type": "text", "text": text}]}]}}}


def test_satisfies_the_agent_backend_protocol(spec):
    assert isinstance(DshAgentBackend(spec), AgentBackend)


def test_sums_the_token_usage_the_session_log_carries(spec, tmp_path):
    backend = DshAgentBackend(spec)
    harness_events = [
        assistant(0, 0, "first", {"inputTokens": 1_000_000, "outputTokens": 0}),
        assistant(0, 1, "second", {"inputTokens": 0, "outputTokens": 1_000_000}),
    ]
    result = _run_with(backend, spec, tmp_path, harness_events)

    # Tokens are observed facts from the log; they are what this backend
    # measures.
    assert result.prompt_tokens == 1_000_000
    assert result.completion_tokens == 1_000_000


def test_reports_no_cost_and_says_so_when_no_price_is_configured(
    spec, tmp_path
):
    backend = DshAgentBackend(spec)
    result = _run_with(
        backend,
        spec,
        tmp_path,
        [assistant(0, 0, "x", {"inputTokens": 1_000, "outputTokens": 1_000})],
    )

    assert result.cost_usd == 0.0
    # Zero cost beside nonzero tokens means unpriced, not free; the trace has
    # to distinguish the two.
    assert result.events[-1].data["cost_priced"] is False
    assert result.prompt_tokens == 1_000


def test_uses_a_caller_supplied_price_when_one_is_given(tmp_path):
    config = tmp_path / "c.yml"
    config.write_text("- id: stub\n", encoding="utf-8")
    priced = DshRuntimeSpec(
        config_path=config,
        runtime_argv=("node", "bin.js"),
        price_usd_per_mtok=(1.0, 2.0),
    )
    backend = DshAgentBackend(priced)
    result = _run_with(
        backend,
        priced,
        tmp_path,
        [assistant(0, 0, "x", {
            "inputTokens": 1_000_000, "outputTokens": 1_000_000})],
    )

    assert result.cost_usd == pytest.approx(3.0)
    assert result.events[-1].data["cost_priced"] is True


def test_counts_model_responses_as_turns_and_tool_calls_honestly(
    spec, tmp_path
):
    backend = DshAgentBackend(spec)
    harness_events = [
        assistant(0, 0, "thinking"),
        tool_call(0, "c1", "workspace_write"),
        tool_result(0, "c1", "ok"),
        assistant(0, 1, "done"),
    ]
    result = _run_with(backend, spec, tmp_path, harness_events)

    # Reporting fewer turns than actually ran would slip past the proposer's
    # own budget check, which is the failure this backend must not enable.
    assert result.turns == 2
    assert result.tool_calls == 1


# Every member of dsh's `TurnEndReasonMap`. Pinning the whole vocabulary is
# the point: an OpenAI-shaped table ("stop", "length", "content_filter")
# type-checks fine and matches nothing a real session emits.
@pytest.mark.parametrize(
    ("turn_end_kind", "expected"),
    [
        ("completed", AgentTermination.COMPLETED),
        ("aborted", AgentTermination.TIMEOUT),
        ("blocked", AgentTermination.REFUSAL),
        ("error", AgentTermination.BACKEND_ERROR),
        ("max-tokens", AgentTermination.OUTPUT_LIMIT),
        ("interrupted", AgentTermination.PROTOCOL_ERROR),
    ],
)
def test_maps_every_turn_end_kind_dsh_defines(
    spec, tmp_path, turn_end_kind, expected
):
    backend = DshAgentBackend(spec)
    result = _run_with(
        backend, spec, tmp_path, [], finish_reason=turn_end_kind
    )
    assert result.termination is expected


def test_provider_style_finish_reasons_are_not_accepted(spec, tmp_path):
    # These are the values a provider reports, not the values dsh's session log
    # carries. Accepting them would hide a wired-to-nothing mapping table.
    backend = DshAgentBackend(spec)
    for provider_reason in ("stop", "length", "content_filter", "tool_use"):
        result = _run_with(
            backend, spec, tmp_path, [], finish_reason=provider_reason
        )
        assert result.termination is AgentTermination.PROTOCOL_ERROR


def test_an_unknown_kind_is_a_protocol_error_not_success(spec, tmp_path):
    backend = DshAgentBackend(spec)
    result = _run_with(
        backend, spec, tmp_path, [], finish_reason="teleported"
    )
    # dsh's reason map is merge-extensible, so a plugin can add a kind this
    # table has never seen. Folding it into COMPLETED would hand the proposer a
    # workspace that may never have been finished.
    assert result.termination is AgentTermination.PROTOCOL_ERROR


def test_a_missing_turn_end_is_not_treated_as_success(spec, tmp_path):
    backend = DshAgentBackend(spec)
    result = _run_with(backend, spec, tmp_path, [], finish_reason=None)
    # The SDK returns None when no `turn/end` was recorded at all: a turn whose
    # ending is unknown, not a turn that ended well.
    assert result.termination is AgentTermination.PROTOCOL_ERROR


def test_translates_events_and_counts_the_ones_it_cannot(spec, tmp_path):
    sink = RecordingSink()
    backend = DshAgentBackend(spec)
    harness_events = [
        assistant(0, 0, "hello"),
        tool_call(0, "c1", "workspace_write"),
        tool_result(0, "c1", "written"),
        {"type": "turn/start", "data": {"turn": 0}},
        {"type": "turn/start", "data": {"turn": 1}},
    ]
    result = _run_with(backend, spec, tmp_path, harness_events, sink=sink)

    kinds = [event.kind for event in result.events]
    assert kinds == [
        AgentEventKind.SESSION_START,
        AgentEventKind.MODEL_RESPONSE,
        AgentEventKind.TOOL_CALL,
        AgentEventKind.TOOL_RESULT,
        AgentEventKind.TERMINATION,
    ]
    assert [event.sequence for event in result.events] == [0, 1, 2, 3, 4]
    assert sink.events == list(result.events)

    call = result.events[2]
    assert call.tool_name == "workspace_write"
    assert call.call_id == "c1"
    assert result.events[3].call_id == "c1"
    assert result.events[3].content == "written"

    # A lossy translation has to stay visible rather than being inferred from
    # a trace that is quietly shorter than the session.
    assert result.events[-1].data["untranslated_event_types"] == {
        "turn/start": 2
    }


def test_names_the_limits_it_cannot_enforce(spec, tmp_path):
    backend = DshAgentBackend(spec)
    result = _run_with(backend, spec, tmp_path, [])
    start = result.events[0]
    assert start.kind is AgentEventKind.SESSION_START
    assert start.data["unsupported_limits"] == list(UNSUPPORTED_LIMITS)
    assert "max_turns" in UNSUPPORTED_LIMITS


def test_reuses_one_runtime_across_repair_rounds(spec, tmp_path):
    backend = DshAgentBackend(spec)
    request = make_request(tmp_path)
    first = backend.run(request)

    issue = PreflightIssue(
        validator="run", code="crash", message="boom"
    )
    resumed = backend.run(
        make_request(
            tmp_path,
            session_id=first.session_id,
            feedback=(issue,),
        )
    )

    assert resumed.session_id == first.session_id
    assert len(FakeHarness.instances) == 1
    harness = FakeHarness.instances[0]
    assert [session for _, session in harness.prompts] == [
        first.session_id,
        first.session_id,
    ]
    # The first round carries the task; the repair round carries only the
    # structured feedback, so it does not read as a competing request.
    assert harness.prompts[0][0] == "Improve the candidate."
    assert json.loads(harness.prompts[1][0])["type"] == "preflight_feedback"
    assert resumed.events[0].kind is AgentEventKind.SESSION_RESUME


def test_carries_the_system_prompt_as_the_deployment_persona(spec, tmp_path):
    backend = DshAgentBackend(spec)
    backend.run(make_request(tmp_path, system="Be terse."))
    kwargs = FakeHarness.instances[0].kwargs
    assert kwargs["env"]["DSH_SYSTEM_PROMPT"] == "Be terse."
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["cordis"] == str(spec.config_path)
    assert kwargs["request_timeout_seconds"] == 30


def test_a_launch_failure_is_a_terminal_result_not_an_exception(
    spec, tmp_path, monkeypatch
):
    def explode(**_kwargs):
        raise RuntimeError("runtime refused to boot")

    monkeypatch.setattr(
        dsh_backend_module,
        "DshAgentBackend",
        DshAgentBackend,
    )
    module = type("_Module", (), {"DeepSeekHarness": explode})
    monkeypatch.setitem(__import__("sys").modules, "deepseek_harness", module)

    backend = DshAgentBackend(spec)
    result = backend.run(make_request(tmp_path))

    # Raising would lose the round's accounting: the proposer absorbs usage
    # before it inspects termination.
    assert result.termination is AgentTermination.BACKEND_ERROR
    assert "runtime refused to boot" in result.final_message
    assert result.events[-1].kind is AgentEventKind.TERMINATION


def test_a_transport_failure_mid_session_is_also_terminal(spec, tmp_path):
    backend = DshAgentBackend(spec)
    request = make_request(tmp_path)
    backend._session_for(  # noqa: SLF001 - exercising the cached-session path
        "proposal-fixed", request
    )
    FakeHarness.instances[0].raise_on_run = OSError("pipe closed")

    result = backend.run(make_request(tmp_path, session_id="proposal-fixed"))

    assert result.termination is AgentTermination.BACKEND_ERROR
    assert "pipe closed" in result.final_message


def test_release_reaps_the_runtime(spec, tmp_path):
    backend = DshAgentBackend(spec)
    result = backend.run(make_request(tmp_path))
    harness = FakeHarness.instances[0]

    assert harness.closed is False
    backend.release(result.session_id)

    # With one runtime per session this is where the process tree dies, so a
    # missed release leaks a process rather than a dictionary entry.
    assert harness.closed is True
    backend.release(result.session_id)  # idempotent


def test_resuming_against_a_moved_workdir_is_refused(spec, tmp_path):
    backend = DshAgentBackend(spec)
    first = backend.run(make_request(tmp_path))
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()

    result = backend.run(
        make_request(elsewhere, session_id=first.session_id)
    )

    # The runtime's cwd is fixed at construction, so silently continuing would
    # edit the previous candidate's directory.
    assert result.termination is AgentTermination.BACKEND_ERROR
    assert "cannot resume" in result.final_message


def test_identity_covers_the_deployment_the_candidate_runs_under(spec):
    backend = DshAgentBackend(spec)
    before = backend.spec.identity()
    spec.config_path.write_text("- id: stub\n- id: extra\n", encoding="utf-8")
    after = backend.spec.identity()

    # The cordis file names every plugin the candidate can reach, so editing it
    # changes the experiment.
    assert before["config_hash"] != after["config_hash"]
    assert before["unsupported_limits"] == list(UNSUPPORTED_LIMITS)


def test_any_model_id_is_accepted(tmp_path):
    config = tmp_path / "c.yml"
    config.write_text("- id: stub\n", encoding="utf-8")
    # Pricing is a secondary convenience, so an unknown model must not be able
    # to block a run.
    spec = DshRuntimeSpec(
        config_path=config,
        runtime_argv=("node",),
        model="some-new-model",
    )
    assert spec.price_usd_per_mtok is None


def _run_with(
    backend, spec, tmp_path, harness_events, finish_reason=None, sink=None
):
    request = make_request(tmp_path, event_sink=sink)
    live = backend._session_for("proposal-fixed", request)  # noqa: SLF001
    live.harness.results.append(
        FakeRunResult(harness_events, finish_reason=finish_reason)
    )
    return backend.run(make_request(
        tmp_path, session_id="proposal-fixed", event_sink=sink
    ))
