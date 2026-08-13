"""Native agent session lifecycle tests."""

import json
from dataclasses import replace

import pytest

from evoharness.core import (
    AgentEventKind,
    AgentSessionLimits,
    AgentSessionRequest,
    Candidate,
    PreflightIssue,
    PreflightPipeline,
    ProposalPreflight,
)
from evoharness.core.agent.feedback import (
    preflight_issue_to_payload,
    render_preflight_feedback,
)
from evoharness.core.agent.runtime import _SessionStore


def make_parent(parent_id="parent"):
    return Candidate(
        id=parent_id,
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
        "limits": AgentSessionLimits(),
        "preflight": ProposalPreflight(PreflightPipeline()),
    }
    values.update(changes)
    return AgentSessionRequest(**values)


def make_issue():
    return PreflightIssue(
        validator="compile",
        code="syntax-error",
        message="invalid syntax",
        path="main.py",
        line=2,
        command=("python", "-m", "compileall"),
        stderr="traceback",
    )


def test_new_session_has_exact_initial_history_and_start_event(tmp_path):
    store = _SessionStore(
        clock=lambda: 12.5,
        session_id_factory=lambda: "session-1",
    )
    request = make_request(tmp_path)

    state, event = store.open(
        request,
        model="model-a",
        tool_names=("workspace_read", "workspace_edit"),
    )

    assert state.session_id == "session-1"
    assert state.workdir == tmp_path.resolve()
    assert [(message.role, message.content) for message in state.messages] == [
        ("system", request.system),
        ("user", request.user),
    ]
    assert state.created_at == state.last_active_at == 12.5
    assert event.kind is AgentEventKind.SESSION_START
    assert event.session_id == "session-1"
    assert event.round_index == 0
    assert event.sequence == 0
    assert event.data["tool_names"] == (
        "workspace_read",
        "workspace_edit",
    )
    assert event.data["system"] == request.system
    assert event.data["user"] == request.user


def test_resume_appends_only_rendered_feedback(tmp_path):
    now = 10.0

    def clock():
        return now

    store = _SessionStore(
        clock=clock,
        session_id_factory=lambda: "session-1",
    )
    initial = make_request(tmp_path)
    state, _ = store.open(initial, model="model-a", tool_names=())
    initial_messages = tuple(state.messages)
    state.lifetime_turns = 3
    now = 20.0

    resumed, event = store.open(
        replace(
            initial,
            session_id=state.session_id,
            feedback=(make_issue(),),
        ),
        model="model-a",
        tool_names=(),
    )

    assert resumed is state
    assert tuple(state.messages[:2]) == initial_messages
    assert len(state.messages) == 3
    assert state.messages[-1].role == "user"
    payload = json.loads(state.messages[-1].content)
    assert payload["type"] == "preflight_feedback"
    assert payload["issues"][0]["code"] == "syntax-error"
    assert state.last_active_at == 20.0
    assert event.kind is AgentEventKind.SESSION_RESUME
    assert event.session_id == "session-1"
    assert event.round_index == 1
    assert event.sequence == 1
    assert event.turn == 3
    assert event.data["issue_count"] == 1
    assert json.loads(event.data["feedback"])["issues"][0]["code"] == (
        "syntax-error"
    )


@pytest.mark.parametrize(
    ("case", "expected_name"),
    [
        ("workdir", "workdir"),
        ("parent", "parent_id"),
        ("operator", "operator"),
        ("model", "model"),
        ("tools", "tool_names"),
        ("system", "system"),
        ("user", "user"),
    ],
)
def test_resume_rejects_fingerprint_changes_without_mutation(
    tmp_path,
    case,
    expected_name,
):
    store = _SessionStore(session_id_factory=lambda: "session-1")
    initial = make_request(tmp_path)
    state, _ = store.open(
        initial,
        model="model-a",
        tool_names=("workspace_read",),
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    changes = {
        "session_id": state.session_id,
        "feedback": (make_issue(),),
    }
    model = "model-a"
    tool_names = ("workspace_read",)

    if case == "workdir":
        changes["workdir"] = other_dir
    elif case == "parent":
        changes["parent"] = make_parent("other-parent")
    elif case == "operator":
        changes["operator"] = "repair"
    elif case == "model":
        model = "model-b"
    elif case == "tools":
        tool_names = ("workspace_edit",)
    elif case == "system":
        changes["system"] = "Different system prompt."
    elif case == "user":
        changes["user"] = "Different user prompt."

    with pytest.raises(ValueError, match=expected_name):
        store.open(
            replace(initial, **changes),
            model=model,
            tool_names=tool_names,
        )

    assert len(state.messages) == 2
    assert state.next_event_sequence == 1


def test_resume_requires_known_session_and_nonempty_feedback(tmp_path):
    store = _SessionStore(session_id_factory=lambda: "session-1")
    initial = make_request(tmp_path)
    state, _ = store.open(initial, model="model-a", tool_names=())

    with pytest.raises(ValueError, match="requires preflight feedback"):
        store.open(
            replace(initial, session_id=state.session_id),
            model="model-a",
            tool_names=(),
        )

    with pytest.raises(ValueError, match="unknown agent session"):
        store.open(
            replace(
                initial,
                session_id="missing",
                feedback=(make_issue(),),
            ),
            model="model-a",
            tool_names=(),
        )

    assert len(state.messages) == 2


def test_release_is_idempotent(tmp_path):
    store = _SessionStore(session_id_factory=lambda: "session-1")
    state, _ = store.open(
        make_request(tmp_path),
        model="model-a",
        tool_names=(),
    )

    store.release(state.session_id)
    store.release(state.session_id)

    with pytest.raises(ValueError, match="unknown agent session"):
        store.get(state.session_id)


def test_usage_snapshot_reports_only_lifetime_delta(tmp_path):
    store = _SessionStore(session_id_factory=lambda: "session-1")
    state, _ = store.open(
        make_request(tmp_path),
        model="model-a",
        tool_names=(),
    )
    before = state.usage_snapshot()

    state.lifetime_turns += 2
    state.lifetime_tool_calls += 3
    state.lifetime_cost_usd += 0.25
    state.lifetime_prompt_tokens += 11
    state.lifetime_completion_tokens += 7

    delta = state.usage_snapshot().delta_from(before)

    assert delta.turns == 2
    assert delta.tool_calls == 3
    assert delta.cost_usd == 0.25
    assert delta.prompt_tokens == 11
    assert delta.completion_tokens == 7


def test_feedback_renderer_is_shared_bounded_and_does_not_mutate_ir():
    issue = replace(
        make_issue(),
        stdout="a" * 100,
        stderr="b" * 100,
    )

    rendered = preflight_issue_to_payload(issue, max_output_chars=40)
    feedback = json.loads(
        render_preflight_feedback((issue,), max_output_chars=40)
    )

    assert rendered == feedback["issues"][0]
    assert rendered["stdout_truncated"]
    assert rendered["stderr_truncated"]
    assert len(rendered["stdout"]) == 40
    assert issue.stdout == "a" * 100
