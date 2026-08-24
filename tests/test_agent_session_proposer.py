"""End-to-end state-machine tests for AgentSessionProposer."""

from __future__ import annotations

import json
import shutil

import pytest

from evoharness.core import (
    AgentSessionLimits,
    AgentSessionProposer,
    AgentSessionResult,
    AgentTermination,
    AgentEvent,
    AgentEventKind,
    Candidate,
    PreflightIssue,
    PreflightPipeline,
    PreflightResult,
    ProposalPreflight,
    JsonlEventSinkFactory,
)
from evoharness.core.llm import LLMBillingError
from evoharness.core.workspace import GitWorkspace


class MemorySink:
    trace_path = "memory://proposal/events"

    def __init__(self, *, flush_error=None):
        self.events = []
        self.preflight_records = []
        self.finalizations = []
        self.flush_error = flush_error
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
        if self.flush_error is not None:
            raise self.flush_error

    def close(self):
        self.closed = True


class MemorySinkFactory:
    def __init__(self, sink=None):
        self.sink = sink or MemorySink()
        self.calls = []

    def open(self, *, proposal_id, parent_id, operator):
        self.calls.append((proposal_id, parent_id, operator))
        return self.sink


class QueueBackend:
    def __init__(self, *steps, release_error=None):
        self.steps = list(steps)
        self.requests = []
        self.released = []
        self.release_error = release_error

    def run(self, request):
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("unexpected backend run")
        return self.steps.pop(0)(request)

    def release(self, session_id):
        self.released.append(session_id)
        if self.release_error is not None:
            raise self.release_error


class StaticValidator:
    name = "compile"

    def __init__(self, issues=()):
        self.issues = tuple(issues)
        self.calls = 0

    def validate(self, ctx):
        self.calls += 1
        return PreflightResult(stage=self.name, issues=self.issues)


class FixableValidator:
    name = "compile"

    def validate(self, ctx):
        if (ctx.workdir / "main.py").read_text() == "x = 3\n":
            return PreflightResult(stage=self.name)
        return PreflightResult(
            stage=self.name,
            issues=(
                PreflightIssue(
                    validator=self.name,
                    code="syntax-error",
                    message="invalid syntax",
                    path="main.py",
                    line=1,
                ),
            ),
        )


class CountingPreflight(ProposalPreflight):
    def __init__(self):
        super().__init__(PreflightPipeline())
        self.calls = 0

    def check(self, ctx):
        self.calls += 1
        return super().check(ctx)


def make_parent(*, workspace=None):
    if workspace is None:
        return Candidate(
            id="parent-1",
            code="x = 1\n",
            generation=0,
            parent_id=None,
            island_idx=0,
            operator="seed",
        )
    return Candidate(
        id="parent-1",
        code=workspace.serialize(),
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
        workspace_kind=workspace.kind,
    )


def session_result(
    *,
    termination=AgentTermination.COMPLETED,
    session_id="session-1",
    final_message=(
        "TITLE: improve value\n"
        "SUMMARY: Updated the implementation."
    ),
    model="fake-model",
    cost_usd=0.1,
    turns=1,
    tool_calls=1,
):
    return AgentSessionResult(
        termination=termination,
        session_id=session_id,
        final_message=final_message,
        model=model,
        cost_usd=cost_usd,
        turns=turns,
        tool_calls=tool_calls,
        prompt_tokens=10,
        completion_tokens=4,
    )


def edit_main(text, **result_kwargs):
    def step(request):
        (request.workdir / "main.py").write_text(text)
        return session_result(**result_kwargs)

    return step


def no_change(**result_kwargs):
    def step(request):
        return session_result(**result_kwargs)

    return step


def make_proposer(
    tmp_path,
    backend,
    *,
    preflight=None,
    sink_factory=None,
    limits=None,
    max_repair_rounds=2,
):
    return AgentSessionProposer(
        backend=backend,
        preflight=preflight or ProposalPreflight(PreflightPipeline()),
        limits=limits
        or AgentSessionLimits(
            max_turns=6,
            max_tool_calls=6,
            timeout_s=30,
            max_cost_usd=1.0,
        ),
        event_sink_factory=sink_factory or MemorySinkFactory(),
        max_repair_rounds=max_repair_rounds,
        work_root=tmp_path / "sessions",
        proposal_id_factory=lambda: "proposal-1",
    )


def propose(proposer, parent=None):
    return proposer.propose(
        "rewrite",
        parent or make_parent(),
        "You are a coding agent.",
        "Improve the candidate.",
    )


def test_first_run_success_builds_workspace_metadata_and_cleans_up(tmp_path):
    backend = QueueBackend(edit_main("x = 2\n"))
    sink_factory = MemorySinkFactory()
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            sink_factory=sink_factory,
        )
    )

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.code == "x = 2\n"
    assert result.proposal.workspace is not None
    assert result.proposal.workspace.main_text() == "x = 2\n"
    assert result.proposal.title == "improve value"
    assert result.proposal.summary == "Updated the implementation."
    assert result.proposal.model == "fake-model"
    assert result.proposal.metadata["proposal_id"] == "proposal-1"
    assert result.proposal.metadata["session_id"] == "session-1"
    assert result.proposal.metadata["attempts"] == 1
    assert result.proposal.metadata["repair_rounds"] == 0
    assert result.proposal.metadata["termination"] == "completed"
    assert result.llm_cost == pytest.approx(0.1)
    assert result.trace_path == "memory://proposal/events"
    assert backend.released == ["session-1"]
    assert sink_factory.sink.flushed
    assert sink_factory.sink.closed
    assert not any((tmp_path / "sessions").iterdir())


def test_no_change_resumes_same_session_with_structured_feedback(tmp_path):
    backend = QueueBackend(
        no_change(turns=2, tool_calls=2, cost_usd=0.2),
        edit_main("x = 3\n", cost_usd=0.15),
    )
    result = propose(make_proposer(tmp_path, backend))

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.code == "x = 3\n"
    assert result.attempts == 2
    assert result.llm_cost == pytest.approx(0.35)
    assert result.proposal.metadata["repair_rounds"] == 1
    assert len(backend.requests) == 2
    first, second = backend.requests
    assert first.workdir == second.workdir
    assert first.session_id is None
    assert first.feedback == ()
    assert second.session_id == "session-1"
    assert [issue.code for issue in second.feedback] == ["no-changes"]
    assert second.limits.max_turns == 4
    assert second.limits.max_tool_calls == 4
    assert second.limits.max_cost_usd == pytest.approx(0.8)
    assert backend.released == ["session-1"]


def test_task_failure_resumes_with_validator_issue_and_then_passes(tmp_path):
    backend = QueueBackend(
        edit_main("x = 2\n"),
        edit_main("x = 3\n"),
    )
    preflight = ProposalPreflight(
        PreflightPipeline([FixableValidator()])
    )

    result = propose(
        make_proposer(tmp_path, backend, preflight=preflight)
    )

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.code == "x = 3\n"
    assert len(backend.requests) == 2
    issue = backend.requests[1].feedback[0]
    assert issue.validator == "compile"
    assert issue.code == "syntax-error"
    assert issue.path == "main.py"
    assert issue.line == 1


@pytest.mark.parametrize(
    "termination",
    [
        AgentTermination.TIMEOUT,
        AgentTermination.TURN_LIMIT,
        AgentTermination.TOOL_LIMIT,
        AgentTermination.COST_LIMIT,
        AgentTermination.CONTEXT_LIMIT,
    ],
)
def test_hard_termination_still_accepts_valid_workspace(
    tmp_path,
    termination,
):
    backend = QueueBackend(
        edit_main("x = 2\n", termination=termination)
    )
    result = propose(make_proposer(tmp_path, backend))

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.metadata["termination"] == termination.value
    assert len(backend.requests) == 1


def test_hard_termination_does_not_resume_invalid_workspace(tmp_path):
    backend = QueueBackend(
        no_change(termination=AgentTermination.CONTEXT_LIMIT)
    )
    result = propose(make_proposer(tmp_path, backend))

    assert not result.ok
    assert result.failure_reason == "context_limit"
    assert result.attempts == 1
    assert len(backend.requests) == 1
    assert backend.released == ["session-1"]


def test_nonrepairable_preflight_stops_without_resume(tmp_path):
    issue = PreflightIssue(
        validator="compile",
        code="policy-denied",
        message="This change is forbidden",
        repairable=False,
    )
    validator = StaticValidator((issue,))
    preflight = ProposalPreflight(PreflightPipeline([validator]))
    backend = QueueBackend(edit_main("x = 2\n"))

    result = propose(
        make_proposer(tmp_path, backend, preflight=preflight)
    )

    assert not result.ok
    assert result.failure_reason == "nonrepairable-preflight"
    assert validator.calls == 1
    assert len(backend.requests) == 1


def test_backend_error_termination_skips_final_preflight(tmp_path):
    preflight = CountingPreflight()
    backend = QueueBackend(
        no_change(termination=AgentTermination.BACKEND_ERROR)
    )

    result = propose(
        make_proposer(tmp_path, backend, preflight=preflight)
    )

    assert not result.ok
    assert result.failure_reason == "backend-error"
    assert preflight.calls == 0
    assert result.attempts == 1


def test_missing_session_id_prevents_repair_resume(tmp_path):
    backend = QueueBackend(no_change(session_id=None))
    result = propose(make_proposer(tmp_path, backend))

    assert not result.ok
    assert result.failure_reason == "backend-session-error"
    assert result.attempts == 1
    assert backend.released == []


def test_repair_limit_stops_after_initial_invalid_attempt(tmp_path):
    backend = QueueBackend(no_change())
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            max_repair_rounds=0,
        )
    )

    assert not result.ok
    assert result.failure_reason == "repair-limit"
    assert result.attempts == 1
    assert len(backend.requests) == 1


def test_total_turn_limit_blocks_next_repair_run(tmp_path):
    backend = QueueBackend(no_change(turns=2))
    limits = AgentSessionLimits(
        max_turns=2,
        max_tool_calls=6,
        timeout_s=30,
    )
    result = propose(
        make_proposer(tmp_path, backend, limits=limits)
    )

    assert not result.ok
    assert result.failure_reason == "turn-limit"
    assert result.attempts == 1
    assert len(backend.requests) == 1


def test_cost_overshoot_is_preserved_and_blocks_next_run(tmp_path):
    backend = QueueBackend(no_change(cost_usd=0.6))
    limits = AgentSessionLimits(
        max_turns=6,
        max_tool_calls=6,
        timeout_s=30,
        max_cost_usd=0.5,
    )
    result = propose(
        make_proposer(tmp_path, backend, limits=limits)
    )

    assert not result.ok
    assert result.failure_reason == "cost-limit"
    assert result.llm_cost == pytest.approx(0.6)
    assert result.attempts == 1
    assert len(backend.requests) == 1


def test_sink_cleanup_failure_overrides_otherwise_valid_proposal(tmp_path):
    sink = MemorySink(flush_error=OSError("disk full"))
    sink_factory = MemorySinkFactory(sink)
    backend = QueueBackend(edit_main("x = 2\n"))
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            sink_factory=sink_factory,
        )
    )

    assert not result.ok
    assert result.failure_reason == "sink-flush-error"
    assert result.llm_cost == pytest.approx(0.1)
    assert result.attempts == 1
    assert sink.closed
    assert backend.released == ["session-1"]
    assert not any((tmp_path / "sessions").iterdir())
    summary, final_patch = sink.finalizations[0]
    assert summary.success is False
    assert summary.failure_reason == "sink-flush-error"
    assert final_patch is not None


def test_billing_failure_escapes_instead_of_becoming_a_proposal_failure(tmp_path):
    # ETP run 12 (2026-08-21): the provider answered HTTP 403 "insufficient
    # balance", the broad backend-error handler turned it into an ordinary
    # failed proposal, and the run planned five more generations before
    # stopping on "proposer_dead". The SearchLoop has a handler that stops on
    # the first one and names the real cause -- it only ever fired in the
    # non-agentic lane, which ETP does not use.
    def fail(request):
        raise LLMBillingError("account cannot pay for the call (HTTP 403)")

    sink_factory = MemorySinkFactory()
    backend = QueueBackend(fail)
    proposer = make_proposer(tmp_path, backend, sink_factory=sink_factory)

    with pytest.raises(LLMBillingError):
        propose(proposer)

    # Escaping must not leak the session's resources.
    assert sink_factory.sink.flushed
    assert sink_factory.sink.closed
    assert not any((tmp_path / "sessions").iterdir())


def test_backend_exception_is_classified_and_resources_are_closed(tmp_path):
    def fail(request):
        raise OSError("provider unavailable")

    sink_factory = MemorySinkFactory()
    backend = QueueBackend(fail)
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            sink_factory=sink_factory,
        )
    )

    assert not result.ok
    assert result.failure_reason == "backend-error"
    assert result.attempts == 0
    assert result.llm_cost == 0
    assert backend.released == []
    assert sink_factory.sink.flushed
    assert sink_factory.sink.closed
    assert not any((tmp_path / "sessions").iterdir())


def test_backend_release_failure_overrides_otherwise_valid_proposal(tmp_path):
    backend = QueueBackend(
        edit_main("x = 2\n"),
        release_error=OSError("release failed"),
    )
    result = propose(make_proposer(tmp_path, backend))

    assert not result.ok
    assert result.failure_reason == "backend-release-error"
    assert result.attempts == 1
    assert result.llm_cost == pytest.approx(0.1)
    assert backend.released == ["session-1"]
    assert not any((tmp_path / "sessions").iterdir())
    summary, final_patch = backend.requests[0].event_sink.finalizations[0]
    assert summary.success is False
    assert summary.failure_reason == "backend-release-error"
    assert final_patch is not None


def test_preflight_exception_is_classified_and_session_is_released(tmp_path):
    class BrokenPreflight(ProposalPreflight):
        def check(self, ctx):
            raise RuntimeError("preflight unavailable")

    backend = QueueBackend(edit_main("x = 2\n"))
    preflight = BrokenPreflight(PreflightPipeline())
    result = propose(
        make_proposer(tmp_path, backend, preflight=preflight)
    )

    assert not result.ok
    assert result.failure_reason == "preflight-error"
    assert result.attempts == 1
    assert backend.released == ["session-1"]


def test_jsonl_transcript_persists_successful_proposal_artifacts(tmp_path):
    backend = QueueBackend(edit_main("x = 2\n"))
    sink_factory = JsonlEventSinkFactory(tmp_path / "run")
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            sink_factory=sink_factory,
        )
    )

    assert result.ok
    assert result.proposal is not None
    directory = tmp_path / "run" / "agent_sessions" / "proposal-1"
    assert result.trace_path == str(directory.resolve())
    assert result.proposal.metadata["trace_path"] == str(
        directory.resolve()
    )
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["success"] is True
    assert summary["attempts"] == 1
    assert summary["termination"] == "completed"
    assert len((directory / "preflight.jsonl").read_text().splitlines()) == 1
    assert "-x = 1" in (directory / "final.patch").read_text()
    assert "+x = 2" in (directory / "final.patch").read_text()


def test_backend_exception_keeps_prior_events_and_failure_summary(tmp_path):
    def emit_then_fail(request):
        request.event_sink.emit(
            AgentEvent(
                session_id="session-before-error",
                round_index=0,
                sequence=0,
                kind=AgentEventKind.SESSION_START,
                turn=0,
                elapsed_s=0,
                data={"model": "fake-model"},
            )
        )
        raise OSError("provider unavailable")

    backend = QueueBackend(emit_then_fail)
    result = propose(
        make_proposer(
            tmp_path,
            backend,
            sink_factory=JsonlEventSinkFactory(tmp_path / "run"),
        )
    )

    assert not result.ok
    assert result.failure_reason == "backend-error"
    directory = tmp_path / "run" / "agent_sessions" / "proposal-1"
    events = (directory / "events.jsonl").read_text().splitlines()
    assert len(events) == 1
    assert json.loads(events[0])["event"]["kind"] == "session_start"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["success"] is False
    assert summary["failure_reason"] == "backend-error"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_git_workspace_captures_multi_file_agent_changes(tmp_path):
    workspace = GitWorkspace(
        base_files={
            "main.py": "from helper import value\nprint(value)\n",
            "helper.py": "value = 1\n",
        }
    )
    parent = make_parent(workspace=workspace)

    def edit_multiple_files(request):
        (request.workdir / "main.py").write_text(
            "from helper import value\nprint(value * 2)\n"
        )
        (request.workdir / "helper.py").write_text("value = 3\n")
        (request.workdir / "new.py").write_text("enabled = True\n")
        return session_result()

    backend = QueueBackend(edit_multiple_files)
    result = propose(make_proposer(tmp_path, backend), parent)

    assert result.ok
    assert result.proposal is not None
    assert result.proposal.workspace is not None
    assert result.proposal.workspace.kind == "git"
    assert result.proposal.workspace.texts() == {
        "helper.py": "value = 3\n",
        "main.py": "from helper import value\nprint(value * 2)\n",
        "new.py": "enabled = True\n",
    }
    assert parent.workspace.texts() == {
        "helper.py": "value = 1\n",
        "main.py": "from helper import value\nprint(value)\n",
    }


def test_system_prompt_carries_session_budget_note(tmp_path):
    captured = {}

    def capture(request):
        captured["system"] = request.system
        (request.workdir / "main.py").write_text("x = 2\n")
        return session_result()

    result = propose(make_proposer(tmp_path, QueueBackend(capture)))
    assert result.ok
    assert "# Session budget" in captured["system"]
    assert "at most 6 turns" in captured["system"]  # from the fixture limits
    # the original system content is preserved ahead of the note
    assert captured["system"].startswith("You are a coding agent.")
