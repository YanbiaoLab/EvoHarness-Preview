"""Resource lifecycle tests for AgentSessionProposer."""

import pytest

from evoharness.evocore import (
    Candidate,
    EventSinkFactory,
    ManagedEventSink,
)
from evoharness.evocore.agent.session_proposer import (
    _ProposalResourceError,
    _ProposalResources,
)


class FakeBackend:
    def __init__(self, *, release_error=None):
        self.released = []
        self.release_error = release_error

    def run(self, request):
        raise AssertionError("run is not used by resource tests")

    def release(self, session_id):
        self.released.append(session_id)
        if self.release_error is not None:
            raise self.release_error


class MemorySink:
    trace_path = "memory://proposal/events"

    def __init__(self, *, flush_error=None, close_error=None):
        self.events = []
        self.preflight_records = []
        self.finalizations = []
        self.flushed = False
        self.closed = False
        self.flush_error = flush_error
        self.close_error = close_error

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
        if self.close_error is not None:
            raise self.close_error


class MemorySinkFactory:
    def __init__(self, sink=None):
        self.sink = sink or MemorySink()
        self.calls = []

    def open(self, *, proposal_id, parent_id, operator):
        self.calls.append(
            {
                "proposal_id": proposal_id,
                "parent_id": parent_id,
                "operator": operator,
            }
        )
        return self.sink


def make_parent():
    return Candidate(
        id="parent-1",
        code="x = 1\n",
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )


def make_resources(
    tmp_path,
    *,
    backend=None,
    sink_factory=None,
    proposal_id="proposal-1",
):
    return _ProposalResources(
        proposal_id=proposal_id,
        parent=make_parent(),
        operator="rewrite",
        backend=backend or FakeBackend(),
        event_sink_factory=sink_factory or MemorySinkFactory(),
        work_root=tmp_path / "sessions",
    )


def test_resource_contract_fakes_are_valid():
    sink = MemorySink()
    factory = MemorySinkFactory(sink)

    assert isinstance(sink, ManagedEventSink)
    assert isinstance(factory, EventSinkFactory)


def test_resources_materialize_once_and_cleanup_everything(tmp_path):
    backend = FakeBackend()
    sink = MemorySink()
    factory = MemorySinkFactory(sink)
    resources = make_resources(
        tmp_path,
        backend=backend,
        sink_factory=factory,
    )

    with resources as opened:
        assert opened.workdir is not None
        workdir = opened.workdir

        assert workdir.is_dir()
        assert (workdir / "main.py").read_text() == "x = 1\n"
        assert opened.trace_path == "memory://proposal/events"

        opened.bind_session("session-1")
        opened.bind_session("session-1")

    assert resources.workdir is None
    assert backend.released == ["session-1"]
    assert sink.flushed
    assert sink.closed
    assert not workdir.exists()
    assert (tmp_path / "sessions").is_dir()
    assert factory.calls == [
        {
            "proposal_id": "proposal-1",
            "parent_id": "parent-1",
            "operator": "rewrite",
        }
    ]


def test_resources_reject_session_id_changes_and_still_cleanup(tmp_path):
    backend = FakeBackend()
    resources = make_resources(tmp_path, backend=backend)

    with pytest.raises(
        _ProposalResourceError,
        match="backend-session-error",
    ):
        with resources as opened:
            assert opened.workdir is not None
            workdir = opened.workdir
            opened.bind_session("session-1")
            opened.bind_session("session-2")

    assert backend.released == ["session-1"]
    assert not workdir.exists()


def test_body_exception_is_not_hidden_by_cleanup(tmp_path):
    backend = FakeBackend(release_error=OSError("release failed"))
    sink = MemorySink(
        flush_error=OSError("flush failed"),
        close_error=OSError("close failed"),
    )
    resources = make_resources(
        tmp_path,
        backend=backend,
        sink_factory=MemorySinkFactory(sink),
    )

    with pytest.raises(RuntimeError, match="body failed"):
        with resources as opened:
            assert opened.workdir is not None
            workdir = opened.workdir
            opened.bind_session("session-1")
            raise RuntimeError("body failed")

    assert backend.released == ["session-1"]
    assert sink.flushed
    assert sink.closed
    assert not workdir.exists()


def test_cleanup_failure_propagates_after_successful_body(tmp_path):
    sink = MemorySink(flush_error=OSError("disk full"))
    resources = make_resources(
        tmp_path,
        sink_factory=MemorySinkFactory(sink),
    )

    with pytest.raises(
        _ProposalResourceError,
        match="sink-flush-error",
    ):
        with resources as opened:
            assert opened.workdir is not None
            workdir = opened.workdir

    assert sink.closed
    assert not workdir.exists()


def test_invalid_sink_is_rejected_before_materialization(tmp_path):
    class InvalidFactory:
        def open(self, *, proposal_id, parent_id, operator):
            return object()

    resources = make_resources(
        tmp_path,
        sink_factory=InvalidFactory(),
    )

    with pytest.raises(
        _ProposalResourceError,
        match="sink-open-error",
    ):
        with resources:
            pass

    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize("proposal_id", ["../escape", "a/b", ""])
def test_resources_reject_unsafe_proposal_ids(tmp_path, proposal_id):
    with pytest.raises(ValueError, match="unsafe characters"):
        make_resources(tmp_path, proposal_id=proposal_id)


def test_resources_are_one_shot(tmp_path):
    resources = make_resources(tmp_path)

    with resources:
        pass

    with pytest.raises(RuntimeError, match="cannot be reused"):
        with resources:
            pass
