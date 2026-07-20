"""Native agent runtime state and session lifecycle."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Protocol

from ..llm import (
    LLMClient,
    LLMMessage,
    LLMProtocolError,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolResult,
    LLMTransientError,
)
from .contracts import (
    AgentEvent,
    AgentEventKind,
    AgentSessionRequest,
    AgentSessionResult,
    AgentTermination,
    EventSink,
)
from .feedback import render_preflight_feedback
from .tools import AgentToolContext, AgentToolRegistry, make_tool_error


_MAX_TOKENS_CONTINUATION = (
    "Continue exactly where the previous response stopped. "
    "Do not repeat completed content."
)


def _new_session_id() -> str:
    return uuid.uuid4().hex


class TokenEstimator(Protocol):
    """Estimate provider input tokens for messages and tool schemas."""

    def __call__(
        self,
        messages: tuple[LLMMessage, ...],
        tools: tuple[LLMToolDefinition, ...],
    ) -> int:
        ...


@dataclass(frozen=True)
class _UsageSnapshot:
    """Lifetime accounting captured at one run boundary."""

    turns: int
    tool_calls: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int

    def delta_from(self, earlier: _UsageSnapshot) -> _UsageSnapshot:
        delta = _UsageSnapshot(
            turns=self.turns - earlier.turns,
            tool_calls=self.tool_calls - earlier.tool_calls,
            cost_usd=self.cost_usd - earlier.cost_usd,
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=(
                self.completion_tokens - earlier.completion_tokens
            ),
        )
        if (
            delta.turns < 0
            or delta.tool_calls < 0
            or delta.cost_usd < 0
            or delta.prompt_tokens < 0
            or delta.completion_tokens < 0
        ):
            raise RuntimeError("session lifetime accounting moved backwards")
        return delta


@dataclass
class _RunState:
    """State belonging only to one backend.run() invocation."""

    started_at: float
    deadline: float
    usage_before: _UsageSnapshot
    events: list[AgentEvent]
    sink: EventSink | None
    recoveries: int = 0
    sink_error: str | None = None


@dataclass(frozen=True)
class _ScheduledToolResults:
    """Ordered results from one provider-issued tool-call group."""

    results: tuple[LLMToolResult, ...]
    started_count: int
    timed_out: bool


@dataclass
class _SessionState:
    """Mutable state retained across runs of one agent session."""

    session_id: str
    messages: list[LLMMessage]

    workdir: Path
    parent_id: str
    operator: str
    model: str
    tool_names: tuple[str, ...]

    lifetime_turns: int = 0
    lifetime_tool_calls: int = 0
    lifetime_cost_usd: float = 0.0
    lifetime_prompt_tokens: int = 0
    lifetime_completion_tokens: int = 0

    output_recoveries: int = 0
    round_index: int = 0
    next_event_sequence: int = 0
    created_at: float = 0.0
    last_active_at: float = 0.0

    def usage_snapshot(self) -> _UsageSnapshot:
        return _UsageSnapshot(
            turns=self.lifetime_turns,
            tool_calls=self.lifetime_tool_calls,
            cost_usd=self.lifetime_cost_usd,
            prompt_tokens=self.lifetime_prompt_tokens,
            completion_tokens=self.lifetime_completion_tokens,
        )

    def next_event(
        self,
        kind: AgentEventKind,
        *,
        elapsed_s: float = 0.0,
        call_id: str | None = None,
        tool_name: str | None = None,
        content: str = "",
        data: Mapping[str, object] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            session_id=self.session_id,
            round_index=self.round_index,
            sequence=self.next_event_sequence,
            kind=kind,
            turn=self.lifetime_turns,
            elapsed_s=elapsed_s,
            call_id=call_id,
            tool_name=tool_name,
            content=content,
            data={} if data is None else data,
        )
        self.next_event_sequence += 1
        return event


class _SessionStore:
    """In-memory session lifecycle owned by one native agent backend."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        session_id_factory: Callable[[], str] = _new_session_id,
    ):
        self._clock = clock
        self._session_id_factory = session_id_factory
        self._sessions: dict[str, _SessionState] = {}

    def open(
        self,
        request: AgentSessionRequest,
        *,
        model: str,
        tool_names: tuple[str, ...],
    ) -> tuple[_SessionState, AgentEvent]:
        """Create a new session or resume an existing one."""

        workdir = Path(request.workdir).resolve()
        if not workdir.is_dir():
            raise ValueError(
                "agent session workdir must be an existing directory"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("agent session model must be non-empty")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in tool_names
        ):
            raise ValueError("agent session tool names must be non-empty")
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("agent session tool names must be unique")

        if request.session_id is None:
            return self._create(
                request,
                workdir=workdir,
                model=model,
                tool_names=tool_names,
            )
        return self._resume(
            request,
            workdir=workdir,
            model=model,
            tool_names=tool_names,
        )

    def get(self, session_id: str) -> _SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown agent session: {session_id!r}"
            ) from exc

    def release(self, session_id: str) -> None:
        """Idempotently remove one session from memory."""

        self._sessions.pop(session_id, None)

    def _create(
        self,
        request: AgentSessionRequest,
        *,
        workdir: Path,
        model: str,
        tool_names: tuple[str, ...],
    ) -> tuple[_SessionState, AgentEvent]:
        if not request.system.strip():
            raise ValueError(
                "new agent session requires a non-empty system prompt"
            )
        if not request.user.strip():
            raise ValueError(
                "new agent session requires a non-empty user prompt"
            )
        if request.feedback:
            raise ValueError(
                "new agent session cannot contain repair feedback"
            )

        session_id = self._session_id_factory()
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError("session_id_factory returned an invalid ID")
        if session_id in self._sessions:
            raise RuntimeError(
                "session_id_factory returned duplicate ID: "
                f"{session_id!r}"
            )

        now = self._clock()
        state = _SessionState(
            session_id=session_id,
            messages=[
                LLMMessage(role="system", content=request.system),
                LLMMessage(role="user", content=request.user),
            ],
            workdir=workdir,
            parent_id=request.parent.id,
            operator=request.operator,
            model=model,
            tool_names=tool_names,
            created_at=now,
            last_active_at=now,
        )
        self._sessions[session_id] = state

        event = state.next_event(
            AgentEventKind.SESSION_START,
            data={
                "parent_id": state.parent_id,
                "operator": state.operator,
                "model": state.model,
                "tool_names": state.tool_names,
                "system": request.system,
                "user": request.user,
            },
        )
        return state, event

    def _resume(
        self,
        request: AgentSessionRequest,
        *,
        workdir: Path,
        model: str,
        tool_names: tuple[str, ...],
    ) -> tuple[_SessionState, AgentEvent]:
        assert request.session_id is not None
        state = self.get(request.session_id)
        self._validate_resume(
            state,
            request,
            workdir=workdir,
            model=model,
            tool_names=tool_names,
        )

        feedback_content = render_preflight_feedback(request.feedback)
        state.messages.append(
            LLMMessage(
                role="user",
                content=feedback_content,
            )
        )
        state.last_active_at = self._clock()
        state.round_index += 1

        event = state.next_event(
            AgentEventKind.SESSION_RESUME,
            data={
                "issue_count": len(request.feedback),
                "feedback": feedback_content,
            },
        )
        return state, event

    @staticmethod
    def _validate_resume(
        state: _SessionState,
        request: AgentSessionRequest,
        *,
        workdir: Path,
        model: str,
        tool_names: tuple[str, ...],
    ) -> None:
        expected = {
            "workdir": state.workdir,
            "parent_id": state.parent_id,
            "operator": state.operator,
            "model": state.model,
            "tool_names": state.tool_names,
            "system": state.messages[0].content,
            "user": state.messages[1].content,
        }
        received = {
            "workdir": workdir,
            "parent_id": request.parent.id,
            "operator": request.operator,
            "model": model,
            "tool_names": tool_names,
            "system": request.system,
            "user": request.user,
        }
        mismatches = [
            name
            for name, expected_value in expected.items()
            if received[name] != expected_value
        ]
        if mismatches:
            raise ValueError(
                "agent session fingerprint mismatch: "
                + ", ".join(mismatches)
            )
        if not request.feedback:
            raise ValueError(
                "resumed agent session requires preflight feedback"
            )

class _ToolScheduler:
    """Execute safe tool batches concurrently and unsafe calls exclusively."""

    def __init__(
        self,
        *,
        registry: AgentToolRegistry,
        max_workers: int,
        clock: Callable[[], float] = monotonic,
    ):
        self.registry = registry
        self.max_workers = max_workers
        self.clock = clock


    def execute(
        self,
        calls: tuple[LLMToolCall, ...],
        *,
        context_factory: Callable[[float], AgentToolContext],
        deadline: float,
    ) -> _ScheduledToolResults:
        results: list[LLMToolResult | None] = [None] * len(calls)
        started_count = 0
        timed_out = False
        index = 0

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="evoharness-agent-tool",
        ) as executor:
            while index < len(calls):
                if self.clock() >= deadline:
                    self._fill_timeout_results(
                        calls,
                        results,
                        start=index,
                    )
                    timed_out = True
                    break
                safe_batch: list[
                    tuple[int, LLMToolCall, AgentToolContext]
                ] = []

                while index < len(calls):
                    remaining_s = deadline - self.clock()
                    if remaining_s <= 0:
                        self._fill_timeout_results(
                            calls,
                            results,
                            start=index,
                        )
                        timed_out = True
                        break

                    call = calls[index]
                    ctx = context_factory(remaining_s)

                    if not self.registry.is_concurrency_safe(call, ctx):
                        break

                    safe_batch.append((index, call, ctx))
                    index += 1

                if safe_batch:
                    futures = [
                        (
                            result_index,
                            executor.submit(
                                self.registry.invoke,
                                call,
                                ctx,
                            ),
                        )
                        for result_index, call, ctx in safe_batch
                    ]

                    started_count += len(futures)

                    for result_index, future in futures:
                        results[result_index] = future.result()
                    continue

                if index >= len(calls):
                    break

                remaining_s = deadline - self.clock()
                if remaining_s <= 0:
                    self._fill_timeout_results(
                        calls,
                        results,
                        start=index,
                    )
                    timed_out = True
                    break

                call = calls[index]
                ctx = context_factory(remaining_s)

                # Unsafe calls run on the coordinating thread, after all
                # previous safe futures have completed.
                results[index] = self.registry.invoke(call, ctx)
                started_count += 1
                index += 1

        if any(result is None for result in results):
            raise RuntimeError(
                "tool scheduler left calls without results"
            )

        return _ScheduledToolResults(
            results=tuple(
                result
                for result in results
                if result is not None
            ),
            started_count=started_count,
            timed_out=timed_out,
        )

    @staticmethod
    def _fill_timeout_results(
        calls: tuple[LLMToolCall, ...],
        results: list[LLMToolResult | None],
        *,
        start: int,
    ) -> None:
        for index in range(start, len(calls)):
            call = calls[index]
            results[index] = make_tool_error(
                call.call_id,
                "timeout",
                (
                    "tool call was not started because the agent "
                    "session deadline expired"
                ),
            )


class NativeToolAgentBackend:
    """Provider-neutral native model/tool agent runtime."""

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        registry: AgentToolRegistry,
        max_input_tokens: int,
        token_estimator: TokenEstimator,
        max_parallel_tools: int = 4,
        clock: Callable[[], float] = monotonic,
        session_id_factory: Callable[[], str] = _new_session_id,
        recent_tool_results_to_keep: int = 8,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("agent model must be non-empty")
        if not isinstance(registry, AgentToolRegistry):
            raise TypeError("registry must be AgentToolRegistry")
        if (
            isinstance(max_input_tokens, bool)
            or not isinstance(max_input_tokens, int)
            or max_input_tokens < 1
        ):
            raise ValueError("max_input_tokens must be a positive integer")
        if (
            isinstance(max_parallel_tools, bool)
            or not isinstance(max_parallel_tools, int)
            or max_parallel_tools < 1
        ):
            raise ValueError(
                "max_parallel_tools must be a positive integer"
            )
        if not callable(token_estimator):
            raise TypeError("token_estimator must be callable")

        if (
            isinstance(recent_tool_results_to_keep, bool)
            or not isinstance(recent_tool_results_to_keep, int)
            or recent_tool_results_to_keep < 0
        ):
            raise ValueError(
                "recent_tool_results_to_keep must be a nonnegative integer"
            )
        self.recent_tool_results_to_keep = recent_tool_results_to_keep

        self.client = client
        self.model = model
        self.registry = registry
        self.max_input_tokens = max_input_tokens
        self.token_estimator = token_estimator
        self.max_parallel_tools = max_parallel_tools
        self.clock = clock
        self._sessions = _SessionStore(
            clock=clock,
            session_id_factory=session_id_factory,
        )

        self.scheduler = _ToolScheduler(
            registry=registry,
            max_workers=max_parallel_tools,
            clock=clock,
        )

    def release(self, session_id: str) -> None:
        self._sessions.release(session_id)

    def run(
        self,
        request: AgentSessionRequest,
    ) -> AgentSessionResult | None:
        started_at = self.clock()
        state, session_event = self._sessions.open(
            request=request,
            model=self.model,
            tool_names=tuple(
                definition.name
                for definition in self.registry.definitions
            ),
        )

        run = _RunState(
            started_at=started_at,
            deadline=started_at + request.limits.timeout_s,
            usage_before=state.usage_snapshot(),
            events=[],
            sink=request.event_sink,
        )
        self._record_event(run, session_event)

        if run.sink_error is not None:
            return self._finish(
                state,
                run,
                termination=AgentTermination.BACKEND_ERROR,
                final_message=run.sink_error,
            )
        while True:
            try:
                admission = self._admit_model_query(
                    state,
                    request,
                    run,
                )
            except Exception as exc:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=f"{type(exc).__name__}: {exc}",
                )

            if admission is not None:
                return admission

            remaining_s = run.deadline - self.clock()
            if remaining_s <= 0:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.TIMEOUT,
                )

            try:
                response = self.client.query_messages(
                    messages=tuple(state.messages),
                    model=self.model,
                    tools=self.registry.definitions,
                    parallel_tool_calls=True,
                    timeout_s=remaining_s,
                )
            except LLMProtocolError as exc:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.PROTOCOL_ERROR,
                    final_message=str(exc),
                )
            except LLMTransientError as exc:
                return self._finish(
                    state,
                    run,
                    termination=(
                        AgentTermination.TIMEOUT
                        if self.clock() >= run.deadline
                        else AgentTermination.BACKEND_ERROR
                    ),
                    final_message=str(exc),
                )
            except Exception as exc:
                return self._finish(
                    state,
                    run,
                    termination=(
                        AgentTermination.TIMEOUT
                        if self.clock() >= run.deadline
                        else AgentTermination.BACKEND_ERROR
                    ),
                    final_message=f"{type(exc).__name__}: {exc}",
                )

            self._record_model_response(state, run, response)
            if run.sink_error is not None:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=run.sink_error,
                    model=response.model,
                )

            if response.stop_reason is LLMStopReason.TOOL_CALLS:
                tool_result = self._handle_tool_calls(
                    state,
                    request,
                    run,
                    response,
                )
                if tool_result is not None:
                    return tool_result
                continue

            if self.clock() >= run.deadline:
                if response.text.strip():
                    state.messages.append(
                        LLMMessage(
                            role="assistant",
                            content=response.text,
                        )
                    )
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.TIMEOUT,
                    final_message=response.text,
                    model=response.model,
                )

            decision = self._decide_response(state, run, response)
            if decision is not None:
                return decision

    def _handle_tool_calls(
        self,
        state: _SessionState,
        request: AgentSessionRequest,
        run: _RunState,
        response: LLMResponse,
    ) -> AgentSessionResult | None:
        calls = response.tool_calls
        usage = state.usage_snapshot().delta_from(run.usage_before)

        if self.clock() >= run.deadline:
            self._record_tool_calls(
                state,
                run,
                calls,
                admitted_count=0,
                rejected_reason="timeout",
            )
            results = self._synthetic_results(
                calls,
                code="timeout",
                message=(
                    "tool call was not started because the agent "
                    "session deadline expired"
                ),
            )
            self._append_tool_exchange(
                state,
                run,
                response,
                results,
                started_count=0,
            )
            if run.sink_error is not None:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=run.sink_error,
                    model=response.model,
                )
            return self._finish(
                state,
                run,
                termination=AgentTermination.TIMEOUT,
                model=response.model,
            )

        if (
            request.limits.max_cost_usd is not None
            and usage.cost_usd >= request.limits.max_cost_usd
        ):
            self._record_tool_calls(
                state,
                run,
                calls,
                admitted_count=0,
                rejected_reason="cost-limit",
            )
            results = self._synthetic_results(
                calls,
                code="cost-limit",
                message=(
                    "tool call was not started because the run "
                    "cost budget was exhausted"
                ),
            )
            self._append_tool_exchange(
                state,
                run,
                response,
                results,
                started_count=0,
            )
            if run.sink_error is not None:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=run.sink_error,
                    model=response.model,
                )
            return self._finish(
                state,
                run,
                termination=AgentTermination.COST_LIMIT,
                model=response.model,
            )

        remaining_tool_calls = max(
            0,
            request.limits.max_tool_calls - usage.tool_calls,
        )
        admitted_calls = calls[:remaining_tool_calls]
        rejected_calls = calls[remaining_tool_calls:]

        self._record_tool_calls(
            state,
            run,
            calls,
            admitted_count=len(admitted_calls),
            rejected_reason="tool-limit",
        )
        if run.sink_error is not None:
            results = self._synthetic_results(
                calls,
                code="event-sink-error",
                message="tool call was not started because transcript failed",
            )
            self._append_tool_exchange(
                state,
                run,
                response,
                results,
                started_count=0,
            )
            return self._finish(
                state,
                run,
                termination=AgentTermination.BACKEND_ERROR,
                final_message=run.sink_error,
                model=response.model,
            )

        try:
            scheduled = self.scheduler.execute(
                admitted_calls,
                context_factory=lambda remaining_s: AgentToolContext(
                    workdir=state.workdir,
                    parent=request.parent,
                    operator=request.operator,
                    preflight=request.preflight,
                    remaining_timeout_s=remaining_s,
                ),
                deadline=run.deadline,
            )
        except Exception as exc:
            results = self._synthetic_results(
                calls,
                code="tool-scheduler-error",
                message=f"{type(exc).__name__}: {exc}",
            )
            self._append_tool_exchange(
                state,
                run,
                response,
                results,
                started_count=0,
            )
            if run.sink_error is not None:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=run.sink_error,
                    model=response.model,
                )
            return self._finish(
                state,
                run,
                termination=AgentTermination.BACKEND_ERROR,
                final_message=f"{type(exc).__name__}: {exc}",
                model=response.model,
            )

        rejected_results = self._synthetic_results(
            rejected_calls,
            code="tool-limit",
            message=(
                "tool call was not executed because the run "
                "tool budget was exhausted"
            ),
        )

        all_results = (*scheduled.results, *rejected_results)

        self._append_tool_exchange(
            state,
            run,
            response,
            all_results,
            started_count=scheduled.started_count,
        )

        if run.sink_error is not None:
            return self._finish(
                state,
                run,
                termination=AgentTermination.BACKEND_ERROR,
                final_message=run.sink_error,
                model=response.model,
            )

        if scheduled.timed_out or self.clock() >= run.deadline:
            return self._finish(
                state,
                run,
                termination=AgentTermination.TIMEOUT,
                model=response.model,
            )

        if rejected_calls:
            return self._finish(
                state,
                run,
                termination=AgentTermination.TOOL_LIMIT,
                model=response.model,
            )

        return None

    def _append_tool_exchange(
        self,
        state: _SessionState,
        run: _RunState,
        response: LLMResponse,
        results: tuple[LLMToolResult, ...],
        *,
        started_count: int,
    ) -> None:
        calls = response.tool_calls

        if len(calls) != len(results):
            raise RuntimeError(
                "every tool call must have exactly one tool result"
            )

        for call, result in zip(calls, results, strict=True):
            if call.call_id != result.call_id:
                raise RuntimeError(
                    "tool result order or call ID does not match"
                )

        # Mutate history only after every result exists and pairing is valid.
        state.messages.extend(
            (
                LLMMessage(
                    role="assistant",
                    content=response.text,
                    tool_calls=calls,
                ),
                LLMMessage(
                    role="tool",
                    tool_results=results,
                ),
            )
        )
        state.lifetime_tool_calls += started_count
        state.last_active_at = self.clock()

        for index, (call, result) in enumerate(
            zip(calls, results, strict=True)
        ):
            executed = index < started_count
            elapsed_s = max(0.0, self.clock() - run.started_at)

            self._record_event(
                run,
                state.next_event(
                    AgentEventKind.TOOL_RESULT,
                    elapsed_s=elapsed_s,
                    call_id=result.call_id,
                    tool_name=call.name,
                    content=result.content,
                    data={
                        "is_error": result.is_error,
                        "executed": executed,
                    },
                ),
            )

    def _record_tool_calls(
        self,
        state: _SessionState,
        run: _RunState,
        calls: tuple[LLMToolCall, ...],
        *,
        admitted_count: int,
        rejected_reason: str,
    ) -> None:
        """Persist tool intent before any admitted call can execute."""

        for index, call in enumerate(calls):
            admitted = index < admitted_count
            self._record_event(
                run,
                state.next_event(
                    AgentEventKind.TOOL_CALL,
                    elapsed_s=max(0.0, self.clock() - run.started_at),
                    call_id=call.call_id,
                    tool_name=call.name,
                    data={
                        "arguments": dict(call.arguments),
                        "admitted": admitted,
                        "admission_reason": (
                            "execute" if admitted else rejected_reason
                        ),
                    },
                ),
            )

    @staticmethod
    def _synthetic_results(
        calls: tuple[LLMToolCall, ...],
        *,
        code: str,
        message: str,
    ) -> tuple[LLMToolResult, ...]:
        return tuple(
            make_tool_error(call.call_id, code, message)
            for call in calls
        )

    @staticmethod
    def _record_event(
        run: _RunState,
        event: AgentEvent,
    ) -> None:
        """Record internally first, then fail closed on sink failure."""

        run.events.append(event)

        if run.sink is None or run.sink_error is not None:
            return

        try:
            run.sink.emit(event)
        except Exception as exc:
            run.sink_error = (
                f"event sink failed: {type(exc).__name__}: {exc}"
            )

    def _admit_model_query(
        self,
        state: _SessionState,
        request: AgentSessionRequest,
        run: _RunState,
    ) -> AgentSessionResult | None:
        usage = state.usage_snapshot().delta_from(run.usage_before)

        if self.clock() >= run.deadline:
            return self._finish(
                state,
                run,
                termination=AgentTermination.TIMEOUT,
            )

        if usage.turns >= request.limits.max_turns:
            return self._finish(
                state,
                run,
                termination=AgentTermination.TURN_LIMIT,
            )

        if (
            request.limits.max_cost_usd is not None
            and usage.cost_usd >= request.limits.max_cost_usd
        ):
            return self._finish(
                state,
                run,
                termination=AgentTermination.COST_LIMIT,
            )

        return self._prepare_context(state, run)

    def _record_model_response(
        self,
        state: _SessionState,
        run: _RunState,
        response: LLMResponse,
    ) -> None:
        state.lifetime_turns += 1
        state.lifetime_cost_usd += response.cost
        state.lifetime_prompt_tokens += response.prompt_tokens
        state.lifetime_completion_tokens += response.completion_tokens
        state.last_active_at = self.clock()

        event = state.next_event(
            AgentEventKind.MODEL_RESPONSE,
            elapsed_s=max(0.0, self.clock() - run.started_at),
            content=response.text,
            data={
                "model": response.model,
                "stop_reason": response.stop_reason.value,
                "cost_usd": response.cost,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "tool_call_count": len(response.tool_calls),
            },
        )
        self._record_event(run, event)

    def _decide_response(
        self,
        state: _SessionState,
        run: _RunState,
        response: LLMResponse,
    ) -> AgentSessionResult | None:
        if response.text.strip():
            state.messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.text,
                )
            )

        if response.stop_reason is LLMStopReason.COMPLETED:
            if not response.text.strip():
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.PROTOCOL_ERROR,
                    final_message="completed model response was empty",
                    model=response.model,
                )
            return self._finish(
                state,
                run,
                termination=AgentTermination.COMPLETED,
                final_message=response.text,
                model=response.model,
            )

        if response.stop_reason is LLMStopReason.REFUSAL:
            return self._finish(
                state,
                run,
                termination=AgentTermination.REFUSAL,
                final_message=response.text,
                model=response.model,
            )

        if response.stop_reason is LLMStopReason.CONTENT_FILTER:
            return self._finish(
                state,
                run,
                termination=AgentTermination.CONTENT_FILTER,
                final_message=response.text,
                model=response.model,
            )

        if response.stop_reason is LLMStopReason.MAX_TOKENS:
            if not response.text.strip():
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.PROTOCOL_ERROR,
                    final_message="max-token model response was empty",
                    model=response.model,
                )

            if run.recoveries >= 1:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.OUTPUT_LIMIT,
                    final_message=response.text,
                    model=response.model,
                )

            run.recoveries += 1
            state.output_recoveries += 1
            state.messages.append(
                LLMMessage(
                    role="user",
                    content=_MAX_TOKENS_CONTINUATION,
                )
            )
            self._record_event(
                run,
                state.next_event(
                    AgentEventKind.RECOVERY,
                    elapsed_s=max(0.0, self.clock() - run.started_at),
                    content=_MAX_TOKENS_CONTINUATION,
                    data={
                        "reason": "output-limit",
                        "attempt": run.recoveries,
                    },
                ),
            )
            if run.sink_error is not None:
                return self._finish(
                    state,
                    run,
                    termination=AgentTermination.BACKEND_ERROR,
                    final_message=run.sink_error,
                    model=response.model,
                )
            return None

        return self._finish(
            state,
            run,
            termination=AgentTermination.PROTOCOL_ERROR,
            final_message=(
                f"unsupported model stop reason: {response.stop_reason.value}"
            ),
            model=response.model,
        )

    def _finish(
        self,
        state: _SessionState,
        run: _RunState,
        *,
        termination: AgentTermination,
        final_message: str = "",
        model: str | None = None,
    ) -> AgentSessionResult:
        elapsed_s = max(0.0, self.clock() - run.started_at)
        state.last_active_at = self.clock()
        usage = state.usage_snapshot().delta_from(run.usage_before)

        event = state.next_event(
            AgentEventKind.TERMINATION,
            elapsed_s=elapsed_s,
            content=final_message,
            data={
                "termination": termination.value,
                "model": self.model if model is None else model,
                "cost_usd": usage.cost_usd,
                "turns": usage.turns,
                "tool_calls": usage.tool_calls,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "elapsed_s": elapsed_s,
            },
        )

        sink_error_before = run.sink_error

        self._record_event(run, event)

        if run.sink_error != sink_error_before:
            termination = AgentTermination.BACKEND_ERROR
            final_message = run.sink_error or "event sink failed"
            run.events[-1] = replace(
                event,
                content=final_message,
                data={
                    **event.data,
                    "termination": termination.value,
                    "sink_error": final_message,
                },
            )

        return AgentSessionResult(
            termination=termination,
            session_id=state.session_id,
            final_message=final_message,
            model=self.model if model is None else model,
            cost_usd=usage.cost_usd,
            turns=usage.turns,
            tool_calls=usage.tool_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            elapsed_s=elapsed_s,
            events=tuple(run.events),
        )

    def _estimate_input_tokens(
        self,
        state: _SessionState,
    ) -> int:
        estimated = self.token_estimator(
            tuple(state.messages),
            self.registry.definitions,
        )
        if (
            isinstance(estimated, bool)
            or not isinstance(estimated, int)
            or estimated < 0
        ):
            raise RuntimeError(
                "token_estimator must return a nonnegative integer"
            )
        return estimated

    @staticmethod
    def _compacted_tool_content(is_error: bool) -> str:
        return json.dumps(
            {
                "ok": not is_error,
                "compacted": True,
                "message": "older tool result removed",
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _is_compacted_result(
        cls,
        result: LLMToolResult,
    ) -> bool:
        return result.content == cls._compacted_tool_content(
            result.is_error
        )

    def _prepare_context(
        self,
        state: _SessionState,
        run: _RunState,
    ) -> AgentSessionResult | None:
        estimated_before = self._estimate_input_tokens(state)
        if estimated_before <= self.max_input_tokens:
            return None

        locations: list[tuple[int, int, LLMToolResult]] = []

        for message_index, message in enumerate(state.messages):
            if message.role != "tool":
                continue

            for result_index, result in enumerate(message.tool_results):
                if not self._is_compacted_result(result):
                    locations.append(
                        (message_index, result_index, result)
                    )

        keep = self.recent_tool_results_to_keep
        compactable = locations[:-keep] if keep else locations

        if not compactable:
            return self._finish(
                state,
                run,
                termination=AgentTermination.CONTEXT_LIMIT,
                final_message=(
                    f"estimated input tokens {estimated_before} exceed "
                    f"limit {self.max_input_tokens}"
                ),
            )

        compacted_call_ids: list[str] = []
        estimated_after = estimated_before

        for message_index, result_index, result in compactable:
            original_message = state.messages[message_index]
            updated_results = list(original_message.tool_results)
            updated_results[result_index] = LLMToolResult(
                call_id=result.call_id,
                content=self._compacted_tool_content(result.is_error),
                is_error=result.is_error,
            )
            state.messages[message_index] = LLMMessage(
                role="tool",
                tool_results=tuple(updated_results),
            )

            candidate_estimate = self._estimate_input_tokens(state)
            if candidate_estimate >= estimated_after:
                state.messages[message_index] = original_message
                continue

            compacted_call_ids.append(result.call_id)
            estimated_after = candidate_estimate
            if estimated_after <= self.max_input_tokens:
                break

        if not compacted_call_ids:
            return self._finish(
                state,
                run,
                termination=AgentTermination.CONTEXT_LIMIT,
                final_message=(
                    f"estimated input tokens {estimated_before} exceed "
                    f"limit {self.max_input_tokens}; no eligible tool "
                    "result reduced the estimate"
                ),
            )

        event = state.next_event(
            AgentEventKind.CONTEXT_COMPACT,
            elapsed_s=max(0.0, self.clock() - run.started_at),
            data={
                "call_ids": tuple(compacted_call_ids),
                "estimated_tokens_before": estimated_before,
                "estimated_tokens_after": estimated_after,
                "estimated_tokens_released": (
                    estimated_before - estimated_after
                ),
            },
        )
        self._record_event(run, event)

        if run.sink_error is not None:
            return self._finish(
                state,
                run,
                termination=AgentTermination.BACKEND_ERROR,
                final_message=run.sink_error,
            )

        if estimated_after > self.max_input_tokens:
            return self._finish(
                state,
                run,
                termination=AgentTermination.CONTEXT_LIMIT,
                final_message=(
                    f"estimated input tokens {estimated_after} still exceed "
                    f"limit {self.max_input_tokens} after compaction"
                ),
            )

        return None
