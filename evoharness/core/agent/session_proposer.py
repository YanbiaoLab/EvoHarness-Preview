"""AgentSessionProposer outer-loop orchestration."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from .runtime import NativeToolAgentBackend
from ..operators import parse_change_header
from ..population import Candidate
from ..preflight import PreflightContext, PreflightIssue, ProposalPreflight
from ..proposer import Proposal, ProposeResult, Proposer
from ..workspace import Workspace, WorkspaceError
from .contracts import (
    AgentBackend,
    AgentSessionLimits,
    AgentSessionRequest,
    AgentSessionResult,
    AgentTermination,
    EventSinkFactory,
    ManagedEventSink,
    PreflightTraceRecord,
    ProposalTraceSummary,
)
from .transcript import render_workspace_patch


_PROPOSAL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)


logger = logging.getLogger(__name__)


def _new_proposal_id() -> str:
    return uuid.uuid4().hex


class _ProposalResourceError(RuntimeError):
    """Classified failure while opening or closing proposal resources."""

    def __init__(self, code: str, cause: Exception):
        super().__init__(f"{code}: {type(cause).__name__}: {cause}")
        self.code = code
        self.cause = cause


class _ProposalLoopError(RuntimeError):
    """Classified terminal condition in the proposer outer loop."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class _ProposalUsage:
    """Accounting accumulated across all backend runs."""

    attempts: int = 0
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    last_termination: AgentTermination | None = None
    last_model: str = ""

    def absorb(
        self,
        result: AgentSessionResult,
        run_limits: AgentSessionLimits,
    ) -> None:
        if not isinstance(result, AgentSessionResult):
            raise _ProposalLoopError(
                "backend-error",
                "backend.run() must return AgentSessionResult",
            )

        if result.turns > run_limits.max_turns:
            raise _ProposalLoopError(
                "backend-error",
                "backend exceeded its turn budget",
            )

        if result.tool_calls > run_limits.max_tool_calls:
            raise _ProposalLoopError(
                "backend-error",
                "backend exceeded its tool budget",
            )

        self.attempts += 1
        self.turns += result.turns
        self.tool_calls += result.tool_calls
        self.cost_usd += result.cost_usd
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.last_termination = result.termination
        self.last_model = result.model or ""

    def remaining_limits(
        self,
        total: AgentSessionLimits,
        *,
        remaining_timeout_s: float,
    ) -> AgentSessionLimits:
        if remaining_timeout_s <= 0:
            raise _ProposalLoopError(
                "timeout",
                "proposal deadline was exhausted",
            )

        remaining_turns = total.max_turns - self.turns
        if remaining_turns <= 0:
            raise _ProposalLoopError(
                "turn-limit",
                "proposal turn budget was exhausted",
            )

        remaining_tools = total.max_tool_calls - self.tool_calls
        if remaining_tools < 0:
            raise _ProposalLoopError(
                "backend-error",
                "tool accounting exceeded the total budget",
            )

        remaining_cost: float | None = None
        if total.max_cost_usd is not None:
            remaining_cost = total.max_cost_usd - self.cost_usd
            if remaining_cost <= 0:
                raise _ProposalLoopError(
                    "cost-limit",
                    "proposal cost budget was exhausted",
                )

        return AgentSessionLimits(
            max_turns=remaining_turns,
            max_tool_calls=remaining_tools,
            timeout_s=remaining_timeout_s,
            max_cost_usd=remaining_cost,
        )


class _ProposalResources:
    """Own one proposal's sink, backend session, and temporary workspace."""

    def __init__(
        self,
        *,
        proposal_id: str,
        parent: Candidate,
        operator: str,
        backend: AgentBackend,
        event_sink_factory: EventSinkFactory,
        work_root: Path | None,
    ):
        if (
            not isinstance(proposal_id, str)
            or _PROPOSAL_ID_PATTERN.fullmatch(proposal_id) is None
        ):
            raise ValueError("proposal_id contains unsafe characters")
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator must be non-empty")

        self.proposal_id = proposal_id
        self.parent = parent
        self.operator = operator
        self.backend = backend
        self.event_sink_factory = event_sink_factory
        self.work_root = (
            None if work_root is None else Path(work_root).resolve()
        )

        self.sink: ManagedEventSink | None = None
        self.workdir: Path | None = None
        self.session_id: str | None = None
        self.trace_path: str | None = None

        self._temporary_directory: TemporaryDirectory | None = None
        self._entered = False
        self._release_attempted = False
        self._flush_attempted = False

    def __enter__(self) -> _ProposalResources:
        if self._entered:
            raise RuntimeError("proposal resources cannot be reused")
        self._entered = True

        self._open_sink()

        try:
            self._materialize_workspace()
        except Exception as exc:
            self._cleanup()
            if isinstance(exc, _ProposalResourceError):
                raise
            raise _ProposalResourceError(
                "workspace-materialize-error",
                exc,
            ) from exc

        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        cleanup_error = self._cleanup()

        # Never hide an exception raised by the proposal loop itself.
        if exc is None and cleanup_error is not None:
            raise cleanup_error

        return False

    def bind_session(self, session_id: str | None) -> None:
        """Remember the backend session that must be released on exit."""

        if session_id is None:
            return

        if not isinstance(session_id, str) or not session_id.strip():
            raise _ProposalResourceError(
                "backend-session-error",
                ValueError("backend returned an invalid session_id"),
            )

        if self.session_id is None:
            self.session_id = session_id
            return

        if self.session_id != session_id:
            raise _ProposalResourceError(
                "backend-session-error",
                ValueError(
                    "backend changed session_id across repair rounds"
                ),
            )

    def record_preflight(self, record: PreflightTraceRecord) -> None:
        if self.sink is None:
            raise _ProposalResourceError(
                "sink-preflight-error",
                RuntimeError("proposal sink is not open"),
            )
        try:
            self.sink.record_preflight(record)
        except Exception as exc:
            raise _ProposalResourceError(
                "sink-preflight-error",
                exc,
            ) from exc

    def finalize(
        self,
        summary: ProposalTraceSummary,
        final_patch: str | None,
    ) -> None:
        if self.sink is None:
            raise _ProposalResourceError(
                "sink-finalize-error",
                RuntimeError("proposal sink is not open"),
            )
        try:
            self.sink.finalize(summary, final_patch)
        except Exception as exc:
            raise _ProposalResourceError(
                "sink-finalize-error",
                exc,
            ) from exc

    def release_session(self) -> None:
        """Release once while retaining the session ID as provenance."""

        if self.session_id is None or self._release_attempted:
            return
        self._release_attempted = True
        try:
            self.backend.release(self.session_id)
        except Exception as exc:
            raise _ProposalResourceError(
                "backend-release-error",
                exc,
            ) from exc

    def flush_sink(self) -> None:
        """Flush once before the terminal summary is assembled."""

        if self.sink is None or self._flush_attempted:
            return
        self._flush_attempted = True
        try:
            self.sink.flush()
        except Exception as exc:
            raise _ProposalResourceError(
                "sink-flush-error",
                exc,
            ) from exc

    def _open_sink(self) -> None:
        try:
            sink = self.event_sink_factory.open(
                proposal_id=self.proposal_id,
                parent_id=self.parent.id,
                operator=self.operator,
            )
        except Exception as exc:
            raise _ProposalResourceError(
                "sink-open-error",
                exc,
            ) from exc

        if not isinstance(sink, ManagedEventSink):
            raise _ProposalResourceError(
                "sink-open-error",
                TypeError(
                    "event sink factory must return ManagedEventSink"
                ),
            )

        trace_path = sink.trace_path
        if trace_path is not None and (
            not isinstance(trace_path, str)
            or not trace_path.strip()
        ):
            try:
                sink.close()
            finally:
                raise _ProposalResourceError(
                    "sink-open-error",
                    ValueError(
                        "managed event sink trace_path must be non-empty"
                    ),
                )

        self.sink = sink
        self.trace_path = trace_path

    def _materialize_workspace(self) -> None:
        if self.work_root is not None:
            self.work_root.mkdir(parents=True, exist_ok=True)

        temporary = TemporaryDirectory(
            prefix=f"evoharness-{self.proposal_id}-",
            dir=self.work_root,
        )
        self._temporary_directory = temporary

        temporary_root = Path(temporary.name).resolve()
        workdir = Path(
            self.parent.workspace.materialize(temporary_root)
        ).resolve()

        if workdir != temporary_root:
            raise _ProposalResourceError(
                "workspace-materialize-error",
                ValueError(
                    "workspace.materialize() must return its destination"
                ),
            )

        if not workdir.is_dir():
            raise _ProposalResourceError(
                "workspace-materialize-error",
                ValueError(
                    "workspace.materialize() did not create a directory"
                ),
            )

        self.workdir = workdir

    def _cleanup(self) -> _ProposalResourceError | None:
        first_error: _ProposalResourceError | None = None
        sink = self.sink

        if sink is not None and not self._flush_attempted:
            try:
                self.flush_sink()
            except _ProposalResourceError as exc:
                first_error = exc

        if self.session_id is not None and not self._release_attempted:
            try:
                self.release_session()
            except _ProposalResourceError as exc:
                if first_error is None:
                    first_error = exc

        if sink is not None:
            self.sink = None
            try:
                sink.close()
            except Exception as exc:
                if first_error is None:
                    first_error = _ProposalResourceError(
                        "sink-close-error",
                        exc,
                    )

        temporary = self._temporary_directory
        self.workdir = None
        if temporary is not None:
            self._temporary_directory = None
            try:
                temporary.cleanup()
            except Exception as exc:
                if first_error is None:
                    first_error = _ProposalResourceError(
                        "workspace-cleanup-error",
                        exc,
                    )

        return first_error


def _budget_note(limits: AgentSessionLimits) -> str:
    """Turn-budget awareness for the system prompt. Smoke-run postmortem:
    deterministic sessions burned 11/12 turns on per-item trace inspection
    and never edited a file; the agent must know the ceiling it is under."""
    return (
        "\n\n# Session budget\n"
        f"You have at most {limits.max_turns} turns and "
        f"{limits.max_tool_calls} tool calls; every tool call costs one "
        "turn. Budget them: spend only the first few turns inspecting "
        "evaluation results (prefer batch/digest tool actions over reading "
        "items one by one), then use the remaining turns to read and EDIT "
        "the workspace. Always reserve enough turns to complete your edits "
        "and the final TITLE/SUMMARY response."
    )


class AgentSessionProposer(Proposer):
    """Run an agent session until its workspace passes final preflight."""

    def __init__(
        self,
        *,
        backend: AgentBackend | NativeToolAgentBackend ,
        preflight: ProposalPreflight,
        limits: AgentSessionLimits,
        event_sink_factory: EventSinkFactory,
        max_repair_rounds: int = 3,
        work_root: Path | None = None,
        clock: Callable[[], float] = monotonic,
        proposal_id_factory: Callable[[], str] = _new_proposal_id,
    ):
        if not isinstance(backend, AgentBackend):
            raise TypeError("backend must implement AgentBackend")
        if not isinstance(preflight, ProposalPreflight):
            raise TypeError("preflight must be ProposalPreflight")
        if not isinstance(limits, AgentSessionLimits):
            raise TypeError("limits must be AgentSessionLimits")
        if not isinstance(event_sink_factory, EventSinkFactory):
            raise TypeError(
                "event_sink_factory must implement EventSinkFactory"
            )
        if (
            isinstance(max_repair_rounds, bool)
            or not isinstance(max_repair_rounds, int)
            or max_repair_rounds < 0
        ):
            raise ValueError(
                "max_repair_rounds must be a nonnegative integer"
            )
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(proposal_id_factory):
            raise TypeError("proposal_id_factory must be callable")

        self.backend = backend
        self.preflight = preflight
        self.limits = limits
        self.event_sink_factory = event_sink_factory
        self.max_repair_rounds = max_repair_rounds
        self.work_root = (
            None if work_root is None else Path(work_root).resolve()
        )
        self.clock = clock
        self.proposal_id_factory = proposal_id_factory

    def propose(
        self,
        operator: str,
        parent: Candidate,
        system: str,
        user: str,
    ) -> ProposeResult | None:
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator must be non-empty")
        if not isinstance(system, str) or not system.strip():
            raise ValueError("system prompt must be non-empty")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("user prompt must be non-empty")

        system = system + _budget_note(self.limits)
        usage = _ProposalUsage()
        started_at = self.clock()
        deadline = started_at + self.limits.timeout_s

        try:
            proposal_id = self.proposal_id_factory()
            resources = _ProposalResources(
                proposal_id=proposal_id,
                parent=parent,
                operator=operator,
                backend=self.backend,
                event_sink_factory=self.event_sink_factory,
                work_root=self.work_root,
            )
        except Exception:
            logger.exception("failed to initialize agent proposal resources")
            return self._failure(
                usage,
                reason="proposal-id-error",
                trace_path=None,
            )

        try:
            with resources as opened:
                outcome = self._capture_run(
                    resources=opened,
                    usage=usage,
                    deadline=deadline,
                    started_at=started_at,
                    operator=operator,
                    parent=parent,
                    system=system,
                    user=user,
                )
                outcome = self._settle_resources(
                    resources=opened,
                    usage=usage,
                    outcome=outcome,
                )
                return self._finalize_transcript(
                    resources=opened,
                    usage=usage,
                    outcome=outcome,
                    parent=parent,
                    operator=operator,
                    started_at=started_at,
                )
        except _ProposalResourceError as exc:
            logger.warning("agent proposal resource failure: %s", exc)
            return self._failure(
                usage,
                reason=exc.code,
                trace_path=resources.trace_path,
            )
        except _ProposalLoopError as exc:
            logger.warning("agent proposal loop failure: %s", exc)
            return self._failure(
                usage,
                reason=exc.code,
                trace_path=resources.trace_path,
            )
        except Exception:
            logger.exception("unexpected AgentSessionProposer failure")
            return self._failure(
                usage,
                reason="proposer-error",
                trace_path=resources.trace_path,
            )

    @staticmethod
    def _settle_resources(
        *,
        resources: _ProposalResources,
        usage: _ProposalUsage,
        outcome: ProposeResult,
    ) -> ProposeResult:
        """Surface release/flush failures before writing final summary."""

        for action in (resources.release_session, resources.flush_sink):
            try:
                action()
            except _ProposalResourceError as exc:
                logger.warning("agent proposal resource failure: %s", exc)
                outcome = AgentSessionProposer._failure(
                    usage,
                    reason=exc.code,
                    trace_path=resources.trace_path,
                )
        return outcome

    def _capture_run(
        self,
        *,
        resources: _ProposalResources,
        usage: _ProposalUsage,
        deadline: float,
        started_at: float,
        operator: str,
        parent: Candidate,
        system: str,
        user: str,
    ) -> ProposeResult:
        try:
            return self._run_loop(
                resources=resources,
                usage=usage,
                deadline=deadline,
                started_at=started_at,
                operator=operator,
                parent=parent,
                system=system,
                user=user,
            )
        except _ProposalResourceError as exc:
            logger.warning("agent proposal resource failure: %s", exc)
            return self._failure(
                usage,
                reason=exc.code,
                trace_path=resources.trace_path,
            )
        except _ProposalLoopError as exc:
            logger.warning("agent proposal loop failure: %s", exc)
            return self._failure(
                usage,
                reason=exc.code,
                trace_path=resources.trace_path,
            )
        except Exception:
            logger.exception("unexpected AgentSessionProposer failure")
            return self._failure(
                usage,
                reason="proposer-error",
                trace_path=resources.trace_path,
            )

    def _run_loop(
        self,
        *,
        resources: _ProposalResources,
        usage: _ProposalUsage,
        deadline: float,
        started_at: float,
        operator: str,
        parent: Candidate,
        system: str,
        user: str,
    ) -> ProposeResult:
        if resources.workdir is None or resources.sink is None:
            raise _ProposalLoopError(
                "resource-error",
                "proposal resources were not opened",
            )

        feedback: tuple[PreflightIssue, ...] = ()

        for repair_round in range(self.max_repair_rounds + 1):
            remaining_timeout_s = deadline - self.clock()
            run_limits = usage.remaining_limits(
                self.limits,
                remaining_timeout_s=remaining_timeout_s,
            )

            request = AgentSessionRequest(
                system=system,
                user=user,
                parent=parent,
                operator=operator,
                workdir=resources.workdir,
                limits=run_limits,
                preflight=self.preflight,
                event_sink=resources.sink,
                session_id=resources.session_id,
                feedback=feedback,
            )

            try:
                result = self.backend.run(request)
            except Exception as exc:
                raise _ProposalLoopError(
                    "backend-error",
                    f"{type(exc).__name__}: {exc}",
                ) from exc

            usage.absorb(result, run_limits)
            resources.bind_session(result.session_id)

            if result.termination is AgentTermination.BACKEND_ERROR:
                return self._failure(
                    usage,
                    reason="backend-error",
                    trace_path=resources.trace_path,
                )

            try:
                checked = self.preflight.check(
                    PreflightContext(
                        parent=parent,
                        operator=operator,
                        workdir=resources.workdir,
                    )
                )
            except Exception as exc:
                raise _ProposalLoopError(
                    "preflight-error",
                    f"{type(exc).__name__}: {exc}",
                ) from exc

            resources.record_preflight(
                PreflightTraceRecord(
                    round_index=repair_round,
                    session_id=resources.session_id,
                    report=checked.report,
                )
            )

            if checked.ok:
                assert checked.child_workspace is not None
                return self._success(
                    usage=usage,
                    resources=resources,
                    result=result,
                    child_workspace=checked.child_workspace,
                    operator=operator,
                    started_at=started_at,
                )

            report = checked.report
            if not report.repairable:
                return self._failure(
                    usage,
                    reason="nonrepairable-preflight",
                    trace_path=resources.trace_path,
                )

            if result.termination is not AgentTermination.COMPLETED:
                return self._failure(
                    usage,
                    reason=result.termination.value,
                    trace_path=resources.trace_path,
                )

            if repair_round >= self.max_repair_rounds:
                return self._failure(
                    usage,
                    reason="repair-limit",
                    trace_path=resources.trace_path,
                )

            if resources.session_id is None:
                return self._failure(
                    usage,
                    reason="backend-session-error",
                    trace_path=resources.trace_path,
                )

            feedback = report.issues

        # The bounded for-loop should always return from one of the branches.
        raise _ProposalLoopError(
            "proposer-error",
            "repair loop ended without a result",
        )

    def _finalize_transcript(
        self,
        *,
        resources: _ProposalResources,
        usage: _ProposalUsage,
        outcome: ProposeResult,
        parent: Candidate,
        operator: str,
        started_at: float,
    ) -> ProposeResult:
        final_patch: str | None = None

        if outcome.proposal is not None:
            child_workspace = outcome.proposal.workspace
            assert child_workspace is not None
            try:
                final_patch = render_workspace_patch(
                    parent.workspace,
                    child_workspace,
                )
            except Exception as exc:
                logger.warning("failed to render final proposal patch: %s", exc)
                outcome = self._failure(
                    usage,
                    reason="final-patch-error",
                    trace_path=resources.trace_path,
                )
            else:
                if not final_patch:
                    outcome = self._failure(
                        usage,
                        reason="final-patch-error",
                        trace_path=resources.trace_path,
                    )
        elif resources.workdir is not None:
            try:
                partial_workspace = parent.workspace.capture_child(
                    resources.workdir
                )
                final_patch = render_workspace_patch(
                    parent.workspace,
                    partial_workspace,
                ) or None
            except (OSError, WorkspaceError):
                final_patch = None

        elapsed_s = max(0.0, self.clock() - started_at)
        failure_reason = outcome.failure_reason
        summary = ProposalTraceSummary(
            proposal_id=resources.proposal_id,
            parent_id=parent.id,
            operator=operator,
            success=outcome.ok,
            session_id=resources.session_id,
            attempts=usage.attempts,
            repair_rounds=max(0, usage.attempts - 1),
            turns=usage.turns,
            tool_calls=usage.tool_calls,
            cost_usd=usage.cost_usd,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            elapsed_s=elapsed_s,
            termination=(
                None
                if usage.last_termination is None
                else usage.last_termination.value
            ),
            failure_reason=failure_reason,
            model=usage.last_model,
        )
        resources.finalize(summary, final_patch)
        return outcome

    def _success(
        self,
        *,
        usage: _ProposalUsage,
        resources: _ProposalResources,
        result: AgentSessionResult,
        child_workspace: Workspace,
        operator: str,
        started_at: float,
    ) -> ProposeResult:
        title, summary = parse_change_header(result.final_message)

        metadata: dict[str, object] = {
            "proposal_id": resources.proposal_id,
            "session_id": resources.session_id,
            "attempts": usage.attempts,
            "repair_rounds": max(0, usage.attempts - 1),
            "turns": usage.turns,
            "tool_calls": usage.tool_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": usage.cost_usd,
            "termination": result.termination.value,
            "agent_elapsed_s": max(
                0.0,
                self.clock() - started_at,
            ),
            "operator": operator,
            "trace_path": resources.trace_path,
        }

        proposal = Proposal(
            code=child_workspace.main_text(),
            title=title,
            summary=summary,
            model=result.model or "",
            workspace=child_workspace,
            metadata=metadata,
        )

        return ProposeResult(
            proposal=proposal,
            llm_cost=usage.cost_usd,
            attempts=usage.attempts,
            trace_path=resources.trace_path,
        )

    @staticmethod
    def _failure(
        usage: _ProposalUsage,
        *,
        reason: str,
        trace_path: str | None,
    ) -> ProposeResult:
        return ProposeResult(
            proposal=None,
            llm_cost=usage.cost_usd,
            attempts=usage.attempts,
            failure_reason=reason,
            trace_path=trace_path,
        )
