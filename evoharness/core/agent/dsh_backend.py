"""Run one proposal session inside a dedicated deepseek-harness runtime.

The backend owns nothing but the runtime: `AgentSessionProposer` materializes
the parent workspace into `request.workdir`, reads the child back, and runs
preflight. This adapter points one dsh runtime at that directory and returns a
normalized `AgentSessionResult`.

One runtime per session is deliberate. The dsh deployment named by
`DshRuntimeSpec.config_path` IS the candidate's sandbox — its tool set, its
filesystem confinement, and the monotonic tool guard that denies capabilities
which could rewrite the search itself. Sharing a runtime across candidates
would mean sharing that boundary, and a candidate is not a trusted tenant.

Three limits have no enforcement point in dsh and are reported as unsupported
rather than silently dropped; see `UNSUPPORTED_LIMITS`.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import (
    AgentEvent,
    AgentEventKind,
    AgentSessionRequest,
    AgentSessionResult,
    AgentTermination,
)

# The complete `turn/end` reason vocabulary, mirroring dsh's merge-extensible
# `TurnEndReasonMap`. These are harness turn outcomes, NOT provider finish
# reasons — an OpenAI-shaped table of "stop"/"length"/"content_filter" matches
# nothing and sends every real session to PROTOCOL_ERROR.
#
# Any value absent here is a protocol change, not a normal outcome: the backend
# must not fold an unknown reason into COMPLETED, because the proposer reads
# `completed` to decide whether the workspace is worth grading.
_TERMINATION_BY_TURN_END_KIND: Mapping[str, AgentTermination] = {
    "completed": AgentTermination.COMPLETED,
    # Cancellation. This backend's only cancellation channel is the SDK request
    # timeout, so the cause is a deadline; a caller that gains another way to
    # cancel has to read `data.reason.reason.kind` to keep this honest.
    "aborted": AgentTermination.TIMEOUT,
    # A policy or approval decision stopped the turn — the session ended
    # without doing the work, which is the outcome REFUSAL describes even
    # though the refusal came from the harness rather than the model.
    "blocked": AgentTermination.REFUSAL,
    "error": AgentTermination.BACKEND_ERROR,
    "max-tokens": AgentTermination.OUTPUT_LIMIT,
    # A persistence backend closed a crash-orphaned turn on reload. The loop
    # never emits it live, so seeing it means the runtime died mid-turn.
    "interrupted": AgentTermination.PROTOCOL_ERROR,
}

# Limits the caller may set that this backend cannot enforce. dsh exposes no
# turn or tool-call budget through the SDK, and no spend ceiling at all. They
# are named here so the gap is visible in the trace instead of being inferred
# from a session that ran longer than it was allowed to. The proposer still
# fails closed: `_ProposalUsage.absorb` rejects a result that overran its
# budget, so an overrun costs one wasted session rather than an unbounded one.
UNSUPPORTED_LIMITS: tuple[str, ...] = (
    "max_turns",
    "max_tool_calls",
    "max_cost_usd",
)



def _content_id(part: str) -> str:
    """An argv element reduced to something location-independent.

    An element naming a readable file becomes `name@sha256:<12>`; anything
    else passes through. This is what lets two checkouts of the same runtime
    compare equal while a changed runtime entry does not — matching on the
    path instead would get both cases backwards.
    """

    try:
        candidate = Path(part)
        if not candidate.is_file():
            return part
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return part
    return f"{candidate.name}@sha256:{digest[:12]}"


class DshBackendError(RuntimeError):
    """The dsh runtime could not be launched or driven."""


def _harness_class() -> Any:
    """The SDK entry point, or a `DshBackendError` naming what to do.

    Imported through a function so the spec can try it at construction and the
    backend can use it at launch, with one message between them. Not imported
    at module scope: the SDK lives outside this package and importing it to
    define a class would make the whole agent module unimportable without it.
    """

    try:
        from deepseek_harness import DeepSeekHarness
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DshBackendError(
            "the deepseek-harness Python SDK is not importable; install it "
            "or add its src directory to PYTHONPATH "
            "(<deepseek-harness>/python/sdk/src)"
        ) from exc
    return DeepSeekHarness


@dataclass(frozen=True)
class DshRuntimeSpec:
    """Everything that decides what a candidate's runtime can do.

    This is the identity of the proposal environment, not merely its
    configuration: the cordis file names every plugin the candidate can reach,
    so a change to it is a change of experiment. `identity()` is what belongs
    in `RunSpec.proposer_backend`'s `ComponentSpec.config`.
    """

    config_path: Path
    runtime_argv: tuple[str, ...]
    provider: str = "deepseek-official"
    model: str = "deepseek-v4-flash"
    runtime_cwd: Path | None = None
    session_root: Path | None = None
    #: The run directory, handed to the runtime as `EVO_RUN_DIR` so a
    #: peer-reading tool knows where to ask. Not part of `fingerprint()` for
    #: the same reason `RunSpec` drops `output_dir`: the same experiment moved
    #: to another directory is still the same experiment.
    run_dir: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    #: The name the candidate's cordis config gives its peer-reading tool, or
    #: None when that deployment mounts none. A DECLARATION, not a probe —
    #: Python cannot see the runtime's tool table, so naming a tool the config
    #: does not mount makes the prompt lie about it. It enters `fingerprint()`
    #: so at least two deployments that differ here are not compared as one.
    peer_fetch_tool: str | None = None
    # Optional USD per million tokens for this spec's model, as
    # `(input, output)`. Left unset the backend reports tokens only: the
    # trusted spend ledger is BudgetMeter's `budget.json`, so a price here is a
    # convenience for callers that want a rough per-session figure, never the
    # number a budget decision rests on.
    price_usd_per_mtok: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, Path):
            raise TypeError("config_path must be a Path")
        if not self.config_path.is_file():
            raise ValueError(
                f"cordis config does not exist: {self.config_path}"
            )
        if not isinstance(self.runtime_argv, tuple) or not self.runtime_argv:
            raise ValueError("runtime_argv must be a non-empty tuple")
        if not all(
            isinstance(part, str) and part.strip()
            for part in self.runtime_argv
        ):
            raise ValueError("runtime_argv entries must be non-empty text")
        for name in ("provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.peer_fetch_tool is not None and (
            not isinstance(self.peer_fetch_tool, str)
            or not self.peer_fetch_tool.strip()
        ):
            raise ValueError("peer_fetch_tool must be non-empty when set")
        # Checked here rather than at the first session. Without it a missing
        # SDK is a normal terminal result for every proposal in turn, so the
        # run spends its whole circuit-breaker budget before stopping, and
        # what it reports is `proposer_dead` — a verdict about the model.
        _harness_class()
        if self.price_usd_per_mtok is not None and (
            not isinstance(self.price_usd_per_mtok, tuple)
            or len(self.price_usd_per_mtok) != 2
            or not all(
                isinstance(part, (int, float))
                and not isinstance(part, bool)
                and math.isfinite(part)
                and part >= 0
                for part in self.price_usd_per_mtok
            )
        ):
            raise ValueError(
                "price_usd_per_mtok must be two nonnegative finite numbers"
            )

    @property
    def unsupported_limits(self) -> tuple[str, ...]:
        """Caller limits this runtime cannot enforce.

        Exposed as a plain attribute, not only inside `identity()`, so the
        manifest can print it beside the limits themselves. A reader who finds
        `max_turns: 48` in one place and its unenforceability three levels
        away in another has to know to look.
        """

        return UNSUPPORTED_LIMITS

    def config_hash(self) -> str:
        """Content hash of the cordis deployment the candidate runs under."""

        digest = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        return f"sha256:{digest}"

    def identity(self) -> dict[str, Any]:
        """The frozen description that makes this environment comparable.

        A v1 approximation of the composed plugin tree: the cordis file's own
        content rather than `dsh --dump-config` output. It catches every edit
        to the deployment this backend launches, but not a change inside a
        plugin package the file names by version range.
        """

        return {
            "kind": "dsh-sdk",
            "config_path": str(self.config_path),
            "config_hash": self.config_hash(),
            "runtime_argv": list(self.runtime_argv),
            "provider": self.provider,
            "model": self.model,
            "peer_fetch_tool": self.peer_fetch_tool,
            "unsupported_limits": list(UNSUPPORTED_LIMITS),
        }

    def fingerprint(self) -> dict[str, Any]:
        """The subset of `identity()` that decides whether two runs match.

        Everything here goes into `RunSpec.proposer_backend`'s config and
        therefore into the checkpoint fingerprint, so a resume against a
        changed deployment is refused rather than quietly comparing two
        different experiments.

        Absolute paths are deliberately excluded and replaced by content: the
        same checkout at a different location is the same experiment, and
        hashing its path would refuse a legitimate resume after a move. What
        remains is what actually decides behaviour — the cordis file's
        content, the runtime entry's content, the endpoint, and the limits
        this backend cannot enforce.
        """

        return {
            "kind": "dsh-sdk",
            "config_hash": self.config_hash(),
            "runtime_argv": [_content_id(part) for part in self.runtime_argv],
            "provider": self.provider,
            "model": self.model,
            # What the candidate can reach is part of what the experiment IS,
            # and this also decides what the prompt promises it.
            "peer_fetch_tool": self.peer_fetch_tool,
            "unsupported_limits": list(UNSUPPORTED_LIMITS),
        }


@dataclass
class _LiveSession:
    """One dsh runtime bound to one proposal's workdir."""

    harness: Any
    workdir: Path
    rounds: int = 0
    sequence: int = 0


class DshAgentBackend:
    """`AgentBackend` over the deepseek-harness Python SDK."""

    def __init__(
        self,
        spec: DshRuntimeSpec,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        if not isinstance(spec, DshRuntimeSpec):
            raise TypeError("spec must be a DshRuntimeSpec")
        self.spec = spec
        self.clock = clock
        self._sessions: dict[str, _LiveSession] = {}

    # -- AgentBackend ----------------------------------------------------

    def run(self, request: AgentSessionRequest) -> AgentSessionResult:
        if not isinstance(request, AgentSessionRequest):
            raise TypeError("request must be an AgentSessionRequest")

        started = self.clock()
        session_id = request.session_id or f"proposal-{uuid.uuid4().hex}"
        try:
            live = self._session_for(session_id, request)
        except Exception as exc:
            return self._failed(
                session_id,
                started,
                f"{type(exc).__name__}: {exc}",
                request,
            )

        resumed = live.rounds > 0
        live.rounds += 1

        try:
            result = live.harness.run(
                _compose_prompt(request, resumed=resumed),
                session_id=session_id,
            )
        except Exception as exc:
            return self._failed(
                session_id,
                started,
                f"{type(exc).__name__}: {exc}",
                request,
                round_index=live.rounds - 1,
            )

        return self._normalize(
            live=live,
            session_id=session_id,
            request=request,
            events=list(getattr(result, "events", ()) or ()),
            final_message=str(getattr(result, "final_response", "") or ""),
            finish_reason=getattr(result, "finish_reason", None),
            started=started,
            resumed=resumed,
        )

    def release(self, session_id: str) -> None:
        """Reap the runtime that ran this session.

        With one runtime per session this is where the dsh process dies, so a
        caller that forgets to release leaks a process tree rather than a
        dictionary entry.
        """

        live = self._sessions.pop(session_id, None)
        if live is None:
            return
        try:
            live.harness.close()
        except Exception:
            # A runtime that already died is released; re-raising here would
            # mask the real proposal outcome behind a teardown failure.
            pass

    def close(self) -> None:
        """Release every live session; safe to call more than once."""

        for session_id in list(self._sessions):
            self.release(session_id)

    # -- internals -------------------------------------------------------

    def _session_for(
        self, session_id: str, request: AgentSessionRequest
    ) -> _LiveSession:
        live = self._sessions.get(session_id)
        if live is not None:
            if live.workdir != request.workdir:
                # The runtime's cwd is fixed when it is constructed, so a
                # moved workdir would silently edit the previous directory.
                raise DshBackendError(
                    f"session {session_id} is bound to {live.workdir}, "
                    f"cannot resume against {request.workdir}"
                )
            return live

        harness = self._build_harness(request)
        live = _LiveSession(harness=harness, workdir=request.workdir)
        self._sessions[session_id] = live
        return live

    def _build_harness(self, request: AgentSessionRequest) -> Any:
        DeepSeekHarness = _harness_class()

        env = dict(self.spec.env)
        # The candidate's persona is the caller's system prompt. The cordis
        # deployment reads DSH_SYSTEM_PROMPT, so the runtime carries it for
        # every turn including resumed ones.
        env["DSH_SYSTEM_PROMPT"] = request.system
        # What a peer-reading tool in the runtime needs to ask Python back.
        # All three come from this process rather than being discovered on the
        # other side: the run directory is not derivable from the runtime's
        # cwd (that is a temporary directory holding one candidate's files),
        # and an interpreter found on PATH is not necessarily the one this
        # harness is installed in.
        if self.spec.run_dir is not None:
            import sys

            import evoharness

            env["EVO_RUN_DIR"] = str(self.spec.run_dir)
            env["EVO_PYTHON"] = sys.executable
            env["EVO_HARNESS_ROOT"] = str(
                Path(evoharness.__file__).resolve().parent.parent
            )

        harness = DeepSeekHarness(
            provider=self.spec.provider,
            model=self.spec.model,
            cwd=str(request.workdir),
            runtime_cwd=(
                str(self.spec.runtime_cwd)
                if self.spec.runtime_cwd is not None
                else None
            ),
            session_root=(
                str(self.spec.session_root)
                if self.spec.session_root is not None
                else None
            ),
            cordis=str(self.spec.config_path),
            launch_args_override=self.spec.runtime_argv,
            env=env,
            request_timeout_seconds=request.limits.timeout_s,
        )
        harness.start()
        return harness

    def _failed(
        self,
        session_id: str,
        started: float,
        detail: str,
        request: AgentSessionRequest,
        *,
        round_index: int = 0,
    ) -> AgentSessionResult:
        """Report a launch or transport failure as a normal terminal result.

        Raising would lose the round's accounting: the proposer absorbs usage
        before it inspects the termination, so a returned BACKEND_ERROR keeps
        a failed session's spend on the books.
        """

        elapsed = max(0.0, self.clock() - started)
        event = AgentEvent(
            session_id=session_id,
            round_index=round_index,
            sequence=0,
            kind=AgentEventKind.TERMINATION,
            turn=0,
            elapsed_s=elapsed,
            content=detail,
            data={"termination": AgentTermination.BACKEND_ERROR.value},
        )
        _emit(request.event_sink, event)
        return AgentSessionResult(
            termination=AgentTermination.BACKEND_ERROR,
            session_id=session_id,
            final_message=detail,
            model=self.spec.model,
            elapsed_s=elapsed,
            events=(event,),
        )

    def _normalize(
        self,
        *,
        live: _LiveSession,
        session_id: str,
        request: AgentSessionRequest,
        events: Sequence[Mapping[str, Any]],
        final_message: str,
        finish_reason: Any,
        started: float,
        resumed: bool,
    ) -> AgentSessionResult:
        round_index = live.rounds - 1
        elapsed = max(0.0, self.clock() - started)

        translated: list[AgentEvent] = []

        def add(
            kind: AgentEventKind,
            *,
            turn: int = 0,
            call_id: str | None = None,
            tool_name: str | None = None,
            content: str = "",
            data: Mapping[str, Any] | None = None,
        ) -> None:
            event = AgentEvent(
                session_id=session_id,
                round_index=round_index,
                sequence=live.sequence,
                kind=kind,
                turn=turn,
                elapsed_s=max(0.0, self.clock() - started),
                call_id=call_id,
                tool_name=tool_name,
                content=content,
                data=dict(data or {}),
            )
            live.sequence += 1
            translated.append(event)
            _emit(request.event_sink, event)

        add(
            AgentEventKind.SESSION_RESUME
            if resumed
            else AgentEventKind.SESSION_START,
            data={
                "workdir": str(request.workdir),
                "operator": request.operator,
                "identity": self.spec.identity(),
                "unsupported_limits": list(UNSUPPORTED_LIMITS),
                # What the candidate was actually told, recorded the way the
                # in-process runtime records it. Without it the cold trace
                # answers what the candidate DID and not what it was asked,
                # and every audit of the prompt reads as "found nothing"
                # rather than "could not look". A prompt that named a tool
                # this deployment does not mount was invisible for exactly
                # that reason. Once per session, not per turn.
                "system": request.system,
            },
        )

        prompt_tokens = 0
        completion_tokens = 0
        model_responses = 0
        tool_calls = 0
        dropped: dict[str, int] = {}

        for raw in events:
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("type", ""))
            data = raw.get("data")
            data = data if isinstance(data, Mapping) else {}
            turn = _none_int(data.get("turn"))

            if kind == "assistant/message":
                model_responses += 1
                usage = data.get("usage")
                if isinstance(usage, Mapping):
                    prompt_tokens += _none_int(
                        usage.get("inputTokens", usage.get("promptTokens"))
                    )
                    completion_tokens += _none_int(
                        usage.get("outputTokens", usage.get("completionTokens"))
                    )
                add(
                    AgentEventKind.MODEL_RESPONSE,
                    turn=turn,
                    content=_message_text(data.get("message")),
                    data={"step": _none_int(data.get("step"))},
                )
            elif kind == "tool/call":
                tool_calls += 1
                add(
                    AgentEventKind.TOOL_CALL,
                    turn=turn,
                    call_id=_optional_text(data.get("callId")),
                    tool_name=_optional_text(data.get("name")),
                    content=str(data.get("arguments", "")),
                )
            elif kind == "tool/result":
                message = data.get("message")
                source = (
                    message.get("source")
                    if isinstance(message, Mapping)
                    else None
                )
                add(
                    AgentEventKind.TOOL_RESULT,
                    turn=turn,
                    call_id=_optional_text(
                        source.get("callId")
                        if isinstance(source, Mapping)
                        else None
                    ),
                    content=_message_text(message),
                )
            elif kind.startswith("compaction/"):
                add(AgentEventKind.CONTEXT_COMPACT, turn=turn)
            else:
                # Unmapped session events are counted rather than translated:
                # forcing them into a neighbouring kind would put invented
                # facts into the durable trace. The tally rides on the
                # termination event so a lossy translation stays visible.
                dropped[kind] = dropped.get(kind, 0) + 1

        termination = _termination_for(finish_reason)
        price = self.spec.price_usd_per_mtok
        cost_usd = (
            0.0
            if price is None
            else (prompt_tokens * price[0] + completion_tokens * price[1])
            / 1_000_000
        )

        add(
            AgentEventKind.TERMINATION,
            data={
                "termination": termination.value,
                "finish_reason": finish_reason,
                "untranslated_event_types": dict(sorted(dropped.items())),
                # Zero cost with nonzero tokens means unpriced, not free. The
                # distinction has to be readable from the trace so nobody reads
                # an absent price as a cheap session.
                "cost_priced": price is not None,
            },
        )

        return AgentSessionResult(
            termination=termination,
            session_id=session_id,
            final_message=final_message,
            model=self.spec.model,
            cost_usd=cost_usd,
            # A dsh step is one model round trip, which is what the caller's
            # turn budget counts. Reporting the honest number lets the
            # proposer fail closed on an overrun; under-reporting to slip past
            # `_ProposalUsage.absorb` would be exactly the silent-ignore this
            # backend refuses to do for limits it cannot enforce.
            turns=model_responses,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_s=elapsed,
            events=tuple(translated),
        )


def _compose_prompt(request: AgentSessionRequest, *, resumed: bool) -> str:
    """Build the turn's user message.

    The system prompt travels through the deployment persona, so a resumed
    round carries only the new instruction and its repair feedback — repeating
    the original task would read as a second, competing request.
    """

    parts: list[str] = []
    if not resumed:
        parts.append(request.user)
    if request.feedback:
        from .feedback import render_preflight_feedback

        parts.append(render_preflight_feedback(request.feedback))
    elif resumed:
        parts.append(request.user)
    return "\n\n".join(part for part in parts if part.strip())


def _termination_for(finish_reason: Any) -> AgentTermination:
    if finish_reason is None:
        # The SDK returns None when the run interval carried no `turn/end`
        # event at all. That is not a completed turn — it is a turn whose
        # ending was never recorded, and treating it as success would grade a
        # workspace the candidate may still have been editing.
        return AgentTermination.PROTOCOL_ERROR
    mapped = _TERMINATION_BY_TURN_END_KIND.get(str(finish_reason))
    if mapped is not None:
        return mapped
    # An unrecognized kind is a protocol change — dsh's reason map is
    # merge-extensible, so a plugin can add one. Calling it COMPLETED would
    # hand the proposer a workspace that may never have been finished.
    return AgentTermination.PROTOCOL_ERROR


def _emit(sink: Any, event: AgentEvent) -> None:
    if sink is None:
        return
    sink.emit(event)


def _message_text(message: Any) -> str:
    """Flatten a session message's text blocks, ignoring non-text content."""

    if isinstance(message, str):
        return message
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return ""
    chunks: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            chunks.append(block["text"])
            continue
        inner = block.get("content")
        if isinstance(inner, Sequence) and not isinstance(inner, (str, bytes)):
            for nested in inner:
                if (
                    isinstance(nested, Mapping)
                    and nested.get("type") == "text"
                    and isinstance(nested.get("text"), str)
                ):
                    chunks.append(nested["text"])
    return "".join(chunks)


def _none_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value
