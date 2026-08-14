# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/llm/llm.py, shinka/llm/client.py (retry with backoff,
#           per-query cost accounting)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Intentional deviation: providers are unified behind litellm instead of
# per-provider client classes; a transport callable can be injected for
# tests and headless runs.
"""Unified LLM client."""

from __future__ import annotations

import http.client
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

# Three attempts assumes a healthy endpoint. Against a degraded one the budget
# has to be sized from its measured success rate, because a long agentic session
# multiplies the per-turn failure: at a 54% success rate, four attempts still fail
# 4.5% of the time, and a 50-turn session then survives only 0.955^50 ~ 10%.
# Holding a 60-turn session at 90% needs the per-turn failure below 0.17%, i.e.
# about nine attempts. Raising it is cheap when the endpoint rejects fast
# (a "busy" reply, not a timeout), so this is an env knob rather than a new default.
MAX_RETRIES = int(os.environ.get("EVOHARNESS_LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF_S = float(os.environ.get("EVOHARNESS_LLM_BACKOFF_S", "1.0"))
# Cap the linear backoff: past ~10 attempts it reaches tens of seconds, while a
# transient busy state clears in about one. Waiting longer buys nothing.
RETRY_BACKOFF_CAP_S = float(os.environ.get("EVOHARNESS_LLM_BACKOFF_CAP_S", "15.0"))

# 4xx bodies that describe a gateway's own routing state rather than anything
# wrong with the request. These clear on their own, so they belong on the
# retry path even though the status code says "your fault".
_ROUTING_FAULT_MARKS = (
    "unknown provider",
    "no healthy upstream",
    "no deployments available",
    "model not available",
)


def _is_routing_fault(code: int, detail: str) -> bool:
    """A 4xx whose body blames routing, not the request.

    Deliberately narrow: matching on status alone would swallow real client
    errors (a bad key, a malformed payload) into an endless retry.
    """
    if not 400 <= code < 500 or code in (401, 403, 429):
        return False
    low = detail.lower()
    return any(mark in low for mark in _ROUTING_FAULT_MARKS)

# A rate limit is not the same kind of failure as a dropped connection, and
# retrying it on the connection schedule does not work: three attempts at
# 1s and 2s spans four seconds, while a provider's rate-limit window is tens
# of seconds. Run genesis_e0_s1 (2026-08-04) died exactly this way — six
# concurrent runs against one endpoint, HTTP 429 on all three attempts inside
# four seconds, five consecutive proposals lost, and the loop correctly
# concluded the proposer was dead after 53 minutes of GPU work. The retry
# budget, not the endpoint, was the defect.
RATE_LIMIT_RETRIES = int(os.environ.get("EVOHARNESS_RATE_LIMIT_RETRIES", "6"))
RATE_LIMIT_BACKOFF_S = float(
    os.environ.get("EVOHARNESS_RATE_LIMIT_BACKOFF_S", "10.0")
)
RATE_LIMIT_BACKOFF_CAP_S = float(
    os.environ.get("EVOHARNESS_RATE_LIMIT_BACKOFF_CAP_S", "90.0")
)

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LLMProtocolError(RuntimeError):
    """A provider response cannot be represented by the normalized IR."""


class LLMTransientError(RuntimeError):
    """A transport failure that may succeed when retried."""


class LLMRateLimitError(LLMTransientError):
    """The endpoint asked us to slow down (HTTP 429).

    Separate from its parent because the right response is different: waiting
    tens of seconds usually works, while retrying immediately never does.
    `retry_after_s` carries the provider's own instruction when it sends one.
    """

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


def _validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError("tool name must match ^[A-Za-z0-9_-]{1,64}$")


class LLMStopReason(str, Enum):
    COMPLETED = "completed"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"


_OPENAI_STOP_REASONS = {
    "stop": LLMStopReason.COMPLETED,
    "tool_calls": LLMStopReason.TOOL_CALLS,
    "length": LLMStopReason.MAX_TOKENS,
    "content_filter": LLMStopReason.CONTENT_FILTER,
}


@dataclass(frozen=True)
class LLMToolResult:
    """One result returned for a provider-issued tool call."""

    call_id: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("tool result call_id must be non-empty")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("tool result content must be non-empty")


@dataclass(frozen=True)
class LLMToolDefinition:
    """Provider-neutral JSON Schema function-tool definition."""

    name: str
    description: str
    input_schema: dict[str, object]
    strict: bool = True

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)

        if (
            not isinstance(self.description, str)
            or not self.description.strip()
        ):
            raise ValueError("tool description must be non-empty")

        if not isinstance(self.input_schema, dict):
            raise ValueError("tool input_schema must be a JSON object")

        if self.input_schema.get("type") != "object":
            raise ValueError(
                "tool input_schema must describe a JSON object"
            )

        properties = self.input_schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(
                "tool input_schema properties must be an object"
            )

        if self.strict:
            if self.input_schema.get("additionalProperties") is not False:
                raise ValueError(
                    "strict tool schemas require "
                    "additionalProperties=false"
                )

            required = self.input_schema.get("required", [])
            if (
                not isinstance(required, list)
                or not all(isinstance(name, str) for name in required)
            ):
                raise ValueError(
                    "strict tool schema required must be a string list"
                )

            if set(required) != set(properties):
                raise ValueError(
                    "strict tool schemas must require every property"
                )


@dataclass(frozen=True)
class LLMToolCall:
    """One structured tool request returned by a model provider."""

    call_id: str
    name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("tool call_id must be non-empty")

        _validate_tool_name(self.name)
        if not isinstance(self.arguments, dict):
            raise ValueError("tool arguments must be a JSON object")


@dataclass(frozen=True)
class LLMMessage:
    """One provider-neutral message in a model conversation."""

    role: str
    content: str = ""
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_results: tuple[LLMToolResult, ...] = ()

    def __post_init__(self) -> None:
        allowed_roles = {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }

        if self.role not in allowed_roles:
            raise ValueError(f"unsupported message role: {self.role!r}")

        if not isinstance(self.content, str):
            raise ValueError("message content must be a string")

        if self.tool_calls and self.role != "assistant":
            raise ValueError(
                "tool_calls are only valid on assistant messages"
            )

        if self.tool_results and self.role != "tool":
            raise ValueError(
                "tool_results are only valid on tool messages"
            )

        if self.role == "tool" and not self.tool_results:
            raise ValueError(
                "tool messages require at least one tool result"
            )

        if self.role == "tool" and self.content:
            raise ValueError(
                "tool messages carry content through tool_results"
            )

        if not self.content.strip() and not (
            self.tool_calls or self.tool_results
        ):
            raise ValueError("message cannot be empty")

        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "assistant tool call IDs must be unique per message"
            )

        result_ids = [result.call_id for result in self.tool_results]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(
                "tool result call IDs must be unique per message"
            )


class LLMToolChoiceMode(str, Enum):
    """Provider-neutral tool-selection policy."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class LLMToolChoice:
    """Controls whether and which tools the model may call."""

    mode: LLMToolChoiceMode = LLMToolChoiceMode.AUTO
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, LLMToolChoiceMode):
            raise ValueError("tool choice mode must be LLMToolChoiceMode")

        if self.mode is LLMToolChoiceMode.SPECIFIC:
            if self.name is None:
                raise ValueError(
                    "specific tool choice requires a tool name"
                )
            _validate_tool_name(self.name)
        elif self.name is not None:
            raise ValueError(
                "tool choice name is only valid in SPECIFIC mode"
            )


def _normalize_tool_request(
    tools: tuple[LLMToolDefinition, ...],
    tool_choice: LLMToolChoice | None,
    parallel_tool_calls: bool,
) -> tuple[tuple[LLMToolDefinition, ...], LLMToolChoice]:
    """Validate and normalize provider-neutral tool request controls."""

    normalized_tools = tuple(tools)

    if not all(
        isinstance(tool, LLMToolDefinition)
        for tool in normalized_tools
    ):
        raise ValueError(
            "tools must contain only LLMToolDefinition values"
        )

    names = tuple(tool.name for tool in normalized_tools)
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique per request")

    if not isinstance(parallel_tool_calls, bool):
        raise ValueError("parallel_tool_calls must be a bool")

    if tool_choice is None:
        normalized_choice = LLMToolChoice()
    elif isinstance(tool_choice, LLMToolChoice):
        normalized_choice = tool_choice
    else:
        raise ValueError("tool_choice must be LLMToolChoice or None")

    if (
        not normalized_tools
        and normalized_choice.mode
        in {
            LLMToolChoiceMode.REQUIRED,
            LLMToolChoiceMode.SPECIFIC,
        }
    ):
        raise ValueError(
            "required or specific tool choice needs at least one tool"
        )

    if (
        normalized_choice.mode is LLMToolChoiceMode.SPECIFIC
        and normalized_choice.name not in names
    ):
        raise ValueError(
            "specific tool choice must name an available tool"
        )

    return normalized_tools, normalized_choice


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: tuple[LLMToolCall, ...] = ()
    stop_reason: LLMStopReason = LLMStopReason.COMPLETED

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("response text must be a string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("response model must be non-empty")
        if not isinstance(self.stop_reason, LLMStopReason):
            raise ValueError("stop_reason must be LLMStopReason")

        accounting = (
            self.cost,
            self.prompt_tokens,
            self.completion_tokens,
        )
        if any(value < 0 for value in accounting):
            raise ValueError("LLM response accounting cannot be negative")

        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("response tool call IDs must be unique")

        if self.tool_calls and self.stop_reason is not LLMStopReason.TOOL_CALLS:
            raise ValueError(
                "responses with tool_calls must use TOOL_CALLS stop reason"
            )

        if (
            self.stop_reason is LLMStopReason.TOOL_CALLS
            and not self.tool_calls
        ):
            raise ValueError(
                "TOOL_CALLS stop reason requires at least one tool call"
            )


class LLMTransport(Protocol):
    """Provider adapter for one multi-message model query."""

    def __call__(
        self,
        *,
        messages: tuple[LLMMessage, ...],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: tuple[LLMToolDefinition, ...],
        tool_choice: LLMToolChoice,
        parallel_tool_calls: bool,
        timeout_s: float | None,
    ) -> LLMResponse:
        ...


def _openai_tool_choice(
    choice: LLMToolChoice,
) -> str | dict[str, object]:
    if choice.mode is LLMToolChoiceMode.SPECIFIC:
        return {
            "type": "function",
            "function": {"name": choice.name},
        }
    return choice.mode.value


def _openai_tools(
    tools: tuple[LLMToolDefinition, ...],
) -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": tool.strict,
            },
        }
        for tool in tools
    ]


def _openai_messages(
    messages: tuple[LLMMessage, ...],
) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []

    for message in messages:
        if message.role == "tool":
            for result in message.tool_results:
                content = result.content
                if result.is_error:
                    content = f"[tool_error]\n{content}"
                serialized.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": content,
                    }
                )
            continue

        item: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            item["content"] = message.content or None
            item["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                }
                for call in message.tool_calls
            ]
        serialized.append(item)

    return serialized


def _openai_chat_request(
    *,
    messages: tuple[LLMMessage, ...],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: tuple[LLMToolDefinition, ...],
    tool_choice: LLMToolChoice,
    parallel_tool_calls: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": _openai_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload.update(
            {
                "tools": _openai_tools(tools),
                "tool_choice": _openai_tool_choice(tool_choice),
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
    return payload


def _parse_openai_tool_calls(
    message: dict[str, object],
) -> tuple[LLMToolCall, ...]:
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise LLMProtocolError("provider tool_calls must be an array")

    calls: list[LLMToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        try:
            if not isinstance(raw_call, dict):
                raise ValueError("tool call must be an object")
            if raw_call.get("type") != "function":
                raise ValueError("unsupported tool call type")
            function = raw_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("tool call function must be an object")
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, str):
                raise ValueError("tool call arguments must be a JSON string")
            arguments = json.loads(raw_arguments)
            calls.append(
                LLMToolCall(
                    call_id=raw_call.get("id"),
                    name=function.get("name"),
                    arguments=arguments,
                )
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMProtocolError(
                f"invalid tool call at index {index}: {exc}"
            ) from exc

    return tuple(calls)


def _parse_openai_chat_response(
    response: dict[str, object],
    *,
    requested_model: str,
    cost: float = 0.0,
) -> LLMResponse:
    if not isinstance(response, dict):
        raise LLMProtocolError("provider response must be an object")

    # Some gateways return HTTP 200 with an error body ("servers are currently
    # overloaded"), so the status-code retry path never fires and the parser
    # would otherwise read a transient hiccup as a broken contract.
    error = response.get("error")
    if isinstance(error, dict):
        raise LLMTransientError(
            f"provider error ({error.get('type') or 'error'}): "
            f"{str(error.get('message'))[:200]}"
        )

    choices = response.get("choices")
    if choices is None:
        raise LLMTransientError("provider response has no choices")
    if not isinstance(choices, list):
        raise LLMProtocolError("provider response choices must be an array")
    if not choices:
        raise LLMTransientError(
            "provider returned an empty choices array (no completion)"
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMProtocolError("provider choice must be an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMProtocolError("provider response is missing message")

    raw_content = message.get("content")
    if raw_content is None:
        text = ""
    elif isinstance(raw_content, str):
        text = raw_content
    else:
        raise LLMProtocolError(
            "provider message content must be a string or null"
        )
    refusal = message.get("refusal")
    if refusal is not None and not isinstance(refusal, str):
        raise LLMProtocolError("provider refusal must be a string")
    if refusal and not text:
        text = refusal
    tool_calls = _parse_openai_tool_calls(message)
    raw_reason = choice.get("finish_reason")
    if not isinstance(raw_reason, str):
        raise LLMProtocolError("provider finish_reason must be a string")
    try:
        stop_reason = _OPENAI_STOP_REASONS[raw_reason]
    except KeyError as exc:
        raise LLMProtocolError(
            f"unsupported provider finish_reason: {raw_reason!r}"
        ) from exc
    if tool_calls:
        if raw_reason != "tool_calls":
            raise LLMProtocolError(
                "provider returned tool calls with an incompatible "
                "finish_reason"
            )
        stop_reason = LLMStopReason.TOOL_CALLS
    elif stop_reason is LLMStopReason.TOOL_CALLS:
        raise LLMProtocolError(
            "provider returned a tool-call finish_reason without tool calls"
        )
    elif refusal and stop_reason is LLMStopReason.COMPLETED:
        stop_reason = LLMStopReason.REFUSAL

    usage = response.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise LLMProtocolError("provider usage must be an object")
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    token_counts = (prompt_tokens, completion_tokens)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in token_counts
    ):
        raise LLMProtocolError(
            "provider token counts must be nonnegative integers"
        )

    provider_model = response.get("model")
    if provider_model is None:
        provider_model = requested_model
    if not isinstance(provider_model, str):
        raise LLMProtocolError("provider model must be a string")

    try:
        return LLMResponse(
            text=text,
            model=provider_model,
            cost=float(cost),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
        )
    except (TypeError, ValueError) as exc:
        raise LLMProtocolError(
            f"invalid provider response: {exc}"
        ) from exc


def _litellm_transport(
    *,
    messages: tuple[LLMMessage, ...],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: tuple[LLMToolDefinition, ...],
    tool_choice: LLMToolChoice,
    parallel_tool_calls: bool,
    timeout_s: float | None = None,
) -> LLMResponse:
    import litellm  # lazy: only needed for real runs

    payload = _openai_chat_request(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    if timeout_s is not None:
        payload["timeout"] = timeout_s
    retryable_errors = tuple(
        getattr(litellm, name)
        for name in (
            "APIConnectionError",
            "InternalServerError",
            "RateLimitError",
            "ServiceUnavailableError",
            "Timeout",
        )
        if isinstance(getattr(litellm, name, None), type)
    )
    try:
        resp = litellm.completion(**payload)
    except retryable_errors as exc:
        raise LLMTransientError(str(exc)) from exc
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception as exc:
        raise LLMProtocolError(
            "LiteLLM could not determine completion cost"
        ) from exc
    try:
        response_data = resp.model_dump()
    except AttributeError as exc:
        raise LLMProtocolError(
            "LiteLLM response must support model_dump()"
        ) from exc
    if not isinstance(response_data, dict):
        raise LLMProtocolError(
            "LiteLLM model_dump() must return an object"
        )
    return _parse_openai_chat_response(
        response_data,
        requested_model=model,
        cost=cost,
    )


# Identify honestly rather than impersonate a browser: measured against a
# Cloudflare-fronted provider, "EvoHarness/0.1" and "curl/8.7.1" both pass
# while "Python-urllib/3.13" and "python-requests/2.31.0" are blocked. The
# rule is about known-default agent strings, not about looking human.
_USER_AGENT = "EvoHarness/0.1"


@dataclass(frozen=True)
class _OpenAICompatTransport:
    """Callable OpenAI-compatible transport with a fallback timeout."""

    url: str
    api_key: str
    default_timeout_s: float

    def __call__(
        self,
        *,
        messages: tuple[LLMMessage, ...],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: tuple[LLMToolDefinition, ...],
        tool_choice: LLMToolChoice,
        parallel_tool_calls: bool,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        import urllib.error
        import urllib.request

        effective_timeout_s = (
            self.default_timeout_s if timeout_s is None else timeout_s
        )
        request_data = _openai_chat_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        payload = json.dumps(request_data).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # urllib sends "Python-urllib/3.x" and Cloudflare's default
                # bot rules 403 it — an HTML challenge page, not JSON, so it
                # surfaced as an unexplained hard failure on every single
                # proposal. Measured 2026-07-27 against a Cloudflare-fronted
                # provider: identical 40 KB request, 403 with the urllib
                # default and 200 with the string below. The size looked
                # like the cause and was not.
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=effective_timeout_s,
            ) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Providers state their own cooldown in Retry-After; obeying
                # it beats any schedule we could guess.
                retry_after = None
                try:
                    header = exc.headers.get("Retry-After")
                    retry_after = float(header) if header else None
                except (AttributeError, TypeError, ValueError):
                    retry_after = None
                raise LLMRateLimitError(str(exc), retry_after) from exc
            if exc.code == 408 or exc.code >= 500:
                raise LLMTransientError(str(exc)) from exc
            # A permanent rejection says only "HTTP Error 403: Forbidden",
            # and the interesting part — a Cloudflare challenge page, a
            # quota message, a model-name typo — is in the body nobody
            # read. Log its head; the diagnosis is usually in the first
            # line.
            try:
                detail = exc.read()[:300].decode("utf-8", "replace")
            except Exception:      # a body that cannot be read is not news
                detail = "<unreadable>"
            logger.warning(
                "LLM endpoint refused the request: HTTP %d %s",
                exc.code, detail.replace("\n", " "),
            )
            # A gateway that answers "unknown provider" for a model it served
            # ten minutes ago is reporting a ROUTING fault, and routing faults
            # clear. Measured on ETP run 10: five of nine proposals died this
            # way on one model name while the same name succeeded in between,
            # so half the run's proposal budget went to a 400 nobody retried.
            # A genuine model-name typo lands here too and now costs a bounded
            # retry sequence before failing with the same body already logged.
            if _is_routing_fault(exc.code, detail):
                raise LLMTransientError(
                    f"provider routing fault (HTTP {exc.code}): {detail}"
                ) from exc
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except http.client.HTTPException as exc:
            # A response cut off mid-body. IncompleteRead does not inherit
            # from URLError or ConnectionError, so it used to escape every
            # retry and kill the run outright: modmul_r1 died at generation 1
            # on "IncompleteRead(12000 bytes read, 49814 more expected)".
            # A truncated read is the most transient fault there is.
            raise LLMTransientError(f"truncated response: {exc}") from exc
        except json.JSONDecodeError as exc:
            # Same cause seen from the other side — enough bytes arrived to
            # return, not enough to parse.
            raise LLMTransientError(f"unparseable response body: {exc}") from exc
        return _parse_openai_chat_response(
            body,
            requested_model=model,
        )


def make_openai_compat_transport(
    api_base: str, api_key: str, timeout_s: float = 120.0
) -> LLMTransport:
    """Build transport for an OpenAI-compatible chat endpoint."""

    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be positive and finite")

    return _OpenAICompatTransport(
        url=api_base.rstrip("/") + "/chat/completions",
        api_key=api_key,
        default_timeout_s=timeout_s,
    )


class LLMClient:
    """Thin client with retries; the SearchLoop charges its budget with the
    returned cost."""

    def __init__(
        self,
        temperature: float = 0.75,
        max_tokens: int = 4096,
        transport: LLMTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.transport = transport or _litellm_transport
        self.sleep = sleep
        self.clock = clock

    def query(self, system: str, user: str, model: str) -> LLMResponse:
        """Single-shot entry point implemented through message history."""

        return self.query_messages(
            messages=(
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ),
            model=model,
        )

    def query_messages(
        self,
        messages: tuple[LLMMessage, ...],
        model: str,
        *,
        tools: tuple[LLMToolDefinition, ...] = (),
        tool_choice: LLMToolChoice | None = None,
        parallel_tool_calls: bool = True,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Query a model, sharing one deadline across all retries."""

        if not messages:
            raise ValueError("messages must contain at least one message")
        if timeout_s is not None and (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")

        normalized_tools, normalized_choice = _normalize_tool_request(
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        last_err: Exception | None = None
        deadline = (
            None if timeout_s is None else self.clock() + timeout_s
        )

        attempt = 0
        rate_limited = 0
        while True:
            remaining_timeout_s = self._remaining_timeout(
                deadline,
                last_err,
            )
            try:
                return self.transport(
                    messages=messages,
                    model=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=normalized_tools,
                    tool_choice=normalized_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    timeout_s=remaining_timeout_s,
                )
            except (LLMTransientError, ConnectionError, TimeoutError) as e:
                last_err = e
                is_rate_limit = isinstance(e, LLMRateLimitError)
                # A rate limit does not consume the ordinary retry budget: it
                # says nothing about whether the request itself is sound, only
                # that the endpoint is busy right now.
                if is_rate_limit:
                    rate_limited += 1
                    exhausted = rate_limited >= RATE_LIMIT_RETRIES
                    backoff_s = min(
                        e.retry_after_s
                        or RATE_LIMIT_BACKOFF_S * (2 ** (rate_limited - 1)),
                        RATE_LIMIT_BACKOFF_CAP_S,
                    )
                    label = f"rate limited (wait {backoff_s:.0f}s)"
                else:
                    attempt += 1
                    exhausted = attempt >= MAX_RETRIES
                    backoff_s = min(RETRY_BACKOFF_S * attempt,
                                    RETRY_BACKOFF_CAP_S)
                    label = "transient"
                logger.warning(
                    "LLM query failed, %s (attempt %d, rate-limited %d): %s",
                    label,
                    attempt,
                    rate_limited,
                    e,
                )
                if exhausted:
                    break
                if deadline is not None:
                    remaining_s = deadline - self.clock()
                    if remaining_s <= backoff_s:
                        raise LLMTransientError(
                            "LLM query deadline exhausted"
                        ) from last_err
                self.sleep(backoff_s)

        raise RuntimeError(
            f"LLM query failed after {attempt} transient and "
            f"{rate_limited} rate-limited attempts"
        ) from last_err

    def _remaining_timeout(
        self,
        deadline: float | None,
        last_err: Exception | None,
    ) -> float | None:
        if deadline is None:
            return None

        remaining_s = deadline - self.clock()
        if remaining_s <= 0:
            raise LLMTransientError(
                "LLM query deadline exhausted"
            ) from last_err
        return remaining_s
