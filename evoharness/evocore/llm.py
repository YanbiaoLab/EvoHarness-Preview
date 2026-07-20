# Portions derived from SakanaAI/ShinkaEvolve (Apache-2.0)
# Upstream: shinka/llm/llm.py, shinka/llm/client.py (retry with backoff,
#           per-query cost accounting)
# Upstream revision: 7939f6b44046a2b92e4baa6687b52b23e6236898
# Intentional deviation: providers are unified behind litellm instead of
# per-provider client classes; a transport callable can be injected for
# tests and headless runs.
"""Unified LLM client."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.0

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LLMProtocolError(RuntimeError):
    """A provider response cannot be represented by the normalized IR."""


class LLMTransientError(RuntimeError):
    """A transport failure that may succeed when retried."""


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

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMProtocolError(
            "provider response choices must be a non-empty array"
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
            },
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=effective_timeout_s,
            ) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in {408, 429} or exc.code >= 500:
                raise LLMTransientError(str(exc)) from exc
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise LLMTransientError(str(exc)) from exc
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

        for attempt in range(MAX_RETRIES):
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
                logger.warning(
                    "LLM query failed (attempt %d): %s",
                    attempt + 1,
                    e,
                )
                if attempt == MAX_RETRIES - 1:
                    break
                backoff_s = RETRY_BACKOFF_S * (attempt + 1)
                if deadline is not None:
                    remaining_s = deadline - self.clock()
                    if remaining_s <= backoff_s:
                        raise LLMTransientError(
                            "LLM query deadline exhausted"
                        ) from last_err
                self.sleep(backoff_s)

        raise RuntimeError(
            f"LLM query failed after {MAX_RETRIES} attempts"
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
