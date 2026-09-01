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


MAX_RETRIES = int(os.environ.get("EVOHARNESS_LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF_S = float(os.environ.get("EVOHARNESS_LLM_BACKOFF_S", "1.0"))
# Cap the linear backoff: past ~10 attempts it reaches tens of seconds, while a
# transient busy state clears in about one. Waiting longer buys nothing.
RETRY_BACKOFF_CAP_S = float(os.environ.get("EVOHARNESS_LLM_BACKOFF_CAP_S", "15.0"))


_ROUTING_FAULT_MARKS = (
    "unknown provider",
    "no healthy upstream",
    "no deployments available",
    "model not available",
)


# Bodies that mean "the account cannot pay", across the phrasings providers use.
# Not every provider follows the OpenAI error envelope: the one used for ETP
# answers HTTP 403 with a bare {"code": "INSUFFICIENT_BALANCE", ...}.
_BILLING_MARKS = (
    "insufficient balance",
    "insufficient_balance",
    "insufficient quota",
    "insufficient_quota",
    "billing_error",
    "exceeded your current quota",
    "payment required",
)


def _is_billing_failure(code: int, detail: str) -> bool:
    if code not in (400, 402, 403, 429):
        return False
    low = detail.lower()
    return any(mark in low for mark in _BILLING_MARKS)


def _is_routing_fault(code: int, detail: str) -> bool:
    """A 4xx whose body blames routing, not the request.

    Deliberately narrow: matching on status alone would swallow real client
    errors (a bad key, a malformed payload) into an endless retry.
    """
    if not 400 <= code < 500 or code in (401, 403, 429):
        return False
    low = detail.lower()
    return any(mark in low for mark in _ROUTING_FAULT_MARKS)


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


class LLMToolCallFormatError(LLMProtocolError):
    """A tool call arrived with arguments that are not parseable JSON.

    Separated from the rest of LLMProtocolError because the recovery differs.
    Most protocol errors say the provider and this client disagree about the
    wire, which no retry fixes. This one is per-turn flakiness -- a model that
    wrote a raw newline inside a JSON string, or truncated the object -- and
    the same model reissues a well-formed call when told what was wrong.
    Measured on ETP run 14: three of six terminated sessions died here, one of
    them after 135 turns and 145 tool calls, all from a single bad call.
    """


class LLMTransientError(RuntimeError):
    """A transport failure that may succeed when retried."""


class LLMBillingError(RuntimeError):
    """The account cannot pay for the call.

    Deliberately NOT an LLMTransientError: a rate limit clears on its own, an
    empty balance does not. Retrying it burns the schedule and reports the
    wrong cause -- a run that stops on "the proposer is dead" sends someone to
    debug the proposer instead of topping up the account.
    """


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
    cached_prompt_tokens: int = 0
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
            self.cached_prompt_tokens,
        )
        if any(value < 0 for value in accounting):
            raise ValueError("LLM response accounting cannot be negative")
        if self.cached_prompt_tokens > self.prompt_tokens:
            raise ValueError(
                "cached_prompt_tokens cannot exceed prompt_tokens"
            )

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
            raise LLMToolCallFormatError(
                f"invalid tool call at index {index}: {exc}"
            ) from exc

    return tuple(calls)




def _responses_input(messages: tuple[LLMMessage, ...]) -> list[dict[str, object]]:
    """Flatten a chat-shaped history into Responses `input` items.

    Three shapes do not map one to one:

    - An assistant tool call is a sibling item (`function_call`), not a field
      on the message, so one assistant message with n calls becomes 1+n items.
    - A tool result is a `function_call_output` item keyed by `call_id`, not a
      message with role "tool".
    - `system` is rewritten to `developer`. A gateway may override the
      `instructions` field with its own prompt, which leaves `input` as the
      only way in — and inside `input`, `system` loses to `instructions` in
      the Responses instruction hierarchy while `developer` wins. Getting this
      wrong dilutes the task prompt silently: every call still returns 200.
    """
    items: list[dict[str, object]] = []
    for message in messages:
        if message.role == "tool":
            for result in message.tool_results:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": result.content,
                    }
                )
            continue
        if message.content:
            role = "developer" if message.role == "system" else message.role
            items.append({"role": role, "content": message.content})
        for call in message.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    # Arguments travel as a JSON string, not an object, both ways.
                    "arguments": json.dumps(call.arguments),
                }
            )
    return items


def _responses_tools(
    tools: tuple[LLMToolDefinition, ...],
) -> list[dict[str, object]]:
    """Responses tool schemas are flat: no chat-style `function` nesting."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": tool.strict,
        }
        for tool in tools
    ]


def _responses_tool_choice(choice: LLMToolChoice) -> str | dict[str, object]:
    """Flat shape for a named tool; auto/none/required are spelled the same
    in both protocols and are reused as-is."""
    if choice.mode is LLMToolChoiceMode.SPECIFIC:
        return {"type": "function", "name": choice.name}
    return choice.mode.value


def _responses_request(
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
        "input": _responses_input(messages),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if tools:
        payload.update(
            {
                "tools": _responses_tools(tools),
                "tool_choice": _responses_tool_choice(tool_choice),
                "parallel_tool_calls": parallel_tool_calls,
            }
        )
    return payload


def _parse_responses_response(
    response: dict[str, object],
    *,
    requested_model: str,
    cost: float = 0.0,
) -> LLMResponse:
    if not isinstance(response, dict):
        raise LLMProtocolError("provider response must be an object")

    output = response.get("output")
    if not isinstance(output, list):
        # Transient, not a protocol error: a proxy that truncates the body can
        # leave a prefix that still parses as JSON but has no `output`. Raising
        # LLMProtocolError here ends the agent session and discards its work,
        # while a genuine shape mismatch still surfaces after the retries are
        # exhausted — with the body head carried in the message.
        head = json.dumps(response, ensure_ascii=False)[:300]
        raise LLMTransientError(
            f"provider response has no output array: {head}"
        )

    chunks: list[str] = []
    calls: list[LLMToolCall] = []
    for index, item in enumerate(output):
        if not isinstance(item, dict):
            raise LLMProtocolError("provider output item must be an object")
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or ():
                if isinstance(part, dict) and part.get("type") == "output_text":
                    chunks.append(str(part.get("text") or ""))
        elif kind == "function_call":
            try:
                raw_arguments = item.get("arguments")
                if not isinstance(raw_arguments, str):
                    raise ValueError("tool call arguments must be a JSON string")
                calls.append(
                    LLMToolCall(
                        call_id=item.get("call_id", ""),
                        name=item.get("name", ""),
                        arguments=json.loads(raw_arguments),
                    )
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMToolCallFormatError(
                    f"invalid tool call at index {index}: {exc}"
                ) from exc
        # `reasoning` items are dropped: LLMMessage cannot carry them, and
        # omitting them from the echoed history is accepted by the provider.

    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}

    # Truncation is reported through status/incomplete_details rather than a
    # finish_reason; missing it would pass a half-written body off as complete.
    stop_reason = LLMStopReason.COMPLETED
    if calls:
        stop_reason = LLMStopReason.TOOL_CALLS
    if response.get("status") == "incomplete":
        incomplete = response.get("incomplete_details")
        reason = (incomplete or {}).get("reason") if isinstance(incomplete, dict) else None
        if reason == "max_output_tokens":
            stop_reason = LLMStopReason.MAX_TOKENS
        elif reason == "content_filter":
            stop_reason = LLMStopReason.CONTENT_FILTER

    return LLMResponse(
        text="".join(chunks),
        model=str(response.get("model") or requested_model),
        cost=cost,
        prompt_tokens=int(usage.get("input_tokens") or 0),
        completion_tokens=int(usage.get("output_tokens") or 0),
        cached_prompt_tokens=int(details.get("cached_tokens") or 0),
        tool_calls=tuple(calls),
        stop_reason=stop_reason,
    )


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
    # OpenAI-compatible providers report the prefix-cache hit here. Absent or
    # malformed means "no cache reporting", not an error: this number informs
    # a decision, it never gates one.
    details = usage.get("prompt_tokens_details")
    cached_prompt_tokens = 0
    if isinstance(details, dict):
        candidate = details.get("cached_tokens", 0)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            cached_prompt_tokens = max(0, min(candidate, prompt_tokens
                                              if isinstance(prompt_tokens, int)
                                              else 0))
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
            cached_prompt_tokens=cached_prompt_tokens,
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


# `urlopen(timeout=)` is a **socket** timeout: it bounds the gap between two
# reads, not the request. A server that trickles bytes resets it forever.
#
# Measured 2026-08-27 on run18: one call sat for 8,141 seconds — 2h16m, a single
# attempt, no retries — against a 400s socket timeout, and only ended when the
# agent session's own 9,000s budget expired. The session had done 16 healthy
# turns in 859s (model calls: median 12s) and lost all of it. 449k input tokens,
# zero output.
#
# So the read needs a deadline of its own. The factor is relative to the socket
# timeout rather than absolute, so the two cannot drift apart: whoever tunes the
# socket timeout to measured latency gets a proportional ceiling for free. 3x
# leaves room for a genuinely slow-but-progressing response while capping a
# trickle at minutes instead of hours.
HARD_TIMEOUT_FACTOR = float(
    os.environ.get("EVOHARNESS_LLM_HARD_TIMEOUT_FACTOR", "3.0")
)


def _read_within(resp, deadline_s: float) -> bytes:
    """Read a response body under a total wall-clock deadline.

    Chunked rather than one `resp.read()`: the whole point is to be able to
    look at the clock between reads. Raising `LLMTransientError` puts a hung
    call onto the path that already exists for a flaky provider — bounded
    retries with backoff — instead of letting it consume the session.
    """
    import time as _time

    end = _time.monotonic() + max(1.0, deadline_s)
    chunks: list[bytes] = []
    while True:
        # Checked before the read, not after: the rule is "never start a wait
        # we already know runs past the deadline".
        #
        # A body that *completed* on the read that crossed the deadline is
        # discarded too, and that is deliberate — HTTP gives no way to tell
        # "done" from "still trickling" without one more read, so the two are
        # indistinguishable from in here. A response that needed longer than
        # the ceiling is late whether or not its last chunk happened to finish
        # it.
        if _time.monotonic() >= end:
            raise LLMTransientError(
                f"response body exceeded {deadline_s:.0f}s wall clock after "
                f"{sum(map(len, chunks))} bytes; treating as a hung call"
            )
        chunk = resp.read(65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_sse_within(resp, deadline_s: float) -> dict:
    """Consume a Responses SSE stream and return the terminal snapshot.

    `response.completed` carries the whole `response` object — output items and
    usage included — so the stream does not have to be reassembled from deltas
    and the non-streaming parser can be reused unchanged.

    Streaming exists here for delivery robustness, not for incremental output:
    a proxy that cuts the body mid-flight yields a stream that simply ends
    without its terminal event, which is a retryable transport fault rather
    than an unparseable blob.
    """
    import time as _time

    end = _time.monotonic() + max(1.0, deadline_s)
    terminal: dict | None = None
    seen = 0
    for raw in resp:
        if _time.monotonic() >= end:
            raise LLMTransientError(
                f"event stream exceeded {deadline_s:.0f}s wall clock after "
                f"{seen} events; treating as a hung call"
            )
        seen += 1
        if not raw.startswith(b"data:"):
            continue  # `event:` lines and blank separators carry no payload
        payload = raw[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError:
            # One malformed frame does not condemn the stream; the terminal
            # event is what decides whether this call produced a result.
            continue
        if isinstance(event, dict) and event.get("type") == "response.completed":
            terminal = event.get("response")

    if not isinstance(terminal, dict):
        raise LLMTransientError(
            f"event stream ended after {seen} events without "
            f"response.completed"
        )
    return terminal


@dataclass(frozen=True)
class _OpenAICompatTransport:
    """Callable OpenAI-compatible transport with a fallback timeout.

    The protocol only affects `build_request` and `parse_response`. The error
    handling below (429 Retry-After, 5xx, challenge pages, truncated reads,
    billing failures, routing faults) is shared by both, so it cannot drift
    between them.
    """

    url: str
    api_key: str
    default_timeout_s: float
    build_request: Callable[..., dict] = _openai_chat_request
    parse_response: Callable[..., LLMResponse] = _parse_openai_chat_response
    stream: bool = False

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
        request_data = self.build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        if self.stream:
            request_data = dict(request_data, stream=True)
        payload = json.dumps(request_data).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **({"Accept": "text/event-stream"} if self.stream else {}),
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
                hard_deadline_s = effective_timeout_s * HARD_TIMEOUT_FACTOR
                body = (
                    _read_sse_within(resp, hard_deadline_s)
                    if self.stream
                    else json.loads(_read_within(resp, hard_deadline_s))
                )
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
            if _is_billing_failure(exc.code, detail):
                raise LLMBillingError(
                    f"account cannot pay for the call (HTTP {exc.code}): {detail}"
                ) from exc
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
        return self.parse_response(
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


def make_openai_responses_transport(
    api_base: str, api_key: str, timeout_s: float = 120.0, *,
    stream: bool = False,
) -> LLMTransport:
    """Build transport for an OpenAI **Responses** endpoint.

    A separate constructor rather than autodetection: a gateway that serves
    only one protocol rejects the other with the same 403 it uses for an
    unauthorized token, so a silent fallback would pick a protocol nobody
    chose and report nothing. The protocol is explicit configuration.
    """
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be positive and finite")

    return _OpenAICompatTransport(
        url=api_base.rstrip("/") + "/responses",
        api_key=api_key,
        default_timeout_s=timeout_s,
        build_request=_responses_request,
        parse_response=_parse_responses_response,
        stream=stream,
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
