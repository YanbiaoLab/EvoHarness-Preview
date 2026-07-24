"""OpenAI-compatible transport for live IMO experiment runs."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

from evoharness.evocore.llm import (
    LLMMessage,
    LLMProtocolError,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinition,
    LLMTransientError,
    _openai_chat_request,
    _parse_openai_chat_response,
)


_LEADING_THINK_RE = re.compile(
    r"\A\s*<think>.*?</think>\s*",
    re.DOTALL | re.IGNORECASE,
)


def _without_inline_reasoning(response: LLMResponse) -> LLMResponse:
    """Keep provider reasoning out of task-visible model output."""

    text = response.text
    if not text.lstrip().lower().startswith("<think>"):
        return response
    match = _LEADING_THINK_RE.match(text)
    if match is None:
        raise LLMProtocolError("provider returned an unterminated <think> block")
    final_text = text[match.end():].strip()
    if not final_text and not response.tool_calls:
        raise LLMProtocolError(
            "provider returned reasoning without a final answer"
        )
    return replace(response, text=final_text)


@dataclass(frozen=True)
class SpecOpenAITransport:
    api_base: str
    api_key: str
    enable_thinking: bool
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    default_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not self.api_base.strip() or not self.api_key.strip():
            raise ValueError("api_base and api_key must be non-empty")
        if (
            not math.isfinite(self.default_timeout_s)
            or self.default_timeout_s <= 0
        ):
            raise ValueError("default_timeout_s must be positive and finite")
        if (self.input_cost_per_million is None) != (
            self.output_cost_per_million is None
        ):
            raise ValueError("input and output prices must be set together")

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
        payload = _openai_chat_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        if model.startswith("openai/"):
            payload["model"] = model.removeprefix("openai/")
        if "minimax" in str(payload["model"]).lower():
            # DashScope's direct MiniMax models ignore enable_thinking.
            payload["thinking"] = {
                "type": "adaptive" if self.enable_thinking else "disabled"
            }
        else:
            payload["enable_thinking"] = self.enable_thinking
        request = urllib.request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.default_timeout_s if timeout_s is None else timeout_s,
            ) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:  # pragma: no cover - body may be unavailable
                pass
            message = f"HTTP Error {exc.code}: {exc.reason}"
            if detail.strip():
                message += f" | {detail.strip()}"
            if exc.code in {408, 429} or exc.code >= 500:
                raise LLMTransientError(message) from exc
            raise LLMProtocolError(message) from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise LLMTransientError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise LLMProtocolError("provider returned invalid JSON") from exc
        parsed = _without_inline_reasoning(
            _parse_openai_chat_response(value, requested_model=model)
        )
        if self.input_cost_per_million is None:
            return parsed
        cost = (
            parsed.prompt_tokens * self.input_cost_per_million
            + parsed.completion_tokens * self.output_cost_per_million
        ) / 1_000_000
        return replace(parsed, cost=cost)
