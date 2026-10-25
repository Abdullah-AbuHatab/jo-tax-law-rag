"""Chat completion clients: OpenRouter over REST, and a scriptable fake.

OpenRouter contract used here (checked 2026-10-24):
  * https://openrouter.ai/docs/api-reference/chat-completion
  * Fallbacks: https://openrouter.ai/docs/guides/routing/model-fallbacks
    ("models": [...] in priority order; the response "model" field says
    which one answered; any error, including rate limits, triggers the next.)

Upstream error bodies are logged (status + OpenRouter error code only) and
never put into exception messages that reach API clients.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(Exception):
    """A failed LLM call with a stable, client-safe error code."""

    def __init__(self, code: str, http_status: int) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


# Stable error codes. Clients see the code and a fixed message, nothing else.
LLM_TIMEOUT = "LLM_TIMEOUT"
LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
LLM_CONFIG_ERROR = "LLM_CONFIG_ERROR"
LLM_BAD_RESPONSE = "LLM_BAD_RESPONSE"


def map_status_to_error(status: int) -> LLMError:
    if status == 429:
        return LLMError(LLM_RATE_LIMITED, 503)
    if status in (401, 402, 403):  # bad key, no credits, key not allowed
        return LLMError(LLM_CONFIG_ERROR, 503)
    if status == 408:
        return LLMError(LLM_TIMEOUT, 504)
    if status >= 500:
        return LLMError(LLM_UNAVAILABLE, 503)
    return LLMError(LLM_BAD_RESPONSE, 502)  # 400 / 404 / other: our request or their reply is wrong


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user"
    content: str


@dataclass(frozen=True)
class Completion:
    content: str
    model: str


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message], timeout_seconds: float | None = None) -> Completion:
        """``timeout_seconds`` (when given) caps this call's total wall-clock time."""
        ...


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: Sequence[str] = (),
        timeout_seconds: float = 20.0,
        reasoning_effort: str = "low",
        max_tokens: int = 1200,
        app_url: str = "",
        app_title: str = "",
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._api_key = api_key
        self._models = [model, *[m for m in fallback_models if m and m != model]]
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens
        self._app_url = app_url
        self._app_title = app_title
        self._client = client or httpx.Client()
        self._clock = clock

    def request_body(self, messages: Sequence[Message]) -> dict[str, object]:
        body: dict[str, object] = {
            # "model" is the primary; "models" is the full priority list that
            # OpenRouter walks on errors (primary first, then the fallbacks).
            "model": self._models[0],
            "models": self._models,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning_effort:
            # These free models are reasoning models; low effort keeps answers
            # inside the 20 s budget, and exclude keeps the trace out of the reply.
            body["reasoning"] = {"effort": self._reasoning_effort, "exclude": True}
        return body

    def complete(self, messages: Sequence[Message], timeout_seconds: float | None = None) -> Completion:
        """One chat completion within a total deadline.

        The deadline is ``timeout_seconds`` capped by OPENROUTER_TIMEOUT_SECONDS. httpx
        timeouts apply per phase (connect, each read), so the body is read in chunks and
        the wall clock is checked between them: a reply that trickles in slowly cannot
        hold the request (and a concurrency slot) past the deadline.
        """
        budget = self._timeout if timeout_seconds is None else min(self._timeout, timeout_seconds)
        if budget <= 0:
            raise LLMError(LLM_TIMEOUT, 504)
        deadline = self._clock() + budget
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if self._app_url:
            headers["HTTP-Referer"] = self._app_url
        if self._app_title:
            headers["X-Title"] = self._app_title
        try:
            with self._client.stream(
                "POST", OPENROUTER_URL, json=self.request_body(messages), headers=headers, timeout=budget
            ) as response:
                chunks: list[bytes] = []
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    if self._clock() > deadline:
                        raise LLMError(LLM_TIMEOUT, 504)
                status = response.status_code
        except httpx.TimeoutException as exc:
            raise LLMError(LLM_TIMEOUT, 504) from exc
        except httpx.HTTPError as exc:
            logger.warning("openrouter request failed", extra={"error_type": type(exc).__name__})
            raise LLMError(LLM_UNAVAILABLE, 503) from exc

        payload = self._json_or_none(b"".join(chunks))
        if status != 200:
            logger.warning(
                "openrouter returned an error",
                extra={"status": status, "upstream_code": self._upstream_code(payload)},
            )
            raise map_status_to_error(status)
        return self._parse_completion(payload)

    @staticmethod
    def _json_or_none(body: bytes) -> object:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _upstream_code(payload: object) -> object:
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            return payload["error"].get("code")
        return None

    def _parse_completion(self, payload: object) -> Completion:
        if not isinstance(payload, dict):
            raise LLMError(LLM_BAD_RESPONSE, 502)
        if isinstance(payload.get("error"), dict):
            # OpenRouter can report a provider failure inside a 200 response.
            code = payload["error"].get("code")
            logger.warning("openrouter error inside 200 response", extra={"upstream_code": code})
            raise map_status_to_error(code if isinstance(code, int) else 502)
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(LLM_BAD_RESPONSE, 502) from exc
        if choice.get("finish_reason") == "error" or not isinstance(content, str) or not content.strip():
            raise LLMError(LLM_BAD_RESPONSE, 502)
        return Completion(content=content, model=str(payload.get("model") or self._models[0]))


class FakeLLM:
    """Offline LLM for tests and ``--fake`` eval.

    By default it behaves like a well-behaved model: it answers with the first
    sentence of the first article in the prompt and cites that article. Tests
    pass ``responder`` to script any other reply (invalid JSON, bad citations).
    """

    def __init__(self, responder: Callable[[Sequence[Message]], str] | None = None, model: str = "fake-llm") -> None:
        self._responder = responder or self._default_reply
        self.model = model
        self.calls: list[Sequence[Message]] = []
        self.timeouts: list[float | None] = []  # the timeout_seconds each call got

    def complete(self, messages: Sequence[Message], timeout_seconds: float | None = None) -> Completion:
        self.calls.append(messages)
        self.timeouts.append(timeout_seconds)
        return Completion(content=self._responder(messages), model=self.model)

    @staticmethod
    def _default_reply(messages: Sequence[Message]) -> str:
        prompt = messages[-1].content
        match = re.search(r'<article law_id="([^"]+)" article="([^"]+)"[^>]*>\s*(.*?)</article>', prompt, re.S)
        if not match:
            return json.dumps({"answer": "", "citations": [], "refused": True}, ensure_ascii=False)
        law_id, article, text = match.groups()
        first_sentence = re.split(r"(?<=[.:؛])\s", text.strip(), maxsplit=1)[0][:300]
        reply = {
            "answer": f"وفقاً للمادة ({article}): {first_sentence}",
            "citations": [{"law_id": law_id, "article": article}],
            "refused": False,
        }
        return json.dumps(reply, ensure_ascii=False)
