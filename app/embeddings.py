"""Embedding providers: Gemini over REST, and a deterministic fake for tests.

Gemini REST contract used here (checked 2026-10-24):
  * API reference: https://ai.google.dev/api/embeddings
  * Guide with REST examples: https://ai.google.dev/gemini-api/docs/embeddings

  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents
  header  x-goog-api-key: <key>
  body    {"requests": [{"model": "models/{model}",
                         "content": {"parts": [{"text": "..."}]},
                         "taskType": "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY",
                         "outputDimensionality": 768}, ...]}
  reply   {"embeddings": [{"values": [...]}, ...]}   (same order as requests)

The guide's REST examples put ``taskType`` / ``outputDimensionality`` at the
top level of each request; the API reference now marks those top-level
fields deprecated in favour of an ``embedContentConfig`` object. Both are
accepted today. If Google removes the top-level form, only
``GeminiEmbedder._request_item`` has to change.

The guide also says gemini-embedding-001 vectors are only normalized at
3072 dimensions, so smaller vectors are L2-normalized here.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import httpx

from app.arabic import tokenize

logger = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(Exception):
    """Embedding failed. The message is safe to log; it never contains the API key."""


class Embedder(Protocol):
    model: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        batch_size: int = 50,
        timeout_seconds: float = 15.0,
        max_attempts: int = 4,
        base_backoff_seconds: float = 1.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        query_max_attempts: int = 2,
        query_max_backoff_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self._api_key = api_key
        self._batch_size = batch_size
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff_seconds
        self._client = client or httpx.Client()
        self._sleep = sleep
        self._jitter = jitter
        self._query_max_attempts = query_max_attempts
        self._query_max_backoff = query_max_backoff_seconds
        self._clock = clock

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Ingest path: up to ``max_attempts`` per batch, ``timeout_seconds`` per attempt."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Request path (one /ask): at most ``query_max_attempts`` (2) calls, short
        backoff, and ``timeout_seconds`` for all attempts together, not per attempt."""
        deadline = self._clock() + self._timeout
        return self._embed_batch(
            [text],
            "RETRIEVAL_QUERY",
            max_attempts=self._query_max_attempts,
            max_backoff=self._query_max_backoff,
            deadline=deadline,
        )[0]

    def _request_item(self, text: str, task_type: TaskType) -> dict[str, object]:
        return {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dimension,
        }

    def _embed_batch(
        self,
        texts: Sequence[str],
        task_type: TaskType,
        max_attempts: int | None = None,
        max_backoff: float = 30.0,
        deadline: float | None = None,
    ) -> list[list[float]]:
        url = f"{GEMINI_BASE_URL}/models/{self.model}:batchEmbedContents"
        body = {"requests": [self._request_item(t, task_type) for t in texts]}
        response = self._post_with_retry(url, body, max_attempts or self._max_attempts, max_backoff, deadline)
        try:
            items = response.json()["embeddings"]
            vectors = [[float(v) for v in item["values"]] for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Gemini returned an unexpected response body") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Gemini returned {len(vectors)} vectors for {len(texts)} texts")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(f"Gemini returned a {len(vector)}-d vector, expected {self.dimension}")
        return [l2_normalize(v) for v in vectors]

    def _post_with_retry(
        self, url: str, body: dict[str, object], max_attempts: int, max_backoff: float, deadline: float | None
    ) -> httpx.Response:
        if not self._api_key:
            # Checked per call, not at start-up, so the API still starts (and
            # /health, /laws work) when the key is missing; /ask answers 503.
            raise EmbeddingError("GEMINI_API_KEY is not set")
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        for attempt in range(1, max_attempts + 1):
            timeout = self._timeout
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise EmbeddingError("Gemini query deadline exceeded")
                timeout = min(timeout, remaining)
            try:
                response = self._client.post(url, json=body, headers=headers, timeout=timeout)
            except httpx.TimeoutException as exc:
                # A timeout may still be a transient failure, but we only retry on
                # explicit 429/5xx answers: the policy stays simple to reason about.
                raise EmbeddingError("Gemini request timed out") from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Gemini request failed: {type(exc).__name__}") from exc
            if response.status_code == 200:
                return response
            retryable = response.status_code in RETRYABLE_STATUS
            logger.warning(
                "gemini embedding call failed",
                extra={"status": response.status_code, "attempt": attempt, "retryable": retryable},
            )
            if not retryable or attempt == max_attempts:
                raise EmbeddingError(f"Gemini returned HTTP {response.status_code}")
            pause = self._backoff_seconds(attempt, response, max_backoff)
            if deadline is not None and self._clock() + pause >= deadline:
                raise EmbeddingError(f"Gemini returned HTTP {response.status_code}")  # no time left to retry
            self._sleep(pause)
        raise EmbeddingError("unreachable")  # pragma: no cover

    def _backoff_seconds(self, attempt: int, response: httpx.Response, cap: float = 30.0) -> float:
        """Exponential backoff with full jitter; honour Retry-After when given; never above ``cap``."""
        retry_after = response.headers.get("retry-after", "")
        if retry_after.isdigit():
            return min(float(retry_after), cap)
        return min(self._base_backoff * (2 ** (attempt - 1)) * (0.5 + self._jitter() / 2), cap)


class FakeEmbedder:
    """Deterministic, offline embedder for tests and the ``--fake`` eval run.

    Feature hashing of normalized Arabic tokens (and token bigrams) into a
    fixed-size signed vector. Texts that share words get similar vectors, so
    retrieval behaves sensibly, but this is not a semantic model.
    """

    def __init__(self, dimension: int = 768, model: str = "fake-hashing-v1") -> None:
        self.model = model
        self.dimension = dimension
        self.calls: list[int] = []  # batch sizes, so tests can count re-embedding

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
