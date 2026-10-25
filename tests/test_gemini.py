"""Gemini embedder: request shape, batching, retry policy (httpx mocked)."""

import json
import math

import httpx
import pytest

from app.embeddings import EmbeddingError, FakeEmbedder, GeminiEmbedder

DIM = 768


def vector(seed: float) -> list[float]:
    return [seed] * DIM  # not normalized on purpose: the embedder must normalize


def embeddings_response(request: httpx.Request) -> httpx.Response:
    count = len(json.loads(request.content)["requests"])
    return httpx.Response(200, json={"embeddings": [{"values": vector(i + 1)} for i in range(count)]})


def make_embedder(handler, sleeps=None, **kwargs):
    return GeminiEmbedder(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
        jitter=lambda: 0.5,
        **kwargs,
    )


def test_document_request_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return embeddings_response(request)

    vectors = make_embedder(handler).embed_documents(["نص أول", "نص ثان"])

    request = seen[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    )
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(request.url)  # the key never goes into the URL (it would end up in logs)
    body = json.loads(request.content)
    assert body == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "نص أول"}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 768,
            },
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "نص ثان"}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 768,
            },
        ]
    }
    assert request.extensions["timeout"]["read"] == 15.0
    assert len(vectors) == 2
    assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0)  # L2-normalized


def test_query_uses_retrieval_query_task_type():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return embeddings_response(request)

    make_embedder(handler).embed_query("سؤال")
    assert seen[0]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_documents_are_sent_in_batches():
    sizes = []

    def handler(request):
        sizes.append(len(json.loads(request.content)["requests"]))
        return embeddings_response(request)

    vectors = make_embedder(handler, batch_size=50).embed_documents([f"t{i}" for i in range(120)])
    assert sizes == [50, 50, 20]
    assert len(vectors) == 120


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_on_429_and_5xx_with_backoff(status):
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(status, json={"error": {"message": "busy"}})
        return embeddings_response(request)

    # the ingest path (documents) keeps the full retry policy; the query path is capped (below)
    make_embedder(handler, sleeps=sleeps, base_backoff_seconds=1.0).embed_documents(["نص"])
    assert len(calls) == 3
    # exponential: 1s, 2s, times the jitter factor (0.5 + 0.5/2 = 0.75)
    assert sleeps == [0.75, 1.5]


def test_retry_after_header_is_honoured_and_capped():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        if len(calls) == 2:
            return httpx.Response(429, headers={"Retry-After": "999"})
        return embeddings_response(request)

    make_embedder(handler, sleeps=sleeps).embed_documents(["نص"])
    assert sleeps == [7.0, 30.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_no_retry_on_client_errors(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json={"error": {"message": "bad"}})

    with pytest.raises(EmbeddingError, match=f"HTTP {status}"):
        make_embedder(handler).embed_query("سؤال")
    assert len(calls) == 1


def test_gives_up_after_max_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(EmbeddingError, match="HTTP 503"):
        make_embedder(handler, max_attempts=4).embed_documents(["نص"])
    assert len(calls) == 4


def test_timeout_is_an_embedding_error_without_retry():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(EmbeddingError, match="timed out"):
        make_embedder(handler).embed_query("سؤال")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"embeddings": [{"values": [0.1] * 10}]},  # wrong dimension
        {"embeddings": []},  # wrong count
        {"unexpected": True},
    ],
)
def test_malformed_responses_are_errors(payload):
    with pytest.raises(EmbeddingError):
        make_embedder(lambda r: httpx.Response(200, json=payload)).embed_query("سؤال")


def test_missing_key_fails_on_use_not_on_start():
    embedder = GeminiEmbedder(api_key="", client=httpx.Client(transport=httpx.MockTransport(embeddings_response)))
    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
        embedder.embed_query("سؤال")


def test_fake_embedder_is_deterministic_and_normalized():
    fake = FakeEmbedder()
    a = fake.embed_query("نسبة الضريبة العامة")
    assert a == FakeEmbedder().embed_query("نسبة الضريبة العامة")
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0)
    assert len(a) == DIM


# ---- round 2: the query path (one /ask) is cheap and bounded ---------------------------


def test_query_embedding_makes_at_most_two_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    embedder = make_embedder(handler, max_attempts=4)
    with pytest.raises(EmbeddingError, match="HTTP 503"):
        embedder.embed_query("سؤال")
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["نص"])  # ingest keeps 4
    assert len(calls) == 4


def test_query_retry_after_is_capped_to_two_seconds():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "999"})
        return embeddings_response(request)

    make_embedder(handler, sleeps=sleeps).embed_query("سؤال")
    assert sleeps == [2.0]


def test_query_embedding_has_a_total_deadline():
    now = [0.0]
    seen_timeouts, calls = [], []

    def handler(request):
        calls.append(1)
        seen_timeouts.append(request.extensions["timeout"]["read"])
        now[0] += 10.0  # each attempt takes 10 s
        return httpx.Response(503)

    def sleep(seconds):
        now[0] += seconds

    embedder = GeminiEmbedder(
        api_key="test-key",
        timeout_seconds=15.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
        jitter=lambda: 0.5,
        clock=lambda: now[0],
    )
    with pytest.raises(EmbeddingError):
        embedder.embed_query("سؤال")
    # attempt 1 may use the full 15 s; attempt 2 only what is left (15 - 10 - 0.75 backoff)
    assert seen_timeouts == [15.0, pytest.approx(4.25)]
    assert now[0] <= 15.0 + 10.0  # the mock ignores timeouts; a real call would stop at the deadline

    now[0], calls[:] = 0.0, []

    def slow_handler(request):
        calls.append(1)
        now[0] += 16.0  # the first attempt alone used the whole budget
        return httpx.Response(503)

    embedder._client = httpx.Client(transport=httpx.MockTransport(slow_handler))
    with pytest.raises(EmbeddingError):
        embedder.embed_query("سؤال")
    assert len(calls) == 1  # no time left: no second attempt
