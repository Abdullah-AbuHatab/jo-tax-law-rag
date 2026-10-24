"""OpenRouter client: request body, fallbacks, and error mapping (httpx mocked)."""

import json

import httpx
import pytest

from app.llm import (
    LLM_BAD_RESPONSE,
    LLM_CONFIG_ERROR,
    LLM_RATE_LIMITED,
    LLM_TIMEOUT,
    LLM_UNAVAILABLE,
    OPENROUTER_URL,
    LLMError,
    Message,
    OpenRouterClient,
)

MESSAGES = [Message("system", "rules"), Message("user", "سؤال")]
SECRET_UPSTREAM_TEXT = "provider xyz internal stack trace"


def make_client(handler, **kwargs):
    defaults = {
        "api_key": "test-key",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "fallback_models": ["cohere/north-mini-code:free", "google/gemma-4-31b-it:free"],
    }
    return OpenRouterClient(**{**defaults, **kwargs}, client=httpx.Client(transport=httpx.MockTransport(handler)))


def ok_payload(content='{"answer": "x", "citations": [], "refused": true}', model="google/gemma-4-31b-it:free"):
    return {"id": "gen-1", "model": model, "choices": [{"message": {"role": "assistant", "content": content}}]}


def test_request_body_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200, json=ok_payload())

    completion = make_client(handler, app_url="https://github.com/x/y", app_title="jo-tax-law-rag").complete(MESSAGES)

    assert seen["url"] == OPENROUTER_URL
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert seen["headers"]["http-referer"] == "https://github.com/x/y"
    assert seen["headers"]["x-title"] == "jo-tax-law-rag"
    body = seen["body"]
    assert body["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert body["models"] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cohere/north-mini-code:free",
        "google/gemma-4-31b-it:free",
    ]
    assert body["messages"] == [{"role": "system", "content": "rules"}, {"role": "user", "content": "سؤال"}]
    assert body["temperature"] == 0
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert seen["timeout"]["read"] == 20.0
    # the model that actually answered (a fallback here) is reported
    assert completion.model == "google/gemma-4-31b-it:free"


def test_fallback_list_has_no_duplicates_and_reasoning_can_be_disabled():
    client = make_client(
        lambda r: httpx.Response(200),
        fallback_models=["a", "", "nvidia/nemotron-3-ultra-550b-a55b:free", "b"],
        reasoning_effort="",
    )
    body = client.request_body(MESSAGES)
    assert body["models"] == ["nvidia/nemotron-3-ultra-550b-a55b:free", "a", "b"]
    assert "reasoning" not in body


@pytest.mark.parametrize(
    ("status", "code", "http_status"),
    [
        (400, LLM_BAD_RESPONSE, 502),
        (401, LLM_CONFIG_ERROR, 503),
        (402, LLM_CONFIG_ERROR, 503),
        (403, LLM_CONFIG_ERROR, 503),
        (408, LLM_TIMEOUT, 504),
        (429, LLM_RATE_LIMITED, 503),
        (500, LLM_UNAVAILABLE, 503),
        (502, LLM_UNAVAILABLE, 503),
        (503, LLM_UNAVAILABLE, 503),
    ],
)
def test_http_errors_map_to_stable_codes_without_upstream_text(status, code, http_status):
    def handler(request):
        return httpx.Response(status, json={"error": {"code": status, "message": SECRET_UPSTREAM_TEXT}})

    with pytest.raises(LLMError) as info:
        make_client(handler).complete(MESSAGES)
    assert info.value.code == code
    assert info.value.http_status == http_status
    assert SECRET_UPSTREAM_TEXT not in str(info.value)


def test_timeout_maps_to_llm_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as info:
        make_client(handler).complete(MESSAGES)
    assert info.value.code == LLM_TIMEOUT and info.value.http_status == 504


def test_network_error_maps_to_unavailable():
    def handler(request):
        raise httpx.ConnectError("dns failure", request=request)

    with pytest.raises(LLMError) as info:
        make_client(handler).complete(MESSAGES)
    assert info.value.code == LLM_UNAVAILABLE


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"code": 502, "message": SECRET_UPSTREAM_TEXT}},  # error inside a 200
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"finish_reason": "error", "message": {"content": "partial"}}]},
        ["not", "an", "object"],
    ],
)
def test_unusable_200_responses_are_errors(payload):
    with pytest.raises(LLMError) as info:
        make_client(lambda r: httpx.Response(200, json=payload)).complete(MESSAGES)
    assert info.value.code in (LLM_BAD_RESPONSE, LLM_UNAVAILABLE)
    assert SECRET_UPSTREAM_TEXT not in str(info.value)


def test_non_json_200_is_bad_response():
    with pytest.raises(LLMError) as info:
        make_client(lambda r: httpx.Response(200, text="<html>gateway</html>")).complete(MESSAGES)
    assert info.value.code == LLM_BAD_RESPONSE


def test_missing_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(api_key="", model="m")
