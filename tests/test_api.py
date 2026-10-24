"""HTTP API with fake embedder, fake LLM and the in-memory store."""

import json

from app.db import DatabaseUnavailable
from app.llm import LLM_TIMEOUT, LLMError
from app.rate_limit import SlidingWindowRateLimiter, client_ip

QUESTION = "ما هي نسبة الضريبة العامة على استيراد أي سلعة أو خدمة أو توريد أي منهما؟"


def test_health(make_client):
    client, _, _ = make_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_health_db_ok_and_503_when_down(make_client):
    client, _, store = make_client()
    assert client.get("/health/db").status_code == 200
    store.ping_error = DatabaseUnavailable("down")
    response = client.get("/health/db")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
    store.ping_error = None


def test_ask_returns_answer_with_enriched_citations(make_client):
    client, llm, _ = make_client()
    response = client.post("/ask", json={"question": QUESTION, "law_id": "sales_tax_6_1994", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is False
    assert data["model"] == "fake-llm"
    assert len(data["retrieved"]) == 3
    assert all(r["law_id"] == "sales_tax_6_1994" for r in data["retrieved"])
    citation = data["citations"][0]
    assert set(citation) == {"law_id", "law_name", "article", "article_label", "snippet", "source_url", "score"}
    assert citation["law_name"].startswith("قانون الضريبة العامة على المبيعات")
    assert "ليست استشارة قانونية" in data["disclaimer"]
    assert len(llm.calls) == 1


def test_ask_refuses_below_threshold_without_llm_call(make_client):
    client, llm, _ = make_client(min_similarity=0.99)
    data = client.post("/ask", json={"question": QUESTION}).json()
    assert data["refused"] is True
    assert data["refusal_reason"] == "LOW_RETRIEVAL_SCORE"
    assert data["citations"] == []
    assert data["model"] is None
    assert llm.calls == []


def test_ask_reports_dropped_citations(make_client):
    def responder(messages):
        user = messages[-1].content
        law_id = user.split('law_id="')[1].split('"')[0]
        article = user.split('article="')[1].split('"')[0]
        return json.dumps(
            {
                "answer": "جواب",
                "citations": [{"law_id": law_id, "article": article}, {"law_id": "made_up", "article": "1"}],
                "refused": False,
            }
        )

    client, _, _ = make_client(responder)
    data = client.post("/ask", json={"question": QUESTION}).json()
    assert data["refused"] is False
    assert data["dropped_citations"] == [{"law_id": "made_up", "article": "1"}]


def test_ask_invalid_llm_json_is_a_200_refusal_with_error_code(make_client):
    client, _, _ = make_client(lambda m: "sorry, plain text")
    data = client.post("/ask", json={"question": QUESTION}).json()
    assert data["refused"] is True
    assert data["error_code"] == "LLM_INVALID_OUTPUT"


def test_upstream_llm_error_is_a_stable_code_without_details(make_client):
    class TimeoutLLM:
        def complete(self, messages):
            raise LLMError(LLM_TIMEOUT, 504)

    client, _, _ = make_client(llm=TimeoutLLM())
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 504
    assert response.json() == {
        "error": {"code": "LLM_TIMEOUT", "message": "The language model did not answer in time."}
    }


def test_ask_database_down_is_503(make_client):
    client, _, store = make_client()

    def down(*args, **kwargs):
        raise DatabaseUnavailable("down")

    original = store.search
    store.search = down
    try:
        response = client.post("/ask", json={"question": QUESTION})
    finally:
        store.search = original
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_ask_validation(make_client):
    client, _, _ = make_client()
    cases = [
        {"question": "ab"},
        {"question": "   x   "},
        {"question": "س" * 1001},
        {"question": QUESTION, "top_k": 0},
        {"question": QUESTION, "top_k": 11},
        {"question": QUESTION, "unexpected": 1},
        {},
    ]
    for body in cases:
        response = client.post("/ask", json=body)
        assert response.status_code == 422, body
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_ask_unknown_law_is_422(make_client):
    client, _, _ = make_client()
    response = client.post("/ask", json={"question": QUESTION, "law_id": "no_such_law"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_LAW"


def test_oversized_body_is_413(make_client):
    client, _, _ = make_client()
    response = client.post(
        "/ask", content=json.dumps({"question": "س" * 20000}), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_rate_limit_on_ask(make_client):
    client, _, _ = make_client(rate_limit_per_minute=2)
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert int(response.headers["Retry-After"]) >= 1
    assert client.get("/laws").status_code == 200  # only /ask is limited


def test_laws_and_article_endpoints(make_client):
    client, _, _ = make_client()
    laws = client.get("/laws").json()
    assert {law["law_id"]: law["article_count"] for law in laws} == {
        "income_tax_34_2014": 82,
        "sales_tax_6_1994": 78,
    }
    article = client.get("/laws/sales_tax_6_1994/articles/6").json()
    assert article["article"] == "6" and "(16%)" in article["text"]
    bis = client.get("/laws/sales_tax_6_1994/articles/4 مكرر").json()
    assert bis["article"] == "4bis" and bis["article_label"] == "4 مكرر"
    noted = client.get("/laws/income_tax_34_2014/articles/72").json()
    assert any("المحكمة الدستورية" in note for note in noted["notes"])


def test_article_not_found(make_client):
    client, _, _ = make_client()
    for path in ("/laws/sales_tax_6_1994/articles/999", "/laws/nope/articles/1", "/laws/sales_tax_6_1994/articles/x"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"


def test_index_page_and_static_assets(make_client):
    client, _, _ = make_client()
    page = client.get("/")
    assert page.status_code == 200
    assert 'dir="rtl"' in page.text and "ليست استشارة قانونية" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "Jordan Tax Law RAG"


def test_cors_is_off_by_default_and_configurable(make_client):
    client, _, _ = make_client()
    preflight = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    assert "access-control-allow-origin" not in client.options("/ask", headers=preflight).headers
    client, _, _ = make_client(cors_allow_origins=["https://ui.example"])
    response = client.options("/ask", headers={**preflight, "Origin": "https://ui.example"})
    assert response.headers["access-control-allow-origin"] == "https://ui.example"


# ---- rate limiter unit tests ------------------------------------------------


def test_sliding_window_limiter_frees_slots_after_the_window():
    now = [0.0]
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=lambda: now[0])
    assert limiter.allow("ip")[0] and limiter.allow("ip")[0]
    allowed, retry_after = limiter.allow("ip")
    assert not allowed and retry_after == 61
    assert limiter.allow("other")[0]  # per key
    now[0] = 60.0
    assert limiter.allow("ip")[0]


def test_client_ip_ignores_forwarded_for_unless_proxies_are_trusted():
    assert client_ip("10.0.0.1", "1.2.3.4", trusted_proxy_hops=0) == "10.0.0.1"
    # one trusted proxy appends the real client: take the right-most entry
    assert client_ip("10.0.0.1", "6.6.6.6, 1.2.3.4", trusted_proxy_hops=1) == "1.2.3.4"
    assert client_ip("10.0.0.1", None, trusted_proxy_hops=1) == "10.0.0.1"
    assert client_ip(None, None, trusted_proxy_hops=0) == "unknown"
