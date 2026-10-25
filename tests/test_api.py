"""HTTP API with fake embedder, fake LLM and the in-memory store."""

import base64
import hashlib
import inspect
import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

from app.db import DatabaseUnavailable
from app.llm import LLM_TIMEOUT, LLMError
from app.logs import KeyValueFormatter
from app.main import BodySizeLimitMiddleware, loggable_path
from app.rate_limit import ClientIpSettings, SlidingWindowRateLimiter, client_ip

QUESTION = "ما هي نسبة الضريبة العامة على استيراد أي سلعة أو خدمة أو توريد أي منهما؟"


def test_health(make_client):
    client, _, _ = make_client()
    assert client.get("/health").json() == {"status": "ok"}


HEALTH = {"X-Health-Token": "s3cret-health-token"}


def test_health_db_ok_and_503_when_down(make_client):
    client, _, store = make_client(health_token="s3cret-health-token")
    assert client.get("/health/db", headers=HEALTH).status_code == 200
    store.ping_error = DatabaseUnavailable("down")
    response = client.get("/health/db", headers=HEALTH)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
    store.ping_error = None


def test_health_db_is_private_404_without_the_right_token(make_client):
    client, _, store = make_client(health_token="s3cret-health-token")
    store.ping_error = DatabaseUnavailable("down")  # a 404 must not even touch the database
    for token in (None, "", "wrong", "s3cret-health-tokenX"):
        headers = {} if token is None else {"X-Health-Token": token}
        response = client.get("/health/db", headers=headers)
        assert response.status_code == 404, headers
        assert response.json()["error"]["code"] == "NOT_FOUND"
    store.ping_error = None
    assert "/health/db" not in client.get("/openapi.json").json()["paths"]

    unset, _, _ = make_client()  # no HEALTH_TOKEN configured: always 404, even with a header
    assert unset.get("/health/db", headers={"X-Health-Token": ""}).status_code == 404
    assert unset.get("/health/db", headers=HEALTH).status_code == 404


def test_health_liveness_needs_no_worker_thread(make_client):
    client, _, _ = make_client()
    route = next(r for r in client.app.routes if getattr(r, "path", "") == "/health")
    assert inspect.iscoroutinefunction(route.endpoint)
    assert client.head("/health").status_code == 200


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
        def complete(self, messages, timeout_seconds=None):
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


def _chunks(payload: bytes, size: int = 1024):
    for start in range(0, len(payload), size):
        yield payload[start : start + size]


def test_chunked_body_over_limit_returns_413(make_client):
    client, llm, _ = make_client()
    payload = json.dumps({"question": "س" * 20000}).encode()
    response = client.post("/ask", content=_chunks(payload), headers={"Content-Type": "application/json"})
    assert "content-length" not in response.request.headers  # streamed: only the byte count can catch it
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert llm.calls == []


def test_chunked_body_under_limit_is_accepted(make_client):
    client, _, _ = make_client()
    payload = json.dumps({"question": QUESTION}).encode()
    response = client.post("/ask", content=_chunks(payload, size=16), headers={"Content-Type": "application/json"})
    assert "content-length" not in response.request.headers
    assert response.status_code == 200


def test_body_limit_middleware_answers_413_itself_when_unhandled():
    async def raw_app(scope, receive, send):  # reads the body with no FastAPI handler in between
        more = True
        while more:
            message = await receive()
            more = message.get("more_body", False)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    client = TestClient(BodySizeLimitMiddleware(raw_app, max_bytes=10))
    assert client.post("/", content=_chunks(b"x" * 10, size=3)).status_code == 200
    response = client.post("/", content=_chunks(b"x" * 11, size=3))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_head_root_returns_200(make_client):
    client, _, _ = make_client()
    for path in ("/", "/health"):
        response = client.head(path)
        assert response.status_code == 200, path
        assert response.content == b""


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
    no_proxy, one_proxy = ClientIpSettings(trusted_proxy_hops=0), ClientIpSettings(trusted_proxy_hops=1)
    assert client_ip("10.0.0.1", {"x-forwarded-for": "1.2.3.4"}, no_proxy) == "10.0.0.1"
    # one trusted proxy appends the real client: take the right-most entry
    assert client_ip("10.0.0.1", {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}, one_proxy) == "1.2.3.4"
    assert client_ip("10.0.0.1", {}, one_proxy) == "10.0.0.1"
    assert client_ip(None, {}, no_proxy) == "unknown"


# ---- security headers ---------------------------------------------------------

API_CSP = "default-src 'none'; frame-ancestors 'none'"
COMMON_HEADERS = {
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("GET", "/", 200),
        ("HEAD", "/", 200),
        ("GET", "/health", 200),
        ("GET", "/laws", 200),
        ("GET", "/docs", 200),
        ("GET", "/static/app.js", 200),
        ("GET", "/no/such/page", 404),
        ("POST", "/ask", 422),
    ],
)
def test_every_response_has_hsts_and_frame_deny(make_client, method, path, status):
    client, _, _ = make_client()
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code == status
    for name, value in COMMON_HEADERS.items():
        assert response.headers.get(name) == value, (path, name)
    assert "content-security-policy" in response.headers
    assert "server" not in response.headers


def test_early_413_still_has_security_headers(make_client):
    client, _, _ = make_client()
    response = client.post("/ask", content=b"x" * 20000, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"] == API_CSP


def test_api_csp_is_default_none(make_client):
    client, _, _ = make_client()
    for response in (
        client.get("/health"),
        client.get("/laws"),
        client.get("/openapi.json"),
        client.get("/laws/sales_tax_6_1994/articles/6"),
        client.post("/ask", json={"question": QUESTION}),
        client.get("/nope"),
    ):
        assert response.headers["content-security-policy"] == API_CSP, response.request.url
    index_csp = client.get("/").headers["content-security-policy"]
    assert "script-src 'self'" in index_csp and "frame-ancestors 'none'" in index_csp
    assert "default-src 'self'" in index_csp


SWAGGER_PREFIX = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.33.3/"
# Computed from the files served by jsdelivr, which were checked byte for byte against
# the npm tarball whose sha512 matches the registry's published integrity.
SWAGGER_JS_SRI = "sha384-h00llMzD1afyNow1pDmnZKlHsUZgp9KE9HXLFXF2CvbwsBY0+WBO4f75SfGwvM4K"
SWAGGER_CSS_SRI = "sha384-Ov4/wv3j2bmct8cDc5X4ngJZohVPzEmc6uDPH8WeljUxO5vtoykvMEfbu9Vh6RaW"


def test_docs_assets_are_pinned_with_sri_and_csp_allows_only_that_path(make_client):
    client, _, _ = make_client()
    page = client.get("/docs")
    assert page.status_code == 200
    html = page.text
    assert (
        f'<script src="{SWAGGER_PREFIX}swagger-ui-bundle.js" integrity="{SWAGGER_JS_SRI}" '
        'crossorigin="anonymous"></script>'
    ) in html
    assert (
        f'<link rel="stylesheet" href="{SWAGGER_PREFIX}swagger-ui.css" integrity="{SWAGGER_CSS_SRI}" '
        'crossorigin="anonymous">'
    ) in html
    assert 'href="/favicon.ico"' in html and "fastapi.tiangolo.com" not in html
    assert html.count("cdn.jsdelivr.net") == 2
    inline = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(inline) == 1 and "SwaggerUIBundle(" in inline[0] and "validatorUrl: null" in inline[0]
    digest = base64.b64encode(hashlib.sha256(inline[0].encode()).digest()).decode()
    assert page.headers["content-security-policy"] == (
        f"default-src 'none'; script-src {SWAGGER_PREFIX} 'sha256-{digest}'; "
        f"style-src {SWAGGER_PREFIX} 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    )


def test_redoc_disabled(make_client):
    client, _, _ = make_client()
    assert client.get("/redoc").status_code == 404
    assert client.get("/docs/oauth2-redirect").status_code == 404


# ---- UI ----------------------------------------------------------------------


def test_favicon_served(make_client):
    client, _, _ = make_client()
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.text.lstrip().startswith("<svg")
    assert response.headers["content-security-policy"] == API_CSP  # an SVG served alone may not run scripts
    assert "<script" not in response.text.lower()
    assert client.head("/favicon.ico").status_code == 200
    assert client.get("/static/favicon.svg").content == response.content
    assert 'rel="icon"' in client.get("/").text


def test_ui_has_arabic_text_for_every_ask_error_code():
    from app.main import ERROR_MESSAGES, STATIC_DIR

    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    block = script.split("const ERROR_TEXT = {", 1)[1].split("};", 1)[0]
    ui_codes = set(re.findall(r"^\s*([A-Z_]+):", block, re.M))
    server_only = {"NOT_FOUND", "INTERNAL_ERROR"}  # /ask never needs a dedicated text for these
    assert set(ERROR_MESSAGES) - server_only <= ui_codes
    for code in ("RATE_LIMITED", "DAILY_LIMIT_REACHED", "DAILY_BUDGET_EXHAUSTED", "BUSY"):
        assert code in ui_codes
    assert "retry-after" in script.lower()


def test_ui_stays_csp_clean():
    from app.main import STATIC_DIR

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", page) == []  # no inline <script>
    assert "<style" not in page and " style=" not in page
    assert re.search(r"\son[a-z]+=", page) is None  # no inline event handlers
    assert re.search(r"\.(innerHTML|outerHTML|insertAdjacentHTML)\b|document\.write", script) is None
    assert "eval(" not in script and "new Function" not in script
    assert len(re.findall(r'class="example\b', page)) == 4


def test_ui_loads_fonts_and_styles_only_from_its_own_origin():
    from app.main import STATIC_DIR

    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    assert "@import" not in css
    urls = re.findall(r"url\(\s*['\"]?([^'\")]+)", css)
    assert urls and all(url.startswith("fonts/") and url.endswith(".woff2") for url in urls)
    for url in urls:
        assert (STATIC_DIR / url).is_file(), url
    links = re.findall(r"<link\b[^>]*\bhref=\"([^\"]+)\"", page)
    assert links and all(href.startswith("/static/") for href in links)


def test_self_hosted_fonts_are_served_as_woff2_with_their_licences(make_client):
    from app.main import STATIC_DIR

    client, _, _ = make_client()
    fonts = sorted((STATIC_DIR / "fonts").glob("*.woff2"))
    assert fonts
    for font in fonts:
        response = client.get(f"/static/fonts/{font.name}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "font/woff2"
        assert response.content[:4] == b"wOF2"
    for family in ("IBM-Plex-Sans-Arabic", "Noto-Naskh-Arabic"):
        assert "SIL OPEN FONT LICENSE Version 1.1" in (STATIC_DIR / "fonts" / f"OFL-{family}.txt").read_text(
            encoding="utf-8"
        )


def test_static_assets_are_revalidated_so_a_deploy_never_mixes_old_js_with_new_html(make_client):
    client, _, _ = make_client()
    for path in ("/static/app.js", "/static/style.css", "/"):
        response = client.get(path)
        assert response.headers["cache-control"] == "no-cache", path
        assert response.headers["etag"]
    etag = client.get("/static/app.js").headers["etag"]
    assert client.get("/static/app.js", headers={"If-None-Match": etag}).status_code == 304


# ---- round 2: unhandled errors and log hygiene --------------------------------------


def test_unhandled_error_is_a_500_with_every_header_and_a_request_log(make_client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    built, _, store = make_client()

    def boom():
        raise RuntimeError("secret internal detail")

    built.app.add_api_route("/boom", boom)
    client = TestClient(built.app, raise_server_exceptions=False)
    response = client.get("/boom", headers={"X-Request-ID": "req-123"})
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "INTERNAL_ERROR", "message": "Internal server error."}}
    assert "secret internal detail" not in response.text
    for name, value in COMMON_HEADERS.items():
        assert response.headers[name] == value, name
    assert response.headers["content-security-policy"] == API_CSP
    assert response.headers["x-request-id"] == "req-123"
    request_lines = [r for r in caplog.records if r.getMessage() == "request" and r.request_id == "req-123"]
    assert len(request_lines) == 1 and request_lines[0].status == 500 and len(request_lines[0].client_key) == 12
    errors = [r for r in caplog.records if r.getMessage() == "unhandled error"]
    assert len(errors) == 1 and errors[0].exc_info and errors[0].request_id == "req-123"

    def broken_search(*args, **kwargs):  # e.g. a psycopg error that is not "database unavailable"
        raise RuntimeError("relation does not exist")

    original, store.search = store.search, broken_search
    try:
        ask = client.post("/ask", json={"question": QUESTION})
    finally:
        store.search = original
    assert ask.status_code == 500 and ask.json()["error"]["code"] == "INTERNAL_ERROR"
    assert ask.headers["x-frame-options"] == "DENY" and ask.headers["x-request-id"]


def test_logged_path_cannot_forge_fields_or_hide_text(make_client, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client, _, _ = make_client()
    client.get("/x%0aINJECTED%20status=200%20client_key=forged")
    client.get("/x%E2%80%AEevil")
    client.get("/" + "a" * 1000)
    client.get("/laws")
    lines = [r for r in caplog.records if r.getMessage() == "request"]
    assert len(lines) == 4
    formatter = KeyValueFormatter("%(message)s")
    for record in lines:
        line = formatter.format(record)
        assert "\n" not in line and "\t" not in line and "\u202e" not in line
        assert line.count(" status=") == 1 and line.count(" client_key=") == 1 and line.count(" path=") == 1
        assert len(record.path) <= 200
    assert lines[0].path.endswith("INJECTED%20status%3D200%20client_key%3Dforged")
    assert lines[1].path == "/x%E2%80%AEevil"  # RLO shown as an escape, never raw
    assert lines[2].path.endswith("...")
    assert lines[3].path == "/laws"
    # a raw newline (possible over a socket) is encoded too, and Arabic stays readable as UTF-8 escapes
    assert loggable_path("/x\nINJECTED status=200\r\t") == "/x%0AINJECTED%20status%3D200%0D%09"
    assert loggable_path("/laws/x/articles/4 مكرر") == "/laws/x/articles/4%20%D9%85%D9%83%D8%B1%D8%B1"


def test_ask_gives_the_llm_the_rest_of_the_request_deadline(make_client):
    client, llm, _ = make_client(ask_deadline_seconds=12.0)
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200
    assert len(llm.timeouts) == 1 and 0 < llm.timeouts[0] <= 12.0


def test_ui_tells_users_that_questions_go_to_third_party_ai_providers(make_client):
    client, _, _ = make_client()
    page = client.get("/").text
    notice = re.search(r'<p id="privacy-notice"[^>]*>(.*?)</p>', page, re.S)
    assert notice, "privacy notice missing"
    text = notice.group(1)
    assert "Google Gemini" in text and "OpenRouter" in text and "بيانات شخصية" in text


# ---- round 3: response details --------------------------------------------------------------


def test_405_keeps_the_allow_header(make_client):
    client, _, _ = make_client()
    for method, path, allowed in (("PUT", "/laws", "GET"), ("GET", "/ask", "POST")):
        response = client.request(method, path)
        assert response.status_code == 405, path
        assert allowed in response.headers["allow"], (path, response.headers.get("allow"))
        assert response.json()["error"]["code"] == "HTTP_ERROR"
        assert response.headers["x-frame-options"] == "DENY"
    # GET and HEAD /health are two routes; Starlette names the methods of the last one it matched
    health = client.delete("/health")
    assert health.status_code == 405 and set(health.headers["allow"].split(", ")) <= {"GET", "HEAD"}


def test_validation_message_echoes_only_a_safe_field_path(make_client):
    client, _, _ = make_client()
    weird_key = "<img src=x onerror=alert(1)>\u202e" + "k" * 200
    response = client.post("/ask", json={"question": QUESTION, weird_key: 1})
    assert response.status_code == 422
    message = response.json()["error"]["message"]
    field, _, reason = message.partition(": ")
    assert re.fullmatch(r"[A-Za-z0-9_.\[\]]{0,64}", field), field
    assert field.startswith("imgsrcxonerroralert1") and len(field) == 64
    assert reason == "Extra inputs are not permitted"
    plain = client.post("/ask", json={"question": QUESTION, "top_k": 0}).json()["error"]["message"]
    assert plain.startswith("top_k: ")  # ordinary field names are unchanged


INDEX_CSP_EXPECTED = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


def test_page_csp_has_no_data_images_and_no_plugins(make_client):
    client, _, _ = make_client()
    assert client.get("/").headers["content-security-policy"] == INDEX_CSP_EXPECTED
    assert client.head("/").headers["content-security-policy"] == INDEX_CSP_EXPECTED
