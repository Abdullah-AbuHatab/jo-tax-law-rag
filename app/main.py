"""FastAPI application.

    uvicorn app.main:create_app_from_env --factory --port 8000

``create_app`` takes optional collaborators so tests can pass fakes; in
production everything is built from environment variables. Building the
app never opens a network connection: the database pool opens on first use.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import logging
import mimetypes
import re
import secrets
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.answering import AnswerResult, AnswerService, ServiceError
from app.captcha import CaptchaUnavailable, TurnstileVerifier
from app.config import Settings
from app.db import DatabaseUnavailable
from app.dependencies import build_database, build_embedder, build_llm, build_vector_store
from app.embeddings import Embedder
from app.guards import AnswerCache, ConcurrencyGate, DailyBudget, GateFull, PerKeyDailyLimit, SingleFlight
from app.ingest import ingest_laws
from app.llm import LLMClient
from app.logs import configure_logging
from app.parsing import Law, load_laws, normalize_article_ref
from app.rate_limit import ClientIpSettings, SlidingWindowRateLimiter, client_ip, client_key, client_network
from app.schemas import (
    ArticleOut,
    ArticleSummaryOut,
    AskRequest,
    AskResponse,
    CitationOut,
    DroppedCitationOut,
    ErrorResponse,
    LawOut,
    PublicConfigOut,
    RetrievedOut,
)
from app.store import InMemoryStore, VectorStore

logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Python's MIME table depends on the OS and may not know .woff2; register it so the
# self-hosted fonts in static/fonts/ are served as font/woff2, not octet-stream.
mimetypes.add_type("font/woff2", ".woff2")
MAX_BODY_BYTES = 16 * 1024
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
APP_TITLE = "Jordan Tax Law RAG"
OPENAPI_URL = "/openapi.json"

# Swagger UI for /docs: one exact release, loaded with Subresource Integrity. The
# sha384 values were computed from the files cdn.jsdelivr.net serves, after checking
# them byte for byte against the npm tarball (whose sha512 matches the registry's
# published integrity). Bump the version and both hashes together.
SWAGGER_UI_VERSION = "5.33.3"
SWAGGER_UI_PREFIX = f"https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}/"
SWAGGER_UI_JS_SRI = "sha384-h00llMzD1afyNow1pDmnZKlHsUZgp9KE9HXLFXF2CvbwsBY0+WBO4f75SfGwvM4K"
SWAGGER_UI_CSS_SRI = "sha384-Ov4/wv3j2bmct8cDc5X4ngJZohVPzEmc6uDPH8WeljUxO5vtoykvMEfbu9Vh6RaW"
# The one inline script; the CSP allows it by hash, not with 'unsafe-inline'.
_DOCS_INIT_SCRIPT = (
    f'SwaggerUIBundle({{url: "{OPENAPI_URL}", dom_id: "#swagger-ui", layout: "BaseLayout", '
    "deepLinking: true, showExtensions: true, showCommonExtensions: true, validatorUrl: null, "
    "presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset]});"
)
DOCS_HTML = (
    "<!doctype html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f"<title>{APP_TITLE} - docs</title>\n"
    '<link rel="icon" href="/favicon.ico" type="image/svg+xml">\n'
    f'<link rel="stylesheet" href="{SWAGGER_UI_PREFIX}swagger-ui.css" integrity="{SWAGGER_UI_CSS_SRI}" '
    'crossorigin="anonymous">\n'
    "</head>\n"
    "<body>\n"
    '<div id="swagger-ui"></div>\n'
    f'<script src="{SWAGGER_UI_PREFIX}swagger-ui-bundle.js" integrity="{SWAGGER_UI_JS_SRI}" '
    'crossorigin="anonymous"></script>\n'
    f"<script>{_DOCS_INIT_SCRIPT}</script>\n"
    "</body>\n"
    "</html>\n"
).encode()
_DOCS_SCRIPT_HASH = base64.b64encode(hashlib.sha256(_DOCS_INIT_SCRIPT.encode()).digest()).decode()

COMMON_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
API_CSP = "default-src 'none'; frame-ancestors 'none'"
INDEX_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
# Scoped to the pinned package path, not all of cdn.jsdelivr.net (which serves any npm
# package); Swagger UI needs inline style attributes, hence 'unsafe-inline' for styles only.
DOCS_CSP = (
    f"default-src 'none'; script-src {SWAGGER_UI_PREFIX} 'sha256-{_DOCS_SCRIPT_HASH}'; "
    f"style-src {SWAGGER_UI_PREFIX} 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


MAX_LOGGED_PATH_CHARS = 200
MAX_ECHOED_FIELD_CHARS = 64
_UNSAFE_FIELD_CHARS = re.compile(r"[^A-Za-z0-9_.\[\]]")


def loggable_path(path: str) -> str:
    """The (already percent-decoded) path made safe for key=value log lines.

    Percent-encodes everything outside a small URL-safe set, so control
    characters, spaces, "=" and bidi characters cannot forge fields or hide
    text; then caps the length.
    """
    encoded = quote(path, safe="/-._~!$&'()*+,;:@")
    if len(encoded) <= MAX_LOGGED_PATH_CHARS:
        return encoded
    return encoded[: MAX_LOGGED_PATH_CHARS - 3] + "..."


# With Turnstile on, the page may load Cloudflare's widget script and its iframe; nothing else changes.
INDEX_CSP_TURNSTILE = INDEX_CSP.replace(
    "script-src 'self';",
    "script-src 'self' https://challenges.cloudflare.com; frame-src https://challenges.cloudflare.com;",
)


def security_headers(path: str, turnstile: bool = False) -> dict[str, str]:
    """Headers for every response: the common set plus the CSP for this path."""
    page_csp = INDEX_CSP_TURNSTILE if turnstile else INDEX_CSP
    csp = {"/": page_csp, "/docs": DOCS_CSP}.get(path, API_CSP)
    return {**COMMON_SECURITY_HEADERS, "Content-Security-Policy": csp}


# Fixed client-facing messages per error code; upstream text is never forwarded.
ERROR_MESSAGES = {
    "VALIDATION_ERROR": "The request is invalid.",
    "UNKNOWN_LAW": "Unknown law_id. See GET /laws for the available laws.",
    "NOT_FOUND": "Not found.",
    "RATE_LIMITED": "Too many requests. Please wait and try again.",
    "DAILY_LIMIT_REACHED": "You have reached today's question limit. Please try again tomorrow (UTC).",
    "DAILY_BUDGET_EXHAUSTED": "The service has used today's capacity. Please try again tomorrow (UTC).",
    "BUSY": "The service is busy. Please try again in a few seconds.",
    "CAPTCHA_REQUIRED": "Complete the verification challenge, then ask again.",
    "CAPTCHA_FAILED": "The verification challenge could not be confirmed. Please try again.",
    "PAYLOAD_TOO_LARGE": "The request body is too large.",
    "DATABASE_UNAVAILABLE": "The database is temporarily unavailable.",
    "EMBEDDING_UNAVAILABLE": "The embedding service is temporarily unavailable.",
    "LLM_TIMEOUT": "The language model did not answer in time.",
    "LLM_RATE_LIMITED": "The language model is rate-limited. Please try again later.",
    "LLM_UNAVAILABLE": "The language model is temporarily unavailable.",
    "LLM_CONFIG_ERROR": "The language model is not configured correctly.",
    "LLM_BAD_RESPONSE": "The language model returned an unusable response.",
    "INTERNAL_ERROR": "Internal server error.",
}


def error_response(code: str, status: int, headers: Mapping[str, str] | None = None) -> JSONResponse:
    message = ERROR_MESSAGES.get(code, ERROR_MESSAGES["INTERNAL_ERROR"])
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class _BodyTooLarge(StarletteHTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413)


class BodySizeLimitMiddleware:
    """Caps the request body by counting the bytes actually received.

    The Content-Length check in ``request_guard`` rejects declared oversized
    bodies before any byte is read; this also catches chunked bodies, which
    carry no Content-Length. It raises an HTTPException from ``receive``, the
    one exception FastAPI re-raises while reading a body, so the app's own
    handlers answer 413 (with the usual headers and request log). If nothing
    handled it, this middleware answers the 413 itself.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if response_started:
                raise
            await error_response("PAYLOAD_TOO_LARGE", 413)(scope, receive, send)


def to_response(result: AnswerResult) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        error_code=result.error_code,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        dropped_citations=[DroppedCitationOut(**d) for d in result.dropped_citations],
        retrieved=[
            RetrievedOut(
                chunk_id=r.id, law_id=r.law_id, article=r.article, chunk_index=r.chunk_index, score=round(r.score, 4)
            )
            for r in result.retrieved
        ],
        model=result.model,
    )


def create_app(
    settings: Settings | None = None,
    *,
    laws: Mapping[str, Law] | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
    captcha: TurnstileVerifier | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    # Turnstile is on only when both keys are configured; ``captcha`` lets tests mock siteverify.
    if not settings.turnstile_enabled:
        captcha = None
    elif captcha is None:
        captcha = TurnstileVerifier(settings.turnstile_secret_key.get_secret_value())
    laws = dict(laws) if laws is not None else load_laws(settings.data_dir)
    database = build_database(settings) if store is None and settings.vector_store == "pgvector" else None
    embedder = embedder or build_embedder(settings)
    store = store or build_vector_store(settings, database)
    if llm is None:
        llm = build_llm(settings)  # None when no key: /ask then answers 503 LLM_CONFIG_ERROR
    llm_budget = DailyBudget(settings.daily_llm_budget)
    service = AnswerService(
        embedder,
        store,
        llm,
        laws,
        settings.min_similarity,
        settings.default_top_k,
        llm_budget=llm_budget,
        deadline_seconds=settings.ask_deadline_seconds,
        llm_per_client=PerKeyDailyLimit(settings.daily_llm_calls_per_client),
    )
    limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute, window_seconds=60)
    daily_per_client = PerKeyDailyLimit(settings.daily_ask_limit_per_client)
    embedding_budget = DailyBudget(settings.daily_embedding_budget)
    in_flight = ConcurrencyGate(settings.max_concurrent_asks)
    answers = AnswerCache(settings.answer_cache_size, settings.answer_cache_ttl_seconds)
    flights = SingleFlight(max_wait_seconds=30.0, max_waiters=8)
    ip_settings = ClientIpSettings(header=settings.client_ip_header, trusted_proxy_hops=settings.trusted_proxy_hops)
    # Random per process: client keys only need to compare clients within one deploy.
    client_key_salt = secrets.token_hex(16)

    def identify(request: Request) -> str:
        """The caller's pseudonymous key. Raw IPs are never logged or stored."""
        ip = client_ip(request.client.host if request.client else None, request.headers, ip_settings)
        return client_key(client_network(ip), client_key_salt)  # one IPv6 /64 = one client

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if isinstance(store, InMemoryStore) and len(store) == 0:
            # Offline demo mode: fill the memory store from the cleaned laws.
            for report in ingest_laws(laws, embedder, store):
                logger.info("in-memory ingest", extra={"law_id": report.law_id, "chunks": report.chunks})
        yield
        if database is not None:
            database.close()

    app = FastAPI(
        title=APP_TITLE,
        version="1.0.0",
        description=(
            "Arabic question answering over Jordanian tax laws with article-level citations. Not legal advice."
        ),
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
        openapi_url=OPENAPI_URL,
        docs_url=None,  # served below with pinned assets and its own CSP
        redoc_url=None,
    )
    app.state.single_flight = flights  # observable (tests, debugging)
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "CF-Turnstile-Response"],
        )

    # Added before request_guard, so it sits directly inside it: request_guard is a
    # BaseHTTPMiddleware, whose receive() runs in a task group that would wrap the
    # 413 exception in an ExceptionGroup (FastAPI turns that into a 400). Here,
    # FastAPI reads the body through this cap with nothing in between.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    @app.middleware("http")
    async def request_guard(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        # accept a caller's id only if it is short and plain, so it is safe to log
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        started = time.perf_counter()
        key = identify(request)
        request.state.client_key = key
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            response = error_response("PAYLOAD_TOO_LARGE", 413)  # declared too large: nothing is read
        else:
            try:
                response = await call_next(request)
            except Exception:
                # Answered here, inside request_guard, so a 500 still gets the security
                # headers, X-Request-ID and a request log line; Starlette's own 500
                # handler sits outside this middleware. No detail reaches the client.
                logger.exception("unhandled error", extra={"request_id": request_id, "client_key": key})
                response = error_response("INTERNAL_ERROR", 500)
        response.headers["X-Request-ID"] = request_id
        response.headers.update(security_headers(request.url.path, turnstile=captcha is not None))
        if request.url.path.startswith("/static/"):
            # Revalidate (ETag, cheap 304s): after a deploy the page must never
            # run a heuristically cached old app.js against the new index.html.
            response.headers.setdefault("Cache-Control", "no-cache")
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "client_key": key,
                "method": request.method,
                "path": loggable_path(request.url.path),
                "status": response.status_code,
                "ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        # The path can hold a caller-chosen JSON key: echo only a plain, short version of it.
        field = _UNSAFE_FIELD_CHARS.sub("", field)[:MAX_ECHOED_FIELD_CHARS]
        body = {"error": {"code": "VALIDATION_ERROR", "message": f"{field}: {first.get('msg', 'invalid')}"}}
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 429: "RATE_LIMITED", 413: "PAYLOAD_TOO_LARGE"}.get(exc.status_code)
        if code is None:
            # e.g. 405: keep Starlette's headers, which carry Allow.
            return JSONResponse(
                {"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}}, exc.status_code, headers=exc.headers
            )
        return error_response(code, exc.status_code, headers=exc.headers)

    @app.exception_handler(ServiceError)
    async def service_handler(_: Request, exc: ServiceError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return error_response(exc.code, exc.http_status, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return error_response("INTERNAL_ERROR", 500)

    def rate_limited(request: Request) -> None:
        key = getattr(request.state, "client_key", None) or identify(request)
        allowed, retry_after = limiter.allow(key)
        if not allowed:
            raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after)})

    @app.get("/health", tags=["health"])
    @app.head("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Liveness: the process is up. Does not touch the database, and as an
        async route it needs no worker thread, so a busy pool cannot stall it."""
        return {"status": "ok"}

    health_token = settings.health_token.get_secret_value().encode()

    @app.get("/health/db", include_in_schema=False)
    def health_db(request: Request):
        """Readiness of the database (private): a real round trip, 503 when it is down.

        404 unless X-Health-Token matches HEALTH_TOKEN, so the public cannot use it
        to keep Neon awake or to queue requests on a stalled database.
        """
        presented = request.headers.get("x-health-token", "").encode()
        if not health_token or not hmac.compare_digest(presented, health_token):
            raise HTTPException(status_code=404)
        try:
            store.ping()
        except DatabaseUnavailable:
            return error_response("DATABASE_UNAVAILABLE", 503)
        return {"status": "ok"}

    def _mask_ip(value: str | None) -> str | None:
        """An address shortened for display: IPv4 without its last octet, IPv6 to its /64."""
        if not value:
            return None
        try:
            address = ipaddress.ip_address(value.strip())
        except ValueError:
            return "invalid"
        if address.version == 4:
            return ".".join(str(address).split(".")[:3]) + ".x"
        return str(ipaddress.ip_network(f"{address}/64", strict=False))

    @app.get("/health/client", include_in_schema=False)
    def health_client(request: Request) -> dict[str, object]:
        """Private diagnostic (same token as /health/db): which client-IP headers reach
        the app, masked, so the owner can check the rate-limit key source after a deploy."""
        presented = request.headers.get("x-health-token", "").encode()
        if not health_token or not hmac.compare_digest(presented, health_token):
            raise HTTPException(status_code=404)
        forwarded = request.headers.get("x-forwarded-for", "")
        return {
            "configured_header": settings.client_ip_header or None,
            "trusted_proxy_hops": settings.trusted_proxy_hops,
            "cf_connecting_ip": _mask_ip(request.headers.get("cf-connecting-ip")),
            "true_client_ip": _mask_ip(request.headers.get("true-client-ip")),
            "x_real_ip": _mask_ip(request.headers.get("x-real-ip")),
            "x_forwarded_for": [_mask_ip(part) for part in forwarded.split(",") if part.strip()],
            "peer": _mask_ip(request.client.host if request.client else None),
        }

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["ask"],
        dependencies=[Depends(rate_limited)],
        responses={429: {"model": ErrorResponse}},
    )
    def ask(body: AskRequest, request: Request) -> AskResponse:
        """Answer from the laws, with validated article citations, or refuse.

        Guards, in order: the per-minute limit (dependency), the per-client
        daily limit, the answer cache (a hit has already been counted by
        both limits), single-flight (an identical question already being
        answered is waited for, at most 30 s), the concurrency gate, the daily
        embedding budget, the client's daily LLM allowance and the daily LLM
        budget (both spent only when the LLM is actually called). BUSY,
        DAILY_BUDGET_EXHAUSTED and an exhausted LLM allowance give the client's
        daily ask back: nothing was served.
        """
        if body.law_id is not None and body.law_id not in laws:
            raise ServiceError("UNKNOWN_LAW", 422)
        if captcha is not None:
            # After the per-minute limit (so verify calls are throttled too), before anything is counted.
            token = request.headers.get("cf-turnstile-response", "")
            if not token:
                raise ServiceError("CAPTCHA_REQUIRED", 403)
            try:
                verified = captcha.verify(token)
            except CaptchaUnavailable:
                raise ServiceError("CAPTCHA_FAILED", 503, retry_after=5) from None  # fail closed
            if not verified:
                raise ServiceError("CAPTCHA_FAILED", 403)
        key = getattr(request.state, "client_key", None) or identify(request)
        if not daily_per_client.allow(key):
            raise ServiceError("DAILY_LIMIT_REACHED", 429, retry_after=daily_per_client.seconds_until_reset())
        top_k = body.top_k or settings.default_top_k
        cache_key = AnswerCache.key_for(body.question, body.law_id, top_k)
        cached = answers.get(cache_key)
        if cached is not None:
            return cached
        leader = flights.lead(cache_key)
        if not leader:
            flights.wait(cache_key)
            cached = answers.get(cache_key)
            if cached is not None:
                return cached
            # The first request failed, gave an uncacheable result or took too long: answer normally.
        try:
            with in_flight:
                if not embedding_budget.try_consume():
                    raise ServiceError(
                        "DAILY_BUDGET_EXHAUSTED", 503, retry_after=embedding_budget.seconds_until_reset()
                    )
                result = service.ask(body.question, law_id=body.law_id, top_k=top_k, client_key=key)
                response = to_response(result)
            answers.put(cache_key, response)  # before finish(): waiters read it from the cache
        except GateFull:
            daily_per_client.refund(key)
            raise ServiceError("BUSY", 503, retry_after=5) from None
        except ServiceError as exc:
            # The per-client ask check ran before this block, so a DAILY_LIMIT_REACHED here is
            # the LLM allowance: like an exhausted budget, nothing was served.
            if exc.code in ("DAILY_BUDGET_EXHAUSTED", "DAILY_LIMIT_REACHED"):
                daily_per_client.refund(key)
            raise
        finally:
            if leader:
                flights.finish(cache_key)
        return response

    @app.get("/config", response_model=PublicConfigOut, tags=["config"])
    def public_config() -> PublicConfigOut:
        """What the page needs to know: the Turnstile site key (public) and gate threshold."""
        return PublicConfigOut(
            turnstile_site_key=settings.turnstile_site_key if captcha is not None else None,
            min_similarity=settings.min_similarity,
        )

    @app.get("/laws", response_model=list[LawOut], tags=["laws"])
    def list_laws() -> list[LawOut]:
        return [
            LawOut(
                law_id=law.law_id,
                name=law.name,
                short_name=law.short_name,
                number=law.number,
                year=law.year,
                version=law.version,
                source_url=law.source_url,
                retrieved=law.retrieved,
                article_count=len(law.articles),
                version_ar=law.version_ar,
                coverage_note_ar=law.coverage_note_ar,
            )
            for law in laws.values()
        ]

    @app.get("/laws/{law_id}/articles", response_model=list[ArticleSummaryOut], tags=["laws"])
    def list_articles(law_id: str) -> list[ArticleSummaryOut]:
        """Article keys and printed labels in document order for one law."""
        law = laws.get(law_id)
        if law is None:
            raise HTTPException(status_code=404)
        return [ArticleSummaryOut(article=a.key, article_label=a.label) for a in law.articles]

    @app.get("/laws/{law_id}/articles/{n}", response_model=ArticleOut, tags=["laws"])
    def get_article(law_id: str, n: str) -> ArticleOut:
        """Full text of one article. ``n`` is "6", "4bis" or "4 مكرر"."""
        law = laws.get(law_id)
        key = normalize_article_ref(n[:32])
        article = law.article(key) if law and key else None
        if law is None or article is None:
            raise HTTPException(status_code=404)
        prev_a, next_a = law.neighbours(article.key)
        return ArticleOut(
            law_id=law.law_id,
            law_name=law.name,
            article=article.key,
            article_label=article.label,
            text=article.text,
            notes=list(article.notes),
            source_url=law.source_url,
            prev_article=prev_a.key if prev_a else None,
            prev_article_label=prev_a.label if prev_a else None,
            next_article=next_a.key if next_a else None,
            next_article_label=next_a.label if next_a else None,
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index() -> FileResponse:
        # Its CSP (INDEX_CSP) is set by request_guard, like every other header.
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/docs", include_in_schema=False)
    def docs() -> HTMLResponse:
        return HTMLResponse(DOCS_HTML)

    @app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
    def favicon() -> FileResponse:
        # Browsers ask for /favicon.ico on their own; the page links the same SVG.
        return FileResponse(
            STATIC_DIR / "favicon.svg", media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"}
        )

    return app


def create_app_from_env() -> FastAPI:
    """Entry point for uvicorn --factory: configure logging, then build the app."""
    configure_logging()
    return create_app()
