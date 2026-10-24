"""FastAPI application.

    uvicorn app.main:create_app_from_env --factory --port 8000

``create_app`` takes optional collaborators so tests can pass fakes; in
production everything is built from environment variables. Building the
app never opens a network connection: the database pool opens on first use.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.answering import AnswerResult, AnswerService, ServiceError
from app.config import Settings
from app.db import DatabaseUnavailable
from app.dependencies import build_database, build_embedder, build_llm, build_vector_store
from app.embeddings import Embedder
from app.ingest import ingest_laws
from app.llm import LLMClient
from app.logs import configure_logging
from app.parsing import Law, load_laws, normalize_article_ref
from app.rate_limit import SlidingWindowRateLimiter, client_ip
from app.schemas import (
    ArticleOut,
    AskRequest,
    AskResponse,
    CitationOut,
    DroppedCitationOut,
    ErrorResponse,
    LawOut,
    RetrievedOut,
)
from app.store import InMemoryStore, VectorStore

logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 16 * 1024
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Fixed client-facing messages per error code; upstream text is never forwarded.
ERROR_MESSAGES = {
    "VALIDATION_ERROR": "The request is invalid.",
    "UNKNOWN_LAW": "Unknown law_id. See GET /laws for the available laws.",
    "NOT_FOUND": "Not found.",
    "RATE_LIMITED": "Too many requests. Please wait and try again.",
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
) -> FastAPI:
    settings = settings or Settings.from_env()
    laws = dict(laws) if laws is not None else load_laws(settings.data_dir)
    database = build_database(settings) if store is None and settings.vector_store == "pgvector" else None
    embedder = embedder or build_embedder(settings)
    store = store or build_vector_store(settings, database)
    if llm is None:
        llm = build_llm(settings)  # None when no key: /ask then answers 503 LLM_CONFIG_ERROR
    service = AnswerService(embedder, store, llm, laws, settings.min_similarity, settings.default_top_k)
    limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute, window_seconds=60)

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
        title="Jordan Tax Law RAG",
        version="1.0.0",
        description=(
            "Arabic question answering over Jordanian tax laws with article-level citations. Not legal advice."
        ),
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def request_guard(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        # accept a caller's id only if it is short and plain, so it is safe to log
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:
            return error_response("PAYLOAD_TOO_LARGE", 413)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        body = {"error": {"code": "VALIDATION_ERROR", "message": f"{field}: {first.get('msg', 'invalid')}"}}
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 429: "RATE_LIMITED", 413: "PAYLOAD_TOO_LARGE"}.get(exc.status_code)
        if code is None:
            return JSONResponse({"error": {"code": "HTTP_ERROR", "message": str(exc.detail)}}, exc.status_code)
        return error_response(code, exc.status_code, headers=exc.headers)

    @app.exception_handler(ServiceError)
    async def service_handler(_: Request, exc: ServiceError) -> JSONResponse:
        return error_response(exc.code, exc.http_status)

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return error_response("INTERNAL_ERROR", 500)

    def rate_limited(request: Request) -> None:
        ip = client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            settings.trusted_proxy_hops,
        )
        allowed, retry_after = limiter.allow(ip)
        if not allowed:
            raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after)})

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        """Liveness: the process is up. Does not touch the database."""
        return {"status": "ok"}

    @app.get("/health/db", tags=["health"], responses={503: {"model": ErrorResponse}})
    def health_db():
        """Readiness of the database: a real round trip, 503 when it is down."""
        try:
            store.ping()
        except DatabaseUnavailable:
            return error_response("DATABASE_UNAVAILABLE", 503)
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, tags=["ask"], dependencies=[Depends(rate_limited)])
    def ask(body: AskRequest) -> AskResponse:
        """Answer from the laws, with validated article citations, or refuse."""
        if body.law_id is not None and body.law_id not in laws:
            raise ServiceError("UNKNOWN_LAW", 422)
        return to_response(service.ask(body.question, law_id=body.law_id, top_k=body.top_k))

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
            )
            for law in laws.values()
        ]

    @app.get("/laws/{law_id}/articles/{n}", response_model=ArticleOut, tags=["laws"])
    def get_article(law_id: str, n: str) -> ArticleOut:
        """Full text of one article. ``n`` is "6", "4bis" or "4 مكرر"."""
        law = laws.get(law_id)
        key = normalize_article_ref(n[:32])
        article = law.article(key) if law and key else None
        if law is None or article is None:
            raise HTTPException(status_code=404)
        return ArticleOut(
            law_id=law.law_id,
            law_name=law.name,
            article=article.key,
            article_label=article.label,
            text=article.text,
            notes=list(article.notes),
            source_url=law.source_url,
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
                ),
                "Cache-Control": "no-cache",
            },
        )

    return app


def create_app_from_env() -> FastAPI:
    """Entry point for uvicorn --factory: configure logging, then build the app."""
    configure_logging()
    return create_app()
