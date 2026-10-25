"""Application settings, read once from environment variables.

Plain pydantic model + ``os.environ`` on purpose: every setting is visible in
one place, with its default, and nothing is read implicitly elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIMENSION = 768  # must match vector(768) in migrations/001_init.sql


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseModel):
    # Providers. The defaults are the real services; "fake"/"memory" give an
    # offline demo (EMBEDDER=fake LLM=fake VECTOR_STORE=memory) that needs no keys.
    embedder: Literal["gemini", "fake"] = "gemini"
    llm: Literal["openrouter", "fake"] = "openrouter"
    vector_store: Literal["pgvector", "memory"] = "pgvector"

    # Database (Neon / local pgvector). Empty means "no database configured".
    database_url: SecretStr = SecretStr("")
    db_connect_timeout_seconds: float = 5.0
    db_pool_max_size: int = 4
    # The web service connects with a SELECT-only role and must not migrate;
    # python -m app.ingest (the owner path) always applies migrations.
    run_migrations: bool = False

    # Embeddings (Gemini REST)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_batch_size: int = Field(default=50, ge=1, le=100)
    gemini_timeout_seconds: float = 15.0

    # Chat completions (OpenRouter REST)
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_fallback_models: list[str] = ["cohere/north-mini-code:free", "google/gemma-4-31b-it:free"]
    openrouter_timeout_seconds: float = 20.0
    openrouter_reasoning_effort: str = "low"  # "" disables the reasoning hint
    openrouter_app_url: str = "https://github.com/Abdullah-AbuHatab/jo-tax-law-rag"
    openrouter_app_title: str = "jo-tax-law-rag"

    # Retrieval and answering
    min_similarity: float = Field(default=0.60, ge=-1.0, le=1.0)
    default_top_k: int = Field(default=5, ge=1, le=10)
    # Upstream time one /ask may use in total: the query embedding has at most
    # GEMINI_TIMEOUT_SECONDS (2 attempts), and the LLM gets what is left, at most
    # OPENROUTER_TIMEOUT_SECONDS. 35 = 15 + 20.
    ask_deadline_seconds: float = Field(default=35.0, ge=5.0, le=120.0)

    # HTTP layer
    cors_allow_origins: list[str] = []
    rate_limit_per_minute: int = Field(default=10, ge=1)
    # A header the edge overwrites with the real client IP (production:
    # cf-connecting-ip). Empty = not used; then trusted_proxy_hops applies.
    client_ip_header: str = Field(default="", max_length=64)
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)
    # /health/db answers only with header X-Health-Token equal to this; empty = always 404.
    health_token: SecretStr = SecretStr("")

    # Abuse and cost guards (app/guards.py). Daily counters reset at 00:00 UTC.
    daily_ask_limit_per_client: int = Field(default=30, ge=1)
    # LLM calls per client per day (cache hits and low-score refusals do not count), so one
    # client cannot use up DAILY_LLM_BUDGET alone; rotating addresses is the accepted residue.
    daily_llm_calls_per_client: int = Field(default=5, ge=1)
    # Below the providers' own free-tier caps, so this app runs out before the key does:
    # Gemini gemini-embedding-001 free tier is reportedly ~1,000 requests a day, and
    # OpenRouter's free models allow ~50 a day without purchased credits.
    daily_embedding_budget: int = Field(default=800, ge=0)  # query embeddings per day, all clients
    daily_llm_budget: int = Field(default=45, ge=0)  # LLM calls per day, all clients
    max_concurrent_asks: int = Field(default=4, ge=1, le=64)
    answer_cache_size: int = Field(default=500, ge=0)  # 0 disables the cache
    answer_cache_ttl_seconds: int = Field(default=86400, ge=1)

    # Optional Cloudflare Turnstile on /ask: on only when both keys are set.
    turnstile_site_key: str = Field(default="", max_length=128)
    turnstile_secret_key: SecretStr = SecretStr("")

    data_dir: Path = REPO_ROOT / "data" / "clean"

    @property
    def turnstile_enabled(self) -> bool:
        return bool(self.turnstile_site_key and self.turnstile_secret_key.get_secret_value())

    @field_validator("client_ip_header")
    @classmethod
    def _header_name(cls, value: str) -> str:
        value = value.strip().lower()
        if value and not all(ch.isascii() and (ch.isalnum() or ch == "-") for ch in value):
            raise ValueError("CLIENT_IP_HEADER must be a plain header name, e.g. cf-connecting-ip")
        return value

    @model_validator(mode="after")
    def _memory_store_needs_fake_embedder(self) -> Settings:
        # The memory store is filled at start-up; with Gemini that would spend
        # API quota on every restart, so only the fake embedder may use it.
        if self.vector_store == "memory" and self.embedder != "fake":
            raise ValueError("VECTOR_STORE=memory requires EMBEDDER=fake")
        # Half a Turnstile configuration is a mistake, not "off": refuse to start.
        if bool(self.turnstile_site_key) != bool(self.turnstile_secret_key.get_secret_value()):
            raise ValueError("set both TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY, or neither")
        return self

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ if env is None else env)
        values: dict[str, object] = {}
        simple = {
            "EMBEDDER": "embedder",
            "LLM": "llm",
            "VECTOR_STORE": "vector_store",
            "DATABASE_URL": "database_url",
            "DB_CONNECT_TIMEOUT_SECONDS": "db_connect_timeout_seconds",
            "DB_POOL_MAX_SIZE": "db_pool_max_size",
            "RUN_MIGRATIONS": "run_migrations",
            "GEMINI_API_KEY": "gemini_api_key",
            "GEMINI_EMBED_MODEL": "gemini_embed_model",
            "GEMINI_EMBED_BATCH_SIZE": "gemini_embed_batch_size",
            "GEMINI_TIMEOUT_SECONDS": "gemini_timeout_seconds",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OPENROUTER_MODEL": "openrouter_model",
            "OPENROUTER_TIMEOUT_SECONDS": "openrouter_timeout_seconds",
            "OPENROUTER_REASONING_EFFORT": "openrouter_reasoning_effort",
            "OPENROUTER_APP_URL": "openrouter_app_url",
            "OPENROUTER_APP_TITLE": "openrouter_app_title",
            "MIN_SIMILARITY": "min_similarity",
            "DEFAULT_TOP_K": "default_top_k",
            "ASK_DEADLINE_SECONDS": "ask_deadline_seconds",
            "RATE_LIMIT_PER_MINUTE": "rate_limit_per_minute",
            "TRUSTED_PROXY_HOPS": "trusted_proxy_hops",
            "CLIENT_IP_HEADER": "client_ip_header",
            "HEALTH_TOKEN": "health_token",
            "TURNSTILE_SITE_KEY": "turnstile_site_key",
            "TURNSTILE_SECRET_KEY": "turnstile_secret_key",
            "DAILY_ASK_LIMIT_PER_CLIENT": "daily_ask_limit_per_client",
            "DAILY_LLM_CALLS_PER_CLIENT": "daily_llm_calls_per_client",
            "DAILY_EMBEDDING_BUDGET": "daily_embedding_budget",
            "DAILY_LLM_BUDGET": "daily_llm_budget",
            "MAX_CONCURRENT_ASKS": "max_concurrent_asks",
            "ANSWER_CACHE_SIZE": "answer_cache_size",
            "ANSWER_CACHE_TTL_SECONDS": "answer_cache_ttl_seconds",
            "DATA_DIR": "data_dir",
        }
        for env_name, field_name in simple.items():
            if env.get(env_name, "") != "":
                values[field_name] = env[env_name]
        if "OPENROUTER_FALLBACK_MODELS" in env:
            values["openrouter_fallback_models"] = _list(env["OPENROUTER_FALLBACK_MODELS"])
        if env.get("CORS_ALLOW_ORIGINS"):
            values["cors_allow_origins"] = _list(env["CORS_ALLOW_ORIGINS"])
        return cls.model_validate(values)
