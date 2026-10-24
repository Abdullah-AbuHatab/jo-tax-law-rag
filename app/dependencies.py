"""Build the app's collaborators from Settings (one place, no hidden globals)."""

from __future__ import annotations

from app.config import EMBEDDING_DIMENSION, Settings
from app.db import Database
from app.embeddings import Embedder, FakeEmbedder, GeminiEmbedder
from app.llm import FakeLLM, LLMClient, OpenRouterClient
from app.store import InMemoryStore, PgVectorStore, VectorStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder(dimension=EMBEDDING_DIMENSION)
    return GeminiEmbedder(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_embed_model,
        dimension=EMBEDDING_DIMENSION,
        batch_size=settings.gemini_embed_batch_size,
        timeout_seconds=settings.gemini_timeout_seconds,
    )


def build_llm(settings: Settings) -> LLMClient | None:
    """None when no key is configured: /ask then answers 503 LLM_CONFIG_ERROR."""
    if settings.llm == "fake":
        return FakeLLM()
    key = settings.openrouter_api_key.get_secret_value()
    if not key:
        return None
    return OpenRouterClient(
        api_key=key,
        model=settings.openrouter_model,
        fallback_models=settings.openrouter_fallback_models,
        timeout_seconds=settings.openrouter_timeout_seconds,
        reasoning_effort=settings.openrouter_reasoning_effort,
        app_url=settings.openrouter_app_url,
        app_title=settings.openrouter_app_title,
    )


def build_database(settings: Settings) -> Database:
    return Database(
        settings.database_url.get_secret_value(),
        connect_timeout_seconds=settings.db_connect_timeout_seconds,
        pool_max_size=settings.db_pool_max_size,
    )


def build_vector_store(settings: Settings, database: Database | None = None) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryStore()
    return PgVectorStore(database or build_database(settings))
