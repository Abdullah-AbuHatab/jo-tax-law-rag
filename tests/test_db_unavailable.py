"""The app must start and degrade to 503s when the database is unreachable (cold Neon)."""

import pytest
from fastapi.testclient import TestClient

from app.db import Database, DatabaseUnavailable
from app.embeddings import FakeEmbedder
from app.llm import FakeLLM
from app.main import create_app
from tests.conftest import fake_settings

UNREACHABLE = "postgresql://nobody:nothing@127.0.0.1:1/none"  # port 1: connection refused at once


def test_unreachable_database_raises_database_unavailable_every_time():
    db = Database(UNREACHABLE, connect_timeout_seconds=1)
    for _ in range(2):  # the second call retries instead of caching a broken pool
        with pytest.raises(DatabaseUnavailable):
            db.ping()


def test_unconfigured_database_is_unavailable_not_a_crash():
    with pytest.raises(DatabaseUnavailable):
        Database("").ping()


def test_app_starts_without_database_and_returns_503s(laws):
    settings = fake_settings(vector_store="pgvector", database_url=UNREACHABLE, db_connect_timeout_seconds=1)
    app = create_app(settings, laws=laws, embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/laws").status_code == 200  # served from the cleaned files
        db_health = client.get("/health/db")
        assert db_health.status_code == 503
        assert db_health.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
        ask = client.post("/ask", json={"question": "ما هي نسبة الضريبة العامة على المبيعات؟"})
        assert ask.status_code == 503
        assert ask.json()["error"]["code"] == "DATABASE_UNAVAILABLE"
