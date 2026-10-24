"""pgvector integration tests. Run only when TEST_DATABASE_URL is set, e.g.

    docker run -d --rm -p 5433:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:test@localhost:5433/postgres pytest -m integration

The tests drop and recreate the law_chunks table: never point this at real data.
"""

import os

import psycopg
import pytest

from app.chunking import chunk_law
from app.db import Database, apply_migrations
from app.embeddings import FakeEmbedder
from app.ingest import ingest_law, ingest_laws
from app.store import PgVectorStore

URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set"),
]


@pytest.fixture
def database():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS law_chunks")
        conn.execute("DROP TABLE IF EXISTS schema_migrations")
    db = Database(URL)
    yield db
    db.close()


def test_migrations_are_applied_once(database):
    database.ping()  # first use applies them
    with psycopg.connect(URL) as conn:
        assert apply_migrations(conn) == []  # nothing left to apply
        index = conn.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'law_chunks_embedding_hnsw'").fetchone()
    assert index is not None and "hnsw" in index[0] and "vector_cosine_ops" in index[0]


def test_ingest_search_and_filter(database, laws):
    store, embedder = PgVectorStore(database), FakeEmbedder()
    reports = ingest_laws(laws, embedder, store)
    expected_rows = sum(len(chunk_law(law)) for law in laws.values())
    assert sum(r.embedded for r in reports) == expected_rows

    query = embedder.embed_query("الضريبة العامة على استيراد أي سلعة أو خدمة بنسبة 16%")
    results = store.search(query, k=5)
    assert len(results) == 5
    assert results == sorted(results, key=lambda r: -r.score)
    assert all(-1.0 <= r.score <= 1.0 for r in results)

    filtered = store.search(query, k=5, law_id="income_tax_34_2014")
    assert len(filtered) == 5
    assert {r.law_id for r in filtered} == {"income_tax_34_2014"}
    assert filtered[0].metadata["law_name"].startswith("قانون ضريبة الدخل")


def test_same_ranking_as_the_in_memory_store(database, laws, filled_store, embedder):
    store = PgVectorStore(database)
    ingest_laws(laws, embedder, store)
    query = embedder.embed_query("الإعفاء الشخصي للشخص الطبيعي المقيم")
    in_db = [r.id for r in store.search(query, k=3)]
    in_memory = [r.id for r in filled_store.search(query, k=3)]
    assert in_db == in_memory


def test_reingest_is_idempotent_and_deletes_stale_rows(database, laws):
    store, embedder = PgVectorStore(database), FakeEmbedder()
    law = laws["sales_tax_6_1994"]
    ingest_law(law, embedder, store)
    again = ingest_law(law, embedder, store)
    assert again.embedded == 0 and again.unchanged == again.chunks

    store.upsert([])  # no-op
    store.delete(["sales_tax_6_1994:77:0"])
    repaired = ingest_law(law, embedder, store)
    assert repaired.embedded == 1

    with psycopg.connect(URL) as conn:
        conn.execute(
            "INSERT INTO law_chunks (id, law_id, article, chunk_index, text, text_normalized, content_hash, embedding)"
            " SELECT 'sales_tax_6_1994:999:0', law_id, '999', 0, text, text_normalized, content_hash, embedding"
            " FROM law_chunks WHERE id = 'sales_tax_6_1994:1:0'"
        )
    cleaned = ingest_law(law, embedder, store)
    assert cleaned.deleted == 1
    assert "sales_tax_6_1994:999:0" not in store.existing_hashes(law.law_id)
