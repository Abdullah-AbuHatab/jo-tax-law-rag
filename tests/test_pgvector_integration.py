"""pgvector integration tests. Run only when TEST_DATABASE_URL is set, e.g.

    docker run -d --rm -p 5433:5432 -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
    TEST_DATABASE_URL=postgresql://postgres:test@localhost:5433/postgres pytest -m integration

The tests drop and recreate the law_chunks table, create (then drop) the roles
rag_reader, rag_test_owner and rag_extra_group, and briefly make rag_test_owner
the owner of the database: never point this at real data.
"""

import base64
import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_law
from app.db import Database, apply_migrations
from app.embeddings import FakeEmbedder
from app.ingest import ingest_law, ingest_laws
from app.llm import FakeLLM
from app.main import create_app
from app.store import PgVectorStore
from tests.conftest import fake_settings
from tools import create_reader_role

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
    db = Database(URL, run_migrations=True)  # the owner path (ingest)
    yield db
    db.close()


def _tables_present() -> set[str]:
    with psycopg.connect(URL) as conn:
        rows = conn.execute(
            "SELECT relname FROM pg_class WHERE relname IN ('law_chunks', 'schema_migrations') AND relkind = 'r'"
        ).fetchall()
    return {row[0] for row in rows}


def _env_value(path, key: str) -> str:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(f"{key}=")]
    assert len(lines) == 1, lines
    return lines[0].split("=", 1)[1]


def test_migrations_are_applied_once(database):
    database.ping()  # first use applies them (run_migrations=True)
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


def test_web_does_not_run_migrations_by_default():
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS law_chunks")
        conn.execute("DROP TABLE IF EXISTS schema_migrations")
    db = Database(URL)
    try:
        db.ping()  # the pool opens, but nothing is created
    finally:
        db.close()
    assert _tables_present() == set()

    settings = fake_settings(vector_store="pgvector", database_url=URL, health_token="t")
    app = create_app(settings, embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app) as client:
        assert client.get("/health/db", headers={"X-Health-Token": "t"}).status_code == 200
    assert _tables_present() == set()


def test_reader_role_can_search_but_not_write(database, laws, tmp_path, capsys):
    store, embedder = PgVectorStore(database), FakeEmbedder()
    ingest_law(laws["sales_tax_6_1994"], embedder, store)
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING=keep\n", encoding="utf-8")
    try:
        assert create_reader_role.main(["--env-file", str(env_file)], env={"OWNER_DATABASE_URL": URL}) == 0
        reader_url = _env_value(env_file, "DATABASE_URL_READER")
        assert capsys.readouterr().out == "role ready\n"  # the URL and password never reach stdout
        assert "EXISTING=keep" in env_file.read_text(encoding="utf-8")
        assert reader_url.startswith("postgresql://rag_reader:")

        reader_db = Database(reader_url)  # run_migrations=False: the web default
        try:
            reader_store = PgVectorStore(reader_db)
            query = embedder.embed_query("الضريبة العامة على استيراد أي سلعة أو خدمة بنسبة 16%")
            assert len(reader_store.search(query, k=3)) == 3
        finally:
            reader_db.close()

        writes = [
            "INSERT INTO law_chunks (id, law_id, article, chunk_index, text, text_normalized, content_hash, embedding)"
            " SELECT 'x', law_id, '999', 0, text, text_normalized, content_hash, embedding FROM law_chunks LIMIT 1",
            "UPDATE law_chunks SET text = 'x'",
            "DELETE FROM law_chunks",
            "TRUNCATE law_chunks",
            "DROP TABLE law_chunks",
            "INSERT INTO schema_migrations (filename) VALUES ('999_evil.sql')",
            "CREATE TABLE evil (id int)",
        ]
        for statement in writes:
            with psycopg.connect(reader_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)

        # the web app starts and answers with the SELECT-only role
        settings = fake_settings(vector_store="pgvector", database_url=reader_url, health_token="t")
        app = create_app(settings, laws=laws, embedder=embedder, llm=FakeLLM())
        with TestClient(app) as client:
            assert client.get("/health/db", headers={"X-Health-Token": "t"}).status_code == 200
            response = client.post("/ask", json={"question": "ما هي نسبة الضريبة العامة على المبيعات؟"})
            assert response.status_code == 200 and response.json()["retrieved"]

        # a second run rotates the password and rewrites the same .env line
        assert create_reader_role.main(["--env-file", str(env_file)], env={"OWNER_DATABASE_URL": URL}) == 0
        rotated = _env_value(env_file, "DATABASE_URL_READER")
        assert rotated != reader_url
        with pytest.raises(psycopg.OperationalError):
            psycopg.connect(reader_url, connect_timeout=5).close()
        with psycopg.connect(rotated) as conn:
            assert conn.execute("SELECT count(*) FROM law_chunks").fetchone()[0] > 0
    finally:
        with psycopg.connect(URL, autocommit=True) as conn:
            conn.execute("DROP OWNED BY rag_reader")
            conn.execute("DROP ROLE IF EXISTS rag_reader")


def test_create_reader_role_refuses_without_owner_url_or_tables(database, tmp_path, capsys):
    env_file = tmp_path / ".env"
    assert create_reader_role.main(["--env-file", str(env_file)], env={}) == 2
    assert create_reader_role.main(["--env-file", str(env_file)], env={"OWNER_DATABASE_URL": URL}) == 2  # no tables
    assert not env_file.exists()
    captured = capsys.readouterr()
    assert URL not in captured.out + captured.err


# ---- round 2: reader role tool, as on Neon (a non-superuser owner) ---------------------------

OWNER_ROLE = "rag_test_owner"
OWNER_PASSWORD = "owner-local-test-only"


def _owner_url() -> str:
    parts = urlsplit(URL)
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, f"{OWNER_ROLE}:{OWNER_PASSWORD}@{host}", parts.path, parts.query, ""))


def _reader_attributes() -> tuple:
    with psycopg.connect(URL) as conn:
        return conn.execute(
            "SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolreplication, rolbypassrls,"
            " left(rolpassword, 20) FROM pg_authid WHERE rolname = 'rag_reader'"
        ).fetchone()


@pytest.fixture
def neon_like_owner(database, laws):
    """Tables and database owned by a LOGIN CREATEROLE CREATEDB role that is not a superuser."""
    ingest_law(laws["sales_tax_6_1994"], FakeEmbedder(), PgVectorStore(database))
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"DROP ROLE IF EXISTS {OWNER_ROLE}")
        conn.execute(f"CREATE ROLE {OWNER_ROLE} LOGIN CREATEROLE CREATEDB PASSWORD '{OWNER_PASSWORD}'")
        conn.execute(f"ALTER DATABASE postgres OWNER TO {OWNER_ROLE}")
        conn.execute(f"ALTER TABLE law_chunks OWNER TO {OWNER_ROLE}")
        conn.execute(f"ALTER TABLE schema_migrations OWNER TO {OWNER_ROLE}")
    yield _owner_url()
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("ALTER DATABASE postgres OWNER TO postgres")
        conn.execute("ALTER TABLE IF EXISTS law_chunks OWNER TO postgres")
        conn.execute("ALTER TABLE IF EXISTS schema_migrations OWNER TO postgres")
        conn.execute("DROP ROLE IF EXISTS rag_extra_group")
        for role in ("rag_reader", OWNER_ROLE):
            if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
                conn.execute(f"DROP OWNED BY {role}")
                conn.execute(f"DROP ROLE {role}")


def test_scram_verifier_matches_libpq():
    password = create_reader_role.new_password()
    with psycopg.connect(URL) as conn:
        from_libpq = conn.pgconn.encrypt_password(password.encode(), b"rag_reader", b"scram-sha-256").decode()
    iterations, rest = from_libpq.removeprefix("SCRAM-SHA-256$").split(":", 1)
    salt = base64.b64decode(rest.split("$", 1)[0])
    assert create_reader_role.scram_sha256_verifier(password, salt=salt, iterations=int(iterations)) == from_libpq


def test_reader_role_tool_enforces_least_privilege_as_a_non_superuser_owner(neon_like_owner, tmp_path, capsys):
    env_file = tmp_path / ".env"
    env = {"OWNER_DATABASE_URL": neon_like_owner}
    # a role created by another role (here the superuser): this owner has no ADMIN OPTION on it
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("CREATE ROLE rag_reader LOGIN PASSWORD 'old-password'")
    assert create_reader_role.main(["--env-file", str(env_file)], env=env) == 2
    assert "ADMIN OPTION" in capsys.readouterr().err and not env_file.exists()
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("DROP ROLE rag_reader")

    # a pre-existing rag_reader (created by this owner) with more than it should have
    with psycopg.connect(neon_like_owner, autocommit=True) as conn:
        conn.execute("CREATE ROLE rag_reader LOGIN CREATEDB PASSWORD 'old-password'")
        conn.execute("GRANT INSERT, UPDATE, DELETE ON law_chunks TO rag_reader")
        conn.execute("GRANT CREATE ON SCHEMA public TO rag_reader")
    assert create_reader_role.main(["--env-file", str(env_file)], env=env) == 0
    assert capsys.readouterr().out == "role ready\n"
    login, superuser, createdb, createrole, replication, bypassrls, stored = _reader_attributes()
    assert login and not (superuser or createdb or createrole or replication or bypassrls)
    assert stored.startswith("SCRAM-SHA-256$4096:")
    with psycopg.connect(URL) as conn:
        assert conn.execute("SELECT rolconnlimit FROM pg_roles WHERE rolname = 'rag_reader'").fetchone()[0] == 20
        settings = conn.execute(
            "SELECT s.setconfig FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole"
            " WHERE r.rolname = 'rag_reader' AND s.setdatabase = 0"
        ).fetchone()[0]
    assert sorted(settings) == ["idle_in_transaction_session_timeout=10s", "statement_timeout=5s"]
    reader_url = _env_value(env_file, "DATABASE_URL_READER")
    with psycopg.connect(reader_url) as conn:  # the generated password logs in
        assert conn.execute("SELECT count(*) FROM law_chunks").fetchone()[0] > 0
        assert conn.execute("SHOW statement_timeout").fetchone()[0] == "5s"  # applied to every session
        assert conn.execute("SHOW idle_in_transaction_session_timeout").fetchone()[0] == "10s"
    for statement in (
        "INSERT INTO law_chunks (id, law_id, article, chunk_index, text, text_normalized, content_hash, embedding)"
        " SELECT 'x', law_id, '9', 0, text, text_normalized, content_hash, embedding FROM law_chunks LIMIT 1",
        "UPDATE law_chunks SET text = 'x'",
        "DELETE FROM law_chunks",
        "CREATE TABLE evil (id int)",
    ):
        with psycopg.connect(reader_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(statement)

    # rotation by the same non-superuser owner works (T4 sent NOSUPERUSER and PostgreSQL refused it)
    assert create_reader_role.main(["--env-file", str(env_file)], env=env) == 0
    rotated = _env_value(env_file, "DATABASE_URL_READER")
    assert rotated != reader_url
    with psycopg.connect(rotated) as conn:
        conn.execute("SELECT 1")

    # an attribute only a superuser may remove, or a role membership: refuse, change nothing
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("ALTER ROLE rag_reader REPLICATION")
    capsys.readouterr()
    assert create_reader_role.main(["--env-file", str(env_file)], env=env) == 2
    assert "REPLICATION" in capsys.readouterr().err
    assert _env_value(env_file, "DATABASE_URL_READER") == rotated
    with psycopg.connect(rotated) as conn:  # rolled back: the previous password still works
        conn.execute("SELECT 1")
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("ALTER ROLE rag_reader NOREPLICATION")
        conn.execute("CREATE ROLE rag_extra_group NOLOGIN")
        conn.execute("GRANT rag_extra_group TO rag_reader")
    assert create_reader_role.main(["--env-file", str(env_file)], env=env) == 2
    assert "rag_extra_group" in capsys.readouterr().err
