"""Postgres access: lazy connection pool and plain-SQL migrations.

Nothing here runs at import or at app start. The first request that needs
the database opens the pool; if Neon is cold or unreachable that request
gets ``DatabaseUnavailable`` (the API turns it into a 503) and the next
request tries again. The app therefore always starts, even with no DB.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout

from app.config import REPO_ROOT

logger = logging.getLogger(__name__)
MIGRATIONS_DIR = REPO_ROOT / "migrations"
# Any constant works; it only has to be the same for every process.
_MIGRATION_LOCK_ID = 7_112_014


class DatabaseUnavailable(Exception):
    """The database could not be reached or did not answer in time."""


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply migrations/*.sql in name order, each once, inside one transaction.

    An advisory lock makes concurrent runners (two app instances, or the app
    and ``python -m app.ingest``) wait for each other instead of racing.
    """
    applied_now: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " filename text PRIMARY KEY,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in done:
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            applied_now.append(path.name)
    return applied_now


class Database:
    def __init__(self, url: str, connect_timeout_seconds: float = 5.0, pool_max_size: int = 4) -> None:
        self._url = url
        self._connect_timeout = connect_timeout_seconds
        self._pool_max_size = pool_max_size
        self._pool: ConnectionPool | None = None
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._url)

    def _open_pool(self) -> ConnectionPool:
        with self._lock:
            if self._pool is not None:
                return self._pool
            if not self._url:
                raise DatabaseUnavailable("DATABASE_URL is not set")
            try:
                # Migrations first, on a plain connection: register_vector() below
                # needs the vector extension to exist already.
                with psycopg.connect(
                    self._url, connect_timeout=int(self._connect_timeout), prepare_threshold=None
                ) as conn:
                    applied = apply_migrations(conn)
                    if applied:
                        logger.info("applied migrations", extra={"migrations": applied})
                pool = ConnectionPool(
                    self._url,
                    min_size=1,
                    max_size=self._pool_max_size,
                    # autocommit: reads need no transaction, writes open one explicitly
                    # (conn.transaction()), and register_vector leaves no open transaction.
                    # prepare_threshold=None: no server-side prepared statements, which a
                    # transaction-mode pooler (Neon's "-pooler" endpoint) may not keep.
                    kwargs={
                        "connect_timeout": int(self._connect_timeout),
                        "autocommit": True,
                        "prepare_threshold": None,
                    },
                    configure=register_vector,
                    open=False,
                )
                pool.open(wait=True, timeout=self._connect_timeout * 2)
            except (psycopg.Error, PoolTimeout, OSError) as exc:
                logger.warning("database unavailable", extra={"error_type": type(exc).__name__})
                raise DatabaseUnavailable("database is unavailable") from exc
            self._pool = pool
            return pool

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        pool = self._open_pool()
        try:
            with pool.connection(timeout=self._connect_timeout) as conn:
                yield conn
        except (psycopg.OperationalError, PoolTimeout) as exc:
            logger.warning("database call failed", extra={"error_type": type(exc).__name__})
            raise DatabaseUnavailable("database is unavailable") from exc

    def ping(self) -> None:
        with self.connection() as conn:
            conn.execute("SELECT 1")

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None
