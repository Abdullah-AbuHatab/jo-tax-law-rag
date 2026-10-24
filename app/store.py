"""Vector stores with one interface: pgvector for real use, in-memory for tests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.chunking import Chunk
from app.db import Database


@dataclass(frozen=True)
class StoredChunk:
    """A chunk ready to be written: the chunk, its vector and its content hash."""

    chunk: Chunk
    embedding: list[float]
    content_hash: str


@dataclass(frozen=True)
class SearchResult:
    id: str
    law_id: str
    article: str
    chunk_index: int
    text: str
    score: float  # cosine similarity, 1.0 = same direction
    metadata: dict[str, object] = field(default_factory=dict, hash=False, compare=False)


class VectorStore(Protocol):
    def existing_hashes(self, law_id: str) -> dict[str, str]: ...

    def upsert(self, items: Sequence[StoredChunk]) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def search(self, query_vec: Sequence[float], k: int, law_id: str | None = None) -> list[SearchResult]: ...

    def ping(self) -> None: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryStore:
    """Same behaviour as PgVectorStore (exact search, no index). For tests and --fake eval."""

    def __init__(self) -> None:
        self._rows: dict[str, StoredChunk] = {}
        self.ping_error: Exception | None = None  # tests set this to simulate an outage

    def existing_hashes(self, law_id: str) -> dict[str, str]:
        return {i: r.content_hash for i, r in self._rows.items() if r.chunk.law_id == law_id}

    def upsert(self, items: Sequence[StoredChunk]) -> None:
        for item in items:
            self._rows[item.chunk.id] = item

    def delete(self, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            self._rows.pop(chunk_id, None)

    def search(self, query_vec: Sequence[float], k: int, law_id: str | None = None) -> list[SearchResult]:
        if self.ping_error:
            raise self.ping_error
        rows = [r for r in self._rows.values() if law_id is None or r.chunk.law_id == law_id]
        scored = sorted(
            ((cosine_similarity(query_vec, r.embedding), r) for r in rows),
            key=lambda pair: (-pair[0], pair[1].chunk.id),
        )
        return [
            SearchResult(
                id=r.chunk.id,
                law_id=r.chunk.law_id,
                article=r.chunk.article,
                chunk_index=r.chunk.chunk_index,
                text=r.chunk.text,
                score=score,
                metadata=dict(r.chunk.metadata),
            )
            for score, r in scored[:k]
        ]

    def ping(self) -> None:
        if self.ping_error:
            raise self.ping_error

    def __len__(self) -> int:
        return len(self._rows)


class PgVectorStore:
    def __init__(self, database: Database, hnsw_ef_search: int = 100) -> None:
        self._db = database
        self._ef_search = hnsw_ef_search

    def existing_hashes(self, law_id: str) -> dict[str, str]:
        with self._db.connection() as conn:
            rows = conn.execute("SELECT id, content_hash FROM law_chunks WHERE law_id = %s", (law_id,)).fetchall()
        return {row[0]: row[1] for row in rows}

    def upsert(self, items: Sequence[StoredChunk]) -> None:
        if not items:
            return
        sql = """
            INSERT INTO law_chunks
                (id, law_id, article, chunk_index, text, text_normalized, content_hash, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                law_id = EXCLUDED.law_id,
                article = EXCLUDED.article,
                chunk_index = EXCLUDED.chunk_index,
                text = EXCLUDED.text,
                text_normalized = EXCLUDED.text_normalized,
                content_hash = EXCLUDED.content_hash,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        params = [
            (
                item.chunk.id,
                item.chunk.law_id,
                item.chunk.article,
                item.chunk.chunk_index,
                item.chunk.text,
                item.chunk.text_normalized,
                item.content_hash,
                Jsonb(item.chunk.metadata),
                Vector(item.embedding),
            )
            for item in items
        ]
        with self._db.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.executemany(sql, params)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self._db.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM law_chunks WHERE id = ANY(%s)", (list(ids),))

    def search(self, query_vec: Sequence[float], k: int, law_id: str | None = None) -> list[SearchResult]:
        # HNSW filters *after* the index scan; a larger ef_search keeps a
        # filtered query (law_id) from coming back with fewer than k rows.
        sql = """
            SELECT id, law_id, article, chunk_index, text, metadata, 1 - (embedding <=> %(q)s) AS score
            FROM law_chunks
            WHERE (%(law_id)s::text IS NULL OR law_id = %(law_id)s::text)
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
        """
        with self._db.connection() as conn, conn.transaction():
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(self._ef_search)}")
            rows = conn.execute(sql, {"q": Vector(list(query_vec)), "law_id": law_id, "k": k}).fetchall()
        return [
            SearchResult(
                id=row[0],
                law_id=row[1],
                article=row[2],
                chunk_index=row[3],
                text=row[4],
                metadata=row[5] or {},
                score=float(row[6]),
            )
            for row in rows
        ]

    def ping(self) -> None:
        self._db.ping()
