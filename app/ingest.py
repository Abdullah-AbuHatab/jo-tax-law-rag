"""Ingest the cleaned laws into the vector store.

    python -m app.ingest                 # all laws, real embeddings, DATABASE_URL
    python -m app.ingest --law sales_tax_6_1994
    python -m app.ingest --dry-run       # parse + chunk only, no DB, no API calls

Idempotent: chunk ids are deterministic and each row stores a content hash
(model + dimension + embedded text). A re-run embeds only new or changed
chunks, leaves unchanged rows alone and deletes rows whose chunk no longer
exists (e.g. an article that is now split differently).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from app.chunking import chunk_law, content_hash
from app.config import EMBEDDING_DIMENSION, Settings
from app.embeddings import Embedder
from app.logs import configure_logging
from app.parsing import Law, load_laws
from app.store import StoredChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LawIngestReport:
    law_id: str
    articles: int
    chunks: int
    embedded: int
    unchanged: int
    deleted: int


def ingest_law(law: Law, embedder: Embedder, store: VectorStore, batch_size: int = 50) -> LawIngestReport:
    chunks = chunk_law(law)
    hashes = {c.id: content_hash(c.embedding_text, embedder.model, embedder.dimension) for c in chunks}
    existing = store.existing_hashes(law.law_id)

    to_embed = [c for c in chunks if existing.get(c.id) != hashes[c.id]]
    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        vectors = embedder.embed_documents([c.embedding_text for c in batch])
        store.upsert(
            [StoredChunk(chunk=c, embedding=v, content_hash=hashes[c.id]) for c, v in zip(batch, vectors, strict=True)]
        )

    stale = sorted(set(existing) - set(hashes))
    store.delete(stale)
    return LawIngestReport(
        law_id=law.law_id,
        articles=len(law.articles),
        chunks=len(chunks),
        embedded=len(to_embed),
        unchanged=len(chunks) - len(to_embed),
        deleted=len(stale),
    )


def ingest_laws(laws: Mapping[str, Law], embedder: Embedder, store: VectorStore) -> list[LawIngestReport]:
    if embedder.dimension != EMBEDDING_DIMENSION:
        raise ValueError(f"embedder dimension {embedder.dimension} != table dimension {EMBEDDING_DIMENSION}")
    return [ingest_law(law, embedder, store) for law in laws.values()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingest", description="Embed and store the cleaned laws.")
    parser.add_argument("--law", help="only this law_id")
    parser.add_argument("--dry-run", action="store_true", help="parse and chunk only; no database, no API calls")
    args = parser.parse_args(argv)
    configure_logging()

    settings = Settings.from_env()
    laws = load_laws(settings.data_dir)
    if args.law:
        if args.law not in laws:
            print(f"unknown law_id {args.law!r}; known: {', '.join(laws)}", file=sys.stderr)
            return 2
        laws = {args.law: laws[args.law]}

    if args.dry_run:
        for law in laws.values():
            chunks = chunk_law(law)
            split = sum(1 for c in chunks if c.chunk_index == 1)
            print(f"{law.law_id}: {len(law.articles)} articles -> {len(chunks)} chunks ({split} articles split)")
        return 0

    # Imported here so --dry-run works without any provider configuration.
    from app.dependencies import build_embedder, build_vector_store

    embedder = build_embedder(settings)
    # Ingest is the owner path: it always applies migrations, whatever RUN_MIGRATIONS says.
    store = build_vector_store(settings.model_copy(update={"run_migrations": True}))
    for report in ingest_laws(laws, embedder, store):
        print(
            f"{report.law_id}: {report.articles} articles, {report.chunks} chunks, "
            f"{report.embedded} embedded, {report.unchanged} unchanged, {report.deleted} deleted"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
