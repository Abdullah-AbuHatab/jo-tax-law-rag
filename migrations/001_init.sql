-- 001_init.sql: chunk table for article-level retrieval.
-- Applied once by app/db.py:apply_migrations (tracked in schema_migrations).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS law_chunks (
    id              text PRIMARY KEY,           -- "{law_id}:{article}:{chunk_index}"
    law_id          text NOT NULL,
    article         text NOT NULL,              -- article key: "6" or "4bis"
    chunk_index     integer NOT NULL,
    text            text NOT NULL,              -- original text, for display
    text_normalized text NOT NULL,              -- normalized Arabic, for matching
    content_hash    text NOT NULL,              -- sha256(model, dimension, embedded text)
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(768) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT law_chunks_piece_unique UNIQUE (law_id, article, chunk_index)
);

-- Cosine distance (<=>) is what app/store.py orders by.
CREATE INDEX IF NOT EXISTS law_chunks_embedding_hnsw
    ON law_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS law_chunks_law_id_idx ON law_chunks (law_id);
