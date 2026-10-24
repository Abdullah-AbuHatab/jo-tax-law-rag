"""Chunk ids/hashes and idempotent re-ingest."""

from dataclasses import replace

from app.chunking import MAX_ARTICLE_CHARS, chunk_article, chunk_id, chunk_law, content_hash
from app.embeddings import FakeEmbedder
from app.ingest import ingest_law, ingest_laws
from app.store import InMemoryStore


def test_short_article_is_one_chunk(laws):
    law = laws["sales_tax_6_1994"]
    article = law.article("8")
    assert article is not None and len(article.text) <= MAX_ARTICLE_CHARS
    chunks = chunk_article(law, article)
    assert len(chunks) == 1
    assert chunks[0].id == "sales_tax_6_1994:8:0"
    assert chunks[0].text == article.text
    assert chunks[0].embedding_text.startswith("قانون الضريبة العامة على المبيعات - المادة (8)\n")


def test_embedded_text_drops_tatweel_but_display_text_keeps_it(laws):
    law = laws["sales_tax_6_1994"]
    article = law.article("4bis")
    assert article is not None
    chunk = chunk_article(law, article)[0]
    assert "يعتبـر" in chunk.text  # original, as printed in the official PDF
    assert "ـ" not in chunk.embedding_text.split("\n", 1)[1]
    assert "يعتبر" in chunk.embedding_text


def test_long_article_is_split_and_every_piece_keeps_its_article(laws):
    law = laws["income_tax_34_2014"]
    article = law.article("4")  # exemptions: the longest article family in the law
    assert article is not None and len(article.text) > MAX_ARTICLE_CHARS
    chunks = chunk_article(law, article)
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.article == "4"
        assert chunk.metadata["article_label"] == "4"
        assert f"المادة (4) - جزء {chunk.chunk_index + 1} من {len(chunks)}" in chunk.embedding_text
        assert len(chunk.text) <= 1200
    # the pieces cover the article: every word of the article appears in some piece
    joined = " ".join(c.text for c in chunks)
    assert set(article.text.split()) <= set(joined.split())


def test_chunk_ids_are_deterministic_and_unique(laws):
    first = [c.id for law in laws.values() for c in chunk_law(law)]
    second = [c.id for law in laws.values() for c in chunk_law(law)]
    assert first == second
    assert len(first) == len(set(first))
    assert chunk_id("sales_tax_6_1994", "4bis", 0) in first


def test_content_hash_changes_with_text_model_or_dimension():
    base = content_hash("نص", "model-a", 768)
    assert base == content_hash("نص", "model-a", 768)
    assert base != content_hash("نص آخر", "model-a", 768)
    assert base != content_hash("نص", "model-b", 768)
    assert base != content_hash("نص", "model-a", 1536)


def test_reingest_embeds_nothing_when_unchanged(laws):
    store, embedder = InMemoryStore(), FakeEmbedder()
    first = ingest_laws(laws, embedder, store)
    assert all(r.embedded == r.chunks and r.deleted == 0 for r in first)
    calls_after_first = sum(embedder.calls)

    second = ingest_laws(laws, embedder, store)
    assert all(r.embedded == 0 and r.unchanged == r.chunks and r.deleted == 0 for r in second)
    assert sum(embedder.calls) == calls_after_first  # no embedding call at all
    assert len(store) == sum(r.chunks for r in first)


def test_reingest_reembeds_only_the_changed_article_and_deletes_stale_chunks(laws):
    store, embedder = InMemoryStore(), FakeEmbedder()
    law = laws["sales_tax_6_1994"]
    ingest_law(law, embedder, store)
    embedder.calls.clear()

    articles = list(law.articles)
    target = next(i for i, a in enumerate(articles) if a.key == "8")
    articles[target] = replace(articles[target], text=articles[target].text + " (نص معدل للاختبار)")
    dropped = articles.pop()  # pretend the last article was repealed
    changed_law = replace(law, articles=tuple(articles))

    report = ingest_law(changed_law, embedder, store)
    assert report.embedded == 1
    assert sum(embedder.calls) == 1
    assert report.deleted == len(chunk_article(law, dropped))
    assert f"sales_tax_6_1994:{dropped.key}:0" not in store.existing_hashes(law.law_id)


def test_changing_the_embedding_model_reembeds_everything(laws):
    store = InMemoryStore()
    law = laws["sales_tax_6_1994"]
    ingest_law(law, FakeEmbedder(model="fake-a"), store)
    report = ingest_law(law, FakeEmbedder(model="fake-b"), store)
    assert report.embedded == report.chunks
