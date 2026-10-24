"""The golden set is grounded in the real text, and the metrics are computed correctly."""

from app.arabic import normalize_arabic
from app.eval import (
    GOLDEN_PATH,
    build_fake_service,
    load_golden,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
    run,
    summarize,
)
from tests.conftest import fake_settings


def test_golden_set_size_and_shape():
    items = load_golden(GOLDEN_PATH)
    answerable = [i for i in items if not i.should_refuse]
    out_of_scope = [i for i in items if i.should_refuse]
    assert 20 <= len(answerable) <= 30
    assert len(out_of_scope) >= 6
    assert len({i.id for i in items}) == len(items)
    assert all(i.expected for i in answerable)
    assert all(not i.expected for i in out_of_scope)


def test_every_expected_article_exists_and_contains_the_quoted_text(laws):
    """The comment of each answerable item quotes its article; check the quote is really there."""
    for item in load_golden(GOLDEN_PATH):
        for law_id, article_key in item.expected:
            article = laws[law_id].article(article_key)
            assert article is not None, (item.id, law_id, article_key)
            assert normalize_arabic(item.comment) in normalize_arabic(article.text), item.id
        if item.law_id is not None:
            assert item.law_id in laws


def test_recall_and_reciprocal_rank():
    expected = (("law", "6"), ("law", "7"))
    ranked = [("law", "1"), ("law", "7"), ("law", "6")]
    assert recall_at_k(expected, ranked) == 1.0
    assert recall_at_k(expected, ranked[:2]) == 0.5
    assert reciprocal_rank(expected, ranked) == 0.5
    assert reciprocal_rank(expected, [("law", "9")]) == 0.0
    assert recall_at_k((), ranked) == 0.0


def test_fake_eval_runs_end_to_end(laws):
    settings = fake_settings(min_similarity=0.15)
    items = load_golden(GOLDEN_PATH)[:5]
    outcomes = run(items, build_fake_service(laws, settings), k=5)
    summary = summarize(outcomes)
    assert summary.answerable == 5 and summary.errors == 0
    assert 0.0 <= summary.recall_at_k <= 1.0
    report = render_markdown(outcomes, summary, {"mode": "fake", "top_k": 5})
    assert "FAKE SMOKE RUN" in report
    assert report.count("| inc-0") == 5
