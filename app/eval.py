"""Evaluate retrieval and answering on data/eval/golden.jsonl.

    python -m app.eval --fake       # offline smoke run: fake embedder + fake LLM + in-memory store
    python -m app.eval              # real run: Gemini + OpenRouter + DATABASE_URL (ingest first)

Metrics (written to eval/results.md):
  * recall@k   - share of expected articles found in the top-k retrieved chunks
                 (answerable questions only; article level, pieces deduplicated)
  * MRR        - mean reciprocal rank of the first expected article
  * refusal accuracy - refused exactly when the golden item says it should
  * citation validity - share of model-proposed citations that pointed at a
                 retrieved article (the rest were dropped by the validator)
  * citation hit rate - answered questions whose citations include an expected article

A real run spends API quota (free tiers have per-minute limits), so it waits
``--sleep`` seconds between questions.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.answering import AnswerResult, AnswerService, ServiceError
from app.config import REPO_ROOT, Settings
from app.parsing import Law, load_laws, normalize_article_ref
from app.store import SearchResult

GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "golden.jsonl"
RESULTS_PATH = REPO_ROOT / "eval" / "results.md"


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    expected: tuple[tuple[str, str], ...]  # (law_id, article key)
    should_refuse: bool
    law_id: str | None = None
    comment: str = ""


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        expected = []
        for ref in raw.get("expected", []):
            key = normalize_article_ref(ref["article"])
            if key is None:
                raise ValueError(f"{path}:{line_no}: bad article reference {ref['article']!r}")
            expected.append((ref["law_id"], key))
        items.append(
            GoldenItem(
                id=raw["id"],
                question=raw["question"],
                expected=tuple(expected),
                should_refuse=bool(raw["should_refuse"]),
                law_id=raw.get("law_id"),
                comment=raw.get("comment", ""),
            )
        )
    return items


def ranked_articles(results: list[SearchResult]) -> list[tuple[str, str]]:
    """Retrieved (law_id, article) in rank order, each article once."""
    seen: list[tuple[str, str]] = []
    for result in results:
        key = (result.law_id, result.article)
        if key not in seen:
            seen.append(key)
    return seen


def recall_at_k(expected: tuple[tuple[str, str], ...], ranked: list[tuple[str, str]]) -> float:
    if not expected:
        return 0.0
    return sum(1 for e in expected if e in ranked) / len(expected)


def reciprocal_rank(expected: tuple[tuple[str, str], ...], ranked: list[tuple[str, str]]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in expected:
            return 1.0 / rank
    return 0.0


@dataclass
class ItemOutcome:
    item: GoldenItem
    ranked: list[tuple[str, str]]
    best_score: float | None
    result: AnswerResult | None
    error: str | None = None


@dataclass
class Summary:
    answerable: int = 0
    out_of_scope: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    citation_validity: float | None = None
    citation_hit_rate: float | None = None
    errors: int = 0
    extra: dict[str, object] = field(default_factory=dict)


def summarize(outcomes: list[ItemOutcome]) -> Summary:
    answerable = [o for o in outcomes if not o.item.should_refuse]
    summary = Summary(answerable=len(answerable), out_of_scope=len(outcomes) - len(answerable))
    if answerable:
        summary.recall_at_k = sum(recall_at_k(o.item.expected, o.ranked) for o in answerable) / len(answerable)
        summary.mrr = sum(reciprocal_rank(o.item.expected, o.ranked) for o in answerable) / len(answerable)
    judged = [o for o in outcomes if o.result is not None]
    summary.errors = len(outcomes) - len(judged)
    if judged:
        summary.refusal_accuracy = sum(o.result.refused == o.item.should_refuse for o in judged) / len(judged)
    proposed = sum(len(o.result.citations) + len(o.result.dropped_citations) for o in judged)
    if proposed:
        summary.citation_validity = sum(len(o.result.citations) for o in judged) / proposed
    answered = [o for o in judged if not o.result.refused and not o.item.should_refuse]
    if answered:
        hits = sum(any((c.law_id, c.article) in o.item.expected for c in o.result.citations) for o in answered)
        summary.citation_hit_rate = hits / len(answered)
    return summary


def run(items: list[GoldenItem], service: AnswerService, k: int, sleep_seconds: float = 0.0) -> list[ItemOutcome]:
    outcomes = []
    for index, item in enumerate(items):
        if index and sleep_seconds:
            time.sleep(sleep_seconds)
        try:
            results = service.retrieve(item.question, k=k, law_id=item.law_id)
            answer = service.answer_from_results(item.question, results)
            outcomes.append(ItemOutcome(item, ranked_articles(results), results[0].score if results else None, answer))
        except ServiceError as exc:
            outcomes.append(ItemOutcome(item, [], None, None, error=exc.code))
    return outcomes


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown(outcomes: list[ItemOutcome], summary: Summary, meta: Mapping[str, object]) -> str:
    lines = ["# Evaluation results", ""]
    if meta.get("mode") == "fake":
        lines += [
            "> **FAKE SMOKE RUN.** Fake hashing embedder + scripted fake LLM + in-memory store.",
            "> These numbers only prove the pipeline runs end to end; they say nothing about answer quality.",
            "",
        ]
    lines += [f"- {key}: `{value}`" for key, value in meta.items()]
    lines += [
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| questions (answerable / out of scope) | {summary.answerable} / {summary.out_of_scope} |",
        f"| recall@{meta.get('top_k')} | {_pct(summary.recall_at_k)} |",
        f"| MRR | {summary.mrr:.3f} |",
        f"| refusal accuracy | {_pct(summary.refusal_accuracy)} |",
        f"| citation validity | {_pct(summary.citation_validity)} |",
        f"| citation hit rate (answered questions citing an expected article) | {_pct(summary.citation_hit_rate)} |",
        f"| errors (dependency failures) | {summary.errors} |",
        "",
        "## Per question",
        "",
        "| id | expected | top retrieved | best score | refused (expected) | reason | cited |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        expected = ", ".join(f"{law}:{art}" for law, art in o.item.expected) or "-"
        top = ", ".join(f"{law}:{art}" for law, art in o.ranked[:3]) or "-"
        best = "-" if o.best_score is None else f"{o.best_score:.3f}"
        if o.result is None:
            lines.append(f"| {o.item.id} | {expected} | {top} | {best} | error | {o.error} | - |")
            continue
        cited = ", ".join(f"{c.law_id}:{c.article}" for c in o.result.citations) or "-"
        refused = f"{'yes' if o.result.refused else 'no'} ({'yes' if o.item.should_refuse else 'no'})"
        lines.append(
            f"| {o.item.id} | {expected} | {top} | {best} | {refused} | {o.result.refusal_reason or '-'} | {cited} |"
        )
    return "\n".join(lines) + "\n"


def build_fake_service(laws: Mapping[str, Law], settings: Settings) -> AnswerService:
    from app.embeddings import FakeEmbedder
    from app.ingest import ingest_laws
    from app.llm import FakeLLM
    from app.store import InMemoryStore

    embedder, store = FakeEmbedder(), InMemoryStore()
    ingest_laws(laws, embedder, store)
    return AnswerService(embedder, store, FakeLLM(), laws, settings.min_similarity, settings.default_top_k)


def build_real_service(laws: Mapping[str, Law], settings: Settings) -> AnswerService:
    from app.dependencies import build_embedder, build_llm, build_vector_store

    return AnswerService(
        build_embedder(settings),
        build_vector_store(settings),
        build_llm(settings),
        laws,
        settings.min_similarity,
        settings.default_top_k,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval", description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="offline smoke run, no keys or database needed")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, help="override MIN_SIMILARITY for this run")
    parser.add_argument("--sleep", type=float, default=None, help="seconds between questions (real run: 4)")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    env_settings = Settings.from_env()
    overrides: dict[str, object] = {}
    if args.min_similarity is not None:
        overrides["min_similarity"] = args.min_similarity
    elif args.fake:
        overrides["min_similarity"] = 0.15  # the hashing embedder's scores live on a different scale
    settings = env_settings.model_copy(update=overrides)

    laws = load_laws(settings.data_dir)
    items = load_golden(args.golden)
    service = build_fake_service(laws, settings) if args.fake else build_real_service(laws, settings)
    sleep_seconds = args.sleep if args.sleep is not None else (0.0 if args.fake else 4.0)
    outcomes = run(items, service, k=args.top_k, sleep_seconds=sleep_seconds)
    summary = summarize(outcomes)

    meta = {
        "mode": "fake" if args.fake else "real",
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "top_k": args.top_k,
        "min_similarity": settings.min_similarity,
        "embedder": "fake-hashing-v1" if args.fake else settings.gemini_embed_model,
        "llm": "fake-llm"
        if args.fake
        else ", ".join([settings.openrouter_model, *settings.openrouter_fallback_models]),
        "golden": args.golden.relative_to(REPO_ROOT).as_posix()
        if args.golden.is_relative_to(REPO_ROOT)
        else str(args.golden),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(outcomes, summary, meta), encoding="utf-8")
    print(
        f"recall@{args.top_k}={summary.recall_at_k:.3f} mrr={summary.mrr:.3f} "
        f"refusal_accuracy={summary.refusal_accuracy:.3f} citation_validity={_pct(summary.citation_validity)} "
        f"errors={summary.errors} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
