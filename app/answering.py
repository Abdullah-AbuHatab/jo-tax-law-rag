"""Question answering: retrieve, gate, generate, then validate.

"The model proposes, code decides":
  * retrieval decides whether the LLM is called at all (similarity gate);
  * the model's JSON is parsed and validated, never trusted as-is;
  * a citation survives only if it points at an article that was actually
    retrieved for this question; an answer with no surviving citation is
    turned into a refusal.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.arabic import is_mostly_arabic
from app.db import DatabaseUnavailable
from app.embeddings import Embedder, EmbeddingError
from app.llm import LLMClient, LLMError
from app.parsing import Law, normalize_article_ref
from app.prompts import build_messages
from app.store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 280
REFUSAL_AR = "لا تتضمن نصوص القوانين المتوفرة لدي إجابة عن هذا السؤال، لذلك لا أستطيع الإجابة عنه."
REFUSAL_EN = "The laws available to me do not answer this question, so I cannot answer it."

# Why an answer was refused (part of the API contract).
LOW_RETRIEVAL_SCORE = "LOW_RETRIEVAL_SCORE"
MODEL_REFUSED = "MODEL_REFUSED"
NO_VALID_CITATIONS = "NO_VALID_CITATIONS"
LLM_INVALID_OUTPUT = "LLM_INVALID_OUTPUT"


class ServiceError(Exception):
    """A request cannot be served (a dependency failed, or bad input).

    ``code`` is stable and safe to show to clients; ``http_status`` is what the
    API returns.
    """

    def __init__(self, code: str, http_status: int) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


class ProposedCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    law_id: str = Field(max_length=64)
    article: str = Field(max_length=32)

    @field_validator("law_id", "article", mode="before")
    @classmethod
    def _to_str(cls, value: object) -> object:
        return str(value).strip() if isinstance(value, int | float | str) and not isinstance(value, bool) else value


class LLMAnswer(BaseModel):
    """The JSON object the model is asked to return."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(max_length=6000)
    citations: list[ProposedCitation] = Field(default_factory=list, max_length=20)
    refused: bool


_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.S)


def parse_llm_output(content: str) -> LLMAnswer | None:
    """Parse the model reply. Returns None when it is not the JSON we asked for.

    Tolerated: a ```json fence around the object, and prose before/after the
    object (we take the outermost {...}). Not tolerated: anything that does
    not validate against ``LLMAnswer``.
    """
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
        return LLMAnswer.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


@dataclass(frozen=True)
class Citation:
    law_id: str
    law_name: str
    article: str  # key, e.g. "4bis"
    article_label: str  # display, e.g. "4 مكرر"
    snippet: str
    source_url: str
    score: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    refused: bool
    refusal_reason: str | None
    error_code: str | None
    citations: list[Citation]
    dropped_citations: list[dict[str, str]]
    retrieved: list[SearchResult]
    model: str | None
    llm_called: bool = field(default=False)


def refusal_text(question: str) -> str:
    return REFUSAL_AR if is_mostly_arabic(question) else REFUSAL_EN


class AnswerService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient | None,
        laws: Mapping[str, Law],
        min_similarity: float,
        default_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._laws = laws
        self._min_similarity = min_similarity
        self._default_top_k = default_top_k

    def retrieve(self, question: str, k: int, law_id: str | None = None) -> list[SearchResult]:
        try:
            query_vec = self._embedder.embed_query(question)
        except EmbeddingError as exc:
            logger.warning("query embedding failed", extra={"error": str(exc)})
            raise ServiceError("EMBEDDING_UNAVAILABLE", 503) from exc
        try:
            return self._store.search(query_vec, k=k, law_id=law_id)
        except DatabaseUnavailable as exc:
            raise ServiceError("DATABASE_UNAVAILABLE", 503) from exc

    def ask(self, question: str, law_id: str | None = None, top_k: int | None = None) -> AnswerResult:
        results = self.retrieve(question, k=top_k or self._default_top_k, law_id=law_id)
        return self.answer_from_results(question, results)

    def answer_from_results(self, question: str, results: list[SearchResult]) -> AnswerResult:
        """Gate, generate and validate for already-retrieved chunks (best first)."""
        best = results[0].score if results else None
        if best is None or best < self._min_similarity:
            # Gate before the LLM: nothing relevant was found, so there is
            # nothing to ground an answer on, and no tokens are spent.
            return self._refusal(question, LOW_RETRIEVAL_SCORE, results, model=None, llm_called=False)

        if self._llm is None:
            raise ServiceError("LLM_CONFIG_ERROR", 503)
        try:
            completion = self._llm.complete(build_messages(question, results))
        except LLMError as exc:
            raise ServiceError(exc.code, exc.http_status) from exc

        parsed = parse_llm_output(completion.content)
        if parsed is None:
            logger.warning("llm output was not valid JSON", extra={"model": completion.model})
            return self._refusal(
                question, LLM_INVALID_OUTPUT, results, model=completion.model, error_code=LLM_INVALID_OUTPUT
            )
        if parsed.refused:
            return self._refusal(question, MODEL_REFUSED, results, model=completion.model)

        valid, dropped = self.validate_citations(parsed.citations, results)
        if not valid or not parsed.answer.strip():
            return self._refusal(question, NO_VALID_CITATIONS, results, model=completion.model, dropped=dropped)
        return AnswerResult(
            answer=parsed.answer.strip(),
            refused=False,
            refusal_reason=None,
            error_code=None,
            citations=valid,
            dropped_citations=dropped,
            retrieved=results,
            model=completion.model,
            llm_called=True,
        )

    def validate_citations(
        self, proposed: list[ProposedCitation], results: list[SearchResult]
    ) -> tuple[list[Citation], list[dict[str, str]]]:
        """Keep citations that match a retrieved (law_id, article); report the rest."""
        best_by_article: dict[tuple[str, str], SearchResult] = {}
        for result in results:  # results are best-first, keep the first per article
            best_by_article.setdefault((result.law_id, result.article), result)

        valid: list[Citation] = []
        dropped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for citation in proposed:
            key = normalize_article_ref(citation.article)
            match = best_by_article.get((citation.law_id, key)) if key else None
            if match is None:
                dropped.append({"law_id": citation.law_id[:64], "article": citation.article[:32]})
                continue
            if (match.law_id, match.article) in seen:
                continue
            seen.add((match.law_id, match.article))
            law = self._laws.get(match.law_id)
            valid.append(
                Citation(
                    law_id=match.law_id,
                    law_name=law.name if law else str(match.metadata.get("law_name", match.law_id)),
                    article=match.article,
                    article_label=str(match.metadata.get("article_label", match.article)),
                    snippet=_snippet(match.text),
                    source_url=law.source_url if law else str(match.metadata.get("source_url", "")),
                    score=round(match.score, 4),
                )
            )
        return valid, dropped

    @staticmethod
    def _refusal(
        question: str,
        reason: str,
        results: list[SearchResult],
        model: str | None,
        llm_called: bool = True,
        error_code: str | None = None,
        dropped: list[dict[str, str]] | None = None,
    ) -> AnswerResult:
        return AnswerResult(
            answer=refusal_text(question),
            refused=True,
            refusal_reason=reason,
            error_code=error_code,
            citations=[],
            dropped_citations=dropped or [],
            retrieved=results,
            model=model,
            llm_called=llm_called,
        )


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "…"
