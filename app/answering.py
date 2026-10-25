"""Question answering: retrieve, gate, generate, then validate.

"The model proposes, code decides":
  * retrieval decides whether the LLM is called at all (similarity gate);
  * the model's JSON is parsed and validated, never trusted as-is;
  * a citation survives only if it points at an article that was actually
    retrieved for this question; an answer with no surviving citation is
    turned into a refusal;
  * the answer text is sanitized (control and bidi-override characters,
    blank-line runs, length) before it is returned.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.arabic import is_mostly_arabic
from app.db import DatabaseUnavailable
from app.embeddings import Embedder, EmbeddingError
from app.guards import DailyBudget, PerKeyDailyLimit
from app.llm import LLMClient, LLMError
from app.parsing import Law, normalize_article_ref
from app.prompts import build_messages
from app.store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 280
MAX_ANSWER_CHARS = 4000
MIN_LLM_SECONDS = 1.0  # below this, an LLM call cannot finish inside the request deadline
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
    API returns; ``retry_after`` (seconds), when set, becomes a Retry-After header.
    """

    def __init__(self, code: str, http_status: int, retry_after: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status
        self.retry_after = retry_after


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


# C0 controls except tab and newline, DEL, C1 controls, the bidi embedding/override
# (U+202A..U+202E) and isolate (U+2066..U+2069) characters, which can make displayed
# text read differently from its logical order, and invisible characters that can hide
# text or split a link so it is not recognised: soft hyphen (U+00AD), zero-width
# space/non-joiner/joiner and LRM/RLM (U+200B..U+200F; legal answers need no ZWNJ/ZWJ),
# word joiner and invisible operators (U+2060..U+2064), BOM (U+FEFF), and Unicode tag
# characters (U+E0000..U+E007F).
_UNSAFE_CHARS = re.compile(
    r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069"
    r"\ufeff\U000e0000-\U000e007f]"
)
_BLANK_LINE_RUN = re.compile(r"\n(?:[ \t]*\n){4,}")  # 4 or more blank lines in a row
# A dot, or a look-alike a renderer may show as one: fullwidth, ideographic, small and
# halfwidth ideographic full stops.
_DOT = "[.\uff0e\u3002\ufe52\uff61]"
# Bare domains count as links only with one of these TLDs, so tokens such as "6.bis",
# "Co.Ltd" or "e.g." are left alone. Common generic TLDs plus the regional ccTLDs.
REAL_TLDS = (
    "com",
    "net",
    "org",
    "info",
    "biz",
    "io",
    "co",
    "me",
    "app",
    "dev",
    "xyz",
    "site",
    "online",
    "link",
    "click",
    "top",
    "shop",
    "store",
    "tech",
    "ai",
    "gov",
    "edu",
    "mil",
    "int",
    "jo",
    "sa",
    "ae",
    "qa",
    "kw",
    "bh",
    "om",
    "eg",
    "uk",
    "us",
    "de",
    "fr",
    "ru",
    "cn",
    "in",
    "tr",
    "ir",
    "iq",
    "sy",
    "lb",
    "ps",
    "ly",
    "tn",
    "ma",
    "dz",
    "sd",
    "ye",
)
_TLD = "(?:" + "|".join(sorted(REAL_TLDS, key=len, reverse=True)) + ")"
# URLs (with a scheme, or starting with www.), e-mail addresses, and bare domains such
# as evil.com/pay. ASCII lookarounds instead of \b, so an Arabic letter written right
# against the domain (a joined "و") does not hide it.
_LINK = re.compile(
    "(?:https?://|www" + _DOT + ")[^\\s<>\"'«»]+"
    "|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:" + _DOT + "[A-Za-z0-9-]+)+"
    "|(?<![A-Za-z0-9-])[A-Za-z0-9][A-Za-z0-9-]*(?:"
    + _DOT
    + "[A-Za-z0-9-]+)*"
    + _DOT
    + _TLD
    + "(?![A-Za-z0-9-])(?:/[^\\s<>\"'«»]*)?",
    re.IGNORECASE,
)
_LINK_TRAILING_PUNCTUATION = ".,;:!?)]}،؛؟»\"'\uff0e\u3002\uff61"
LINK_PLACEHOLDER = "[رابط محذوف]"


def neutralize_links(text: str, allowed_urls: Collection[str]) -> str:
    """Replace every URL, e-mail address and bare domain with LINK_PLACEHOLDER, except
    ``allowed_urls`` (the laws' own source URLs, raw or percent-decoded) and their host
    names written as bare domains (e.g. istd.gov.jo). Trailing punctuation is kept."""
    allowed = set(allowed_urls) | {unquote(url) for url in allowed_urls}
    allowed_hosts = {host.lower() for host in (urlsplit(url).hostname for url in allowed_urls) if host}

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        core = found.rstrip(_LINK_TRAILING_PUNCTUATION)
        if core in allowed or re.sub(_DOT, ".", core).lower() in allowed_hosts:
            return found
        return LINK_PLACEHOLDER + found[len(core) :]

    return _LINK.sub(replace, text)


def sanitize_model_text(
    text: str, max_chars: int = MAX_ANSWER_CHARS, allowed_urls: Collection[str] | None = None
) -> str:
    """Model text made safe to show: no control or bidi-override characters,
    at most 3 blank lines in a row, stripped, and at most ``max_chars`` long
    (cut with an ellipsis). Arabic letters, digits, tabs and newlines are untouched.

    With ``allowed_urls`` (model answers), links and e-mail addresses are also
    replaced by LINK_PLACEHOLDER unless they are one of those URLs: an injected
    question must not make the service show a phishing link or a contact."""
    cleaned = _UNSAFE_CHARS.sub("", text)
    if allowed_urls is not None:
        cleaned = neutralize_links(cleaned, allowed_urls)
    cleaned = _BLANK_LINE_RUN.sub("\n\n\n\n", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


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
        llm_budget: DailyBudget | None = None,
        deadline_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        llm_per_client: PerKeyDailyLimit | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._laws = laws
        self._source_urls = frozenset(law.source_url for law in laws.values() if law.source_url)
        self._min_similarity = min_similarity
        self._default_top_k = default_top_k
        self._llm_budget = llm_budget  # None = unlimited (eval, tests)
        # LLM calls per client per day, so one client cannot use up llm_budget alone.
        # Applies only to calls made with a client_key (the /ask route).
        self._llm_per_client = llm_per_client
        # Upstream time one ask() may use in total (embedding + search + LLM). The LLM
        # call gets only what is left; None = no overall deadline (eval, tests).
        self._deadline_seconds = deadline_seconds
        self._clock = clock

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

    def ask(
        self,
        question: str,
        law_id: str | None = None,
        top_k: int | None = None,
        client_key: str | None = None,
    ) -> AnswerResult:
        deadline = None if self._deadline_seconds is None else self._clock() + self._deadline_seconds
        results = self.retrieve(question, k=top_k or self._default_top_k, law_id=law_id)
        return self.answer_from_results(question, results, deadline=deadline, client_key=client_key)

    def answer_from_results(
        self,
        question: str,
        results: list[SearchResult],
        deadline: float | None = None,
        client_key: str | None = None,
    ) -> AnswerResult:
        """Gate, generate and validate for already-retrieved chunks (best first).

        ``deadline`` (a ``clock()`` value) bounds the LLM call to the time left;
        with less than MIN_LLM_SECONDS left the LLM is not called at all.
        ``client_key`` (None = unlimited) is charged one LLM call against
        ``llm_per_client``, only when the LLM is really called.
        """
        best = results[0].score if results else None
        if best is None or best < self._min_similarity:
            # Gate before the LLM: nothing relevant was found, so there is
            # nothing to ground an answer on, and no tokens are spent.
            return self._refusal(question, LOW_RETRIEVAL_SCORE, results, model=None, llm_called=False)

        if self._llm is None:
            raise ServiceError("LLM_CONFIG_ERROR", 503)
        time_left = None if deadline is None else deadline - self._clock()
        if time_left is not None and time_left < MIN_LLM_SECONDS:
            # Retrieval used up the request's upstream time: do not start (or pay for) a call.
            raise ServiceError("LLM_TIMEOUT", 504)
        # Spent only here, right before the call: gated refusals cost nothing.
        # The client's allowance first; given back if the global budget then says no.
        per_client = self._llm_per_client if client_key is not None else None
        if per_client is not None and not per_client.allow(client_key):
            raise ServiceError("DAILY_LIMIT_REACHED", 429, retry_after=per_client.seconds_until_reset())
        if self._llm_budget is not None and not self._llm_budget.try_consume():
            if per_client is not None:
                per_client.refund(client_key)
            raise ServiceError("DAILY_BUDGET_EXHAUSTED", 503, retry_after=self._llm_budget.seconds_until_reset())
        messages = build_messages(question, results)
        try:
            if time_left is None:
                completion = self._llm.complete(messages)
            else:
                completion = self._llm.complete(messages, timeout_seconds=time_left)
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

        answer = sanitize_model_text(parsed.answer, allowed_urls=self._source_urls)
        if not answer:
            logger.warning("llm answer is empty after sanitizing", extra={"model": completion.model})
            return self._refusal(
                question, LLM_INVALID_OUTPUT, results, model=completion.model, error_code=LLM_INVALID_OUTPUT
            )
        valid, dropped = self.validate_citations(parsed.citations, results)
        if not valid:
            return self._refusal(question, NO_VALID_CITATIONS, results, model=completion.model, dropped=dropped)
        return AnswerResult(
            answer=answer,
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
                dropped.append(  # echoed to clients, so sanitized like the answer
                    {
                        "law_id": sanitize_model_text(citation.law_id, 64),
                        "article": sanitize_model_text(citation.article, 32),
                    }
                )
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
