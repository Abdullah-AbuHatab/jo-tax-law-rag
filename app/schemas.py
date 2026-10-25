"""Request/response models of the HTTP API (the public contract)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

DISCLAIMER_AR = (
    "هذه الإجابة معلومات عامة مستخرجة آلياً من نصوص القوانين وليست استشارة قانونية. "
    "قد تكون النصوص غير محدثة؛ راجع النص الرسمي أو مختصاً قبل الاعتماد عليها."
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000, description="The question, Arabic preferred.")
    law_id: str | None = Field(default=None, max_length=64, description="Restrict retrieval to one law.")
    top_k: int | None = Field(default=None, ge=1, le=10, description="How many chunks to retrieve (default 5).")

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question is too short")
        return value


class CitationOut(BaseModel):
    law_id: str
    law_name: str
    article: str = Field(description='Article key used in URLs, e.g. "6" or "4bis".')
    article_label: str = Field(description='Article number as printed, e.g. "4 مكرر".')
    snippet: str
    source_url: str
    score: float


class RetrievedOut(BaseModel):
    chunk_id: str
    law_id: str
    article: str
    chunk_index: int
    score: float


class DroppedCitationOut(BaseModel):
    law_id: str
    article: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None = Field(
        description="LOW_RETRIEVAL_SCORE | MODEL_REFUSED | NO_VALID_CITATIONS | LLM_INVALID_OUTPUT"
    )
    error_code: str | None
    citations: list[CitationOut]
    dropped_citations: list[DroppedCitationOut]
    retrieved: list[RetrievedOut]
    model: str | None
    disclaimer: str = DISCLAIMER_AR


class ArticleSummaryOut(BaseModel):
    article: str = Field(description='Article key used in URLs, e.g. "6" or "4bis".')
    article_label: str = Field(description='Article number as printed, e.g. "4 مكرر".')


class LawOut(BaseModel):
    law_id: str
    name: str
    short_name: str
    number: str
    year: str
    version: str
    source_url: str
    retrieved: str
    article_count: int
    version_ar: str = Field(default="", description="Consolidated version label in Arabic.")
    coverage_note_ar: str = Field(default="", description="Coverage caveat/note in Arabic.")


class ArticleOut(BaseModel):
    law_id: str
    law_name: str
    article: str
    article_label: str
    text: str
    notes: list[str]
    source_url: str
    prev_article: str | None = Field(default=None, description="Previous article key in document order.")
    prev_article_label: str | None = Field(default=None, description="Previous article label as printed.")
    next_article: str | None = Field(default=None, description="Next article key in document order.")
    next_article_label: str | None = Field(default=None, description="Next article label as printed.")


class PublicConfigOut(BaseModel):
    turnstile_site_key: str | None = Field(description="Cloudflare Turnstile site key; null when Turnstile is off.")


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
