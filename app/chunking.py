"""Turn articles into chunks: one chunk per article, long articles split.

Why article-level: a citation in law is "article N", so the retrieval unit
is the same unit the answer must cite. Only articles longer than
``MAX_ARTICLE_CHARS`` are split, and every piece keeps its law and article
number (in metadata and in the embedded header line), so a piece can never
be cited as some other article.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.arabic import normalize_arabic, strip_tatweel
from app.parsing import Article, Law

MAX_ARTICLE_CHARS = 1500
PIECE_CHARS = 1200
PIECE_OVERLAP = 150
# Arabic-aware separators, coarse to fine: paragraph markers sit on their own
# lines in the cleaned text, then sentence ends (". " "؟" "؛"), then commas
# (Arabic "،" and Latin ","), then words.
ARABIC_SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "! ", "؛ ", "، ", ", ", " ", ""]

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PIECE_CHARS,
    chunk_overlap=PIECE_OVERLAP,
    separators=ARABIC_SEPARATORS,
    keep_separator="end",
    strip_whitespace=True,
)


@dataclass(frozen=True)
class Chunk:
    id: str
    law_id: str
    article: str  # article key: "6" or "4bis"
    chunk_index: int
    text: str  # original text, for display
    embedding_text: str  # header line + text without tatweel: what gets embedded
    metadata: dict[str, object] = field(default_factory=dict, hash=False, compare=False)

    @property
    def text_normalized(self) -> str:
        return normalize_arabic(self.text)


def chunk_id(law_id: str, article: str, chunk_index: int) -> str:
    """Deterministic id: the same article piece always gets the same id."""
    return f"{law_id}:{article}:{chunk_index}"


def content_hash(embedding_text: str, embed_model: str, dimension: int) -> str:
    """Changes when the embedded text, the model or the vector size changes."""
    payload = f"{embed_model}\n{dimension}\n{embedding_text}".encode()
    return hashlib.sha256(payload).hexdigest()


def split_article_text(text: str) -> list[str]:
    if len(text) <= MAX_ARTICLE_CHARS:
        return [text]
    return [piece for piece in _splitter.split_text(text) if piece.strip()]


def chunk_article(law: Law, article: Article) -> list[Chunk]:
    pieces = split_article_text(article.text)
    chunks = []
    for index, piece in enumerate(pieces):
        part = f" - جزء {index + 1} من {len(pieces)}" if len(pieces) > 1 else ""
        header = f"{law.short_name} - المادة ({article.label}){part}"
        chunks.append(
            Chunk(
                id=chunk_id(law.law_id, article.key, index),
                law_id=law.law_id,
                article=article.key,
                chunk_index=index,
                text=piece,
                embedding_text=f"{header}\n{strip_tatweel(piece)}",
                metadata={
                    "law_name": law.name,
                    "law_number": law.number,
                    "law_year": law.year,
                    "article_label": article.label,
                    "source_url": law.source_url,
                    "pieces": len(pieces),
                    "notes": list(article.notes),
                },
            )
        )
    return chunks


def chunk_law(law: Law) -> list[Chunk]:
    return [chunk for article in law.articles for chunk in chunk_article(law, article)]
