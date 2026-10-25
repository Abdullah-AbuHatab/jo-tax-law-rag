"""Parse a cleaned law file (data/clean/*.txt) into articles.

File format (written by tools/build_clean_text.py)::

    # key: value            header metadata
    المادة (6)              one heading line per article
    text ...
    [حاشية] note            amendment note, stored in metadata, not in the text

The heading parser is deliberately tolerant so the same code can read other
consolidated texts: "المادة (12)", "المادة ( 12 )", "المادة ١٢", "مادة 12",
"المادة الأولى", "المادة (4 مكرر)" / "المادة (4) مكرر".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.arabic import normalize_arabic, to_ascii_digits

NOTE_PREFIX = "[حاشية]"
BIS_WORD = "مكرر"

_UNITS = {
    "الاول": 1, "الاولي": 1, "الحاديه": 1, "الحادي": 1,
    "الثاني": 2, "الثانيه": 2, "الثالث": 3, "الثالثه": 3, "الرابع": 4, "الرابعه": 4,
    "الخامس": 5, "الخامسه": 5, "السادس": 6, "السادسه": 6, "السابع": 7, "السابعه": 7,
    "الثامن": 8, "الثامنه": 8, "التاسع": 9, "التاسعه": 9,
}  # fmt: skip
_TENS = {"العشرون": 20, "الثلاثون": 30, "الاربعون": 40, "الخمسون": 50,
         "الستون": 60, "السبعون": 70, "الثمانون": 80, "التسعون": 90}  # fmt: skip

_HEADING = re.compile(
    r"^\s*(?:ال)?ماد[ةه]\s*"  # "المادة" / "مادة"
    r"(?:\(\s*(?P<num>\d+)\s*(?P<bis_in>مكرر)?\s*\)|(?P<bare>\d+)|(?P<words>[^\d()]+?))"
    r"\s*(?P<bis_out>مكرر)?\s*[:.]?\s*$"
)


def ordinal_to_int(words: str) -> int | None:
    """Arabic ordinal words -> number: "الأولى" -> 1, "الحادية عشرة" -> 11, "الخامسة والعشرون" -> 25."""
    # "والعشرون" -> "العشرون": drop the conjunction in front of a tens word
    tokens = [t[1:] if t.startswith("و") and t[1:] in _TENS else t for t in normalize_arabic(words).split()]
    if not tokens:
        return None
    if tokens == ["العاشر"] or tokens == ["العاشره"]:
        return 10
    if len(tokens) == 1 and tokens[0] in _TENS:
        return _TENS[tokens[0]]
    if len(tokens) == 2 and tokens[0] in _UNITS and tokens[1] in ("عشر", "عشره"):
        return 10 + _UNITS[tokens[0]]
    if len(tokens) == 2 and tokens[0] in _UNITS and tokens[1] in _TENS:
        return _TENS[tokens[1]] + _UNITS[tokens[0]]
    if len(tokens) == 1 and tokens[0] in _UNITS:
        return _UNITS[tokens[0]]
    return None


def parse_heading(line: str) -> tuple[int, bool] | None:
    """Return (article number, is_bis) when the line is an article heading."""
    match = _HEADING.match(to_ascii_digits(line))
    if not match:
        return None
    bis = bool(match.group("bis_in") or match.group("bis_out"))
    if match.group("num") or match.group("bare"):
        return int(match.group("num") or match.group("bare")), bis
    number = ordinal_to_int(match.group("words"))
    return (number, bis) if number else None


def article_key(number: int, bis: bool) -> str:
    return f"{number}bis" if bis else str(number)


def article_label(number: int, bis: bool) -> str:
    return f"{number} {BIS_WORD}" if bis else str(number)


def normalize_article_ref(value: object) -> str | None:
    """Turn any reasonable article reference into its key ("6", "4bis")."""
    if isinstance(value, bool) or value is None:
        return None
    text = to_ascii_digits(str(value)).strip()
    text = re.sub(r"^\s*(?:ال)?ماد[ةه]\s*", "", text)
    match = re.fullmatch(r"\(?\s*(\d{1,4})\s*\)?\s*[-_ ]?\s*\(?\s*(مكرر|bis)?\s*\)?", text, flags=re.IGNORECASE)
    if not match:
        return None
    return article_key(int(match.group(1)), bool(match.group(2)))


@dataclass(frozen=True)
class Article:
    law_id: str
    number: int
    bis: bool
    text: str
    notes: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return article_key(self.number, self.bis)

    @property
    def label(self) -> str:
        return article_label(self.number, self.bis)


@dataclass(frozen=True)
class Law:
    law_id: str
    name: str
    short_name: str
    number: str
    year: str
    source_url: str
    version: str
    retrieved: str
    articles: tuple[Article, ...] = field(default=())
    version_ar: str = ""
    coverage_note_ar: str = ""

    def article(self, key: str) -> Article | None:
        return next((a for a in self.articles if a.key == key), None)

    def neighbours(self, key: str) -> tuple[Article | None, Article | None]:
        keys = [a.key for a in self.articles]
        if key not in keys:
            raise KeyError(f"unknown article key {key} in {self.law_id}")
        idx = keys.index(key)
        prev_article = self.articles[idx - 1] if idx > 0 else None
        next_article = self.articles[idx + 1] if idx + 1 < len(self.articles) else None
        return (prev_article, next_article)


def parse_law_text(text: str) -> Law:
    meta: dict[str, str] = {}
    articles: list[Article] = []
    current: tuple[int, bool] | None = None
    body: list[str] = []
    notes: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if current is None:
            return
        number, bis = current
        key = article_key(number, bis)
        if key in seen:
            raise ValueError(f"duplicate article {key} in {meta.get('law_id', '?')}")
        seen.add(key)
        articles.append(
            Article(
                law_id=meta["law_id"],
                number=number,
                bis=bis,
                text="\n".join(body).strip(),
                notes=tuple(notes),
            )
        )

    for raw in text.splitlines():
        line = raw.strip()
        if current is None and line.startswith("#"):
            key, sep, value = line.lstrip("#").partition(":")
            if sep:
                meta[key.strip()] = value.strip()
            continue
        heading = parse_heading(line) if line else None
        if heading:
            if "law_id" not in meta:
                raise ValueError("law file has no '# law_id:' header")
            flush()
            current, body, notes = heading, [], []
        elif line.startswith(NOTE_PREFIX):
            notes.append(line.removeprefix(NOTE_PREFIX).strip())
        elif line and current is not None:
            body.append(line)
    flush()
    required = ("law_id", "name", "source_url")
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise ValueError(f"law file is missing header fields: {missing}")
    return Law(
        law_id=meta["law_id"],
        name=meta["name"],
        short_name=meta.get("short_name", meta["name"]),
        number=meta.get("number", ""),
        year=meta.get("year", ""),
        source_url=meta["source_url"],
        version=meta.get("version", ""),
        retrieved=meta.get("retrieved", ""),
        articles=tuple(articles),
        version_ar=meta.get("version_ar", ""),
        coverage_note_ar=meta.get("coverage_note_ar", ""),
    )


def load_laws(data_dir: Path) -> dict[str, Law]:
    laws: dict[str, Law] = {}
    for path in sorted(data_dir.glob("*.txt")):
        law = parse_law_text(path.read_text(encoding="utf-8"))
        if law.law_id in laws:
            raise ValueError(f"duplicate law_id {law.law_id} in {path}")
        laws[law.law_id] = law
    return laws
