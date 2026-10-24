"""Clean the text that ``pdftotext -layout`` extracts from the repaired sales tax PDF.

Every rule here is mechanical and documented; none of them adds or removes a
word. They fix four artefacts of extracting right-to-left text from a PDF:

1. Unicode forms: NFKC turns presentation forms (U+FExx) into normal
   letters; Times New Roman draws Arabic yeh/heh with the Persian glyphs, so
   U+06CC/U+06BE are folded back to U+064A/U+0647; bidi control characters
   are dropped; Arabic-Indic digits become ASCII digits (the income tax law
   file already uses ASCII digits, and the two laws should match).
2. Number runs: an LTR run such as "(14)" inside Arabic is written with the
   space that followed it placed before it, e.g. "المادة  (14)من". The rules
   in ``fix_number_runs`` move that space back and restore "1- " list
   markers and "16%".
3. Page furniture: page numbers and the footnote block are separated from
   the article text.
4. The definitions table of article 2, whose two columns come out one after
   the other, is paired back into "term: definition" lines.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

ARABIC_LETTER = "ء-ي"
_FOLD = {
    0x06CC: 0x064A,  # Farsi yeh glyph used for Arabic yeh
    0x06BE: 0x0647,  # heh doachashmee glyph used for Arabic heh
    0x200E: None,
    0x200F: None,
    0x202A: None,
    0x202B: None,
    0x202C: None,
    **{0x0660 + d: 0x30 + d for d in range(10)},
}
_LETTER_AHEAD = rf"(?=[{ARABIC_LETTER}])"


def fold_extracted_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_FOLD)


def fix_number_runs(line: str) -> str:
    """Undo the mis-placed spaces/punctuation around digit runs in one line."""
    # "4)مكرر)" -> "(4 مكرر)"  (article "four bis")
    line = re.sub(r"\(?(\d+)\)(مكـ?رر)\)", r"(\1 \2)", line)
    # "-1بيع" / ".1الجدول" -> "1- بيع" / "1. الجدول"
    line = re.sub(rf"(^|\s)([-.])(\d+){_LETTER_AHEAD}", r"\1\3\2 ", line)
    # "لسنة  (1994ويعمل" -> "لسنة 1994) ويعمل"  (a closing paren drawn mirrored)
    line = re.sub(rf"\s\s\((\d+){_LETTER_AHEAD}", r" \1) ", line)
    # "(%16)" -> "(16%)"
    line = re.sub(r"\(%(\d+(?:\.\d+)?)\)", r"(\1%)", line)
    # "X  (14)من" -> "X (14) من" ; "و (3)الملحقين" -> "و(3) الملحقين"
    line = re.sub(rf"\s(\([^()\s]*\d[^()\s]*\)){_LETTER_AHEAD}", r"\1 ", line)
    # a run at the very start of the line lost its trailing space to the previous line
    line = re.sub(rf"^(\s*\([^()\s]*\d[^()\s]*\)){_LETTER_AHEAD}", r"\1 ", line)
    # "رقم  3الملحق" -> "رقم 3 الملحق"
    line = re.sub(rf"\s\s(\d+){_LETTER_AHEAD}", r" \1 ", line)
    # "رقم .(2)" -> "رقم (2)." ; "لسنة .1963" -> "لسنة 1963."
    line = re.sub(r"\s\.(\(\d+\)|\d+)(?=\s|$)", r" \1.", line)
    # "اعتبارا من :2010/1/1" -> "اعتبارا من 2010/1/1:"
    line = re.sub(r":(\d+/\d+/\d+)(?=\s|$)", r"\1:", line)
    return re.sub(r"[ \t]{2,}", " ", line).strip()


# "المادة (12)", "المادة ( ( 1", "المادة (4)مكرر", "1المادة (76)" (footnote mark 1)
_HEADER = re.compile(r"^(?P<mark>\d)?المادة\s*\(\s*\(?\s*(?P<num>\d+)\s*\)?\s*(?P<bis>مكـ?رر)?\s*$")
_FOOTNOTE = re.compile(rf"^(?P<mark>\d)(?=[{ARABIC_LETTER}])")
# Paragraph markers that start a new line in the cleaned text: "أ-", "هـ -", "ه-", "1-", "2."
_MARKER = re.compile(r"\s(?=(?:هـ|[أبجدهوزحطيكلمنسعفصقرش])\s?-|\d{1,2}[-.]\s)")


@dataclass
class RawArticle:
    label: str
    lines: list[str] = field(default_factory=list)
    footnote_marks: list[str] = field(default_factory=list)


def split_pages(text: str) -> list[list[str]]:
    """Split on form feeds and drop each page's trailing page number."""
    pages = []
    for page in text.split("\f"):
        lines = page.split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().isdigit():
            lines.pop()
        pages.append(lines)
    return pages


def parse_header(line: str) -> tuple[str, str | None] | None:
    match = _HEADER.match(line.strip())
    if not match:
        return None
    label = match.group("num") + (" مكرر" if match.group("bis") else "")
    return label, match.group("mark")


def split_articles(pages: list[list[str]]) -> tuple[list[str], list[RawArticle], list[str]]:
    """Return (title lines, articles with their raw layout lines, footnote lines)."""
    title: list[str] = []
    articles: list[RawArticle] = []
    footnotes: list[str] = []
    in_footnote = False
    for lines in pages:
        in_footnote = False  # a footnote block ends with its page
        for line in lines:
            stripped = line.strip()
            header = parse_header(stripped)
            if header:
                label, mark = header
                articles.append(RawArticle(label=label, footnote_marks=[mark] if mark else []))
                continue
            if _FOOTNOTE.match(stripped) and articles:
                in_footnote = True
            if in_footnote:
                footnotes.append(stripped)
            elif articles:
                articles[-1].lines.append(line)
            elif stripped:
                title.append(stripped)
    return title, articles, footnotes


def join_paragraphs(lines: list[str]) -> str:
    """Join soft-wrapped layout lines and put each paragraph marker on its own line."""
    text = " ".join(fix_number_runs(line) for line in lines if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return _MARKER.sub("\n", text)


# Two lines on page 3 of the PDF where the term and definition columns touch,
# split by hand: (exact text) -> (term part, definition part).
_TOUCHING_COLUMNS = {
    "الاقتصادية الخاصة: قـانون منطقـة العقبـة الاقتصـادية الخاصة.": (
        "الاقتصادية الخاصة",
        "قـانون منطقـة العقبـة الاقتصـادية الخاصة.",
    ),
    ": هـو المكـان أو البنـاء الـذي تـودع فيـه البضائع  المستودع:": (
        "المستودع",
        ": هـو المكـان أو البنـاء الـذي تـودع فيـه البضائع",
    ),
}
_TERM_COLUMN = 50  # layout columns: terms start right of this, definitions left of it


def pair_definitions(lines: list[str]) -> list[str]:
    """Rebuild "term: definition" lines from the two-column table of article 2.

    Terms and definitions are both in reading order, so the k-th term belongs
    to the k-th definition that did not already carry its own term. Raises
    ValueError when the counts differ, so a layout change cannot silently
    mis-pair a definition.
    """
    definitions: list[dict[str, list[str] | str | None]] = []
    terms: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped in _TOUCHING_COLUMNS:
            term_part, definition_part = _TOUCHING_COLUMNS[stripped]
            if definition_part.startswith(":"):
                terms.append(term_part)
                definitions.append({"term": None, "text": [definition_part[1:].strip()]})
            else:
                terms[-1] = f"{terms[-1]} {term_part}"
                definitions[-1]["text"].append(definition_part)  # type: ignore[union-attr]
            continue
        for match in re.finditer(r"\S+(?: {1,3}\S+)*", raw):
            segment = match.group(0).strip()
            if match.start() >= _TERM_COLUMN:
                terms.append(segment.rstrip(":").strip())
            elif segment.startswith(":"):
                definitions.append({"term": None, "text": [segment[1:].strip()]})
            elif inline := re.match(rf"^([{ARABIC_LETTER}ـ ]{{2,30}}?)\s*:\s*(\S.*)$", segment):
                definitions.append({"term": inline.group(1).strip(), "text": [inline.group(2)]})
            elif definitions:
                definitions[-1]["text"].append(segment)  # type: ignore[union-attr]
            else:
                raise ValueError(f"definition text before any definition: {segment!r}")
    open_slots = [d for d in definitions if d["term"] is None]
    if len(open_slots) != len(terms):
        raise ValueError(f"{len(terms)} terms but {len(open_slots)} definitions without a term")
    for slot, term in zip(open_slots, terms, strict=True):
        slot["term"] = term
    paired = []
    for definition in definitions:
        body = " ".join(fix_number_runs(part) for part in definition["text"])  # type: ignore[union-attr]
        paired.append(f"{definition['term']}: {re.sub(r'\s+', ' ', body).strip()}")
    return paired
