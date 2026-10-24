"""Arabic text helpers.

``normalize_arabic`` is for matching only (metadata, keyword checks, the
fake embedder, eval evidence checks). The original law text is always kept
for display; normalization is lossy on purpose.
"""

from __future__ import annotations

import re
import unicodedata

# Harakat (fathatan .. sukun), superscript alef, Quranic marks.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_INDIC_DIGITS = {ord("٠") + d: ord("0") + d for d in range(10)}
_EASTERN_ARABIC_INDIC_DIGITS = {ord("۰") + d: ord("0") + d for d in range(10)}
_LETTER_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ی": "ي",  # Persian yeh, common in PDF extractions
        "ک": "ك",  # Persian keheh
        "ھ": "ه",  # heh doachashmee
    }
)


def strip_tatweel(text: str) -> str:
    """Remove kashida (U+0640). It only stretches words for justification, and
    the sales tax PDF is full of it ("يـتم رد الضـريبة"); embedding models see
    it as noise inside the word."""
    return text.replace(_TATWEEL, "")


def to_ascii_digits(text: str) -> str:
    return text.translate(_ARABIC_INDIC_DIGITS).translate(_EASTERN_ARABIC_INDIC_DIGITS)


def normalize_arabic(text: str) -> str:
    """Fold spelling variants so that equal words compare equal.

    Removes diacritics and tatweel, unifies alef/hamza forms, ta marbuta ->
    heh, alef maqsura -> yeh, Arabic-Indic digits -> ASCII, and collapses
    whitespace. Presentation forms are decomposed first (NFKC).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = to_ascii_digits(text).translate(_LETTER_FOLD)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalized word tokens (Arabic letters, Latin letters, digits)."""
    return re.findall(r"[ء-ي]+|[a-z]+|\d+", normalize_arabic(text).lower())


def is_mostly_arabic(text: str) -> bool:
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return False
    arabic = sum(1 for ch in letters if "؀" <= ch <= "ۿ")
    return arabic / len(letters) >= 0.5
