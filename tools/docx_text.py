"""Extract readable paragraphs from a .docx law file (stdlib only).

Word stores the sub-paragraph markers of a law ("أ-", "1.") as automatic
list numbering, not as text, so a plain text dump loses them. This module
re-creates the markers from word/numbering.xml so that the cleaned text
reads like the printed law. It also turns the definitions table into
"term: definition" lines and returns footnotes separately (in the ISTD
consolidated file the footnotes are the amendment notes).

Run only on the official copies in data/raw/, and always with ``python -I``.
The stdlib XML parser is used on purpose (no extra dependency): Python 3.12
ships expat >= 2.4.1, which refuses entity-expansion bombs, and .docx parts
contain no external entities to resolve.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Letter sequences Word uses for the two Arabic list formats (ECMA-376,
# ST_NumberFormat "arabicAbjad" and "arabicAlpha"). Heh is written with a
# tatweel ("هـ") the way Jordanian legislation prints it.
ARABIC_ABJAD = [
    "أ",
    "ب",
    "ج",
    "د",
    "هـ",
    "و",
    "ز",
    "ح",
    "ط",
    "ي",
    "ك",
    "ل",
    "م",
    "ن",
    "س",
    "ع",
    "ف",
    "ص",
    "ق",
    "ر",
    "ش",
    "ت",
    "ث",
    "خ",
    "ذ",
    "ض",
    "ظ",
    "غ",
]
ARABIC_ALPHA = [
    "أ",
    "ب",
    "ت",
    "ث",
    "ج",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ك",
    "ل",
    "م",
    "ن",
    "هـ",
    "و",
    "ي",
]


@dataclass
class Level:
    start: int = 1
    num_format: str = "decimal"
    text: str = "%1."


@dataclass
class Paragraph:
    text: str
    footnote_ids: list[str] = field(default_factory=list)


def _format_number(value: int, num_format: str) -> str:
    if num_format == "arabicAbjad":
        return ARABIC_ABJAD[(value - 1) % len(ARABIC_ABJAD)]
    if num_format == "arabicAlpha":
        return ARABIC_ALPHA[(value - 1) % len(ARABIC_ALPHA)]
    if num_format == "lowerLetter":
        return chr(ord("a") + (value - 1) % 26)
    if num_format == "lowerRoman":
        numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
        return numerals[(value - 1) % len(numerals)]
    return str(value)


def _parse_levels(lvl_parent: ET.Element) -> dict[int, Level]:
    levels: dict[int, Level] = {}
    for lvl in lvl_parent.findall(f"{W}lvl"):
        ilvl = int(lvl.get(f"{W}ilvl", "0"))
        start = lvl.find(f"{W}start")
        fmt = lvl.find(f"{W}numFmt")
        text = lvl.find(f"{W}lvlText")
        levels[ilvl] = Level(
            start=int(start.get(f"{W}val")) if start is not None else 1,
            num_format=fmt.get(f"{W}val") if fmt is not None else "decimal",
            text=text.get(f"{W}val") if text is not None else "",
        )
    return levels


class Numbering:
    """Minimal model of word/numbering.xml.

    Like Word, list instances (w:num) that point at the same abstract list
    share one counter, so a list split by an amendment keeps counting
    (article 4/a of the income tax law runs 1..20 across two instances; its
    own cross-reference to "البند (18)" confirms that). A w:startOverride is
    the only thing that restarts a shared counter.
    """

    def __init__(self, xml: bytes | None) -> None:
        self._levels: dict[str, dict[int, Level]] = {}
        self._counter_key: dict[str, str] = {}
        self._counters: dict[str, dict[int, int]] = {}
        if not xml:
            return
        root = ET.fromstring(xml)  # noqa: S314 - see module docstring
        abstract = {a.get(f"{W}abstractNumId"): _parse_levels(a) for a in root.findall(f"{W}abstractNum")}
        for num in root.findall(f"{W}num"):
            num_id = num.get(f"{W}numId")
            ref = num.find(f"{W}abstractNumId")
            abstract_id = ref.get(f"{W}val") if ref is not None else ""
            levels = dict(abstract.get(abstract_id, {}))
            restarts = False
            for override in num.findall(f"{W}lvlOverride"):
                levels.update(_parse_levels(override))
                restarts = restarts or override.find(f"{W}startOverride") is not None
            self._levels[num_id] = levels
            self._counter_key[num_id] = f"num:{num_id}" if restarts else f"abstract:{abstract_id}"

    def next_label(self, num_id: str, ilvl: int) -> str:
        levels = self._levels.get(num_id)
        if not levels or ilvl not in levels or num_id == "0":
            return ""
        counters = self._counters.setdefault(self._counter_key[num_id], {})
        level = levels[ilvl]
        counters[ilvl] = counters.get(ilvl, level.start - 1) + 1
        for deeper in [k for k in counters if k > ilvl]:
            del counters[deeper]  # a higher-level item restarts the sub-levels
        label = level.text
        for k in range(ilvl + 1):
            if f"%{k + 1}" in label:
                value = counters.get(k, levels.get(k, Level()).start)
                fmt = levels.get(k, Level()).num_format
                label = label.replace(f"%{k + 1}", _format_number(value, fmt))
        return label


def _run_text(element: ET.Element) -> tuple[str, list[str]]:
    parts: list[str] = []
    footnotes: list[str] = []
    for node in element.iter():
        if node.tag == f"{W}t":
            parts.append(node.text or "")
        elif node.tag in (f"{W}tab", f"{W}br"):
            parts.append(" ")
        elif node.tag == f"{W}footnoteReference":
            footnotes.append(node.get(f"{W}id", ""))
    return "".join(parts), footnotes


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _paragraph(p: ET.Element, numbering: Numbering) -> Paragraph:
    text, footnotes = _run_text(p)
    text = _clean_spaces(text)
    num_pr = p.find(f"{W}pPr/{W}numPr")
    # Word prints the number even on an empty list paragraph (the ISTD file
    # uses that for "ب- 1- ..." items whose first line is a sub-list), so an
    # empty numbered paragraph still consumes a counter and keeps its label.
    if num_pr is not None:
        num_id = num_pr.find(f"{W}numId")
        ilvl = num_pr.find(f"{W}ilvl")
        label = numbering.next_label(
            num_id.get(f"{W}val") if num_id is not None else "0",
            int(ilvl.get(f"{W}val")) if ilvl is not None else 0,
        )
        if label:
            text = f"{label} {text}".strip()
    return Paragraph(text=text, footnote_ids=footnotes)


def read_docx(path: str) -> tuple[list[Paragraph], dict[str, str]]:
    """Return the body paragraphs (tables flattened to one line per row) and footnotes."""
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))  # noqa: S314
        names = archive.namelist()
        numbering = Numbering(archive.read("word/numbering.xml") if "word/numbering.xml" in names else None)
        footnotes_xml = archive.read("word/footnotes.xml") if "word/footnotes.xml" in names else None

    paragraphs: list[Paragraph] = []
    body = document.find(f"{W}body")
    for block in list(body) if body is not None else []:
        if block.tag == f"{W}p":
            paragraphs.append(_paragraph(block, numbering))
        elif block.tag == f"{W}tbl":
            for row in block.findall(f"{W}tr"):
                cells = []
                row_footnotes: list[str] = []
                for cell in row.findall(f"{W}tc"):
                    cell_parts = [_paragraph(p, numbering) for p in cell.findall(f"{W}p")]
                    cells.append(" ".join(c.text for c in cell_parts if c.text))
                    for c in cell_parts:
                        row_footnotes.extend(c.footnote_ids)
                line = _clean_spaces(" ".join(c for c in cells if c))
                line = re.sub(r"\s*:\s*", ": ", line, count=1)
                paragraphs.append(Paragraph(text=line, footnote_ids=row_footnotes))

    footnotes: dict[str, str] = {}
    if footnotes_xml:
        for note in ET.fromstring(footnotes_xml).findall(f"{W}footnote"):  # noqa: S314
            note_id = note.get(f"{W}id", "")
            text, _ = _run_text(note)
            text = _clean_spaces(text)
            if text:
                footnotes[note_id] = text
    return paragraphs, footnotes
