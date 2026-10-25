"""Build data/clean/*.txt from the official files in data/raw/ (stdlib only).

Pipeline (see data/SOURCES.md for the sources and README for the commands):

* Income tax law: the ISTD .docx is read directly (tools/docx_text.py).
* Sales tax law: the ISTD PDF is first repaired with
  tools/repair_pdf_tounicode.py, then extracted with
  ``pdftotext -layout -enc UTF-8`` (xpdf 4.x), and that text is cleaned here
  (tools/pdf_text.py).

Output format, one file per law, read by app/parsing.py::

    # law_id: ...            <- header lines "# key: value"
    المادة (1)               <- one heading line per article
    article text ...
    [حاشية] footnote text    <- amendment notes, kept apart from the text

Run with ``python -I tools/build_clean_text.py`` from the repository root.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # our own tools package only; data/ stays off sys.path

from tools.docx_text import read_docx  # noqa: E402
from tools.pdf_text import (  # noqa: E402
    fold_extracted_text,
    join_paragraphs,
    pair_definitions,
    split_articles,
    split_pages,
)

FOOTNOTE_PREFIX = "[حاشية]"
_DOCX_HEADER = re.compile(r"^المادة\s*\(\s*(\d+)\s*\)$")


def _header_block(meta: dict[str, str]) -> list[str]:
    return [f"# {key}: {value}" for key, value in meta.items()]


def build_income_tax(docx_path: Path, meta: dict[str, str]) -> str:
    paragraphs, footnotes = read_docx(str(docx_path))
    out = _header_block(meta)
    title_notes: list[str] = []
    current: str | None = None
    body: list[str] = []
    notes: list[str] = []

    def flush() -> None:
        if current is None:
            return
        out.extend(["", f"المادة ({current})", *body])
        out.extend(f"{FOOTNOTE_PREFIX} {note}" for note in dict.fromkeys(notes))

    for paragraph in paragraphs:
        text = paragraph.text.strip()
        header = _DOCX_HEADER.match(text)
        paragraph_notes = [footnotes[i] for i in paragraph.footnote_ids if i in footnotes]
        if header:
            flush()
            current, body, notes = header.group(1), [], list(paragraph_notes)
            continue
        if current is None:
            title_notes.extend(paragraph_notes)
            continue
        if text:
            body.append(text)
        notes.extend(paragraph_notes)
    flush()
    if title_notes:
        out.insert(len(meta), f"# publication_note: {' '.join(dict.fromkeys(title_notes))}")
    return "\n".join(out) + "\n"


def build_sales_tax(layout_text_path: Path, meta: dict[str, str]) -> str:
    text = fold_extracted_text(layout_text_path.read_text(encoding="utf-8"))
    _title, articles, footnote_lines = split_articles(split_pages(text))
    # one line per footnote: the parser reads a "[حاشية]" line as a single note
    footnote = re.sub(r"^\d", "", " ".join(join_paragraphs(footnote_lines).split())) if footnote_lines else ""
    out = _header_block(meta)
    for article in articles:
        lines = article.lines
        if article.label == "2":
            # article 2 = "أ- intro:", the definitions table, then "ب- ..."
            start = next(i for i, line in enumerate(lines) if line.strip().endswith("غير ذلك:"))
            end = next(i for i, line in enumerate(lines) if line.strip().startswith("ب-"))
            body = (
                join_paragraphs(lines[: start + 1])
                + "\n"
                + "\n".join(pair_definitions(lines[start + 1 : end]))
                + "\n"
                + join_paragraphs(lines[end:])
            )
        else:
            body = join_paragraphs(lines)
        out.extend(["", f"المادة ({article.label})", body])
        if article.footnote_marks and footnote:
            out.append(f"{FOOTNOTE_PREFIX} {footnote}")
    return "\n".join(out) + "\n"


INCOME_META = {
    "law_id": "income_tax_34_2014",
    "name": "قانون ضريبة الدخل رقم (34) لسنة 2014 وتعديلاته",
    "short_name": "قانون ضريبة الدخل",
    "number": "34",
    "year": "2014",
    "source_url": "https://istd.gov.jo/EBV4.0/Root_Storage/AR/EB_Tenders/"
    "%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D8%AF%D8%AE%D9%84_34_"
    "%D9%84%D8%B3%D9%86%D8%A9_2014_%D9%88%D8%AA%D8%B9%D8%AF%D9%8A%D9%84%D8%A7%D8%AA%D9%87.docx",
    "source_file": "data/raw/income_tax_34_2014_istd.docx",
    "version": "ISTD consolidated text: Law No. 34 of 2014 as amended by Law No. 38 of 2018",
    "version_ar": "نص موحّد يشمل تعديلات القانون المعدّل رقم (38) لسنة 2018",
}
SALES_META = {
    "law_id": "sales_tax_6_1994",
    "name": "قانون الضريبة العامة على المبيعات رقم (6) لسنة 1994 وتعديلاته",
    "short_name": "قانون الضريبة العامة على المبيعات",
    "number": "6",
    "year": "1994",
    "source_url": "https://istd.gov.jo/EBV4.0/Root_Storage/AR/EB_Legislation/"
    "%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%B1%D9%82%D9%85_6_%D9%84%D8%B3%D9%86%D8%A9_1994_-_"
    "%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%A7%D9%84%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D8%B9%D8%A7%D9%85%D8%A9_"
    "%D8%B9%D9%84%D9%89_%D8%A7%D9%84%D9%85%D8%A8%D9%8A%D8%B9%D8%A7%D8%AA_"
    "%D9%88%D8%AA%D8%B9%D8%AF%D9%8A%D9%84%D8%A7%D8%AA%D9%87_%D8%A7%D9%8A%D9%84%D9%88%D9%84_2022.pdf",
    "source_file": "data/raw/sales_tax_6_1994_istd_consolidated_2022-09.pdf",
    "version": "ISTD consolidated text dated September 2022 (includes amending Law No. 15 of 2022)",
    "version_ar": "النص الموحّد المنشور على موقع دائرة ضريبة الدخل والمبيعات بتاريخ أيلول 2022",
    "coverage_note_ar": "لا تشمل هذه النصوص الجداول الملحقة بالقانون.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/clean/*.txt")
    parser.add_argument("--retrieved", required=True, help="retrieval date of the raw files, YYYY-MM-DD")
    parser.add_argument(
        "--sales-layout-text",
        type=Path,
        default=REPO_ROOT / "data/interim/sales_tax_6_1994_layout.txt",
        help="pdftotext -layout output of the repaired sales tax PDF",
    )
    args = parser.parse_args()
    clean = REPO_ROOT / "data/clean"
    clean.mkdir(parents=True, exist_ok=True)

    income = build_income_tax(REPO_ROOT / INCOME_META["source_file"], {**INCOME_META, "retrieved": args.retrieved})
    (clean / "income_tax_34_2014.txt").write_text(income, encoding="utf-8", newline="\n")
    sales = build_sales_tax(args.sales_layout_text, {**SALES_META, "retrieved": args.retrieved})
    (clean / "sales_tax_6_1994.txt").write_text(sales, encoding="utf-8", newline="\n")
    print("wrote", clean / "income_tax_34_2014.txt", "and", clean / "sales_tax_6_1994.txt")


if __name__ == "__main__":
    main()
