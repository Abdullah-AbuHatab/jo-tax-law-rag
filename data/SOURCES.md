# Data sources

Both laws are **public legislative texts** of the Hashemite Kingdom of Jordan. The copies here come from the
official website of the Income and Sales Tax Department (ISTD, دائرة ضريبة الدخل والمبيعات). The official
texts stay authoritative. These files were retrieved on **2026-10-24** and may be out of date by the time you read
this. Machine-readable details, including SHA-256 hashes of the raw files, are in [`manifest.json`](manifest.json).

| law_id | Law | Official file | Version |
|---|---|---|---|
| `income_tax_34_2014` | قانون ضريبة الدخل رقم (34) لسنة 2014 وتعديلاته | [ISTD .docx][income] (linked from [this ISTD page][income-page]) | Consolidated: Law 34/2014 as amended by Law 38/2018 |
| `sales_tax_6_1994` | قانون الضريبة العامة على المبيعات رقم (6) لسنة 1994 وتعديلاته | [ISTD PDF, "ايلول 2022"][sales] (linked from [this ISTD page][sales-page]) | Consolidated, dated September 2022 (includes Law 15/2022) |

[income]: https://istd.gov.jo/EBV4.0/Root_Storage/AR/EB_Tenders/%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D8%AF%D8%AE%D9%84_34_%D9%84%D8%B3%D9%86%D8%A9_2014_%D9%88%D8%AA%D8%B9%D8%AF%D9%8A%D9%84%D8%A7%D8%AA%D9%87.docx
[income-page]: https://istd.gov.jo/AR/Pages/2022_%D9%82%D9%88%D8%A7%D9%86%D9%8A%D9%86_%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D8%AF%D8%AE%D9%84
[sales]: https://istd.gov.jo/EBV4.0/Root_Storage/AR/EB_Legislation/%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%B1%D9%82%D9%85_6_%D9%84%D8%B3%D9%86%D8%A9_1994_-_%D9%82%D8%A7%D9%86%D9%88%D9%86_%D8%A7%D9%84%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D8%B9%D8%A7%D9%85%D8%A9_%D8%B9%D9%84%D9%89_%D8%A7%D9%84%D9%85%D8%A8%D9%8A%D8%B9%D8%A7%D8%AA_%D9%88%D8%AA%D8%B9%D8%AF%D9%8A%D9%84%D8%A7%D8%AA%D9%87_%D8%A7%D9%8A%D9%84%D9%88%D9%84_2022.pdf
[sales-page]: https://istd.gov.jo/AR/Pages/%D9%82%D9%88%D8%A7%D9%86%D9%8A%D9%86_%D8%B6%D8%B1%D9%8A%D8%A8%D8%A9_%D8%A7%D9%84%D9%85%D8%A8%D9%8A%D8%B9%D8%A7%D8%AA

## Layout

- `raw/`: the files exactly as downloaded (third-party content; only parse them with `python -I`).
- `clean/`: one UTF-8 text file per law, one `المادة (N)` heading per article, built by
  `tools/build_clean_text.py`. The app reads only these files.
- `interim/` (git-ignored): the repaired PDF and its `pdftotext` output, which can be rebuilt from `raw/`.
- `eval/golden.jsonl`: the evaluation questions. Every answerable item quotes its article in `comment`, and
  `tests/test_eval.py` checks that the quote really appears in that article.

## Income Tax Law (from .docx)

The ISTD file is a native Word document, so its text is exact. `tools/docx_text.py` reads the XML directly and:

- rebuilds the paragraph markers ("أ-", "1-", "أ.") that Word stores as automatic list numbering instead of text.
  It follows Word's rule that list instances sharing one abstract list share a counter. That is why article
  4/أ runs from 1 to 20, as the article's own reference to "البند (18) من الفقرة (أ)" requires. Check: all 39
  references of the form "الفقرة (X) من هذه المادة" resolve to a paragraph that exists;
- turns the definitions table of article 2 into "term: definition" lines;
- keeps the footnotes ("عدل النص بموجب القانون المعدل رقم (38) لسنة 2018", and the Constitutional Court decision
  on article 72) as `[حاشية]` notes. They go into article metadata, never into the article text.

## General Sales Tax Law (from PDF)

ISTD publishes this law only as PDF. Exported from Word 2010, the PDF prints correctly, but its text layer is
garbled: every extractor reads "لبْٔٛ" where the page shows "قانون". The cause is a known Word bug in the PDF's
ToUnicode maps. Word wrote `bfrange` entries meaning "all these glyphs are alef", but a `bfrange` increments
its target, so the forms of one letter come out as a run of consecutive different letters.

Nothing was retyped or guessed. The repair (`tools/repair_pdf_tounicode.py`) works only from data inside the PDF:

1. For each embedded font, read its own `cmap` (letter → glyph) and `GSUB` tables (glyph → initial, medial and
   final forms, and lam-alef ligatures). That yields the true letter for every glyph id.
2. Write a corrected ToUnicode map, then extract with `pdftotext -layout` (xpdf 4.06).

Evidence that the repair is right: for Arabic letters, every *explicit* single-glyph entry in the original maps
agrees with the font tables. The disagreements are the broken ranges, the digit glyphs (whose original entries are
inconsistent too), and parentheses, which right-to-left text draws mirrored (`Times New Roman`: 43 agree, 96
corrected).

`tools/pdf_text.py` then applies mechanical clean-up. No rule adds or removes a word:

- NFKC normalization; Persian yeh and heh glyphs (which Times New Roman uses for Arabic yeh and heh) folded back
  to U+064A and U+0647; bidi control characters dropped.
- Arabic-Indic digits converted to ASCII digits, to match the income tax file.
- Spaces and punctuation around numbers restored. In right-to-left text the extractor moves the space after an
  LTR run in front of it ("المادة  (14)من" → "المادة (14) من", "-1بيع" → "1- بيع", "(%16)" → "(16%)").
- Page numbers removed. The footnote on page 36 (legislation repealed by Law 29 of 2009) is attached to
  article 76 as a note.
- The two-column definitions table of article 2 paired back into "term: definition" lines. Each pair was read
  and checked by hand.

Independent check: a word-level comparison with an unofficial web copy of the law (jordan-lawyer.com, used
only for verification, never as a source) agrees on 96.9% of 9,109 tokens. The disagreements are of two kinds:
articles changed by the 2022 amendment, where the web copy is older (articles 2, 4, 9, 20, and 4 مكرر, which
the web copy lacks), and typos in the web copy ("البندول" vs "البندرول", "السع" vs "السلع"). All 20 intra-article
paragraph references resolve.

## Rebuilding data/clean

```bash
python -m venv .venv-tools && .venv-tools/bin/pip install -r tools/requirements.txt   # pypdf, fonttools
.venv-tools/bin/python -I tools/repair_pdf_tounicode.py     data/raw/sales_tax_6_1994_istd_consolidated_2022-09.pdf data/interim/sales_tax_6_1994_tounicode_repaired.pdf
pdftotext -layout -enc UTF-8     data/interim/sales_tax_6_1994_tounicode_repaired.pdf data/interim/sales_tax_6_1994_layout.txt   # xpdf 4.06
python -I tools/build_clean_text.py --retrieved 2026-10-24
```

The build is deterministic: re-running it reproduces `data/clean/*.txt` byte for byte.

## Known gaps and caveats

- **Currency:** both texts are consolidations published by ISTD. They may not include amendments made after
  they were prepared (income tax file saved 2022-01-04; sales tax file labelled September 2022). Check the
  Legislation and Opinion Bureau (lob.gov.jo) or the Official Gazette before relying on any provision.
- **Sales tax schedules** (الجداول رقم 1-4, the lists of goods and rates) are separate instruments and are not
  included.
- **Punctuation in the sales tax text** around numbers was repaired by rule. The words are verified; a
  parenthesis or a space next to a number may still differ from the printed page in rare places. The official
  PDF is linked from every citation.
- lob.gov.jo was not used: its API is encrypted and protected by reCAPTCHA, and that protection was not bypassed.
