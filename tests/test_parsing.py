"""Article parsing, on real text from data/clean and on heading variants."""

import pytest

from app.parsing import (
    normalize_article_ref,
    ordinal_to_int,
    parse_heading,
    parse_law_text,
)

HEADER = "# law_id: test_law\n# name: قانون تجريبي\n# source_url: https://example.invalid/law\n"


def test_clean_files_have_every_article(laws):
    income = laws["income_tax_34_2014"]
    sales = laws["sales_tax_6_1994"]
    assert [a.number for a in income.articles] == list(range(1, 83))
    # 1..77 plus "4 مكرر" (added by amending Law No. 15 of 2022)
    assert len(sales.articles) == 78
    assert [a.key for a in sales.articles[:6]] == ["1", "2", "3", "4", "4bis", "5"]
    assert all(a.text for law in laws.values() for a in law.articles)


def test_real_article_text_income_tax_personal_exemption(laws):
    article = laws["income_tax_34_2014"].article("9")
    assert article is not None
    assert "عشرة آلاف دينار إعفاءً شخصياً لسنة (2019)" in article.text
    # the docx list numbering was rebuilt: paragraph markers are present
    assert article.text.startswith("أ- للتوصل إلى الدخل الخاضع للضريبة")


def test_real_article_text_sales_tax_rate(laws):
    article = laws["sales_tax_6_1994"].article("6")
    assert article is not None
    assert article.text.startswith(
        "أ- تفرض بمقتضى هذا القانون ضريبة عامة على استيراد أي سلعة أو خدمة أو توريد أي منهما بنسبة (16%)"
    )


def test_real_article_bis_label(laws):
    article = laws["sales_tax_6_1994"].article("4bis")
    assert article is not None
    assert article.label == "4 مكرر"
    assert "يعتبـر توريـد السـلـع قـد تـم داخـل المنطقـة الضـريبية" in article.text


def test_amendment_notes_are_kept_out_of_the_text(laws):
    article = laws["income_tax_34_2014"].article("3")
    assert article is not None
    assert article.notes == ("عدل النص بموجب القانون المعدل رقم (38) لسنة 2018.",)
    assert "[حاشية]" not in article.text


def test_parse_snippet_from_clean_file(clean_text):
    # first two articles of the real sales tax file, parsed on their own
    text = clean_text("sales_tax_6_1994.txt")
    snippet = text[: text.index("المادة (3)")]
    law = parse_law_text(snippet)
    assert law.law_id == "sales_tax_6_1994"
    assert [a.key for a in law.articles] == ["1", "2"]
    assert "الوزير: وزير المالية." in law.articles[1].text


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("المادة (12)", (12, False)),
        ("المادة ( 1 )", (1, False)),
        ("المادة (٣٤)", (34, False)),
        ("المادة ١٢", (12, False)),
        ("مادة 7", (7, False)),
        ("المادة الأولى", (1, False)),
        ("المادة العاشرة", (10, False)),
        ("المادة الحادية عشرة", (11, False)),
        ("المادة الخامسة والعشرون", (25, False)),
        ("المادة (4 مكرر)", (4, True)),
        ("المادة (4) مكرر", (4, True)),
        ("المادة (6) من هذا القانون.", None),  # a reference inside text, not a heading
        ("المادة القيام بأي مما يلي:", None),  # a wrapped line that starts with the word
    ],
)
def test_parse_heading_variants(line, expected):
    assert parse_heading(line) == expected


def test_ordinal_unknown_words_return_none():
    assert ordinal_to_int("القيام بأي") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (6, "6"),
        ("6", "6"),
        ("٦", "6"),
        ("المادة 6", "6"),
        ("المادة (6)", "6"),
        ("4 مكرر", "4bis"),
        ("4مكرر", "4bis"),
        ("4bis", "4bis"),
        ("(4 مكرر)", "4bis"),
        ("abc", None),
        ("12a", None),
        (True, None),
        (None, None),
    ],
)
def test_normalize_article_ref(value, expected):
    assert normalize_article_ref(value) == expected


def test_parse_law_text_with_ordinal_and_indic_headings():
    law = parse_law_text(HEADER + "المادة الأولى\nنص أول.\nالمادة (٢)\nنص ثان.\n[حاشية] ملاحظة.\n")
    assert [(a.key, a.text, a.notes) for a in law.articles] == [
        ("1", "نص أول.", ()),
        ("2", "نص ثان.", ("ملاحظة.",)),
    ]


def test_duplicate_article_is_an_error():
    with pytest.raises(ValueError, match="duplicate article"):
        parse_law_text(HEADER + "المادة (1)\nأ\nالمادة (1)\nب\n")


def test_missing_header_is_an_error():
    with pytest.raises(ValueError, match="law_id"):
        parse_law_text("المادة (1)\nنص\n")


def _keys(pair):
    return tuple(article.key if article else None for article in pair)


def test_law_neighbours_follow_document_order(laws):
    sales, income = laws["sales_tax_6_1994"], laws["income_tax_34_2014"]
    assert _keys(sales.neighbours("4")) == ("3", "4bis")
    assert _keys(sales.neighbours("4bis")) == ("4", "5")
    assert _keys(sales.neighbours("77")) == ("76", None)
    assert _keys(income.neighbours("1")) == (None, "2")
    assert _keys(income.neighbours("82")) == ("81", None)
    with pytest.raises(KeyError):
        sales.neighbours("78")


def test_parse_law_text_reads_optional_arabic_metadata():
    law = parse_law_text(HEADER + "# version_ar: نص تجريبي موحّد\n\nالمادة (1)\nنص المادة.\n")
    assert law.version_ar == "نص تجريبي موحّد"
    assert law.coverage_note_ar == ""


def test_clean_files_carry_arabic_version_labels(laws):
    income, sales = laws["income_tax_34_2014"], laws["sales_tax_6_1994"]
    assert income.version_ar == "نص موحّد يشمل تعديلات القانون المعدّل رقم (38) لسنة 2018"
    assert sales.version_ar == "النص الموحّد المنشور على موقع دائرة ضريبة الدخل والمبيعات بتاريخ أيلول 2022"
    assert sales.coverage_note_ar == "لا تشمل هذه النصوص الجداول الملحقة بالقانون."
    assert income.coverage_note_ar == ""
