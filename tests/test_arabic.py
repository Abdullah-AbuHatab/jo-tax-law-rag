from app.arabic import is_mostly_arabic, normalize_arabic, to_ascii_digits, tokenize


def test_normalize_removes_diacritics_and_tatweel():
    assert normalize_arabic("الضَّرِيبَةُ") == "الضريبه"
    assert normalize_arabic("يعـد بيعـاً") == "يعد بيعا"


def test_normalize_unifies_alef_hamza_forms():
    assert normalize_arabic("أإآٱا") == "ااااا"
    assert normalize_arabic("مسؤول") == normalize_arabic("مسوول")
    assert normalize_arabic("هيئة") == "هييه"


def test_normalize_folds_ta_marbuta_and_alef_maqsura():
    assert normalize_arabic("الدائرة") == normalize_arabic("الدائره")
    assert normalize_arabic("على") == normalize_arabic("علي")


def test_normalize_handles_persian_letters_and_presentation_forms():
    # Persian yeh / keheh / heh-doachashmee come out of some PDF extractions
    assert normalize_arabic("ﻗﺎﻧﻮن") == "قانون"  # presentation forms (U+FExx)
    assert normalize_arabic("یکون") == normalize_arabic("يكون")
    assert normalize_arabic("ھذا") == "هذا"


def test_arabic_indic_digits_become_ascii():
    assert to_ascii_digits("المادة (٣٤) لسنة ٢٠١٤") == "المادة (34) لسنة 2014"
    assert to_ascii_digits("۱۲") == "12"


def test_normalize_collapses_whitespace():
    assert normalize_arabic("  قانون \n\t ضريبة  ") == "قانون ضريبه"


def test_tokenize_matches_spelling_variants():
    assert tokenize("الإقرار الضريبي") == tokenize("الاقرار الضريبى")


def test_is_mostly_arabic():
    assert is_mostly_arabic("ما هي نسبة الضريبة؟")
    assert not is_mostly_arabic("What is the sales tax rate?")
    assert not is_mostly_arabic("123 ???")
