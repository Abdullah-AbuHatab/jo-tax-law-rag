"""Static UI contract: structure, CSS validity, honest copy and data bindings of app/static."""

import json
import re
from html.parser import HTMLParser

from app.config import REPO_ROOT
from app.main import STATIC_DIR

PAGE = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
SCRIPT = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
LAW_IDS = [
    law["law_id"] for law in json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["laws"]
]
RESULTS = (REPO_ROOT / "eval" / "results.md").read_text(encoding="utf-8")


class _StartTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, {name: value or "" for name, value in attrs}))


def page_elements(html: str = PAGE) -> list[tuple[str, dict[str, str]]]:
    parser = _StartTags()
    parser.feed(html)
    return parser.elements


def top_level_preludes(css: str) -> list[str]:
    """Selectors / at-rule preludes at depth 0; asserts every brace is balanced."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    depth, buffer, preludes = 0, [], []
    for char in css:
        if char == "{":
            if depth == 0:
                preludes.append("".join(buffer).strip())
                buffer = []
            depth += 1
        elif char == "}":
            depth -= 1
            assert depth >= 0, "stray closing brace in style.css"
        elif depth == 0:
            buffer.append(char)
    assert depth == 0, "unclosed block in style.css"
    return preludes


def test_css_has_no_stray_braces_or_declarations_outside_rules():
    for prelude in top_level_preludes(CSS):
        assert prelude.startswith("@") or ";" not in prelude, prelude[:80]


def test_css_has_no_important_no_transition_all_and_no_shrunken_targets():
    assert "!important" not in CSS
    assert "transition: all" not in CSS
    assert "calc(var(--target) -" not in CSS


def test_every_css_custom_property_is_defined_and_used():
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", CSS))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", CSS))
    assert used - defined == set()
    assert defined - used == set()


def test_arabic_text_is_never_smaller_than_text_sm():
    assert "--text-xs" not in CSS
    literal = set(re.findall(r"font-size:\s*([0-9.]+(?:rem|em|px))", CSS))
    assert literal <= {"0.875em", "0.8125rem"}  # code and kbd only


def test_index_ids_are_unique_and_every_idref_resolves():
    elements = page_elements()
    ids = [attrs["id"] for _, attrs in elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    keys = ("for", "aria-labelledby", "aria-describedby", "aria-controls")
    refs = {ref for _, attrs in elements for key in keys for ref in attrs.get(key, "").split()}
    assert refs <= set(ids), refs - set(ids)


def test_no_tab_roles_without_tabpanels():
    roles = [attrs.get("role") for _, attrs in page_elements()]
    assert "tab" not in roles or "tabpanel" in roles


def test_ui_hardcodes_no_law_ids():
    for law_id in LAW_IDS:
        assert law_id not in PAGE and law_id not in SCRIPT, law_id


def test_page_keeps_its_honest_caveats():
    english = re.search(r'<section[^>]*lang="en"[^>]*>(.*?)</section>', PAGE, re.S)
    assert english and "Not legal advice" in english.group(1)
    assert "لا تضمن صحة كل إجابة" in PAGE
    for claim in (
        "zero hallucinated",
        "Production-hardened",
        "تحقق برمجي 100%",
        "موثّق",
        "موثق",
        "بدقة مثبتة",
        "Grounded Legal Truth",
        "Zero Hallucination",
        "hallucination",
        "هلوس",
        "مدقق",
        "مضمون",
        "حقيقة قانونية",
        "0% Uncited",
    ):
        assert claim.lower() not in PAGE.lower() and claim.lower() not in SCRIPT.lower(), claim


def test_eval_numbers_on_the_page_match_results_md():
    summary = dict(re.findall(r"^\| ([a-z@0-9 ]+) \| ([0-9.]+)% \|$", RESULTS, re.M))
    date = re.search(r"^- date: `(\d{4}-\d{2}-\d{2})", RESULTS, re.M)
    table = re.search(r'<table class="eval-table">(.*?)</table>', PAGE, re.S)
    assert date and table
    shown = [float(value) for value in re.findall(r"<bdi>([0-9.]+)%</bdi>", table.group(1))]
    assert shown == [float(summary[key]) for key in ("recall@5", "refusal accuracy", "citation validity")]
    assert date.group(1) in table.group(1)


def test_every_ask_trigger_is_locked_while_a_request_runs():
    triggers = [(tag, attrs) for tag, attrs in page_elements() if "data-question" in attrs]
    assert triggers and all(tag == "button" for tag, _ in triggers)
    assert SCRIPT.count('querySelectorAll("button[data-question]")') >= 2  # refreshButtons and the binding
    assert 'querySelectorAll("button.example")' not in SCRIPT


def test_demo_refusal_is_refused_at_the_gate_in_the_eval():
    demos = [attrs for _, attrs in page_elements() if attrs.get("data-demo") == "gate-refusal"]
    assert len(demos) == 1
    golden_path = REPO_ROOT / "data" / "eval" / "golden.jsonl"
    golden = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    case = next(item for item in golden if item["question"] == demos[0]["data-question"])
    assert case["should_refuse"] is True
    row = next(line for line in RESULTS.splitlines() if line.startswith(f"| {case['id']} |"))
    assert "| LOW_RETRIEVAL_SCORE |" in row


def test_answer_actions_are_honest_and_portable():
    copy = next(attrs for _, attrs in page_elements() if attrs.get("id") == "copy-answer")
    assert "aria-label" not in copy
    assert "onrender.com" not in SCRIPT and "location.origin" in SCRIPT
    assert "citations[0]" not in SCRIPT
    assert "matchAll(ARTICLE_MENTION)" in SCRIPT


def test_app_uses_no_client_storage():
    assert "localStorage" not in SCRIPT and "sessionStorage" not in SCRIPT


def test_the_question_never_goes_into_a_url():
    assert 'searchParams.set("q"' not in SCRIPT and 'params.get("q")' not in SCRIPT


def test_privacy_notice_describes_the_question_box():
    box = next(attrs for tag, attrs in page_elements() if tag == "textarea" and attrs.get("id") == "question")
    assert "privacy-notice" in box["aria-describedby"].split()


def test_there_is_exactly_one_law_scope_control():
    controls = [(tag, attrs) for tag, attrs in page_elements() if attrs.get("name") == "law"]
    assert [(tag, attrs.get("type"), attrs.get("value"), attrs.get("id")) for tag, attrs in controls] == [
        ("input", "radio", "", "scope-all")
    ]
    assert "checked" in controls[0][1]
    assert re.search(r'<fieldset id="scope"[^>]*>\s*<legend[^>]*>نطاق البحث</legend>', PAGE)
    form = PAGE.split('id="ask-form"', 1)[1].split("</form>", 1)[0]
    assert "<select" not in form
    assert '$("law")' not in SCRIPT and "law-pill" not in PAGE + SCRIPT + CSS


def test_ask_form_never_falls_back_to_get():
    elements = page_elements()
    form = next(attrs for tag, attrs in elements if tag == "form" and attrs.get("id") == "ask-form")
    submit = next(attrs for _, attrs in elements if attrs.get("id") == "submit")
    assert form.get("method") == "post"
    assert "disabled" in submit


def test_ui_initialises_asking_before_optional_features():
    boot = SCRIPT.split('document.addEventListener("DOMContentLoaded"', 1)[1]
    order = [boot.index(name) for name in ("initAskForm()", "loadLaws()", "setupCaptcha()", "setupArticleDialog")]
    assert order == sorted(order)
    init = SCRIPT.split("function initAskForm()", 1)[1].split("\nfunction ", 1)[0]
    assert '$("submit").disabled = false' in init


def test_reader_navigation_is_driven_by_the_server():
    elements = page_elements()
    by_id = {attrs["id"]: tag for tag, attrs in elements if "id" in attrs}
    assert (by_id["article-prev"], by_id["article-jump"], by_id["article-next"]) == ("button", "select", "button")
    assert any(tag == "label" and attrs.get("for") == "article-jump" for tag, attrs in elements)
    head = re.search(r'<div class="sheet-head">(.*?)<div class="sheet-body">', PAGE, re.S)
    assert head and 'id="article-prev"' in head.group(1)  # navigation sits in the sticky head
    for field in ("prev_article", "prev_article_label", "next_article", "next_article_label", "loadArticleIndex"):
        assert field in SCRIPT, field
    assert "articleNum" not in SCRIPT and 'includes("income")' not in SCRIPT
    assert "closedby" not in PAGE


def test_coverage_offers_browsing_and_currency():
    for needle in ("version_ar", "coverage_note_ar", "تصفّح مواد القانون"):
        assert needle in SCRIPT, needle


def test_ui_hardcodes_no_pipeline_facts():
    from app.config import Settings

    settings = Settings()
    assert f"{settings.min_similarity:.2f}" in PAGE
    for fact in ("text-embedding", "1536", "768", "gemini-2", "nemotron", "pgvector v", "HNSW"):
        assert fact.lower() not in PAGE.lower() and fact.lower() not in SCRIPT.lower(), fact


def test_readout_reads_the_response():
    for field in ("refusal_reason", "dropped_citations", "performance.now()"):
        assert field in SCRIPT, field
    assert ">= 0.6" not in SCRIPT
    assert "< 0.6" not in SCRIPT
    assert ">= 0.60" not in SCRIPT
    assert "< 0.60" not in SCRIPT


def test_no_live_region_on_result():
    elements = page_elements()
    result = next(attrs for tag, attrs in elements if attrs.get("id") == "result")
    assert "aria-live" not in result


def test_token_contrast_pairs():
    def parse_tokens(block: str) -> dict[str, str]:
        return {k: v.strip() for k, v in re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);", block)}

    root_match = re.search(r":root\s*{(.*?)}", CSS, re.S)
    dark_match = re.search(r"@media\s*\(prefers-color-scheme:\s*dark\)\s*{\s*:root\s*{(.*?)}", CSS, re.S)
    assert root_match and dark_match

    root = parse_tokens(root_match.group(1))
    dark = dict(root)
    dark.update(parse_tokens(dark_match.group(1)))

    def hex_to_rgb(h: str) -> tuple[float, float, float]:
        h = h.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore

    def rel_lum(rgb: tuple[float, float, float]) -> float:
        def channel(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = [channel(c) for c in rgb]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(hex1: str, hex2: str) -> float:
        l1, l2 = rel_lum(hex_to_rgb(hex1)), rel_lum(hex_to_rgb(hex2))
        return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

    text_pairs = [
        ("--color-ink", "--color-page"),
        ("--color-ink", "--color-surface"),
        ("--color-muted", "--color-page"),
        ("--color-muted", "--color-surface"),
        ("--color-on-accent", "--color-accent"),
        ("--color-statute", "--color-quote"),
        ("--color-statute", "--color-surface"),
    ]
    non_text_pairs = [
        ("--color-control-border", "--color-page"),
        ("--color-statute-rule", "--color-quote"),
    ]

    for palette in (root, dark):
        for fg, bg in text_pairs:
            assert contrast(palette[fg], palette[bg]) >= 4.5, f"{fg} on {bg}"
        for fg, bg in non_text_pairs:
            assert contrast(palette[fg], palette[bg]) >= 3.0, f"{fg} vs {bg}"
