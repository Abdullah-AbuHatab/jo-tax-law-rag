"""Rebuild the broken ToUnicode maps of a Word-2010 Arabic PDF.

Why this exists
---------------
The official ISTD PDF of the General Sales Tax Law was exported from
Microsoft Word 2010. Its fonts are embedded correctly (the PDF *prints*
fine), but the ToUnicode CMaps that text extractors rely on are wrong: Word
wrote "bfrange" entries such as ``<038E> <0398> <0627>`` meaning "every
contextual form of alef is U+0627", but a bfrange *increments* the target,
so the alef forms come out as alef, beh, teh marbuta, teh, ... Extracted
text is therefore garbled Arabic (e.g. "لبْٔٛ" instead of "قانون").

How it is repaired
------------------
The embedded font itself still knows which glyph is which letter: its
``cmap`` table maps each base letter to a glyph, and its ``GSUB`` table maps
that glyph to the initial/medial/final forms and lam-alef ligatures. This
tool walks those tables to build glyph-id -> text for every glyph, writes a
corrected ToUnicode CMap for each Type0 font, and saves a new PDF. The page
content is not touched, so what you see when printing is unchanged.

It prints, per font, how many original entries agreed with the font tables
(the explicit single-glyph entries agree; the broken ranges do not), which
is the evidence that the repair is right.

Usage (tools-only dependencies, see tools/requirements.txt)::

    python -I tools/repair_pdf_tounicode.py IN.pdf OUT.pdf
"""

from __future__ import annotations

import argparse
import io
import re
import unicodedata

from fontTools.ttLib import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, StreamObject


def parse_tounicode(data: bytes) -> dict[int, tuple[str, str]]:
    """Return glyph id -> (entry kind, text) as the PDF currently claims."""
    entries: dict[int, tuple[str, str]] = {}
    text = data.decode("latin-1")

    def utf16(hex_text: str) -> str:
        return bytes.fromhex(hex_text).decode("utf-16-be")

    for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            entries[int(src, 16)] = ("char", utf16(dst))
    for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
        for lo, hi, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]+>)", block):
            low, high = int(lo, 16), int(hi, 16)
            if dst.startswith("["):
                for offset, value in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", dst)):
                    entries[low + offset] = ("array", utf16(value))
            else:
                base = int(dst[1:-1], 16)
                for offset in range(high - low + 1):
                    entries[low + offset] = ("range", chr(base + offset))
    return entries


def _subtables(lookup):
    for subtable in lookup.SubTable:
        yield subtable.ExtSubTable if lookup.LookupType == 7 else subtable


def glyph_text_from_font(font: TTFont) -> dict[int, str]:
    """Glyph id -> text, derived only from the font's own cmap and GSUB tables."""
    order = font.getGlyphOrder()
    glyph_id = {name: index for index, name in enumerate(order)}
    text_of: dict[str, str] = {}
    for codepoint, name in sorted((font.getBestCmap() or {}).items()):
        text_of.setdefault(name, chr(codepoint))

    if "GSUB" in font:
        lookups = font["GSUB"].table.LookupList.Lookup
        for _ in range(4):  # a few passes reach forms produced from other forms
            for lookup in lookups:
                for subtable in _subtables(lookup):
                    if hasattr(subtable, "ligatures"):
                        for first, ligatures in subtable.ligatures.items():
                            for ligature in ligatures:
                                parts = [first, *ligature.Component]
                                if all(p in text_of for p in parts) and ligature.LigGlyph not in text_of:
                                    text_of[ligature.LigGlyph] = "".join(text_of[p] for p in parts)
                    elif hasattr(subtable, "mapping"):
                        for source, target in subtable.mapping.items():
                            # single substitution only; multiple-substitution values are lists
                            if isinstance(target, str) and source in text_of and target not in text_of:
                                text_of[target] = text_of[source]
                    elif hasattr(subtable, "alternates"):
                        for source, targets in subtable.alternates.items():
                            for target in targets:
                                if source in text_of and target not in text_of:
                                    text_of[target] = text_of[source]

    # Glyph names such as "uniFE8E" for anything GSUB did not reach.
    for name in order:
        match = re.fullmatch(r"uni([0-9A-F]{4})", name)
        if match and name not in text_of:
            text_of[name] = unicodedata.normalize("NFKC", chr(int(match.group(1), 16)))
    return {glyph_id[name]: value for name, value in text_of.items() if name in glyph_id}


def build_cmap(mapping: dict[int, str]) -> bytes:
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        "/CMapName /Adobe-Identity-UCS def",
        "/CMapType 2 def",
        "1 begincodespacerange",
        "<0000> <FFFF>",
        "endcodespacerange",
    ]
    items = sorted(mapping.items())
    for start in range(0, len(items), 100):  # PDF limit: 100 entries per block
        chunk = items[start : start + 100]
        lines.append(f"{len(chunk)} beginbfchar")
        lines.extend(f"<{gid:04X}> <{text.encode('utf-16-be').hex().upper()}>" for gid, text in chunk)
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return "\n".join(lines).encode("ascii")


def repair(source: str, target: str) -> list[str]:
    writer = PdfWriter(clone_from=PdfReader(source))
    report: list[str] = []
    repaired: set[int] = set()
    for page in writer.pages:
        resources = page.get("/Resources")
        fonts = resources.get_object().get("/Font") if resources else None
        if not fonts:
            continue
        for ref in fonts.get_object().values():
            font = ref.get_object()
            if font.get("/Subtype") != "/Type0" or "/ToUnicode" not in font or ref.idnum in repaired:
                continue
            descendant = font["/DescendantFonts"][0].get_object()
            font_file = descendant["/FontDescriptor"].get_object().get("/FontFile2")
            if font_file is None:
                report.append(f"{font.get('/BaseFont')}: no embedded TrueType, left unchanged")
                continue
            from_font = glyph_text_from_font(TTFont(io.BytesIO(font_file.get_object().get_data())))
            claimed = parse_tounicode(font["/ToUnicode"].get_object().get_data())

            agree = disagree = kept = 0
            final: dict[int, str] = {}
            for gid, (_, text) in claimed.items():
                if gid in from_font:
                    same = unicodedata.normalize("NFKC", from_font[gid]) == unicodedata.normalize("NFKC", text)
                    agree, disagree = agree + same, disagree + (not same)
                    final[gid] = from_font[gid]
                else:
                    kept += 1
                    final[gid] = text
            stream = StreamObject()
            stream.set_data(build_cmap(final))
            font[NameObject("/ToUnicode")] = writer._add_object(stream)
            repaired.add(ref.idnum)
            report.append(
                f"{font.get('/BaseFont')}: {len(claimed)} glyphs, {agree} agreed with the font tables, "
                f"{disagree} corrected, {kept} kept as-is (not reachable from the font tables)"
            )
    with open(target, "wb") as handle:
        writer.write(handle)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    for line in repair(args.source, args.target):
        print(line)


if __name__ == "__main__":
    main()
