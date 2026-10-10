"""Glyph oracle: the Stage 1-3 text of every body page must match PyMuPDF's reading of the same
page once whitespace is removed. Issue #62, task A.

PyMuPDF decodes the JetBrains Mono ligatures correctly where pdfplumber's ToUnicode map gives
`=`. It is a reference here, never the parser. Text that Stage 2 moves across a page break (code
and callouts merged onto the previous page) shows up as one large delete plus one large insert;
only replace operations of up to three characters count as glyph differences, and the one that
matters is `=` on our side with no `=` on PyMuPDF's side. Slow: about a minute on the book.
"""

import collections
import difflib
import json
import re
from pathlib import Path

import pytest

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
CLEAN = ROOT / "data" / "clean" / "bca146b9df2d0a2b.jsonl"
MAX_SMALL_REPLACE_CHARS = 3

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent"),
    pytest.mark.skipif(not CLEAN.exists(), reason=f"{CLEAN.relative_to(ROOT)} is absent; run flp ingest"),
]


def pymupdf_body_text(page, settings) -> str:  # type: ignore[no-untyped-def]
    """Characters of one page without the running header and the `CHAPTER NN` label, so the
    comparison sees what Stage 1 keeps and Stage 2 drops."""
    pc, sc = settings.parse, settings.structure
    label = re.compile(sc.label_pattern)
    out: list[str] = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            text = "".join(ch["c"] for sp in spans for ch in sp["chars"])
            if {round(sp["size"]) for sp in spans} == {round(sc.label_size_pt)} and label.fullmatch(text.strip()):
                continue
            for sp in spans:
                if abs(sp["size"] - pc.header_font_size_pt) < 0.01:
                    continue
                out.extend(ch["c"] for ch in sp["chars"] if ch["bbox"][1] >= pc.header_max_top_pt)
    return "".join(out)


def small_replace_pairs(ours: str, theirs: str) -> collections.Counter:
    """Replace operations of up to MAX_SMALL_REPLACE_CHARS on both sides, counted by pair."""
    pairs: collections.Counter = collections.Counter()
    matcher = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and i2 - i1 <= MAX_SMALL_REPLACE_CHARS and j2 - j1 <= MAX_SMALL_REPLACE_CHARS:
            pairs[(ours[i1:i2], theirs[j1:j2])] += 1
    return pairs


def test_clean_text_matches_pymupdf_on_every_body_page() -> None:
    pymupdf = pytest.importorskip("pymupdf")
    settings = load_settings()
    whitespace = re.compile(r"\s+")
    ours_by_page: dict[int, list[str]] = collections.defaultdict(list)
    for line in CLEAN.read_text().splitlines():
        block = json.loads(line)
        ours_by_page[block["page_pdf"]].append(block["text"])

    pairs: collections.Counter = collections.Counter()
    checked = 0
    with pymupdf.open(str(PDF)) as doc:
        for page_no in range(1, doc.page_count + 1):
            if page_no in settings.parse.skip_pages or page_no not in ours_by_page:
                continue
            checked += 1
            ours = whitespace.sub("", "".join(ours_by_page[page_no]))
            theirs = whitespace.sub("", pymupdf_body_text(doc[page_no - 1], settings))
            if ours != theirs:
                pairs.update(small_replace_pairs(ours, theirs))

    assert checked > 250, checked
    glyph_bugs = {pair: n for pair, n in pairs.items() if "=" in pair[0] and "=" not in pair[1]}
    assert not glyph_bugs, glyph_bugs
    assert not pairs, dict(pairs)
