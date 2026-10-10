"""Unit tests for the Level 1 structural gate in eval/chunk_checks.py. Issue #62."""

import re
import sys
from pathlib import Path

import pytest

from flp_rag.contracts import Block, Chapter, Chunk, ParsedPage, Word
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.settings import load_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import chunk_checks as cc

CH5 = Chapter(6, 5, "Property Promotion", "Part 1", "PHP, the Language", 65, 76)
CH6 = Chapter(7, 6, "Readonly Properties", "Part 1", "PHP, the Language", 77, 86)


@pytest.fixture(scope="module")
def cfg():
    return load_settings().chunk


@pytest.fixture(scope="module")
def enc(cfg):
    return tokenizer(cfg)


def block(bid: str, type_: str, text: str, chapter: Chapter = CH5, page: int = 66) -> Block:
    return Block(block_id=bid, type=type_, text=text, part=chapter.part, chapter_no=chapter.chapter_no,  # type: ignore[arg-type]
                 chapter_title=chapter.title, section_title="S", page_pdf=page, page_printed=page - 2,
                 block_index=0, in_rect=False)


def chunk(cid: str, text: str, block_ids: list[str], *, chapter: Chapter = CH5, section: str = "S",
          titles: tuple[str, ...] | None = ("S",), has_code: bool = False, group: str | None = None) -> Chunk:
    payload = {"chapter_no": chapter.chapter_no, "section_title": section, "block_ids": block_ids,
               "has_code": has_code, "code_group_id": group, "page_printed_start": 64, "page_printed_end": 64}
    if titles is not None:
        payload["section_titles"] = list(titles)
    return Chunk(cid, "p", text, text, len(text.split()), payload)


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_sizes_flag_small_and_huge(cfg, enc) -> None:
    ok = cc.check_sizes([chunk("a", words(200), [])], cfg, enc)
    assert ok.passed
    bad = cc.check_sizes([chunk("s", "tiny", []), chunk("h", words(1200), [])], cfg, enc)
    assert not bad.passed and set(bad.details["under_min"]) == {"s"} and set(bad.details["over_max"]) == {"h"}


def _page(words: list[tuple[str, float, float]]) -> ParsedPage:
    return ParsedPage("d", 66, 64, 0, [Word(t, "Inter-Regular", size, 65.0, 90.0, top, top + size, 66) for t, size, top in words])


def test_header_noise_matches_whole_header_lines_and_parsed_leftovers() -> None:
    cfg = load_settings().parse
    patterns = [re.compile(r"\d+ front line php"), re.compile(r"chapter \d+ - readonly properties \d+")]
    body = _page([("Readonly", 10.0, 120.0)])
    clean = [chunk("a", "Readonly properties are set once.\nThe answer is here – Front Line PHP.", [])]
    assert cc.check_header_noise(clean, patterns, [body], cfg).passed
    assert cc.check_glyph_bug(clean).passed
    noisy = [chunk("b", "64 Front Line PHP\nSome text $this=>name", []),
             chunk("c", "Chapter 06 - Readonly Properties 77\nmore", [])]
    r = cc.check_header_noise(noisy, patterns, [body], cfg)
    assert not r.passed and r.details["chunks"] == {"b": ["64 Front Line PHP"], "c": ["Chapter 06 - Readonly Properties 77"]}
    leftover = _page([("Front", 8.0, 32.4), ("deep", 10.0, 40.0)])
    r = cc.check_header_noise(clean, patterns, [leftover], cfg)
    assert not r.passed and [w["text"] for w in r.details["parsed_leftovers"]] == ["Front", "deep"]
    assert not cc.check_header_noise(clean, [], [body], cfg).passed  # no patterns means nothing was checked
    glyph = cc.check_glyph_bug(noisy)
    assert not glyph.passed and glyph.details["chunks"]["$this=>"] == ["b"]


def test_sections_rules() -> None:
    blocks = [block("b1", "heading", "S"), block("b2", "paragraph", "text"), block("b3", "heading", "T"),
              block("b4", "paragraph", "other", chapter=CH6, page=77)]
    good = chunk("a", "S\n\ntext\n\nT", ["b1", "b2", "b3"], titles=("S", "T"))
    assert cc.check_sections([good], blocks).passed
    unset = chunk("u", "text", ["b2"], section="")
    missing = chunk("m", "text", ["b2"], titles=None)
    unlisted = chunk("l", "S\n\ntext\n\nT", ["b1", "b2", "b3"], titles=("S",))
    spans = chunk("x", "text\n\nother", ["b2", "b4"])
    r = cc.check_sections([unset, missing, unlisted, spans], blocks)
    assert not r.passed
    assert r.details["section_title_unset"] == ["u"] and r.details["section_titles_missing"] == ["m"]
    assert r.details["headings_not_listed"] == {"l": ["T"]} and r.details["two_chapters"] == ["x"]


def test_code_integrity_reports_but_never_fails() -> None:
    chunks = [chunk("ok", "class A {\n}", [], has_code=True), chunk("partial", "class B {\n", [], has_code=True),
              chunk("grouped", "{", [], has_code=True, group="g"), chunk("prose", "{ not code", [])]
    r = cc.check_code_integrity(chunks)
    assert r.passed and [m["chunk_id"] for m in r.details["mismatches"]] == ["partial"]


def test_coverage_counts_tokens_of_blocks_found_in_a_chunk(enc) -> None:
    blocks = [block("t", "chapter_title", "Property Promotion"), block("b1", "paragraph", "Alpha beta gamma."),
              block("b2", "code", "class A {}"), block("b3", "paragraph", "Never chunked sentence here.")]
    chunks = [chunk("a", "ALPHA   beta gamma.\n\nclass A {}", ["b1", "b2"])]
    r = cc.check_coverage(chunks, blocks, enc, target=0.99)
    assert not r.passed and [u["block_id"] for u in r.details["uncovered_blocks"]] == ["b3"]
    assert r.details["tokens_total"] == sum(len(enc.encode(b.text)) for b in blocks[1:])
    full = cc.check_coverage(chunks, blocks[:3], enc, target=0.99)
    assert full.passed and full.details["coverage"] == 1.0 and full.details["per_chapter"] == {5: 1.0}


def test_chapters_check_lists_missing_ones() -> None:
    r = cc.check_chapters([chunk("a", "x", [])], [CH5, CH6])
    assert not r.passed and r.details["missing"] == ["06 Readonly Properties"]
    assert cc.check_chapters([chunk("a", "x", []), chunk("b", "y", [], chapter=CH6)], [CH5, CH6]).passed
