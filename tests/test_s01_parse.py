"""Done-when tests for Stage 1, load and parse. Spec Section 5, Stage 1. Issue #6.

Pure functions run on synthetic data. The done-when tests open the real PDF and skip when
`data/raw/` holds no book.
"""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Chapter, ParsedPage, Word, read_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent")

# The 36 bookmarks of the PHP 8.2 edition, as (level, title, pdf page). Spec Appendix A.
BOOKMARKS: list[s1.BookmarkRow] = [
    (1, "Foreword", 9), (2, "Preface", 11),
    (1, "PHP, the Language", 12),
    (2, "PHP Today", 13), (2, "New Versions", 15), (2, "PHP's Type System", 31),
    (2, "Static Analysis", 55), (2, "Property Promotion", 65), (2, "Readonly Properties", 77),
    (2, "Named Arguments", 87), (2, "Attributes", 99), (2, "Enums", 117),
    (2, "Short Closures", 129), (2, "First-class callables", 133),
    (2, "Working with Arrays", 137), (2, "Match", 145), (2, "New in initializers", 151),
    (1, "Building With PHP", 156),
    (2, "Object Oriented PHP", 157), (2, "MVC Frameworks", 179), (2, "Dependency Injection", 197),
    (2, "Collections", 209), (2, "Testing", 213), (2, "Style Guide", 221),
    (1, "PHP In Depth", 232),
    (2, "The JIT", 233), (2, "Preloading", 243), (2, "FFI", 257), (2, "Internals", 261),
    (2, "Type Variance", 267), (2, "Async PHP", 273), (2, "Fibers", 279),
    (2, "Event Driven Development", 285), (2, "Static Analysers In Practice", 297),
    (2, "PHP 8.2", 309), (2, "In Closing", 323),
]
SKIP_PAGES = (1, 2, 3, 4, 5, 6, 7, 8, 12, 156, 232)


def _word(text: str, *, top: float = 100.0, size: float = 10.0, font: str = "X+Inter-Regular",
          page: int = 66) -> dict:
    return {"text": text, "fontname": font, "size": size, "x0": 72.0, "x1": 90.0,
            "top": top, "bottom": top + size, "page_pdf": page}


# --------------------------------------------------------------------------- pure functions


def test_strip_font_prefix() -> None:
    assert s1.strip_font_prefix("MBPBAO+JetBrainsMono-Regular") == "JetBrainsMono-Regular"
    assert s1.strip_font_prefix("Inter-Regular") == "Inter-Regular"
    assert s1.strip_font_prefix("A+B+Staatliches-Regular") == "B+Staatliches-Regular"


def test_header_filter_uses_both_rules() -> None:
    cfg = load_settings().parse
    assert s1.is_header_word(_word("64", top=32.4, size=8.0), cfg)  # both rules
    assert s1.is_header_word(_word("x", top=40.0, size=10.0), cfg)  # top only
    assert s1.is_header_word(_word("x", top=300.0, size=8.0), cfg)  # size only
    assert not s1.is_header_word(_word("class", top=77.0, size=9.0), cfg)
    assert not s1.is_header_word(_word("x", top=50.0, size=10.0), cfg)  # boundary is exclusive


def test_split_header_words_counts_deleted() -> None:
    cfg = load_settings().parse
    raw = [_word("64", top=32.4, size=8.0), _word("Front", top=32.4, size=8.0),
           _word("class", size=9.0, font="X+JetBrainsMono-Regular"), _word("A")]
    kept, deleted = s1.split_header_words(raw, cfg, page_pdf=66)
    assert deleted == 2
    assert [w.text for w in kept] == ["class", "A"]
    assert kept[0].font == "JetBrainsMono-Regular" and kept[0].page_pdf == 66
    assert all(isinstance(w, Word) for w in kept)


def test_chapter_table_matches_appendix_a() -> None:
    table = s1.build_chapter_table(BOOKMARKS, last_page=324, skip_pages=SKIP_PAGES)

    assert len(table) == 33
    by_title = {c.title: c for c in table}
    assert [c.chapter_no for c in table] == [0, 0, *range(1, 32)]
    assert by_title["Foreword"].part == "Front matter" and by_title["Foreword"].part_title == ""
    assert (by_title["Foreword"].page_pdf_start, by_title["Foreword"].page_pdf_end) == (9, 10)
    assert (by_title["Preface"].page_pdf_start, by_title["Preface"].page_pdf_end) == (11, 11)
    assert by_title["PHP Today"].chapter_no == 1
    assert by_title["PHP Today"].part == "Part 1"
    assert by_title["PHP Today"].part_title == "PHP, the Language"
    assert by_title["Property Promotion"].chapter_no == 5
    assert by_title["Type Variance"].chapter_no == 25  # printed label says 21; bookmarks win
    assert by_title["PHP 8.2"].chapter_no == 30
    assert by_title["In Closing"].chapter_no == 31 and by_title["In Closing"].part == "Part 3"
    # End pages never land on a part title page.
    assert by_title["New in initializers"].page_pdf_end == 155
    assert by_title["Style Guide"].page_pdf_end == 231
    assert by_title["In Closing"].page_pdf_end == 324
    assert [c.order for c in table] == list(range(33))


def test_chapter_table_page_ranges_tile_the_body() -> None:
    table = s1.build_chapter_table(BOOKMARKS, last_page=324, skip_pages=SKIP_PAGES)
    covered = sorted(p for c in table for p in range(c.page_pdf_start, c.page_pdf_end + 1))
    body = [p for p in range(9, 325) if p not in SKIP_PAGES]
    assert covered == body


def test_chapter_table_rejects_bookmarks_without_parts() -> None:
    with pytest.raises(ValueError, match="part"):
        s1.build_chapter_table([(1, "Foreword", 9), (2, "PHP Today", 13)], last_page=20,
                               skip_pages=())


def test_chapter_table_rejects_non_increasing_bookmark_pages() -> None:
    rows = [(1, "Foreword", 9), (1, "Part", 12), (2, "A", 20), (2, "B", 20), (2, "C", 30)]
    with pytest.raises(ValueError, match="must increase"):
        s1.build_chapter_table(rows, last_page=40, skip_pages=(12,))


def test_existing_index_is_found_only_on_full_match(tmp_path: Path) -> None:
    def manifest(name: str, **fields: object) -> None:
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"index_version": name, **fields}))

    manifest("v1-a", doc_id="doc", chunk_config_hash="hash", verify={"passed": True})
    manifest("v2-b", doc_id="doc", chunk_config_hash="other", verify={"passed": True})
    manifest("v3-c", doc_id="doc", chunk_config_hash="hash", verify={"passed": False})
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "manifest.json").write_text("{not json")
    (tmp_path / "list").mkdir()
    (tmp_path / "list" / "manifest.json").write_text("[]")

    assert s1.find_existing_index(tmp_path, "doc", "hash") == "v1-a"
    assert s1.find_existing_index(tmp_path, "doc", "nope") is None
    assert s1.find_existing_index(tmp_path, "zzz", "hash") is None
    assert s1.find_existing_index(tmp_path / "missing", "doc", "hash") is None


def test_parsed_page_round_trips_with_nested_words(tmp_path: Path) -> None:
    from flp_rag.contracts import from_json, to_json

    page = ParsedPage(doc_id="d", page_pdf=66, page_printed=64, header_words_deleted=4,
                      words=[Word("a", "Inter-Regular", 10.0, 1, 2, 3, 4, 66)])
    back = from_json(ParsedPage, to_json(page))
    assert back == page and isinstance(back.words[0], Word)


# --------------------------------------------------------------------------- real PDF


@pytest.fixture(scope="module")
def parsed(tmp_path_factory: pytest.TempPathFactory) -> s1.ParseResult:
    if not PDF.exists():
        pytest.skip(f"{PDF.relative_to(ROOT)} is absent")
    out = tmp_path_factory.mktemp("parsed")
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(load_settings(), exporter=exporter, force=True)
    result = s1.parse(PDF, load_settings(), out_dir=out, index_dir=out / "index")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    result_span = spans["ingest.parse"]
    return result, result_span  # type: ignore[return-value]


@needs_pdf
def test_doc_id_is_sixteen_hex(parsed: tuple) -> None:
    result, _ = parsed
    assert result.doc_id == "bca146b9df2d0a2b"


@needs_pdf
def test_page_66_has_prose_in_inter_and_code_in_jetbrainsmono(parsed: tuple) -> None:
    result, _ = parsed
    pages = {p.page_pdf: p for p in read_jsonl(result.output_path, ParsedPage)}
    page = pages[66]
    fonts = {w.font for w in page.words}
    assert "Inter-Regular" in fonts and "JetBrainsMono-Regular" in fonts
    assert not any(f.startswith("MBPBAO+") for f in fonts)
    assert page.page_printed == 64
    assert page.header_words_deleted == 4
    assert all(w.size != 8 and w.top >= 50 for w in page.words)
    assert "CustomerDTO" in {w.text for w in page.words}


@needs_pdf
def test_running_header_is_gone_everywhere(parsed: tuple) -> None:
    result, _ = parsed
    text = result.output_path.read_text()
    assert "Front Line PHP" not in text
    pages = list(read_jsonl(result.output_path, ParsedPage))
    assert all(w.size != 8 for p in pages for w in p.words)


@needs_pdf
def test_pages_and_counts(parsed: tuple) -> None:
    result, span = parsed
    assert result.pages_total == 324
    assert result.pages_parsed == 324 - len(SKIP_PAGES)
    assert result.pages_without_header == [9, 324]
    assert result.header_words_deleted == 1415
    assert span.attributes["rag.pages_parsed"] == result.pages_parsed
    assert span.attributes["rag.header_words_deleted"] == 1415
    assert json.loads(span.attributes["rag.pages_without_header"]) == [9, 324]
    assert span.attributes["rag.chapters_found"] == 33


@needs_pdf
def test_chapter_table_from_real_bookmarks(parsed: tuple) -> None:
    result, _ = parsed
    table = list(read_jsonl(result.chapters_path, Chapter))
    expected = s1.build_chapter_table(BOOKMARKS, last_page=324, skip_pages=SKIP_PAGES)
    assert table == expected


@needs_pdf
def test_second_run_is_already_indexed_when_manifest_matches(tmp_path: Path) -> None:
    settings = load_settings()
    idx = tmp_path / "index" / "v9-test"
    idx.mkdir(parents=True)
    (idx / "manifest.json").write_text(json.dumps({
        "index_version": "v9-test", "doc_id": "bca146b9df2d0a2b",
        "chunk_config_hash": settings.chunk_config_hash, "verify": {"passed": True},
    }))
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)

    result = s1.parse(PDF, settings, out_dir=tmp_path / "out", index_dir=tmp_path / "index")

    assert result.already_indexed == "v9-test"
    assert result.pages_parsed == 0
    assert not (tmp_path / "out").exists()

    forced = s1.parse(PDF, settings, out_dir=tmp_path / "out", index_dir=tmp_path / "index",
                      force=True)
    assert forced.already_indexed is None and forced.pages_parsed > 0
