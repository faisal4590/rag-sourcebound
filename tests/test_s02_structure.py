"""Done-when tests for Stage 2, structure reconstruction. Spec Section 5, Stage 2. Issue #7.

Pure functions run on synthetic words. The done-when tests parse the real PDF first and skip
when `data/raw/` holds no book.
"""

from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Block, Chapter, Rect, Word, read_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.ingest import s02_structure as s2
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent")

INTER, MONO, STA = "Inter-Regular", "JetBrainsMono-Regular", "Staatliches-Regular"
MARGIN, CW = 79.37, 5.4  # code left margin and mono char width measured on page 66


def W(text: str, x0: float, top: float, *, font: str = INTER, size: float = 10.0,
      page: int = 66, width: float | None = None) -> Word:
    w = width if width is not None else len(text) * (CW if "Mono" in font else 5.0)
    return Word(text, font, size, x0, x0 + w, top, top + size, page)


def prose_line(text: str, top: float, *, x0: float = 65.2, page: int = 66,
               font: str = INTER, size: float = 10.0) -> list[Word]:
    words, x = [], x0
    for t in text.split():
        words.append(W(t, x, top, font=font, size=size, page=page))
        x += len(t) * 5.0 + 3.0
    return words


def code_line(text: str, top: float, *, page: int = 66) -> list[Word]:
    """Code words placed on the mono grid: leading spaces become x offset, gaps become spacing."""
    words, col = [], 0
    for token in text.split(" "):
        if token == "":
            col += 1
            continue
        words.append(W(token, MARGIN + col * CW, top, font=MONO, size=9.0, page=page))
        col += len(token) + 1
    return words


def code_rects(tops: list[float], kind: str = "code") -> list[Rect]:
    return [Rect(65.2, 473.4, t - 4.0, t + 12.0, kind) for t in tops]  # type: ignore[arg-type]


@pytest.fixture
def cfg():
    return load_settings().structure


# --------------------------------------------------------------------------- lines


def test_group_lines_by_top_within_tolerance(cfg) -> None:
    words = [W("b", 100, 77.2), W("a", 65, 78.1), W("c", 65, 93.2), W("d", 200, 93.0)]
    lines = s2.group_lines(words, cfg.line_tolerance_pt)
    assert [[w.text for w in ln.words] for ln in lines] == [["a", "b"], ["c", "d"]]
    assert lines[0].page_pdf == 66 and lines[0].top == pytest.approx(77.2)


def test_line_type_by_dominant_font_and_rect(cfg) -> None:
    def kind(words, rects=()):
        return s2.line_type(s2.group_lines(words, 2)[0], list(rects), cfg)

    assert kind(prose_line("Chapter body text here", 100)) == "paragraph"
    assert kind(code_line("class CustomerDTO", 150)) == "code"
    assert kind([W("Property", 65, 88, font=STA, size=34)]) == "chapter_title"
    assert kind([W("Type", 65, 300, font=STA, size=14)]) == "heading"
    assert kind([W("Sub", 65, 300, font=STA, size=16)]) == "heading"
    callout = Rect(65.2, 473.4, 290, 320, "callout")
    assert kind([W("PHP", 79, 300, font=STA, size=14)], [callout]) == "callout"
    assert kind(prose_line("Even though PHP", 305, x0=79.4), [callout]) == "callout"
    assert kind([W("CHAPTER", 65, 70.6, size=9), W("05", 111, 70.6, size=9)]) == "label"
    # Inline code: one mono word on a prose line keeps the line a paragraph.
    mixed = prose_line("you can use the triple", 200) + [W("===", 250, 200, font=MONO, size=9)]
    assert kind(mixed) == "paragraph"
    # IBMPlexMono comments count as code too.
    assert kind([W("//", 79, 150, font="IBMPlexMono-Italic", size=9)]) == "code"


# --------------------------------------------------------------------------- code layout


def test_code_metrics_measure_margin_and_char_width(cfg) -> None:
    words = code_line("class CustomerDTO", 150) + code_line("    public int $x;", 166)
    margin, cw = s2.code_metrics(words, cfg)
    assert margin == pytest.approx(MARGIN)
    assert cw == pytest.approx(CW, abs=0.01)


def test_code_block_keeps_indentation_and_inner_spacing(cfg) -> None:
    tops = [150.0, 166.0, 182.0, 198.0]
    words = (code_line("class CustomerDTO", tops[0])
             + code_line("{", tops[1])
             + code_line("    public function __construct(", tops[2])
             + code_line("        public string $name,  // two spaces", tops[3]))
    page = s2.PageInput(66, words, code_rects(tops))
    blocks = s2.build_page_blocks(page, cfg)
    assert len(blocks) == 1 and blocks[0].type == "code"
    assert blocks[0].text.split("\n") == [
        "class CustomerDTO",
        "{",
        "    public function __construct(",
        "        public string $name,  // two spaces",
    ]
    assert blocks[0].in_rect is True


# --------------------------------------------------------------------------- paragraphs


def test_paragraphs_split_on_large_gap_and_join_hyphens(cfg) -> None:
    words = (prose_line("PHP is an interpreted lan-", 100)
             + prose_line("guage with a compiler.", 116)
             + prose_line("A new paragraph starts here.", 148)  # gap 32 > 1.5 * 16
             + prose_line("It uses a first-", 164)
             + prose_line("Class callable.", 180))  # next starts uppercase: keep hyphen
    blocks = s2.build_page_blocks(s2.PageInput(66, words, []), cfg)
    assert [b.type for b in blocks] == ["paragraph", "paragraph"]
    assert blocks[0].text == "PHP is an interpreted language with a compiler."
    assert blocks[1].text == "A new paragraph starts here. It uses a first- Class callable."
    assert blocks[0].in_rect is False


def test_touching_code_words_get_no_space(cfg) -> None:
    # A font change splits `#[MyAttribute]` into three words with zero gap between them.
    top = 150.0
    a = W("#[", MARGIN, top, font=MONO, size=9.0)
    b = W("MyAttribute", a.x1, top, font=MONO, size=9.0)
    c = W("]", b.x1, top, font=MONO, size=9.0)
    d = W("$a,", c.x1 + CW, top, font=MONO, size=9.0)  # one real space
    blocks = s2.build_page_blocks(s2.PageInput(73, [a, b, c, d], code_rects([top])), cfg)
    assert blocks[0].text == "#[MyAttribute] $a,"


def test_hyphen_join_never_applies_inside_code(cfg) -> None:
    tops = [150.0, 166.0]
    words = code_line("$a = 'foo-", tops[0]) + code_line("bar';", tops[1])
    blocks = s2.build_page_blocks(s2.PageInput(66, words, code_rects(tops)), cfg)
    assert blocks[0].text == "$a = 'foo-\nbar';"


# --------------------------------------------------------------------------- callouts


def test_callout_assembles_title_and_bodies_and_new_title_starts_new_block(cfg) -> None:
    rect1 = [Rect(65.2, 473.4, 466.9, 504.2, "callout"), Rect(65.2, 473.4, 502.5, 591.0, "callout")]
    rect2 = [Rect(65.2, 473.4, 600.0, 660.0, "callout")]
    words = ([W("PHP", 79.4, 483.3, font=STA, size=14), W("Compiler", 100.9, 483.3, font=STA, size=14)]
             + prose_line("Even though PHP is an interpreted language,", 518.2, x0=79.4)
             + prose_line("it still has a compiler.", 534.2, x0=79.4)
             + prose_line("Second paragraph of the same callout.", 566.2, x0=79.4)  # gap 32
             + [W("Automation", 79.4, 610.0, font=STA, size=14)]
             + prose_line("Another callout body.", 630.0, x0=79.4))
    blocks = s2.build_page_blocks(s2.PageInput(61, words, rect1 + rect2), cfg)
    assert [b.type for b in blocks] == ["callout", "callout"]
    assert blocks[0].text == (
        "PHP Compiler\nEven though PHP is an interpreted language, it still has a compiler.\n\n"
        "Second paragraph of the same callout."
    )
    assert blocks[0].starts_with_title and blocks[1].text.startswith("Automation\n")


def test_heading_lines_wrap_into_one_heading(cfg) -> None:
    words = [W("Static", 65, 300, font=STA, size=14), W("Analysers", 110, 300, font=STA, size=14),
             W("In", 65, 318, font=STA, size=14), W("Practice", 85, 318, font=STA, size=14)]
    blocks = s2.build_page_blocks(s2.PageInput(297, words, []), cfg)
    assert [b.type for b in blocks] == ["heading"]
    assert blocks[0].text == "Static Analysers In Practice"


# --------------------------------------------------------------------------- across pages


def test_code_merges_across_pages_but_not_across_a_heading(cfg) -> None:
    a = s2.RawBlock("code", "class A\n{", 10, 500, 530, True, False)
    b = s2.RawBlock("code", "}", 11, 77, 90, True, False)
    h = s2.RawBlock("heading", "Next", 11, 77, 90, False, False)
    c = s2.RawBlock("code", "}", 11, 100, 110, True, False)
    merged, n_code, _ = s2.merge_across_pages([a, b])
    assert n_code == 1 and [x.text for x in merged] == ["class A\n{\n}"]
    merged, n_code, _ = s2.merge_across_pages([a, h, c])
    assert n_code == 0 and [x.type for x in merged] == ["code", "heading", "code"]
    # A blank or skipped page in between means two blocks, not one.
    far = s2.RawBlock("code", "}", 13, 77, 90, True, False)
    merged, n_code, _ = s2.merge_across_pages([a, far])
    assert n_code == 0 and len(merged) == 2


def test_callout_merges_across_pages_only_when_next_has_no_title(cfg) -> None:
    a = s2.RawBlock("callout", "Title\nbody one", 101, 500, 600, True, True)
    cont = s2.RawBlock("callout", "body two", 102, 77, 120, True, False)
    fresh = s2.RawBlock("callout", "Other\nbody", 102, 77, 120, True, True)
    merged, _, n_call = s2.merge_across_pages([a, cont])
    assert n_call == 1 and merged[0].text == "Title\nbody one\n\nbody two"
    merged, _, n_call = s2.merge_across_pages([a, fresh])
    assert n_call == 0 and len(merged) == 2


# --------------------------------------------------------------------------- hierarchy


def test_attach_hierarchy_sets_chapter_section_ids_and_drops_labels() -> None:
    chapters = [
        Chapter(0, 0, "Foreword", "Front matter", "", 9, 10),
        Chapter(1, 5, "Property Promotion", "Part 1", "PHP, the Language", 65, 76),
    ]
    raw = [
        s2.RawBlock("label", "CHAPTER 05", 65, 70, 80, False, False),
        s2.RawBlock("chapter_title", "Property Promo- tion", 65, 88, 120, False, False),
        s2.RawBlock("paragraph", "Intro text.", 65, 140, 160, False, False),
        s2.RawBlock("heading", "Constructor promotion", 66, 77, 90, False, False),
        s2.RawBlock("code", "class CustomerDTO", 66, 120, 400, True, False),
        s2.RawBlock("paragraph", "Foreword text.", 9, 100, 120, False, False),
    ]
    blocks = s2.attach_hierarchy(raw, chapters, doc_id="d", printed_page_offset=2)

    assert [b.type for b in blocks] == ["chapter_title", "paragraph", "heading", "code",
                                         "paragraph"]
    assert blocks[0].text == "Property Promotion"  # bookmark title wins over wrapped print
    assert blocks[0].section_title == "" and blocks[1].section_title == ""
    assert blocks[2].section_title == "Constructor promotion"
    assert blocks[3].section_title == "Constructor promotion"
    assert blocks[3].chapter_no == 5 and blocks[3].chapter_title == "Property Promotion"
    assert blocks[3].part == "Part 1" and blocks[3].page_printed == 64
    assert blocks[4].chapter_no == 0 and blocks[4].chapter_title == "Foreword"
    assert blocks[4].section_title == ""  # section resets per chapter
    assert [b.block_index for b in blocks] == [0, 1, 2, 3, 4]
    assert blocks[0].block_id == "d:065:000" and blocks[1].block_id == "d:065:001"
    assert blocks[2].block_id == "d:066:000"
    assert all(isinstance(b, Block) for b in blocks)


def test_attach_hierarchy_fails_on_page_outside_every_chapter() -> None:
    chapters = [Chapter(0, 1, "A", "Part 1", "P", 13, 14)]
    raw = [s2.RawBlock("paragraph", "x", 99, 0, 10, False, False)]
    with pytest.raises(ValueError, match="99"):
        s2.attach_hierarchy(raw, chapters, doc_id="d", printed_page_offset=2)


# --------------------------------------------------------------------------- real PDF


@pytest.fixture(scope="module")
def structured(tmp_path_factory: pytest.TempPathFactory):
    if not PDF.exists():
        pytest.skip(f"{PDF.relative_to(ROOT)} is absent")
    settings = load_settings()
    data = tmp_path_factory.mktemp("data")
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    parsed = s1.parse(PDF, settings, out_dir=data / "parsed", index_dir=data / "index")
    result = s2.structure(parsed.doc_id, settings, in_dir=data / "parsed", out_dir=data / "blocks")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    return result, spans["ingest.structure"], list(read_jsonl(result.output_path, Block))


@needs_pdf
def test_all_33_chapters_found(structured) -> None:
    result, span, blocks = structured
    assert result.chapters_found == 33
    assert span.attributes["rag.chapters_found"] == 33
    assert {b.chapter_no for b in blocks} == set(range(32))
    assert {b.type for b in blocks} == {"chapter_title", "heading", "callout", "code", "paragraph"}


@needs_pdf
def test_customer_dto_is_one_code_block_with_indentation(structured) -> None:
    _, _, blocks = structured
    # The book reuses CustomerDTO in several chapters. The spec pins the page 66 example.
    hits = [b for b in blocks if "class CustomerDTO" in b.text and b.page_pdf == 66]
    assert len(hits) == 1
    block = hits[0]
    assert block.type == "code" and block.page_printed == 64
    assert block.chapter_no == 5 and block.chapter_title == "Property Promotion"
    lines = block.text.split("\n")
    assert lines[0].startswith("class CustomerDTO")
    assert any(ln.startswith("    public ") for ln in lines), lines
    assert any(ln.startswith("        ") for ln in lines), lines
    assert not any(ln != ln.rstrip() for ln in lines)


@needs_pdf
def test_callout_on_page_61_is_one_block(structured) -> None:
    _, _, blocks = structured
    callouts = [b for b in blocks if b.type == "callout" and b.page_pdf == 61]
    assert len(callouts) == 1
    assert callouts[0].text.startswith("PHP Compiler\nEven though PHP is an interpreted")
    assert callouts[0].in_rect is True


@needs_pdf
def test_no_labels_and_titles_come_from_bookmarks(structured) -> None:
    result, span, blocks = structured
    assert not any(b.text.startswith("CHAPTER ") for b in blocks)
    titles = {b.text for b in blocks if b.type == "chapter_title"}
    assert "Type Variance" in titles and "Static Analysers In Practice" in titles
    assert result.code_blocks_merged_across_pages > 0
    assert span.attributes["rag.blocks_total"] == len(blocks)
