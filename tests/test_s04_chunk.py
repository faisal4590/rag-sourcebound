"""Done-when tests for Stage 4, chunking. Spec Section 5, Stage 4. Issue #9.

Rules run on synthetic blocks with the real tokenizer. The done-when tests run the pipeline on
the real PDF and skip when `data/raw/` holds no book.
"""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Block, Chapter, Chunk, Parent, read_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.ingest import s02_structure as s2
from flp_rag.ingest import s03_clean as s3
from flp_rag.ingest import s04_chunk as s4
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent")

CH5 = Chapter(6, 5, "Property Promotion", "Part 1", "PHP, the Language", 65, 76)
CH6 = Chapter(7, 6, "Readonly Properties", "Part 1", "PHP, the Language", 77, 86)
FOREWORD = Chapter(0, 0, "Foreword", "Front matter", "", 9, 10)
CHAPTERS = [FOREWORD, CH5, CH6]


@pytest.fixture(scope="module")
def cfg():
    return load_settings().chunk


@pytest.fixture(scope="module")
def enc(cfg):
    return s4.tokenizer(cfg)


def words(n: int, seed: str = "alpha") -> str:
    """Prose of about n tokens: distinct short words, one token each."""
    vocabulary = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
    start = vocabulary.index(seed) if seed in vocabulary else 0
    return " ".join(vocabulary[(start + i) % len(vocabulary)] for i in range(n)) + "."


_counter = {"n": 0}


def B(type_: str, text: str, *, chapter: Chapter = CH5, section: str = "Constructor promotion",
      page: int = 66) -> Block:
    _counter["n"] += 1
    return Block(
        block_id=f"d:{page:03d}:{_counter['n']:03d}", type=type_, text=text,  # type: ignore[arg-type]
        part=chapter.part, chapter_no=chapter.chapter_no, chapter_title=chapter.title,
        section_title=section, page_pdf=page, page_printed=page - 2,
        block_index=_counter["n"], in_rect=type_ in ("code", "callout"),
    )


def run(blocks: list[Block], cfg, enc) -> tuple[list[Chunk], list[Parent], s4.ChunkStats]:
    return s4.chunk_blocks(blocks, CHAPTERS, cfg, enc, doc_id="d")


# --------------------------------------------------------------------------- sections


def test_sections_split_on_heading_and_chapter(enc, cfg) -> None:
    blocks = [
        B("chapter_title", "Property Promotion", section=""),
        B("paragraph", words(30), section=""),
        B("heading", "Constructor promotion"),
        B("paragraph", words(30)),
        B("heading", "Defaults", section="Defaults"),
        B("paragraph", words(30), section="Defaults"),
        B("chapter_title", "Readonly Properties", chapter=CH6, section="", page=77),
        B("paragraph", words(30), chapter=CH6, section="", page=77),
    ]
    sections = s4.group_sections(blocks)
    assert [(s.chapter_no, s.section_title, len(s.blocks)) for s in sections] == [
        (5, "", 2), (5, "Constructor promotion", 2), (5, "Defaults", 2), (6, "", 2),
    ]


# --------------------------------------------------------------------------- packing


def test_small_section_is_one_chunk_with_heading_first(cfg, enc) -> None:
    blocks = [B("heading", "Only in Constructors"), B("paragraph", words(25))]
    chunks, parents, stats = run(blocks, cfg, enc)
    assert len(chunks) == 1 and len(parents) == 1
    assert chunks[0].display_text.startswith("Only in Constructors\n\n")
    assert chunks[0].token_count < cfg.min_tokens
    assert stats.chunks_under_min == 1
    assert chunks[0].parent_id == parents[0].parent_id


def test_chapter_title_blocks_are_not_chunk_text(cfg, enc) -> None:
    blocks = [B("chapter_title", "Property Promotion", section=""), B("paragraph", words(40), section="")]
    chunks, _, _ = run(blocks, cfg, enc)
    assert len(chunks) == 1 and "Property Promotion" not in chunks[0].display_text
    assert "Chapter 05: Property Promotion" in chunks[0].embedding_text


def test_paragraphs_pack_to_target_and_never_exceed_max(cfg, enc) -> None:
    blocks = [B("heading", "Long")] + [B("paragraph", words(150, seed=s)) for s in
                                        ("alpha", "beta", "gamma", "delta", "omega")]
    chunks, _, stats = run(blocks, cfg, enc)
    assert len(chunks) == 2
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    assert chunks[0].token_count >= cfg.target_tokens
    assert stats.chunks_over_max == 0


def test_short_last_paragraph_repeats_as_overlap(cfg, enc) -> None:
    p_big, p_small, p_next = words(300), words(50, seed="omega"), words(300, seed="beta")
    blocks = [B("paragraph", p_big), B("paragraph", p_small), B("paragraph", p_next)]
    chunks, _, _ = run(blocks, cfg, enc)
    assert len(chunks) == 2
    assert chunks[0].display_text.endswith(p_small)
    assert chunks[1].display_text.startswith(p_small)  # overlap, whole paragraph
    assert p_next in chunks[1].display_text


def test_long_last_paragraph_gives_no_overlap(cfg, enc) -> None:
    blocks = [B("paragraph", words(200)), B("paragraph", words(200, seed="beta")),
              B("paragraph", words(200, seed="gamma"))]
    chunks, _, _ = run(blocks, cfg, enc)
    assert len(chunks) == 2
    assert not chunks[1].display_text.startswith(words(200, seed="beta")[:40])


def test_trailing_small_remainder_merges_into_previous_chunk(cfg, enc) -> None:
    blocks = [B("paragraph", words(200)), B("paragraph", words(160, seed="beta")),
              B("paragraph", words(30, seed="gamma"))]
    chunks, _, stats = run(blocks, cfg, enc)
    assert len(chunks) == 1 and chunks[0].token_count <= cfg.max_tokens
    assert stats.chunks_under_min == 0


# --------------------------------------------------------------------------- code


def test_code_attaches_to_the_prose_chunk_before_it(cfg, enc) -> None:
    code = "class CustomerDTO\n{\n    public string $name;\n}"
    blocks = [B("paragraph", "Here is an example of a customer data transfer object:"), B("code", code)]
    chunks, _, stats = run(blocks, cfg, enc)
    assert len(chunks) == 1
    assert chunks[0].display_text.endswith(code)
    assert chunks[0].payload["has_code"] is True and chunks[0].payload["code_lang"] == "php"
    assert stats.composition == {"prose_only": 0, "prose_with_code": 1, "code_only": 0, "callout": 0}


def test_code_that_does_not_fit_becomes_its_own_chunk_with_lead_in(cfg, enc) -> None:
    intro = "The full class looks like this:"
    code = "\n".join(f"    public string $field{i};" for i in range(140))  # about 1400 > 900
    blocks = [B("paragraph", words(450)), B("paragraph", intro), B("code", code)]
    chunks, _, stats = run(blocks, cfg, enc)
    kinds = [c.payload["chunk_kind"] for c in chunks]
    assert kinds[0] == "prose_only"
    code_chunks = [c for c in chunks if c.payload["chunk_kind"] == "code_only"]
    assert code_chunks and code_chunks[0].display_text.startswith(intro + "\n\n")
    assert stats.chunks_over_max == 0


def test_huge_code_splits_at_blank_lines_into_a_group(cfg, enc) -> None:
    part = "\n".join(f"$value{i} = compute($value{i});" for i in range(60))  # about 600 tokens
    code = f"{part}\n\n{part}\n\n{part}"
    blocks = [B("paragraph", "Intro."), B("code", code)]
    chunks, _, _ = run(blocks, cfg, enc)
    pieces = [c for c in chunks if c.payload["code_group_id"] is not None]
    assert len(pieces) == 3
    assert len({c.payload["code_group_id"] for c in pieces}) == 1
    assert [c.payload["part_index"] for c in pieces] == [0, 1, 2]
    assert all(c.token_count <= cfg.code_max_tokens for c in pieces)
    assert pieces[0].display_text.startswith("Intro.\n\n")  # lead-in on the first piece only
    assert not pieces[1].display_text.startswith("Intro.")


def test_consecutive_code_blocks_share_a_chunk_when_they_fit(cfg, enc) -> None:
    blocks = [B("paragraph", "Two snippets:"), B("code", "$a = 1;"), B("code", "$b = 2;")]
    chunks, _, _ = run(blocks, cfg, enc)
    assert len(chunks) == 1 and "$a = 1;\n\n$b = 2;" in chunks[0].display_text


# --------------------------------------------------------------------------- callouts


def test_callout_is_its_own_chunk_and_takes_following_code(cfg, enc) -> None:
    blocks = [
        B("paragraph", words(40)),
        B("callout", "PHP Compiler\nEven though PHP is interpreted, it has a compiler."),
        B("code", "php -l file.php"),
        B("paragraph", words(40, seed="beta")),
    ]
    chunks, _, _ = run(blocks, cfg, enc)
    assert len(chunks) == 3
    assert chunks[1].display_text.startswith("PHP Compiler\n") and chunks[1].display_text.endswith("php -l file.php")
    assert chunks[1].payload["chunk_kind"] == "callout"
    assert "PHP Compiler" not in chunks[0].display_text and "PHP Compiler" not in chunks[2].display_text


# --------------------------------------------------------------------------- parents, ids, header


def test_one_parent_per_section_and_windows_for_huge_sections(cfg, enc) -> None:
    small = [B("heading", "A"), B("paragraph", words(50))]
    huge = [B("heading", "B", section="B")] + [B("paragraph", words(190, seed=s), section="B")
                                                for s in ["alpha", "beta", "gamma", "delta", "omega",
                                                          "sigma", "theta", "kappa"] * 2]
    chunks, parents, _ = run(small + huge, cfg, enc)
    small_chunks = [c for c in chunks if c.payload["section_title"] == "A"]
    huge_chunks = [c for c in chunks if c.payload["section_title"] == "B"]
    assert len({c.parent_id for c in small_chunks}) == 1
    assert len({c.parent_id for c in huge_chunks}) == len(huge_chunks) > 3
    by_id = {p.parent_id: p for p in parents}
    mid = huge_chunks[1]
    window = by_id[mid.parent_id]
    assert huge_chunks[0].display_text[:30] in window.text and huge_chunks[2].display_text[:30] in window.text
    assert all(c.parent_id in by_id for c in chunks)


def test_window_parent_does_not_repeat_the_overlap_paragraph(cfg, enc) -> None:
    # Short paragraphs, so every chunk boundary carries an overlap; section over parent_max.
    paras = [words(60, seed=s) + f" mark{i}." for i, s in enumerate(
        ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"] * 6)]
    blocks = [B("heading", "W", section="W")] + [B("paragraph", t, section="W") for t in paras]
    chunks, parents, _ = run(blocks, cfg, enc)
    by_id = {p.parent_id: p for p in parents}
    assert len({c.parent_id for c in chunks}) == len(chunks) > 3
    for c in chunks:
        window = by_id[c.parent_id].text
        for i in range(len(paras)):
            assert window.count(f" mark{i}.") <= 1, (c.chunk_id, i)


def test_token_count_matches_the_display_text(cfg, enc) -> None:
    blocks = [B("paragraph", words(300)), B("paragraph", words(50, seed="omega")),
              B("paragraph", words(300, seed="beta")), B("code", "$a = 1;")]
    chunks, _, _ = run(blocks, cfg, enc)
    for c in chunks:
        assert c.token_count == len(enc.encode(c.display_text))


def test_chunk_ids_header_pages_and_payload(cfg, enc) -> None:
    blocks = [B("heading", "Constructor promotion", page=66), B("paragraph", words(30), page=66),
              B("code", "class A {}", page=67)]
    chunks, parents, _ = run(blocks, cfg, enc)
    c = chunks[0]
    assert c.chunk_id == "d:05:0000"
    assert c.embedding_text.startswith(
        "Front Line PHP > Part 1: PHP, the Language > Chapter 05: Property Promotion > "
        "Constructor promotion (p. 64-65)\n"
    )
    assert c.embedding_text.endswith(c.display_text)
    p = c.payload
    assert p["doc_id"] == "d" and p["parent_id"] == parents[0].parent_id
    assert (p["page_pdf_start"], p["page_pdf_end"], p["page_printed_start"], p["page_printed_end"]) == (66, 67, 64, 65)
    assert p["chapter_no"] == 5 and p["part"] == "Part 1" and p["section_title"] == "Constructor promotion"
    assert p["token_count"] == c.token_count and p["block_ids"] == [b.block_id for b in blocks]
    assert parents[0].page_printed_start == 64 and parents[0].page_printed_end == 65


def test_front_matter_header_has_no_chapter_number(cfg, enc) -> None:
    blocks = [B("paragraph", words(30), chapter=FOREWORD, section="", page=9)]
    chunks, _, _ = run(blocks, cfg, enc)
    assert chunks[0].embedding_text.startswith("Front Line PHP > Front matter > Foreword > Foreword (p. 7)\n")
    assert chunks[0].chunk_id == "d:00:0000"


def test_same_input_gives_same_ids(cfg, enc) -> None:
    blocks = [B("paragraph", words(30)), B("paragraph", words(30, seed="beta"))]
    a, _, _ = run(blocks, cfg, enc)
    b, _, _ = run(blocks, cfg, enc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunks_to_documents() -> None:
    chunk = Chunk(chunk_id="d:05:0001", parent_id="p", display_text="body", embedding_text="hdr\nbody",
                  token_count=2, payload={"chapter_no": 5, "has_code": False})
    docs = s4.chunks_to_documents([chunk])
    assert docs[0].page_content == "hdr\nbody" and docs[0].metadata["chunk_id"] == "d:05:0001"
    assert docs[0].metadata["chapter_no"] == 5


# --------------------------------------------------------------------------- real PDF


@pytest.fixture(scope="module")
def chunked(tmp_path_factory: pytest.TempPathFactory):
    if not PDF.exists():
        pytest.skip(f"{PDF.relative_to(ROOT)} is absent")
    settings = load_settings()
    data = tmp_path_factory.mktemp("data")
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    parsed = s1.parse(PDF, settings, out_dir=data / "parsed", index_dir=data / "index")
    s2.structure(parsed.doc_id, settings, in_dir=data / "parsed", out_dir=data / "blocks")
    s3.clean(parsed.doc_id, settings, in_dir=data / "blocks", out_dir=data / "clean")
    result = s4.chunk(parsed.doc_id, settings, in_dir=data / "clean", parsed_dir=data / "parsed",
                      out_dir=data / "chunks", parents_dir=data / "parents")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    chunks = list(read_jsonl(result.output_path, Chunk))
    parents = list(read_jsonl(result.parents_path, Parent))
    return result, spans["ingest.chunk"], chunks, parents


@needs_pdf
def test_chunk_counts_on_the_real_book(chunked) -> None:
    result, span, chunks, parents = chunked
    assert 150 <= result.chunks_total <= 400, result.chunks_total  # spec estimated 250-350
    assert result.chunks_over_max == 0 and span.attributes["rag.chunks_over_max"] == 0
    assert max(c.token_count for c in chunks) <= 900
    assert result.chunks_under_min == span.attributes["rag.chunks_under_min"]
    # Under-min chunks are complete sections, callouts, or prose beside a callout (decisions
    # in CONTEXT.md). Everything else merged into a neighbor.
    ids = [c.chunk_id for c in chunks]
    for c in chunks:
        if c.token_count >= 120 or c.payload["section_complete"] or c.payload["chunk_kind"] == "callout":
            continue
        i = ids.index(c.chunk_id)
        neighbors = [chunks[j] for j in (i - 1, i + 1) if 0 <= j < len(chunks)
                     and chunks[j].payload["section_title"] == c.payload["section_title"]
                     and chunks[j].payload["chapter_no"] == c.payload["chapter_no"]]
        assert any(n.payload["chunk_kind"] == "callout" for n in neighbors), c.chunk_id
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.parent_id for c in chunks} <= {p.parent_id for p in parents}
    assert span.attributes["rag.chunk_config_hash"] == load_settings().chunk_config_hash
    assert json.loads(span.attributes["rag.composition"])["prose_with_code"] > 100


@needs_pdf
def test_customer_dto_class_is_whole_in_one_chunk(chunked) -> None:
    _, _, chunks, _ = chunked
    hits = [c for c in chunks if "class CustomerDTO" in c.display_text and c.payload["page_pdf_start"] <= 66 <= c.payload["page_pdf_end"]]
    assert len(hits) == 1
    text = hits[0].display_text
    assert "public function __construct(" in text
    # The class closes inside the chunk; the chunk itself continues with the next paragraph.
    assert "$this->birth_date = $birth_date;\n    }\n}" in text
    assert hits[0].payload["has_code"] is True and hits[0].payload["chapter_no"] == 5
    # The book reuses the class in 13 code blocks across chapters, packed into several chunks.
    # The spec's `grep -c "class CustomerDTO"` of 1 assumed one occurrence.
    assert sum(1 for c in chunks if "class CustomerDTO" in c.display_text) >= 5
