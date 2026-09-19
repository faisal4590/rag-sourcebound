"""Done-when tests for Stage 3, cleaning and normalization. Spec Section 5, Stage 3. Issue #8.

Rules run on synthetic blocks. The done-when scan opens the real PDF chain and skips when
`data/raw/` holds no book.
"""

import dataclasses
import re
import unicodedata
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Block, read_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.ingest import s02_structure as s2
from flp_rag.ingest import s03_clean as s3
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent")

ALL_RULES = ("nfc", "nbsp", "zero_width", "whitespace", "page_number_lines")


def block(text: str, type_: str = "paragraph") -> Block:
    return Block(block_id="d:066:000", type=type_, text=text, part="Part 1", chapter_no=5,  # type: ignore[arg-type]
                 chapter_title="Property Promotion", section_title="", page_pdf=66,
                 page_printed=64, block_index=0, in_rect=type_ in ("code", "callout"))


# --------------------------------------------------------------------------- single rules


def test_nfc_normalizes_decomposed_characters() -> None:
    decomposed = "café"  # e + combining acute
    out, counts = s3.clean_text(decomposed, "paragraph", ("nfc",))
    assert out == "café" and unicodedata.is_normalized("NFC", out)
    assert counts == {"nfc": 1}  # one character changed form
    assert s3.clean_text("e\N{COMBINING ACUTE ACCENT}e\N{COMBINING ACUTE ACCENT}", "paragraph",
                         ("nfc",))[1] == {"nfc": 2}
    assert s3.clean_text("café", "paragraph", ("nfc",))[1] == {}


def test_nbsp_becomes_a_space() -> None:
    out, counts = s3.clean_text("PHP\N{NO-BREAK SPACE}8.2\N{NARROW NO-BREAK SPACE}is", "paragraph", ("nbsp",))
    assert out == "PHP 8.2 is" and counts == {"nbsp": 2}


def test_zero_width_and_backspace_are_deleted() -> None:
    out, counts = s3.clean_text(
        "a\N{ZERO WIDTH SPACE}b\N{ZERO WIDTH NO-BREAK SPACE}c\x08d\N{ZERO WIDTH JOINER}",
        "code", ("zero_width",),
    )
    assert out == "abcd" and counts == {"zero_width": 4}


def test_whitespace_runs_collapse_in_prose_but_newlines_survive() -> None:
    out, counts = s3.clean_text("Title \t here\nbody  text   end \n\nnext", "callout", ("whitespace",))
    assert out == "Title here\nbody text end\n\nnext"
    assert counts == {"whitespace": 4}  # three inner runs plus one trailing space


def test_single_trailing_space_counts_as_a_fire() -> None:
    out, counts = s3.clean_text("Line one. \nLine two.\t", "paragraph", ("whitespace",))
    assert out == "Line one.\nLine two." and counts == {"whitespace": 2}
    # A trailing run is counted once, not once as a run and again as trailing.
    assert s3.clean_text("a   ", "paragraph", ("whitespace",)) == ("a", {"whitespace": 1})


def test_whitespace_rule_never_touches_code() -> None:
    code = "    public function __construct(\n        string $name,  // two spaces\n"
    out, counts = s3.clean_text(code, "code", ALL_RULES)
    assert out == code and counts == {}


def test_page_number_lines_are_deleted_from_prose_only() -> None:
    out, counts = s3.clean_text("First line.\n64\nSecond line.", "paragraph", ("page_number_lines",))
    assert out == "First line.\nSecond line." and counts == {"page_number_lines": 1}
    code = "$a = [\n    1,\n    64\n];"
    assert s3.clean_text(code, "code", ("page_number_lines",)) == (code, {})
    # A block that is only a number is content. It is never blanked.
    assert s3.clean_text("42", "paragraph", ("page_number_lines",)) == ("42", {})
    assert s3.clean_text("7\n\n12", "callout", ("page_number_lines",)) == ("7\n\n12", {})


def test_disabled_rules_do_not_fire() -> None:
    out, counts = s3.clean_text("a\N{NO-BREAK SPACE}b", "paragraph", ("nfc",))
    assert out == "a\N{NO-BREAK SPACE}b" and counts == {}


# --------------------------------------------------------------------------- blocks


def test_clean_block_returns_new_block_with_same_metadata() -> None:
    src = block("PHP\N{NO-BREAK SPACE}is  great.")
    out, counts = s3.clean_block(src, ALL_RULES)
    assert out.text == "PHP is great."
    assert dataclasses.replace(out, text=src.text) == src
    assert counts == {"nbsp": 1, "whitespace": 1}
    assert s3.clean_block(block("clean"), ALL_RULES)[1] == {}


def test_code_char_count_drops_only_by_deleted_artifacts() -> None:
    src = block("$a\N{ZERO WIDTH SPACE} = 1;\x08\n$b\N{NO-BREAK SPACE}= 2;", "code")
    out, counts = s3.clean_block(src, ALL_RULES)
    deleted = counts.get("zero_width", 0)
    assert len(src.text) - len(out.text) == deleted == 2
    assert out.text == "$a = 1;\n$b = 2;"


def test_clean_blocks_aggregates_counts_by_rule_and_type() -> None:
    blocks = [block("a\N{NO-BREAK SPACE}b"), block("x  y", "callout"), block("ok", "code"),
              block("z\N{ZERO WIDTH SPACE}", "code")]
    out, result_counts, changed, removed = s3.clean_blocks(blocks, ALL_RULES)
    assert [b.text for b in out] == ["a b", "x y", "ok", "z"]
    assert result_counts == {"nbsp": {"paragraph": 1}, "whitespace": {"callout": 1},
                             "zero_width": {"code": 1}}
    assert changed == 3 and removed == {"code": 1, "paragraph": 0, "callout": 1}


# --------------------------------------------------------------------------- real PDF


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory: pytest.TempPathFactory):
    if not PDF.exists():
        pytest.skip(f"{PDF.relative_to(ROOT)} is absent")
    settings = load_settings()
    data = tmp_path_factory.mktemp("data")
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    parsed = s1.parse(PDF, settings, out_dir=data / "parsed", index_dir=data / "index")
    s2.structure(parsed.doc_id, settings, in_dir=data / "parsed", out_dir=data / "blocks")
    result = s3.clean(parsed.doc_id, settings, in_dir=data / "blocks", out_dir=data / "clean")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    blocks_in = list(read_jsonl(data / "blocks" / f"{parsed.doc_id}.jsonl", Block))
    blocks_out = list(read_jsonl(result.output_path, Block))
    return result, spans["ingest.clean"], blocks_in, blocks_out


@needs_pdf
def test_scan_of_clean_output_finds_no_artifacts(cleaned) -> None:
    result, _, _, blocks_out = cleaned
    text = result.output_path.read_text()
    assert "\\b" not in text and "\x08" not in text  # backspace, raw or JSON-escaped
    assert "\N{NO-BREAK SPACE}" not in text and "\\u00a0" not in text
    assert all(unicodedata.is_normalized("NFC", b.text) for b in blocks_out)
    # The spec scans for "Front Line PHP" and expects zero. The Foreword mentions the book by
    # title in its own prose, so the check is: no header-shaped line survives, and every
    # remaining mention sits inside a sentence of a prose block.
    header = re.compile(r"(^|\n)(\d{1,3} Front Line PHP|Chapter \d+ - .+ \d{1,3})(\n|$)")
    assert not any(header.search(b.text) for b in blocks_out)
    mentions = [b for b in blocks_out if "Front Line PHP" in b.text]
    assert len(mentions) == 1
    assert mentions[0].type == "paragraph" and mentions[0].chapter_title == "Foreword"


@needs_pdf
def test_code_blocks_keep_every_character_but_artifacts(cleaned) -> None:
    result, span, blocks_in, blocks_out = cleaned
    assert len(blocks_in) == len(blocks_out)
    rules = load_settings().clean.rules_enabled
    for before, after in zip(blocks_in, blocks_out, strict=True):
        assert before.block_id == after.block_id
        if before.type == "code":
            _, counts = s3.clean_block(before, rules)
            deleted_here = counts.get("zero_width", 0)
            assert len(before.text) - len(after.text) == deleted_here
    assert span.attributes["rag.blocks_changed"] == result.blocks_changed
    assert "rag.rule_counts" in span.attributes


@needs_pdf
def test_this_edition_is_already_clean(cleaned) -> None:
    # Stage 1 removed the running header and pdfplumber yields NFC text, so no rule fires here.
    result, _, _, _ = cleaned
    assert result.blocks_changed == 0
    assert result.rule_counts == {}
