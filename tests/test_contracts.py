"""Done-when tests for the data contracts. Spec Section 10. Issue #4."""

import dataclasses
import json
from pathlib import Path

import pytest

from flp_rag import contracts as c

PRIMITIVES = (str, int, float, bool)

SAMPLES: list[c.Record] = [
    c.Word(text="readonly", font="Inter-Regular", size=10.0, x0=72.0, x1=110.5, top=120.0,
           bottom=131.0, page_pdf=77),
    c.Block(block_id="3f9a:077:0003", type="code", text="class CustomerDTO {}", part="Part 1",
            chapter_no=5, chapter_title="Property Promotion", section_title="Constructor promotion",
            page_pdf=66, page_printed=64, block_index=3, in_rect=True),
    c.Chunk(chunk_id="3f9a:05:0012", parent_id="3f9a:05:p0004", display_text="A DTO ...",
            embedding_text="Front Line PHP > ... \nA DTO ...", token_count=412,
            payload={"chapter_no": 5, "has_code": True, "php_symbols": ["customerdto"]}),
    c.Parent(parent_id="3f9a:05:p0004", chapter_no=5, section_title="Constructor promotion",
             text="...", page_printed_start=64, page_printed_end=66, token_count=1300),
    c.StandaloneQuery(raw="What about enums?", standalone="How do enums work in PHP 8.1?",
                      intent="book_question", wants_code=False, chapter_hint=9,
                      variants=["php enums", "backed enums"]),
    c.Candidate(chunk_id="3f9a:09:0031", fused_score=0.0321, rerank_score=0.87,
                provenance=["dense:v0", "sparse:v1"], payload={"chapter_no": 9}),
    c.ContextBlock(label="S1", source_ids=["3f9a:06:0020", "3f9a:06:0021"], text="...",
                   chapter_no=6, chapter_title="Readonly Properties", page_printed_start=75,
                   page_printed_end=77, token_count=900),
    c.Source(label="S1", chunk_id="3f9a:06:0020", chapter_no=6, chapter_title="Readonly Properties",
             section_title="Readonly", page_printed_start=75, page_printed_end=77, score=0.91),
    c.Response(answer="Readonly properties ... [S1]", status="answered",
               sources=[c.Source(label="S1", chunk_id="x", chapter_no=6, chapter_title="R",
                                 section_title="s", page_printed_start=75, page_printed_end=76,
                                 score=0.9)],
               trace_id="03922e931537626a84eb066442e5a517", timings_ms={"rerank": 120},
               model={"gen": "m", "embed": "BAAI/bge-m3"}, index_version="v1-bgem3-9c1e0f2a",
               prompt_version="ab12cd34", debug=None),
    c.EvalCase(case_id="g001", group="answerable", question="Show a DTO.",
               expected_answer="A class ...", expected_pages_printed=[64, 65, 66],
               expected_symbol="CustomerDTO", expected_abstain=False, golden_version="v1"),
]


@pytest.mark.parametrize("record", SAMPLES, ids=lambda r: type(r).__name__)
def test_jsonl_round_trip(record: c.Record) -> None:
    line = c.to_json(record)
    assert "\n" not in line
    assert c.from_json(type(record), line) == record


@pytest.mark.parametrize("record", SAMPLES, ids=lambda r: type(r).__name__)
def test_to_attrs_is_flat_primitives(record: c.Record) -> None:
    attrs = record.to_attrs()
    assert attrs, "every record exposes at least one attribute"
    for key, value in attrs.items():
        assert isinstance(key, str) and key.startswith("rag."), key
        assert isinstance(value, PRIMITIVES), (key, type(value))


@pytest.mark.parametrize("record", SAMPLES, ids=lambda r: type(r).__name__)
def test_records_are_frozen(record: c.Record) -> None:
    field = dataclasses.fields(record)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(record, field, "changed")


def test_to_attrs_never_carries_full_text() -> None:
    block = dataclasses.replace(SAMPLES[1], text="x" * 5000)
    chunk = dataclasses.replace(SAMPLES[2], display_text="y" * 5000, embedding_text="z" * 6000)
    for attrs in (block.to_attrs(), chunk.to_attrs()):
        assert all(len(v) < 200 for v in attrs.values() if isinstance(v, str))
    assert block.to_attrs()["rag.text_chars"] == 5000


def test_lists_become_json_strings() -> None:
    attrs = SAMPLES[4].to_attrs()
    assert json.loads(attrs["rag.variants"]) == ["php enums", "backed enums"]
    assert json.loads(SAMPLES[6].to_attrs()["rag.source_ids"]) == ["3f9a:06:0020", "3f9a:06:0021"]


def test_none_fields_are_dropped_from_attrs() -> None:
    q = dataclasses.replace(SAMPLES[4], chapter_hint=None)
    assert "rag.chapter_hint" not in q.to_attrs()
    assert "rag.chapter_hint" in SAMPLES[4].to_attrs()


def test_literal_fields_are_validated() -> None:
    with pytest.raises(ValueError, match="type"):
        dataclasses.replace(SAMPLES[1], type="table")
    with pytest.raises(ValueError, match="status"):
        c.from_json(c.Response, c.to_json(SAMPLES[8]).replace('"answered"', '"maybe"'))
    with pytest.raises(ValueError, match="group"):
        dataclasses.replace(SAMPLES[9], group="unknown")


def test_response_nests_sources_on_read() -> None:
    back = c.from_json(c.Response, c.to_json(SAMPLES[8]))
    assert isinstance(back.sources[0], c.Source)
    assert back.sources[0].page_printed_start == 75


def test_unknown_key_in_json_fails_fast() -> None:
    line = json.dumps({**json.loads(c.to_json(SAMPLES[3])), "colour": "red"})
    with pytest.raises(ValueError, match="colour"):
        c.from_json(c.Parent, line)


def test_write_and_read_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "blocks.jsonl"
    blocks = [dataclasses.replace(SAMPLES[1], block_index=i) for i in range(3)]

    n = c.write_jsonl(path, blocks)

    assert n == 3
    assert path.read_text().count("\n") == 3
    assert list(c.read_jsonl(path, c.Block)) == blocks


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text(c.to_json(SAMPLES[7]) + "\n\n" + c.to_json(SAMPLES[7]) + "\n")
    assert len(list(c.read_jsonl(path, c.Source))) == 2


def test_nan_score_fails_at_write_time() -> None:
    bad = dataclasses.replace(SAMPLES[5], fused_score=float("nan"))
    with pytest.raises(ValueError):
        c.to_json(bad)


def test_source_attrs_carry_titles() -> None:
    attrs = SAMPLES[7].to_attrs()
    assert attrs["rag.chapter_title"] == "Readonly Properties"
    assert attrs["rag.section_title"] == "Readonly"


def test_response_attrs_match_spec_names() -> None:
    attrs = SAMPLES[8].to_attrs()
    assert attrs["rag.status"] == "answered"
    assert attrs["rag.n_sources"] == 1
    assert attrs["rag.answer_chars"] == len(SAMPLES[8].answer)
    assert attrs["rag.index_version"] == "v1-bgem3-9c1e0f2a"
    assert attrs["rag.prompt_version"] == "ab12cd34"
