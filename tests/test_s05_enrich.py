"""Done-when tests for Stage 5, enrichment. Spec Section 5, Stage 5, Milestone 2 rules. Issue #11."""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Chunk, read_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.ingest import s02_structure as s2
from flp_rag.ingest import s03_clean as s3
from flp_rag.ingest import s04_chunk as s4
from flp_rag.ingest import s05_enrich as s5
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason=f"{PDF.relative_to(ROOT)} is absent")

CUSTOMER_DTO = """class CustomerDTO
{
    public string $name;

    public DateTimeImmutable $birth_date;

    public function __construct(
        string $name,
        DateTimeImmutable $birth_date
    ) {
        $this->name = $name;
        $this->birth_date = $birth_date;
    }
}"""


# --------------------------------------------------------------------------- symbols


def test_code_symbols_keywords_declarations_and_attributes() -> None:
    code = (
        "#[Attribute(Attribute::TARGET_CLASS)]\nfinal readonly class Money implements Stringable\n"
        "{\n    public static function fromCents(int $cents): static {}\n}\n"
        "enum Status: string { case Active = 'a'; }\ninterface Loggable {}\ntrait HasUuid {}\n"
        "$fn = fn($x) => match($x) { default => never() };\n#[Route('/blog'), Deprecated]\n"
    )
    symbols = s5.code_symbols(code)
    for expected in ("readonly", "enum", "match", "fn", "static", "attribute", "route", "deprecated",
                     "money", "fromcents", "status", "loggable", "hasuuid"):
        assert expected in symbols, expected
    assert "never" in symbols  # keyword even when used as a call


def test_code_symbols_include_types_after_new_and_typed_parameters() -> None:
    symbols = s5.code_symbols(CUSTOMER_DTO + "\n$d = new DateTimeImmutable('now');\n$c = new \\App\\Cart();")
    assert "customerdto" in symbols and "__construct" in symbols
    assert "datetimeimmutable" in symbols  # spec example: type of $birth_date
    assert "cart" in symbols  # namespace stripped to the last segment


def test_code_rules_ignore_english_prose_in_mixed_chunks() -> None:
    mixed = (
        "Here is a new function that I think will be useful. You cannot use new as a default\n"
        "value, and this function will perform a regex match. If $amount didn't have a type, the\n"
        "class that follows would fail:\n\n"
        "class Money\n{\n    public function __construct(\n        public readonly int $amount,\n"
        "        Currency $currency,\n    ) {}\n}\n"
    )
    symbols = s5.chunk_symbols(mixed, mixed, has_code=True, cap=40)
    for stopword in ("that", "as", "will", "if", "new", "in", "the", "value"):
        assert stopword not in symbols, stopword
    for expected in ("money", "__construct", "readonly", "currency", "amount", "match"):
        assert expected in symbols, expected


def test_attribute_named_arguments_are_not_symbols() -> None:
    symbols = s5.code_symbols("#[Route(path: '/users', methods: ['GET'])]\n#[Deprecated, Pure]")
    assert "route" in symbols and "deprecated" in symbols and "pure" in symbols
    assert "path" not in symbols and "methods" not in symbols


def test_prose_symbols_variables_and_calls() -> None:
    prose = "Use $this->name and call json_encode() on $array; getPaymentDate() returns a date."
    symbols = s5.prose_symbols(prose)
    assert symbols == ["this", "array", "json_encode", "getpaymentdate"]


def test_symbols_are_lowercase_deduplicated_ordered_and_capped() -> None:
    code = "\n".join(f"function fn{i}() {{}}" for i in range(60)) + "\nfunction fn0() {}"
    symbols = s5.chunk_symbols(code, "", has_code=True, cap=40)
    assert len(symbols) == 40 and len(set(symbols)) == 40
    assert symbols[0] == "fn0" and all(s == s.lower() for s in symbols)
    prose_only = s5.chunk_symbols("", "Call foo() then Foo() and $Bar.", has_code=False, cap=40)
    assert prose_only == ["bar", "foo"]


def test_no_symbols_in_plain_prose() -> None:
    assert s5.chunk_symbols("", "PHP is a language with a compiler.", has_code=False, cap=40) == []


# --------------------------------------------------------------------------- versions and payload


def test_next_index_version_counts_existing_runs(tmp_path: Path) -> None:
    assert s5.next_index_version(tmp_path, "BAAI/bge-m3", "9c1e0f2a12345678") == "v1-bgem3-9c1e0f2a"
    (tmp_path / "v1-bgem3-9c1e0f2a").mkdir()
    (tmp_path / "v2-bgem3-deadbeef").mkdir()
    (tmp_path / "junk").mkdir()
    assert s5.next_index_version(tmp_path, "BAAI/bge-m3", "9c1e0f2a12345678") == "v3-bgem3-9c1e0f2a"
    assert s5.embed_model_short("text-embedding-3-small") == "textembedding3small"


def _chunk(payload_extra: dict | None = None, text: str = CUSTOMER_DTO) -> Chunk:
    payload = {
        "doc_id": "d", "chunk_id": "d:05:0046", "parent_id": "d:05:s006", "part": "Part 1",
        "chapter_no": 5, "chapter_title": "Property Promotion", "section_title": "",
        "page_pdf_start": 66, "page_pdf_end": 66, "page_printed_start": 64, "page_printed_end": 64,
        "has_code": True, "code_lang": "php", "code_group_id": None, "part_index": None,
        "token_count": 83, "chunk_kind": "prose_with_code", "section_complete": False,
        "block_ids": ["d:066:001"], **(payload_extra or {}),
    }
    return Chunk(chunk_id="d:05:0046", parent_id="d:05:s006", display_text=text,
                 embedding_text="hdr\n" + text, token_count=83, payload=payload)


def test_enrich_chunk_fills_every_payload_field() -> None:
    settings = load_settings()
    out = s5.enrich_chunk(_chunk(), settings, index_version="v1-bgem3-9c1e0f2a",
                          created_at="2026-09-21T10:00:00Z")
    p = out.payload
    assert p["php_symbols"][:1] == ["customerdto"] and "__construct" in p["php_symbols"]
    assert p["chunk_config_hash"] == settings.chunk_config_hash
    assert p["index_version"] == "v1-bgem3-9c1e0f2a" and p["embed_model"] == "BAAI/bge-m3"
    assert p["created_at"] == "2026-09-21T10:00:00Z"
    s5.validate_payload(p)  # the schema test of the spec
    assert out.display_text == CUSTOMER_DTO and out.embedding_text.endswith(CUSTOMER_DTO)


def test_validate_payload_rejects_missing_or_mistyped_fields() -> None:
    settings = load_settings()
    good = s5.enrich_chunk(_chunk(), settings, index_version="v1-x-y", created_at="2026-09-21T10:00:00Z").payload
    with pytest.raises(ValueError, match="php_symbols"):
        s5.validate_payload({k: v for k, v in good.items() if k != "php_symbols"})
    with pytest.raises(ValueError, match="chapter_no"):
        s5.validate_payload({**good, "chapter_no": "5"})
    with pytest.raises(ValueError, match="created_at"):
        s5.validate_payload({**good, "created_at": "yesterday"})


def test_milestone_7_rules_are_not_wired_yet() -> None:
    settings = load_settings()
    on = settings.model_copy(update={"enrich": settings.enrich.model_copy(update={"contextual_summary": True})})
    with pytest.raises(NotImplementedError, match="#37"):
        s5.enrich_chunks([_chunk()], on, index_version="v", created_at="2026-09-21T10:00:00Z")


# --------------------------------------------------------------------------- real PDF


@pytest.fixture(scope="module")
def enriched(tmp_path_factory: pytest.TempPathFactory):
    if not PDF.exists():
        pytest.skip(f"{PDF.relative_to(ROOT)} is absent")
    settings = load_settings()
    data = tmp_path_factory.mktemp("data")
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    parsed = s1.parse(PDF, settings, out_dir=data / "parsed", index_dir=data / "index")
    s2.structure(parsed.doc_id, settings, in_dir=data / "parsed", out_dir=data / "blocks")
    s3.clean(parsed.doc_id, settings, in_dir=data / "blocks", out_dir=data / "clean")
    s4.chunk(parsed.doc_id, settings, in_dir=data / "clean", parsed_dir=data / "parsed",
             out_dir=data / "chunks", parents_dir=data / "parents")
    result = s5.enrich(parsed.doc_id, settings, in_dir=data / "chunks", out_dir=data / "enriched",
                       index_dir=data / "index")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    return result, spans["ingest.enrich"], list(read_jsonl(result.output_path, Chunk))


@needs_pdf
def test_schema_passes_for_every_chunk(enriched) -> None:
    result, span, chunks = enriched
    assert result.chunks_enriched == len(chunks) == 279
    for c in chunks:
        s5.validate_payload(c.payload)
        assert len(c.payload["php_symbols"]) <= load_settings().enrich.symbols_max
    assert span.attributes["rag.chunks_enriched"] == 279
    assert span.attributes["rag.llm_calls"] == 0 and span.attributes["rag.cost_usd"] == 0.0
    assert result.index_version.startswith("v1-bgem3-")
    assert len({c.payload["created_at"] for c in chunks}) == 1


@needs_pdf
def test_customer_dto_chunk_lists_its_symbols(enriched) -> None:
    _, _, chunks = enriched
    c = next(c for c in chunks if "class CustomerDTO" in c.display_text
             and c.payload["page_pdf_start"] <= 66 <= c.payload["page_pdf_end"])
    assert "customerdto" in c.payload["php_symbols"]
    assert "__construct" in c.payload["php_symbols"]
    assert "datetimeimmutable" in c.payload["php_symbols"]
    assert json.dumps(c.payload)  # serializable for Qdrant
