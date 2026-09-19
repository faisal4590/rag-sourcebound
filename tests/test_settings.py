"""Done-when tests for settings.py. Spec Section 9. Issues #2 and #5."""

import re
from pathlib import Path

import pytest

from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (ROOT / "config.yaml").read_text()


def _load(tmp_path: Path, text: str) -> Settings:
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return load_settings(p)


# --------------------------------------------------------------------------- basics


def test_abstain_text_is_exact() -> None:
    assert ABSTAIN_TEXT == "No information found"


def test_loads_shipped_config() -> None:
    s = load_settings()
    assert s.abstain_text == ABSTAIN_TEXT
    assert s.trace.exporter_endpoint == "http://localhost:6006/v1/traces"
    assert s.trace.capture_content is True
    assert s.trace.retention_days == 30


def test_config_hash_is_eight_hex_chars() -> None:
    s = load_settings()
    assert re.fullmatch(r"[0-9a-f]{8}", s.config_hash)
    assert s.config_hash == load_settings().config_hash


def test_config_hash_tracks_file_bytes(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _load(tmp_path / "a", SHIPPED)
    b = _load(tmp_path / "b", SHIPPED.replace("retention_days: 30", "retention_days: 31"))
    assert a.config_hash != b.config_hash
    assert b.trace.retention_days == 31


def test_missing_required_section_raises(tmp_path: Path) -> None:
    without_trace = SHIPPED[: SHIPPED.index("trace:")] + SHIPPED[SHIPPED.index("prices_usd"):]
    with pytest.raises(ValueError, match="trace"):
        _load(tmp_path, without_trace)


def test_missing_chunk_section_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk"):
        _load(tmp_path, "abstain_text: 'No information found'\n")


@pytest.mark.parametrize(
    "line",
    ["openai_api_key: sk-abc", "anthropic_key: sk-ant", "gen:\n  token: x", "db_password: pw",
     "cohere_credentials: c"],
)
def test_secret_looking_keys_are_rejected(tmp_path: Path, line: str) -> None:
    with pytest.raises(ValueError, match="secret-looking"):
        _load(tmp_path, SHIPPED + "\n" + line + "\n")


# --------------------------------------------------------------------------- every section typed


def test_every_spec_section_is_typed() -> None:
    s = load_settings()
    assert s.parse.header_max_top_pt == 50
    assert s.parse.header_font_size_pt == 8
    assert s.parse.skip_pages == (1, 2, 3, 4, 5, 6, 7, 8, 12, 156, 232)
    assert s.structure.line_tolerance_pt == 2
    assert s.structure.paragraph_gap_factor == 1.5
    assert s.structure.mono_fonts == ("JetBrainsMono", "IBMPlexMono")
    assert s.clean.rules_enabled == ("nfc", "nbsp", "zero_width", "whitespace", "page_number_lines")
    assert s.chunk.tokenizer == "cl100k_base"
    assert (s.chunk.min_tokens, s.chunk.target_tokens, s.chunk.max_tokens) == (120, 350, 600)
    assert s.chunk.code_max_tokens == 900
    assert s.chunk.overlap_paragraph_max_tokens == 80
    assert s.chunk.parent_max_tokens == 2500
    assert s.enrich.symbols_max == 40
    assert s.enrich.contextual_summary is False
    assert s.enrich.hypothetical_questions == 0
    assert s.embed.model == "BAAI/bge-m3"
    assert s.embed.dim == 1024
    assert s.embed.batch_size == 32
    assert s.embed.sparse_model == "Qdrant/bm25"
    assert s.embed.cache_path == Path("data/cache/embeddings.sqlite")
    assert s.index.qdrant_url == "http://localhost:6333"
    assert s.index.alias == "flp_chunks_current"
    assert s.index.keep_previous == 1
    assert s.verify.min_chunks_per_page == 0.5
    assert s.verify.smoke_queries == Path("eval/smoke_queries.jsonl")
    assert s.api.port == 8000
    assert s.api.timeout_s == 20
    assert s.guard.max_chars == 2000
    assert s.guard.rate_limit_per_min == 30
    assert s.query.history_turns == 3
    assert s.query.expansions == 3
    assert s.query.smalltalk_policy == "abstain"
    assert s.query.decompose is False and s.query.hyde is False
    assert s.retrieval.top_k_dense == 30
    assert s.retrieval.top_k_sparse == 30
    assert s.retrieval.rrf_k == 60
    assert s.retrieval.top_k_fused == 30
    assert s.rerank.model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank.top_n == 8
    assert s.rerank.batch_size == 16
    assert (s.gate.tau_low, s.gate.tau_pass, s.gate.tau_high) == (0.20, 0.35, 0.60)
    assert s.context.max_tokens == 6000
    assert s.context.max_blocks == 8
    assert s.context.order == "position"
    assert s.gen.temperature == 0
    assert s.gen.max_output_tokens == 800
    assert s.gen.prompt_file == Path("prompts/answer_v1.md")
    assert s.gen.strict_prompt_file == Path("prompts/answer_strict_v1.md")
    assert s.output.faithfulness_sample_rate == 0.2
    assert s.output.code_verbatim is True
    assert s.feedback.enabled is True
    assert s.cache.semantic_enabled is False
    assert s.cache.similarity_threshold == 0.97
    assert s.cache.ttl_hours == 168
    assert s.trace.content_max_chars == 20_000
    assert s.trace.export_timeout_s == 10


def test_prices_are_typed_per_model() -> None:
    s = load_settings()
    price = s.prices_usd_per_million_tokens[s.gen.model]
    assert price.input == 0.0 and price.output == 0.0


def test_misspelled_section_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="retrieveal"):
        _load(tmp_path, SHIPPED.replace("retrieval:", "retrieveal:"))


def test_misspelled_key_inside_section_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top_k_dens"):
        _load(tmp_path, SHIPPED.replace("top_k_dense: 30", "top_k_dens: 30"))


def test_settings_are_frozen() -> None:
    s = load_settings()
    with pytest.raises(ValueError):
        s.gate.tau_pass = 0.5  # type: ignore[misc]
    with pytest.raises(AttributeError):
        s.parse.skip_pages.append(9999)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- cross-field rules


def test_gate_thresholds_must_be_ordered(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tau_low"):
        _load(tmp_path, SHIPPED.replace("tau_low: 0.20", "tau_low: 0.70"))


def test_chunk_token_bounds_must_be_ordered(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        _load(tmp_path, SHIPPED.replace("min_tokens: 120", "min_tokens: 700"))


def test_configured_models_need_a_price(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="price"):
        _load(tmp_path, SHIPPED.replace("model: <generator-model-name>", "model: gpt-x"))


def test_literal_choices_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="order"):
        _load(tmp_path, SHIPPED.replace("order: position", "order: random"))
    with pytest.raises(ValueError, match="smalltalk_policy"):
        _load(tmp_path, SHIPPED.replace("smalltalk_policy: abstain", "smalltalk_policy: chat"))


# --------------------------------------------------------------------------- chunk_config_hash


def test_chunk_config_hash_is_sixteen_hex_and_stable() -> None:
    s = load_settings()
    assert re.fullmatch(r"[0-9a-f]{16}", s.chunk_config_hash)
    assert s.chunk_config_hash == load_settings().chunk_config_hash


def test_chunk_config_hash_ignores_spelled_out_defaults(tmp_path: Path) -> None:
    # tokenizer has a default. Omitting it or stating it must give the same hash.
    without = _load(tmp_path, SHIPPED.replace("  tokenizer: cl100k_base\n", ""))
    assert without.chunk == load_settings().chunk
    assert without.chunk_config_hash == load_settings().chunk_config_hash


def test_chunk_config_hash_changes_only_with_chunk_section(tmp_path: Path) -> None:
    base = load_settings()
    (tmp_path / "x").mkdir()
    other = _load(tmp_path / "x", SHIPPED.replace("retention_days: 30", "retention_days: 31"))
    assert other.chunk_config_hash == base.chunk_config_hash
    (tmp_path / "y").mkdir()
    changed = _load(tmp_path / "y", SHIPPED.replace("target_tokens: 350", "target_tokens: 400"))
    assert changed.chunk_config_hash != base.chunk_config_hash
