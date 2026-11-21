# Stage 5 - Enrichment. Spec Section 5, Stage 5.
"""Stage 5: add metadata and extra text that make chunks easier to find.

Input: `data/chunks/{doc_id}.jsonl`. Output: `data/enriched/{doc_id}.jsonl`, the same chunks
with a full payload.

Milestone 2 rules:
1. Every payload field in the spec table is present and typed. `validate_payload` is the schema
   test; a missing or mistyped field raises.
2. `php_symbols` from code: keywords (`readonly`, `enum`, `match`, `never`, `fn`, `static`),
   attribute names inside `#[...]`, names after `class`, `interface`, `trait`, `enum`,
   `function`, plus the type after `new` and the type of a typed parameter (the spec's own
   example lists `datetimeimmutable`, which only those two patterns produce).
3. `php_symbols` from prose: `$variables` and `functions()`.
4. Lowercased, deduplicated in order of first appearance, capped at `enrich.symbols_max`.

`index_version` is decided here, once per run, as `v{n}-{embed_model_short}-{chunk_config_hash[:8]}`
with `n` one above the highest run in `data/index/`. Stage 7 reads it from the payload.

Milestone 7 rule 5 (contextual summaries, issue #37) lives in `s05_summary.py`: with
`enrich.contextual_summary` on, `enrich()` builds the summary model (`enrich.summary_model`),
opens the summary cache, and prepends one summary per chunk to `embedding_text`; the calls,
tokens, cost, failures, cache hits, and the prompt version land on the `ingest.enrich` span.
Rule 6 (hypothetical questions) raises when switched on until issue #38 lands.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from flp_rag.contracts import Attrs, Chunk, read_jsonl, write_jsonl
from flp_rag.ingest.s05_summary import Summarizer, Tally, load_summary_prompt
from flp_rag.models import ChatModel, get_chat_model
from flp_rag.settings import PROJECT_ROOT, Settings
from flp_rag.stores.summary_cache import SummaryCache
from flp_rag.tracing import stage

KEYWORDS = ("readonly", "enum", "match", "never", "fn", "static")

_KEYWORD = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b")
_ATTRIBUTE_BLOCK = re.compile(r"#\[(.*?)\]", re.DOTALL)
# A post-comma token is another attribute only when it is not a named argument (`path: ...`).
_ATTRIBUTE_NAME = re.compile(r"(?:^|,)\s*\\?([A-Za-z_][\w\\]*)(?!\s*:)")
# Declarations need their PHP syntax: `class Foo {`, `class Foo extends`, `function foo(`.
# Prose such as "a new function that" must not match.
_TYPE_DECLARATION = re.compile(
    r"\b(?:class|interface|trait|enum)\s+([A-Za-z_]\w*)\s*(?:\{|:|\bextends\b|\bimplements\b|$)",
    re.MULTILINE,
)
_FUNCTION_DECLARATION = re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\(")
_NEW = re.compile(r"\bnew\s+\\?([A-Z][\w\\]*)")
# A typed parameter sits in a parameter list: after `(`, `,`, `|`, `?`, or at a line start.
_TYPED_PARAM = re.compile(r"(?:[(,|?]|^)\s*\\?([A-Z][A-Za-z0-9_\\]*)\s+\$", re.MULTILINE)
_VARIABLE = re.compile(r"\$([a-zA-Z_]\w*)")
_CALL = re.compile(r"\b([a-zA-Z_]\w*)\(\)")
_RUN_DIR = re.compile(r"^v(\d+)-")


# --------------------------------------------------------------------------- symbols


def _last_segment(name: str) -> str:
    return name.rsplit("\\", 1)[-1]


def code_symbols(code: str) -> list[str]:
    """Symbols from PHP code, lowercased, in order of first appearance, duplicates removed."""
    found: list[str] = []
    found.extend(_KEYWORD.findall(code))
    for block in _ATTRIBUTE_BLOCK.findall(code):
        found.extend(_last_segment(n) for n in _ATTRIBUTE_NAME.findall(block))
    found.extend(_TYPE_DECLARATION.findall(code))
    found.extend(_FUNCTION_DECLARATION.findall(code))
    found.extend(_last_segment(n) for n in _NEW.findall(code))
    found.extend(_last_segment(n) for n in _TYPED_PARAM.findall(code))
    return _dedupe_lower(found)


def prose_symbols(prose: str) -> list[str]:
    """`$variables` (without the sigil) and `functions()` (without the parentheses)."""
    return _dedupe_lower([*_VARIABLE.findall(prose), *_CALL.findall(prose)])


def chunk_symbols(code: str, prose: str, *, has_code: bool, cap: int) -> list[str]:
    symbols = code_symbols(code) if has_code else []
    return _dedupe_lower([*symbols, *prose_symbols(prose)])[:cap]


def _dedupe_lower(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        low = name.lower()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return out


# --------------------------------------------------------------------------- versions


def embed_model_short(model: str) -> str:
    """`BAAI/bge-m3` -> `bgem3`."""
    return re.sub(r"[^a-z0-9]", "", _last_segment(model.rsplit("/", 1)[-1]).lower())


def next_index_version(index_dir: Path, embed_model: str, chunk_config_hash: str) -> str:
    """One above the highest run number among `data/index/v{n}-*` directories."""
    highest = 0
    if index_dir.is_dir():
        for child in index_dir.iterdir():
            m = _RUN_DIR.match(child.name)
            if child.is_dir() and m:
                highest = max(highest, int(m.group(1)))
    return f"v{highest + 1}-{embed_model_short(embed_model)}-{chunk_config_hash[:8]}"


# --------------------------------------------------------------------------- payload schema


class ChunkPayload(BaseModel):
    """Every field of the spec's payload table. The schema test of Stage 5."""

    model_config = ConfigDict(extra="allow", strict=True)

    doc_id: str
    chunk_id: str
    parent_id: str
    part: str
    chapter_no: int
    chapter_title: str
    section_title: str
    section_titles: list[str]
    page_pdf_start: int
    page_pdf_end: int
    page_printed_start: int
    page_printed_end: int
    has_code: bool
    code_lang: str
    code_group_id: str | None
    part_index: int | None
    php_symbols: list[str]
    token_count: int
    chunk_config_hash: str
    index_version: str
    embed_model: str
    created_at: str  # ISO 8601 text, kept as a string so the payload stays JSON for Qdrant

    @field_validator("created_at")
    @classmethod
    def _iso_8601(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("must be an ISO 8601 timestamp") from exc
        return value


def validate_payload(payload: dict[str, Any]) -> None:
    try:
        ChunkPayload.model_validate(payload)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        raise ValueError(f"payload fails the schema at {fields}: {exc.errors()[0]['msg']}") from exc


# --------------------------------------------------------------------------- enrichment


@dataclass(frozen=True)
class EnrichResult:
    doc_id: str
    chunks_enriched: int
    index_version: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    failures: int
    output_path: Path
    cache_hits: int = 0
    prompt_version: str | None = None  # the enrichment prompt, with contextual summaries on

    def to_attrs(self) -> Attrs:
        attrs: Attrs = {
            "rag.doc_id": self.doc_id,
            "rag.chunks_enriched": self.chunks_enriched,
            "rag.index_version": self.index_version,
            "rag.llm_calls": self.llm_calls,
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
            "rag.cost_usd": self.cost_usd,
            "rag.failures": self.failures,
            "rag.cache_hits": self.cache_hits,
        }
        if self.prompt_version is not None:
            attrs["rag.prompt_version"] = self.prompt_version
        return attrs


def enrich_chunk(chunk: Chunk, settings: Settings, *, index_version: str, created_at: str) -> Chunk:
    text = chunk.display_text
    payload = {
        **chunk.payload,
        # Code rules run on the whole chunk when it holds code; the prose rules always run.
        "php_symbols": chunk_symbols(
            text, text, has_code=bool(chunk.payload.get("has_code")), cap=settings.enrich.symbols_max
        ),
        "chunk_config_hash": settings.chunk_config_hash,
        "index_version": index_version,
        "embed_model": settings.embed.model,
        "created_at": created_at,
    }
    validate_payload(payload)
    return replace(chunk, payload=payload)


def enrich_chunks(
    chunks: Sequence[Chunk], settings: Settings, *, index_version: str, created_at: str
) -> list[Chunk]:
    """The Milestone 2 enrichment of every chunk (payload and symbols), no model."""
    if settings.enrich.hypothetical_questions:
        raise NotImplementedError("enrich.hypothetical_questions is Milestone 7, issue #38")
    return [enrich_chunk(c, settings, index_version=index_version, created_at=created_at) for c in chunks]


def enrich_all(
    chunks: Sequence[Chunk],
    settings: Settings,
    *,
    index_version: str,
    created_at: str,
    summarizer: Summarizer | None,
) -> tuple[list[Chunk], Tally]:
    """Every chunk enriched and, with a summarizer, summarized (issue #37). More than
    `enrich.summary_max_failures` chunks without a summary fail the run: a half-summarized
    index is a mixed embedding space, like a partial index."""
    enriched = enrich_chunks(chunks, settings, index_version=index_version, created_at=created_at)
    if summarizer is None:
        return enriched, Tally()
    out: list[Chunk] = []
    tally = Tally()
    for chunk in enriched:
        summarized, one = summarizer.apply(chunk)
        out.append(summarized)
        tally += one
        if tally.failures > settings.enrich.summary_max_failures:
            # Stop at the first chunk over the bound: the rest would cost calls and quota for
            # a run that fails anyway, and the cached summaries make the rerun cheap.
            raise RuntimeError(
                f"{tally.failures} chunk(s) have no summary after "
                f"{settings.enrich.summary_retry_attempts} attempts each "
                f"(enrich.summary_max_failures is {settings.enrich.summary_max_failures}); "
                f"stopped at chunk {len(out)} of {len(enriched)}"
            )
    return out, tally


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@stage("ingest.enrich", kind="CHAIN")
def enrich(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/chunks"),
    out_dir: Path = Path("data/enriched"),
    index_dir: Path = Path("data/index"),
    chat: ChatModel | None = None,
) -> EnrichResult:
    """Run Stage 5 on the Stage 4 output of one document. `chat` replaces the configured
    summary model (tests); it is built on first use only when `enrich.contextual_summary` is on."""
    chunks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Chunk))
    index_version = next_index_version(index_dir, settings.embed.model, settings.chunk_config_hash)
    cfg = settings.enrich
    prompt_version: str | None = None
    if cfg.contextual_summary:
        # The prompt is read from the project root, the same root the chunk config hash uses.
        prompt = load_summary_prompt(PROJECT_ROOT / cfg.summary_prompt_file)
        prompt_version = prompt.version
        model = chat if chat is not None else get_chat_model(settings, "enrich")
        with SummaryCache(settings.config_path.parent / cfg.summary_cache_path) as cache:
            enriched, tally = enrich_all(
                chunks,
                settings,
                index_version=index_version,
                created_at=utc_now_iso(),
                summarizer=Summarizer(model, prompt, cfg, cache),
            )
    else:
        enriched, tally = enrich_all(
            chunks, settings, index_version=index_version, created_at=utc_now_iso(), summarizer=None
        )
    output_path = out_dir / f"{doc_id}.jsonl"
    write_jsonl(output_path, enriched)
    return EnrichResult(
        doc_id=doc_id,
        chunks_enriched=len(enriched),
        index_version=index_version,
        llm_calls=tally.calls,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        cost_usd=tally.cost_usd,
        failures=tally.failures,
        output_path=output_path,
        cache_hits=tally.cache_hits,
        prompt_version=prompt_version,
    )


def symbol_stats(chunks: Sequence[Chunk]) -> dict[str, int]:
    counts = [len(c.payload.get("php_symbols", [])) for c in chunks]
    return {
        "chunks_with_symbols": sum(1 for n in counts if n),
        "symbols_total": sum(counts),
        "symbols_max_in_chunk": max(counts, default=0),
        "distinct_symbols": len({s for c in chunks for s in c.payload.get("php_symbols", [])}),
    }

