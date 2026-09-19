# Loads config.yaml into pydantic models and computes config_hash. Spec Section 9.
"""Configuration loader.

One file, `config.yaml`, holds every tunable value. API keys never live there; they come from
environment variables. Every section is typed and closed (`extra="forbid"`), so a misspelled
section or key fails at load time with the offending name.

Hashes stamped on the model:
- `config_hash = sha256(file bytes)[:8]` goes on every trace (`rag.config_hash`).
- `chunk_config_hash = sha256(canonical json of the chunk section)[:16]` names an index version
  and lets Stage 1 skip a PDF that is already indexed with the same chunking.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ABSTAIN_TEXT = "No information found"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_SECRET_KEY = re.compile(r"(^|_)(api_?key|key|secret|token|password|credentials?)(_|$)")


class _Section(BaseModel):
    """Frozen and closed: unknown keys are errors. List-valued keys load as tuples so they cannot
    be edited in place. `prices_usd_per_million_tokens` stays a dict; treat it as read-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- offline sections


class ParseConfig(_Section):
    header_max_top_pt: float = 50
    header_font_size_pt: float = 8
    printed_page_offset: int = 2
    skip_pages: tuple[int, ...]
    rect_min_width_pt: float = Field(default=100, ge=0)
    rect_fills: dict[Literal["code", "callout", "highlight"], tuple[float, float, float]]


class StructureConfig(_Section):
    line_tolerance_pt: float = 2
    paragraph_gap_factor: float = 1.5
    mono_fonts: tuple[str, ...]
    heading_font: str
    heading_sizes_pt: tuple[float, ...]
    chapter_title_size_pt: float
    label_size_pt: float
    label_pattern: str
    mono_char_width_ratio: float = Field(gt=0)
    default_mono_size_pt: float = Field(gt=0)
    code_min_gap_chars: float = Field(ge=0, le=1)


CleanRule = Literal["nfc", "nbsp", "zero_width", "whitespace", "page_number_lines"]


class CleanConfig(_Section):
    rules_enabled: tuple[CleanRule, ...]


class ChunkConfig(_Section):
    tokenizer: str = "cl100k_base"
    target_tokens: int = Field(gt=0)
    min_tokens: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    code_max_tokens: int = Field(gt=0)
    overlap_paragraph_max_tokens: int = Field(ge=0)
    parent_max_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> "ChunkConfig":
        if not self.min_tokens <= self.target_tokens <= self.max_tokens <= self.code_max_tokens:
            raise ValueError(
                "chunk: need min_tokens <= target_tokens <= max_tokens <= code_max_tokens, got "
                f"{self.min_tokens} <= {self.target_tokens} <= {self.max_tokens} "
                f"<= {self.code_max_tokens}"
            )
        return self


class EnrichConfig(_Section):
    symbols_max: int = Field(ge=0)
    contextual_summary: bool = False
    hypothetical_questions: int = Field(default=0, ge=0)


class EmbedConfig(_Section):
    model: str
    dim: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    sparse_model: str
    cache_path: Path


class IndexConfig(_Section):
    qdrant_url: str
    alias: str
    keep_previous: int = Field(ge=0)


class VerifyConfig(_Section):
    min_chunks_per_page: float = Field(ge=0)
    smoke_queries: Path


# --------------------------------------------------------------------------- online sections


class ApiConfig(_Section):
    port: int = Field(gt=0, lt=65536)
    timeout_s: float = Field(gt=0)


class GuardConfig(_Section):
    max_chars: int = Field(gt=0)
    rate_limit_per_min: int = Field(gt=0)


class QueryConfig(_Section):
    history_turns: int = Field(ge=0)
    expansions: int = Field(ge=0)
    smalltalk_policy: Literal["abstain", "greet"]
    decompose: bool = False
    hyde: bool = False
    model: str


class RetrievalConfig(_Section):
    top_k_dense: int = Field(gt=0)
    top_k_sparse: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    top_k_fused: int = Field(gt=0)


class RerankConfig(_Section):
    model: str
    top_n: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class GateConfig(_Section):
    tau_low: float = Field(ge=0, le=1)
    tau_pass: float = Field(ge=0, le=1)
    tau_high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> "GateConfig":
        if not self.tau_low <= self.tau_pass <= self.tau_high:
            raise ValueError(
                "gate: need tau_low <= tau_pass <= tau_high, got "
                f"{self.tau_low} <= {self.tau_pass} <= {self.tau_high}"
            )
        return self


class ContextConfig(_Section):
    max_tokens: int = Field(gt=0)
    max_blocks: int = Field(gt=0)
    order: Literal["position", "sandwich"]


class GenConfig(_Section):
    model: str
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    prompt_file: Path
    strict_prompt_file: Path


class OutputConfig(_Section):
    faithfulness_sample_rate: float = Field(ge=0, le=1)
    code_verbatim: bool = True
    judge_model: str


class FeedbackConfig(_Section):
    enabled: bool = True


class CacheConfig(_Section):
    semantic_enabled: bool = False
    similarity_threshold: float = Field(ge=0, le=1)
    ttl_hours: int = Field(ge=0)


class TraceConfig(_Section):
    exporter_endpoint: str
    capture_content: bool = True
    retention_days: int = Field(default=30, ge=0)
    content_max_chars: int = Field(default=20_000, gt=0)
    attr_string_max_chars: int = Field(default=1_000, gt=0)
    export_timeout_s: float = Field(default=10, gt=0)


class Price(_Section):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


# --------------------------------------------------------------------------- root


class Settings(_Section):
    abstain_text: str = ABSTAIN_TEXT
    parse: ParseConfig
    structure: StructureConfig
    clean: CleanConfig
    chunk: ChunkConfig
    enrich: EnrichConfig
    embed: EmbedConfig
    index: IndexConfig
    verify: VerifyConfig
    api: ApiConfig
    guard: GuardConfig
    query: QueryConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    gate: GateConfig
    context: ContextConfig
    gen: GenConfig
    output: OutputConfig
    feedback: FeedbackConfig
    cache: CacheConfig
    trace: TraceConfig
    prices_usd_per_million_tokens: dict[str, Price]

    config_hash: str = Field(pattern=r"^[0-9a-f]{8}$")
    chunk_config_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    config_path: Path

    @model_validator(mode="before")
    @classmethod
    def _reject_secrets(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaked = sorted(_find_secret_keys(data))
            if leaked:
                raise ValueError(
                    f"config.yaml holds secret-looking keys {leaked}; "
                    "move them to environment variables"
                )
        return data

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "Settings":
        if self.abstain_text != ABSTAIN_TEXT:
            raise ValueError(f"abstain_text must be exactly {ABSTAIN_TEXT!r}")
        unpriced = sorted(
            {self.gen.model, self.query.model, self.output.judge_model}
            - set(self.prices_usd_per_million_tokens)
        )
        if unpriced:
            raise ValueError(
                f"no price for configured model(s) {unpriced}; "
                "add them under prices_usd_per_million_tokens"
            )
        return self

    def price_for(self, model: str) -> Price:
        return self.prices_usd_per_million_tokens[model]


# --------------------------------------------------------------------------- loading


def _find_secret_keys(node: Any, prefix: str = "") -> list[str]:
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _SECRET_KEY.search(str(key).lower()):
            found.append(path)
        found.extend(_find_secret_keys(value, path))
    return found


def chunk_config_hash(chunk: ChunkConfig) -> str:
    """Stable hash of the validated chunk section. Defaults applied, keys sorted, so two files
    with the same effective chunking give the same hash whether or not they spell out defaults."""
    canonical = json.dumps(chunk.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Read `config.yaml`, validate every section, and stamp both hashes."""
    config_path = Path(path)
    raw_bytes = config_path.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must hold a mapping at the top level")
    # Validate the chunk section first: its hash names the index version (Stage 7) and gates
    # the "already indexed" check (Stage 1). A missing section fails here with the field names.
    if "chunk" not in data:
        raise ValueError(f"{config_path}: missing `chunk` section")
    chunk_section = data["chunk"]
    if not isinstance(chunk_section, dict):
        raise TypeError(f"{config_path}: `chunk` must be a mapping")
    chunk = ChunkConfig(**chunk_section)
    return Settings(
        **data,
        config_hash=hashlib.sha256(raw_bytes).hexdigest()[:8],
        chunk_config_hash=chunk_config_hash(chunk),
        config_path=config_path,
    )
