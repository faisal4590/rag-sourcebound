# Data contracts: Word, Block, Chunk, Parent, StandaloneQuery, Candidate, ContextBlock, Source, Response, EvalCase. Spec Section 10.
"""Data contracts.

Every record is a frozen dataclass with a `to_attrs()` method. `to_attrs()` returns a flat
dictionary of primitives and short JSON strings for the `@stage` decorator (spec 7.3). It never
carries full text: `tracing.py` applies the length caps from `config.yaml`, and these methods
only expose ids, counts, scores, and small lists.

JSONL helpers at the bottom read and write one record per line for `data/{stage}/`.

`frozen=True` blocks field reassignment, not in-place edits of a `list` or `dict` field. Treat
`payload`, `variants`, `provenance`, `sources`, and friends as read-only. Build a changed record
with `dataclasses.replace`.
"""

import dataclasses
import json
import typing
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, get_args, get_origin, runtime_checkable

Attrs = dict[str, str | int | float | bool]

BlockType = Literal["chapter_title", "heading", "callout", "code", "paragraph"]
ResponseStatus = Literal["answered", "no_information", "error"]
EvalGroup = Literal["answerable", "in_domain_unanswerable", "out_of_domain"]


@runtime_checkable
class Record(Protocol):
    """Anything the pipeline passes between stages."""

    def to_attrs(self) -> Attrs: ...


# --------------------------------------------------------------------------- helpers


def _attrs(**pairs: Any) -> Attrs:
    """Build the attribute dict: drop None, keep primitives, JSON-encode the rest."""
    out: Attrs = {}
    for key, value in pairs.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            out[f"rag.{key}"] = value
        else:
            out[f"rag.{key}"] = json.dumps(value, default=str)
    return out


def _check_literals(record: Any) -> None:
    """Fail fast when a `Literal` field holds a value outside its set."""
    hints = typing.get_type_hints(type(record))
    for f in dataclasses.fields(record):
        hint = hints[f.name]
        if get_origin(hint) is Literal:
            value = getattr(record, f.name)
            if value not in get_args(hint):
                raise ValueError(
                    f"{type(record).__name__}.{f.name} must be one of {get_args(hint)}, "
                    f"got {value!r}"
                )


# --------------------------------------------------------------------------- offline records


@dataclass(frozen=True)
class Word:
    """Stage 1. One word from pdfplumber with its font and position."""

    text: str
    font: str
    size: float
    x0: float
    x1: float
    top: float
    bottom: float
    page_pdf: int

    def to_attrs(self) -> Attrs:
        return _attrs(page_pdf=self.page_pdf, font=self.font, size=self.size)


@dataclass(frozen=True)
class Block:
    """Stages 2-3. One typed unit of the book with its chapter hierarchy."""

    block_id: str
    type: BlockType
    text: str
    part: str
    chapter_no: int
    chapter_title: str
    section_title: str
    page_pdf: int
    page_printed: int
    block_index: int
    in_rect: bool

    def __post_init__(self) -> None:
        _check_literals(self)

    def to_attrs(self) -> Attrs:
        return _attrs(
            block_id=self.block_id,
            block_type=self.type,
            chapter_no=self.chapter_no,
            section_title=self.section_title,
            page_pdf=self.page_pdf,
            page_printed=self.page_printed,
            block_index=self.block_index,
            in_rect=self.in_rect,
            text_chars=len(self.text),
        )


@dataclass(frozen=True)
class Chunk:
    """Stages 4-7. The unit that the system embeds and retrieves."""

    chunk_id: str
    parent_id: str
    display_text: str
    embedding_text: str
    token_count: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> Attrs:
        return _attrs(
            chunk_id=self.chunk_id,
            parent_id=self.parent_id,
            token_count=self.token_count,
            chapter_no=self.payload.get("chapter_no"),
            has_code=self.payload.get("has_code"),
            display_chars=len(self.display_text),
            embedding_chars=len(self.embedding_text),
        )


@dataclass(frozen=True)
class Parent:
    """Stage 4. The section that contains a chunk. The generator reads parents."""

    parent_id: str
    chapter_no: int
    section_title: str
    text: str
    page_printed_start: int
    page_printed_end: int
    token_count: int

    def to_attrs(self) -> Attrs:
        return _attrs(
            parent_id=self.parent_id,
            chapter_no=self.chapter_no,
            section_title=self.section_title,
            page_printed_start=self.page_printed_start,
            page_printed_end=self.page_printed_end,
            token_count=self.token_count,
        )


# --------------------------------------------------------------------------- online records


@dataclass(frozen=True)
class StandaloneQuery:
    """Stage 11. The raw question, its standalone form, classification, and search variants."""

    raw: str
    standalone: str
    intent: str
    wants_code: bool
    chapter_hint: int | None
    variants: list[str] = field(default_factory=list)

    def to_attrs(self) -> Attrs:
        return _attrs(
            question_raw=self.raw,
            question_standalone=self.standalone,
            intent=self.intent,
            wants_code=self.wants_code,
            chapter_hint=self.chapter_hint,
            variants=self.variants,
            n_variants=len(self.variants),
        )


@dataclass(frozen=True)
class Candidate:
    """Stages 12-14. A chunk that a retriever returned, before or after reranking."""

    chunk_id: str
    fused_score: float
    rerank_score: float | None
    provenance: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> Attrs:
        return _attrs(
            chunk_id=self.chunk_id,
            fused_score=self.fused_score,
            rerank_score=self.rerank_score,
            provenance=self.provenance,
            chapter_no=self.payload.get("chapter_no"),
        )


@dataclass(frozen=True)
class ContextBlock:
    """Stage 15. One labeled block that the generator reads, built from a parent."""

    label: str
    source_ids: list[str]
    text: str
    chapter_no: int
    chapter_title: str
    page_printed_start: int
    page_printed_end: int
    token_count: int

    def to_attrs(self) -> Attrs:
        return _attrs(
            label=self.label,
            source_ids=self.source_ids,
            chapter_no=self.chapter_no,
            chapter_title=self.chapter_title,
            page_printed_start=self.page_printed_start,
            page_printed_end=self.page_printed_end,
            token_count=self.token_count,
        )


@dataclass(frozen=True)
class Source:
    """Stage 18. One cited source with printed pages."""

    label: str
    chunk_id: str
    chapter_no: int
    chapter_title: str
    section_title: str
    page_printed_start: int
    page_printed_end: int
    score: float

    def to_attrs(self) -> Attrs:
        return _attrs(
            label=self.label,
            chunk_id=self.chunk_id,
            chapter_no=self.chapter_no,
            chapter_title=self.chapter_title,
            section_title=self.section_title,
            page_printed_start=self.page_printed_start,
            page_printed_end=self.page_printed_end,
            score=self.score,
        )


@dataclass(frozen=True)
class Response:
    """Stage 18. What the API and the CLI return."""

    answer: str
    status: ResponseStatus
    sources: list[Source]
    trace_id: str
    timings_ms: dict[str, int]
    model: dict[str, str]
    index_version: str
    prompt_version: str
    debug: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_literals(self)

    def to_attrs(self) -> Attrs:
        return _attrs(
            status=self.status,
            n_sources=len(self.sources),
            answer_chars=len(self.answer),
            trace_id=self.trace_id,
            index_version=self.index_version,
            prompt_version=self.prompt_version,
        )


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvalCase:
    """Section 8. One golden dataset case."""

    case_id: str
    group: EvalGroup
    question: str
    expected_answer: str | None
    expected_pages_printed: list[int]
    expected_symbol: str | None
    expected_abstain: bool
    golden_version: str

    def __post_init__(self) -> None:
        _check_literals(self)

    def to_attrs(self) -> Attrs:
        return _attrs(
            case_id=self.case_id,
            group=self.group,
            expected_abstain=self.expected_abstain,
            expected_pages=self.expected_pages_printed,
            expected_symbol=self.expected_symbol,
            golden_version=self.golden_version,
        )


# --------------------------------------------------------------------------- JSONL


def to_dict(record: Any) -> dict[str, Any]:
    return dataclasses.asdict(record)


def from_dict[R](cls: type[R], data: dict[str, Any]) -> R:
    """Build a record from a dict. Nested dataclasses and lists of them are rebuilt. Unknown
    keys raise, so a schema drift in `data/` fails at read time."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {name: _coerce(hints[name], value) for name, value in data.items()}
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(hint)
    if origin is typing.Union or origin is type(int | None):
        inner = [a for a in get_args(hint) if a is not type(None)]
        return _coerce(inner[0], value) if len(inner) == 1 else value
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return from_dict(hint, value)
    if origin is list and isinstance(value, list):
        (item_hint,) = get_args(hint) or (Any,)
        return [_coerce(item_hint, v) for v in value]
    return value


def to_json(record: Any) -> str:
    # allow_nan=False: a NaN score is a bug upstream. Fail at write time, not in a later reader.
    return json.dumps(to_dict(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def from_json[R](cls: type[R], line: str) -> R:
    return from_dict(cls, json.loads(line))


def write_jsonl(path: Path | str, records: Iterable[Any]) -> int:
    """Write one record per line. Returns the number of lines written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(to_json(record))
            fh.write("\n")
            count += 1
    return count


def read_jsonl[R](path: Path | str, cls: type[R]) -> Iterator[R]:
    """Yield one record per non-blank line."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield from_json(cls, line)
