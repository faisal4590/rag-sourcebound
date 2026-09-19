# Stage 4 - Chunking. Ends with chunks_to_documents() -> list[Document]. Spec Section 5, Stage 4.
"""Stage 4: build retrieval units that carry a complete idea and never cut a code example.

Input: `data/clean/{doc_id}.jsonl` and the chapter table `data/parsed/{doc_id}.chapters.jsonl`.
Output: `data/chunks/{doc_id}.jsonl` (`Chunk`) and `data/parents/{doc_id}.jsonl` (`Parent`).

Rules (spec Section 5, Stage 4, plus the decisions in CONTEXT.md):
1. A chunk never crosses a heading and never splits a code block.
2. Paragraphs pack toward `target_tokens`, never past `max_tokens`. A section smaller than
   `min_tokens` is one chunk. A trailing remainder under `min_tokens` merges into the chunk
   before it when the result fits.
3. When a chunk closes, its last paragraph repeats at the start of the next chunk of the same
   section if it has at most `overlap_paragraph_max_tokens`.
4. A code block attaches to the chunk that ends directly before it when the total stays within
   `code_max_tokens`. Otherwise the code becomes its own chunk with the last paragraph before it
   as lead-in. Code over `code_max_tokens` splits at blank lines, then at line ends, into pieces
   that share a `code_group_id`.
5. A callout is its own chunk. Code directly after it attaches to it.
6. One parent per section. A section over `parent_max_tokens` gets one parent per chunk: the
   window of the chunk and its two neighbors.
7. `embedding_text = chunk_header + "\\n" + display_text`. `chunk_id = doc:chapter:seq` with one
   document-wide sequence.
"""

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from langchain_core.documents import Document

from flp_rag.contracts import Attrs, Block, Chapter, Chunk, Parent, read_jsonl, write_jsonl
from flp_rag.settings import ChunkConfig, Settings
from flp_rag.tracing import stage

BOOK_TITLE = "Front Line PHP"
PROSE_KINDS = ("prose_only", "prose_with_code")


def tokenizer(cfg: ChunkConfig) -> tiktoken.Encoding:
    return tiktoken.get_encoding(cfg.tokenizer)


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class Section:
    chapter_no: int
    chapter_title: str
    section_title: str
    blocks: list[Block]

    @property
    def token_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.type != "chapter_title")


def group_sections(blocks: Sequence[Block]) -> list[Section]:
    """Split blocks into sections: a heading starts one, and so does a chapter change."""
    sections: list[Section] = []
    for b in blocks:
        new_chapter = not sections or (sections[-1].chapter_no, sections[-1].chapter_title) != (
            b.chapter_no, b.chapter_title
        )
        if new_chapter or b.type == "heading":
            title = b.text if b.type == "heading" else b.section_title
            sections.append(Section(b.chapter_no, b.chapter_title, title, []))
        sections[-1].blocks.append(b)
    return sections


# --------------------------------------------------------------------------- drafts


@dataclass(frozen=True)
class _Piece:
    kind: str  # heading | paragraph | callout | code
    text: str
    tokens: int
    block: Block


@dataclass
class _Draft:
    pieces: list[_Piece] = field(default_factory=list)
    overlap: _Piece | None = None  # repeated last paragraph of the previous chunk
    lead_in: _Piece | None = None  # paragraph copied in front of a code-only chunk
    code_group_id: str | None = None
    part_index: int | None = None
    callout: bool = False

    def tokens(self, enc: tiktoken.Encoding) -> int:
        """Measured on the final display text, separators included, like Parent.token_count."""
        return len(enc.encode(self.display_text()))

    def own_text(self) -> str:
        """The draft's own pieces, without the overlap or lead-in copied from a neighbor."""
        return "\n\n".join(p.text for p in self.pieces)

    @property
    def has_code(self) -> bool:
        return any(p.kind == "code" for p in self.pieces)

    @property
    def has_prose(self) -> bool:
        return any(p.kind in ("heading", "paragraph") for p in self.pieces)

    @property
    def kind(self) -> str:
        if self.callout:
            return "callout"
        if self.has_code and self.has_prose:
            return "prose_with_code"
        if self.has_code:
            return "code_only"
        return "prose_only"

    def last_paragraph(self) -> _Piece | None:
        for p in reversed(self.pieces):
            if p.kind == "paragraph":
                return p
        return None

    def display_text(self) -> str:
        parts = [p.text for p in (self.overlap, self.lead_in) if p is not None]
        parts += [p.text for p in self.pieces]
        return "\n\n".join(parts)

    def block_ids(self) -> list[str]:
        return [p.block.block_id for p in self.pieces]

    def pages(self) -> tuple[int, int]:
        pages = [p.block.page_pdf for p in self.pieces]
        return min(pages), max(pages)


def split_code(text: str, enc: tiktoken.Encoding, max_tokens: int) -> list[str]:
    """Split code at blank lines into pieces of at most max_tokens. A paragraph of code that is
    still too long splits at line ends. Never inside a line."""
    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        t = len(enc.encode(para))
        if t > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(_split_lines(para, enc, max_tokens))
            continue
        joined = t + (2 if current else 0)
        if current and current_tokens + joined > max_tokens:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += joined
    if current:
        pieces.append("\n\n".join(current))
    # The join estimate is a proxy. Re-check every piece against the cap on its real encoding.
    checked: list[str] = []
    for piece in pieces:
        if len(enc.encode(piece)) <= max_tokens:
            checked.append(piece)
        else:
            checked.extend(_split_lines(piece, enc, max_tokens))
    return checked


def _split_lines(para: str, enc: tiktoken.Encoding, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in para.split("\n"):
        t = len(enc.encode(line)) + 1
        if current and current_tokens + t > max_tokens:
            pieces.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += t
    if current:
        pieces.append("\n".join(current))
    return pieces


class _Packer:
    """Packs the blocks of one section into drafts."""

    def __init__(self, cfg: ChunkConfig, enc: tiktoken.Encoding) -> None:
        self.cfg, self.enc = cfg, enc
        self.drafts: list[_Draft] = []
        self.acc: _Draft | None = None

    def piece(self, block: Block) -> _Piece:
        return _Piece(block.type, block.text, len(self.enc.encode(block.text)), block)

    def flush(self) -> None:
        if self.acc is not None and self.acc.pieces:
            self.drafts.append(self.acc)
        self.acc = None

    def overlap_from_previous(self) -> _Piece | None:
        if not self.drafts or self.drafts[-1].kind not in PROSE_KINDS:
            return None
        last = self.drafts[-1].last_paragraph()
        if last is not None and last.tokens <= self.cfg.overlap_paragraph_max_tokens:
            return last
        return None

    def lead_in_from_previous(self) -> _Piece | None:
        if not self.drafts:
            return None
        last = self.drafts[-1].last_paragraph()
        if last is not None and last.tokens <= self.cfg.overlap_paragraph_max_tokens:
            return last
        return None

    def add_prose(self, p: _Piece) -> None:
        if self.acc is not None and (
            self.acc.callout
            or self.acc.tokens(self.enc) >= self.cfg.target_tokens
            or self.acc.tokens(self.enc) + p.tokens > self.cfg.max_tokens
        ):
            self.flush()
        if self.acc is None:
            self.acc = _Draft(overlap=self.overlap_from_previous())
        self.acc.pieces.append(p)

    def add_callout(self, p: _Piece) -> None:
        self.flush()
        self.acc = _Draft(callout=True, pieces=[p])

    def add_code(self, p: _Piece) -> None:
        limit = self.cfg.code_max_tokens
        if p.tokens > limit:
            self.flush()
            lead = self.lead_in_from_previous()
            group = f"{p.block.block_id}:g"
            for i, text in enumerate(split_code(p.text, self.enc, limit)):
                piece = _Piece("code", text, len(self.enc.encode(text)), p.block)
                self.drafts.append(
                    _Draft(pieces=[piece], lead_in=lead if i == 0 else None,
                           code_group_id=group, part_index=i)
                )
            return
        if self.acc is not None and self.acc.tokens(self.enc) + p.tokens <= limit:
            self.acc.pieces.append(p)
            return
        self.flush()
        self.acc = _Draft(lead_in=self.lead_in_from_previous(), pieces=[p])

    def finish(self) -> list[_Draft]:
        self.flush()
        self._merge_small_drafts()
        return self.drafts

    def _mergeable(self, d: _Draft) -> bool:
        return d.kind in PROSE_KINDS and d.code_group_id is None

    def _fits(self, target: _Draft, pieces: list[_Piece], *, after: bool) -> bool:
        merged = _Draft(
            pieces=[*target.pieces, *pieces] if after else [*pieces, *target.pieces],
            overlap=target.overlap, lead_in=target.lead_in,
        )
        return merged.tokens(self.enc) <= self.cfg.max_tokens

    def _merge_small_drafts(self) -> None:
        """A prose chunk under min_tokens joins a prose neighbor of the same section when the
        result stays within max_tokens: the previous chunk first, else the next. Callouts and
        code groups never take part, so a small chunk beside a callout stays small."""
        changed = True
        while changed:
            changed = False
            for i, d in enumerate(self.drafts):
                if not self._mergeable(d) or d.tokens(self.enc) >= self.cfg.min_tokens:
                    continue
                prev = self.drafts[i - 1] if i > 0 else None
                nxt = self.drafts[i + 1] if i + 1 < len(self.drafts) else None
                if prev is not None and self._mergeable(prev) and self._fits(prev, d.pieces, after=True):
                    prev.pieces.extend(d.pieces)
                    del self.drafts[i]
                elif nxt is not None and self._mergeable(nxt) and self._fits(nxt, d.pieces, after=False):
                    nxt.pieces[:0] = d.pieces
                    nxt.overlap = d.overlap
                    del self.drafts[i]
                else:
                    continue
                changed = True
                break


def pack_section(section: Section, cfg: ChunkConfig, enc: tiktoken.Encoding) -> list[_Draft]:
    packer = _Packer(cfg, enc)
    for block in section.blocks:
        if block.type == "chapter_title":
            continue
        p = packer.piece(block)
        if block.type == "code":
            packer.add_code(p)
        elif block.type == "callout":
            packer.add_callout(p)
        else:
            packer.add_prose(p)
    return packer.finish()


# --------------------------------------------------------------------------- headers and ids


def chapter_label(chapter: Chapter) -> str:
    if 1 <= chapter.chapter_no <= 30:
        return f"Chapter {chapter.chapter_no:02d}: {chapter.title}"
    return chapter.title


def part_label(chapter: Chapter) -> str:
    return f"{chapter.part}: {chapter.part_title}" if chapter.part_title else chapter.part


def page_label(start: int, end: int) -> str:
    return f"p. {start}" if start == end else f"p. {start}-{end}"


def chunk_header(chapter: Chapter, section_title: str, printed_start: int, printed_end: int) -> str:
    section = section_title or chapter.title
    return (
        f"{BOOK_TITLE} > {part_label(chapter)} > {chapter_label(chapter)} > {section} "
        f"({page_label(printed_start, printed_end)})"
    )


def _chapter_for(section: Section, chapters: Sequence[Chapter]) -> Chapter:
    for c in chapters:
        if c.chapter_no == section.chapter_no and c.title == section.chapter_title:
            return c
    raise ValueError(f"chapter {section.chapter_no} {section.chapter_title!r} not in the chapter table")


# --------------------------------------------------------------------------- assembly


@dataclass(frozen=True)
class ChunkStats:
    tokens: list[int]
    chunks_with_code: int
    chunks_over_max: int
    chunks_under_min: int
    composition: dict[str, int]
    parents_total: int


def chunk_blocks(
    blocks: Sequence[Block],
    chapters: Sequence[Chapter],
    cfg: ChunkConfig,
    enc: tiktoken.Encoding,
    *,
    doc_id: str,
    printed_page_offset: int = 2,
) -> tuple[list[Chunk], list[Parent], ChunkStats]:
    """Chunks and parents for a whole document. Deterministic for the same input and config."""
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    tokens: list[int] = []
    composition = {"prose_only": 0, "prose_with_code": 0, "code_only": 0, "callout": 0}
    with_code = over_max = under_min = 0
    seq = 0

    for sec_index, section in enumerate(group_sections(blocks)):
        chapter = _chapter_for(section, chapters)
        drafts = pack_section(section, cfg, enc)
        if not drafts:
            continue
        section_tokens = len(enc.encode(section.token_text))
        windowed = section_tokens > cfg.parent_max_tokens
        section_parent_id = f"{doc_id}:{section.chapter_no:02d}:s{sec_index:03d}"
        content_blocks = [b for b in section.blocks if b.type != "chapter_title"]
        if not windowed and content_blocks:
            parents.append(
                Parent(
                    parent_id=section_parent_id,
                    chapter_no=section.chapter_no,
                    section_title=section.section_title,
                    text=section.token_text,
                    page_printed_start=min(b.page_pdf for b in content_blocks) - printed_page_offset,
                    page_printed_end=max(b.page_pdf for b in content_blocks) - printed_page_offset,
                    token_count=section_tokens,
                )
            )

        section_chunk_ids = [f"{doc_id}:{section.chapter_no:02d}:{seq + i:04d}" for i in range(len(drafts))]
        for i, draft in enumerate(drafts):
            chunk_id = section_chunk_ids[i]
            parent_id = f"{chunk_id}:w" if windowed else section_parent_id
            display = draft.display_text()
            n_tokens = draft.tokens(enc)
            pdf_start, pdf_end = draft.pages()
            printed_start, printed_end = pdf_start - printed_page_offset, pdf_end - printed_page_offset
            header = chunk_header(chapter, section.section_title, printed_start, printed_end)
            kind = draft.kind
            payload: dict[str, Any] = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "part": chapter.part,
                "chapter_no": section.chapter_no,
                "chapter_title": section.chapter_title,
                "section_title": section.section_title,
                "page_pdf_start": pdf_start,
                "page_pdf_end": pdf_end,
                "page_printed_start": printed_start,
                "page_printed_end": printed_end,
                "has_code": draft.has_code,
                "code_lang": "php" if draft.has_code else "",
                "code_group_id": draft.code_group_id,
                "part_index": draft.part_index,
                "token_count": n_tokens,
                "chunk_kind": kind,
                "section_complete": len(drafts) == 1,
                "block_ids": draft.block_ids(),
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    display_text=display,
                    embedding_text=f"{header}\n{display}",
                    token_count=n_tokens,
                    payload=payload,
                )
            )
            tokens.append(n_tokens)
            composition[kind] += 1
            with_code += int(draft.has_code)
            over_max += int(n_tokens > cfg.code_max_tokens)
            under_min += int(n_tokens < cfg.min_tokens)
            if windowed:
                neighbors = drafts[max(0, i - 1) : i + 2]
                # Only the first neighbor keeps its copied overlap or lead-in; later ones would
                # repeat a paragraph that the previous neighbor already ends with.
                text = "\n\n".join(
                    d.display_text() if j == 0 else d.own_text() for j, d in enumerate(neighbors)
                )
                pages = [pg for d in neighbors for pg in d.pages()]
                parents.append(
                    Parent(
                        parent_id=parent_id,
                        chapter_no=section.chapter_no,
                        section_title=section.section_title,
                        text=text,
                        page_printed_start=min(pages) - printed_page_offset,
                        page_printed_end=max(pages) - printed_page_offset,
                        token_count=len(enc.encode(text)),
                    )
                )
        seq += len(drafts)

    stats = ChunkStats(tokens, with_code, over_max, under_min, composition, len(parents))
    return chunks, parents, stats


def chunks_to_documents(chunks: Sequence[Chunk]) -> list[Document]:
    """LangChain documents for the index step: embedding text as content, payload as metadata."""
    return [
        Document(page_content=c.embedding_text, metadata={**c.payload, "chunk_id": c.chunk_id})
        for c in chunks
    ]


# --------------------------------------------------------------------------- the stage


@dataclass(frozen=True)
class ChunkResult:
    doc_id: str
    chunks_total: int
    parents_total: int
    tokens_p50: int
    tokens_p95: int
    tokens_max: int
    chunks_with_code: int
    chunks_over_max: int
    chunks_under_min: int
    composition: dict[str, int]
    histogram: dict[str, int]
    chunk_config_hash: str
    output_path: Path
    parents_path: Path

    def to_attrs(self) -> Attrs:
        return {
            "rag.doc_id": self.doc_id,
            "rag.chunks_total": self.chunks_total,
            "rag.parents_total": self.parents_total,
            "rag.tokens_p50": self.tokens_p50,
            "rag.tokens_p95": self.tokens_p95,
            "rag.tokens_max": self.tokens_max,
            "rag.chunks_with_code": self.chunks_with_code,
            "rag.chunks_over_max": self.chunks_over_max,
            "rag.chunks_under_min": self.chunks_under_min,
            "rag.composition": json.dumps(self.composition, sort_keys=True),
            "rag.token_histogram": json.dumps(self.histogram),
            "rag.chunk_config_hash": self.chunk_config_hash,
        }


def token_histogram(tokens: Sequence[int], bin_size: int = 100) -> dict[str, int]:
    bins: dict[str, int] = {}
    for t in tokens:
        lo = (t // bin_size) * bin_size
        key = f"{lo}-{lo + bin_size - 1}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items(), key=lambda kv: int(kv[0].split("-")[0])))


@stage("ingest.chunk", kind="CHAIN")
def chunk(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/clean"),
    parsed_dir: Path = Path("data/parsed"),
    out_dir: Path = Path("data/chunks"),
    parents_dir: Path = Path("data/parents"),
) -> ChunkResult:
    """Run Stage 4 on the Stage 3 output of one document."""
    cfg = settings.chunk
    enc = tokenizer(cfg)
    blocks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Block))
    chapters = list(read_jsonl(parsed_dir / f"{doc_id}.chapters.jsonl", Chapter))
    chunks, parents, stats = chunk_blocks(
        blocks, chapters, cfg, enc, doc_id=doc_id,
        printed_page_offset=settings.parse.printed_page_offset,
    )
    output_path = out_dir / f"{doc_id}.jsonl"
    parents_path = parents_dir / f"{doc_id}.jsonl"
    write_jsonl(output_path, chunks)
    write_jsonl(parents_path, parents)
    ordered = sorted(stats.tokens)
    return ChunkResult(
        doc_id=doc_id,
        chunks_total=len(chunks),
        parents_total=len(parents),
        tokens_p50=int(statistics.median(ordered)) if ordered else 0,
        tokens_p95=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0,
        tokens_max=ordered[-1] if ordered else 0,
        chunks_with_code=stats.chunks_with_code,
        chunks_over_max=stats.chunks_over_max,
        chunks_under_min=stats.chunks_under_min,
        composition=stats.composition,
        histogram=token_histogram(stats.tokens),
        chunk_config_hash=settings.chunk_config_hash,
        output_path=output_path,
        parents_path=parents_path,
    )
