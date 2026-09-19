# Stage 2 - Structure reconstruction. Spec Section 5, Stage 2.
"""Stage 2: group words into typed blocks and attach the chapter hierarchy to each block.

Input: `data/parsed/{doc_id}.jsonl` and `data/parsed/{doc_id}.chapters.jsonl`.
Output: `data/blocks/{doc_id}.jsonl`, one `Block` per line.

Line types come from the dominant font of a line and from the shaded rectangle under it:
Staatliches 34 is a chapter title, Staatliches 14 or 16 is a heading outside a callout box and a
callout title inside one, any mono font is code, Inter 9 `CHAPTER NN` is a label that is dropped,
Inter 10 inside a callout box is callout body, and everything else is a paragraph.
"""

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import structlog

from flp_rag.contracts import (
    Attrs,
    Block,
    Chapter,
    ParsedPage,
    Rect,
    Word,
    read_jsonl,
    write_jsonl,
)
from flp_rag.ingest.s01_parse import word_in_rect
from flp_rag.settings import Settings, StructureConfig
from flp_rag.tracing import stage

log = structlog.get_logger()

LineType = Literal["chapter_title", "heading", "callout", "code", "paragraph", "label"]
PROSE_TYPES = ("paragraph", "heading", "chapter_title")


@dataclass(frozen=True)
class Line:
    """Words on one page within the line tolerance of each other, ordered by x0."""

    page_pdf: int
    top: float
    bottom: float
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def dominant(self) -> tuple[str, float]:
        """Font and size that cover the most characters on the line."""
        weight: dict[tuple[str, float], int] = {}
        for w in self.words:
            key = (w.font, w.size)
            weight[key] = weight.get(key, 0) + len(w.text)
        return max(weight, key=lambda k: weight[k])


@dataclass(frozen=True)
class PageInput:
    page_pdf: int
    words: Sequence[Word]
    rects: Sequence[Rect]


@dataclass(frozen=True)
class RawBlock:
    """A block before the chapter hierarchy is attached."""

    type: str
    text: str
    page_pdf: int
    top: float
    bottom: float
    in_rect: bool
    starts_with_title: bool


@dataclass(frozen=True)
class StructureResult:
    doc_id: str
    blocks_total: int
    blocks_by_type: dict[str, int]
    chapters_found: int
    code_blocks_merged_across_pages: int
    callouts_merged_across_pages: int
    labels_dropped: int
    output_path: Path

    def to_attrs(self) -> Attrs:
        import json

        return {
            "rag.doc_id": self.doc_id,
            "rag.blocks_total": self.blocks_total,
            "rag.blocks_by_type": json.dumps(self.blocks_by_type, sort_keys=True),
            "rag.chapters_found": self.chapters_found,
            "rag.code_blocks_merged_across_pages": self.code_blocks_merged_across_pages,
            "rag.callouts_merged_across_pages": self.callouts_merged_across_pages,
            "rag.labels_dropped": self.labels_dropped,
        }


# --------------------------------------------------------------------------- lines


def group_lines(words: Iterable[Word], tolerance_pt: float) -> list[Line]:
    """Cluster words into lines by `top`. A new line starts when a word sits more than the
    tolerance below the first word of the current line."""
    ordered = sorted(words, key=lambda w: (w.top, w.x0))
    lines: list[Line] = []
    current: list[Word] = []
    for w in ordered:
        if current and w.top - current[0].top > tolerance_pt:
            lines.append(_make_line(current))
            current = []
        current.append(w)
    if current:
        lines.append(_make_line(current))
    return lines


def _make_line(words: list[Word]) -> Line:
    by_x = tuple(sorted(words, key=lambda w: w.x0))
    return Line(
        page_pdf=by_x[0].page_pdf,
        top=min(w.top for w in by_x),
        bottom=max(w.bottom for w in by_x),
        words=by_x,
    )


def is_mono(font: str, cfg: StructureConfig) -> bool:
    return any(m in font for m in cfg.mono_fonts)


def _in_callout(line: Line, rects: Sequence[Rect]) -> bool:
    callouts = [r for r in rects if r.kind == "callout"]
    return any(word_in_rect(w, r) for w in line.words for r in callouts)


def _is_heading_font(font: str, cfg: StructureConfig) -> bool:
    return cfg.heading_font in font


def line_type(line: Line, rects: Sequence[Rect], cfg: StructureConfig) -> LineType:
    font, size = line.dominant()
    if is_mono(font, cfg):
        return "code"
    if _is_heading_font(font, cfg):
        if abs(size - cfg.chapter_title_size_pt) < 0.5:
            return "chapter_title"
        if any(abs(size - s) < 0.5 for s in cfg.heading_sizes_pt):
            return "callout" if _in_callout(line, rects) else "heading"
    if abs(size - cfg.label_size_pt) < 0.5 and re.fullmatch(cfg.label_pattern, line.text):
        return "label"
    if _in_callout(line, rects):
        return "callout"
    return "paragraph"


# --------------------------------------------------------------------------- code layout


def code_metrics(words: Iterable[Word], cfg: StructureConfig) -> tuple[float, float]:
    """Left margin and character width of the mono font on a page."""
    mono = [w for w in words if is_mono(w.font, cfg)]
    if not mono:
        return 0.0, cfg.mono_char_width_ratio * cfg.default_mono_size_pt
    margin = min(w.x0 for w in mono)
    widths = [(w.x1 - w.x0) / len(w.text) for w in mono if len(w.text) >= 3]
    if widths:
        return margin, statistics.median(widths)
    size = statistics.median(w.size for w in mono)
    return margin, cfg.mono_char_width_ratio * size


def code_line_text(line: Line, margin: float, char_width: float, min_gap_chars: float) -> str:
    """One code line with its indentation and inner spacing rebuilt from x positions.

    Words that touch (a font change split them, as in `#[MyAttribute]` or `*/;`) are joined with
    no space: a gap below `min_gap_chars` character widths is zero spaces."""
    first = line.words[0]
    cols = max(0, round((first.x0 - margin) / char_width))
    out = " " * cols + first.text
    for prev, nxt in pairwise(line.words):
        gap_pt = nxt.x0 - prev.x1
        gap = round(gap_pt / char_width) if gap_pt >= min_gap_chars * char_width else 0
        out += " " * gap + nxt.text
    return out.rstrip()


# --------------------------------------------------------------------------- blocks on a page


def join_prose(lines: Sequence[str]) -> str:
    """Join wrapped prose lines. A line that ends with a hyphen joins the next line without a
    space when that line starts with a lowercase letter."""
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        if len(out) > 1 and out.endswith("-") and nxt[:1].islower():
            out = out[:-1] + nxt
        else:
            out = out + " " + nxt
    return out


def _line_pitch(lines: Sequence[Line]) -> float:
    """The tightest positive distance between consecutive line tops on the page."""
    gaps = [b.top - a.top for a, b in pairwise(lines) if b.top - a.top > 0.5]
    return min(gaps) if gaps else (lines[0].height if lines else 0.0)


def _rect_for(line: Line, rects: Sequence[Rect], kinds: tuple[str, ...]) -> Rect | None:
    for r in rects:
        if r.kind in kinds and any(word_in_rect(w, r) for w in line.words):
            return r
    return None


class _Builder:
    """Accumulates the lines of one block and emits it."""

    def __init__(self, cfg: StructureConfig, margin: float, char_width: float, pitch: float):
        self.cfg, self.margin, self.cw, self.pitch = cfg, margin, char_width, pitch
        self.type: str | None = None
        self.lines: list[Line] = []
        self.paragraphs: list[list[str]] = []  # callout body paragraphs
        self.title: str | None = None
        self.in_rect = False
        self.blocks: list[RawBlock] = []

    def start(self, kind: str, line: Line, *, title: bool = False, in_rect: bool = False) -> None:
        self.flush()
        self.type, self.lines, self.in_rect = kind, [line], in_rect
        self.paragraphs, self.title = ([] if title else [[line.text]]), (line.text if title else None)

    def extend(self, line: Line, *, new_paragraph: bool = False) -> None:
        self.lines.append(line)
        if self.type == "callout":
            if new_paragraph or not self.paragraphs:
                self.paragraphs.append([line.text])
            else:
                self.paragraphs[-1].append(line.text)

    def gap_breaks(self, line: Line) -> bool:
        prev = self.lines[-1]
        threshold = self.cfg.paragraph_gap_factor * max(self.pitch, prev.height)
        return line.top - prev.top > threshold

    def flush(self) -> None:
        if self.type is None or not self.lines:
            return
        text = self._text()
        self.blocks.append(
            RawBlock(
                type=self.type,
                text=text,
                page_pdf=self.lines[0].page_pdf,
                top=self.lines[0].top,
                bottom=self.lines[-1].bottom,
                in_rect=self.in_rect,
                starts_with_title=self.title is not None,
            )
        )
        self.type, self.lines, self.paragraphs, self.title = None, [], [], None

    def _text(self) -> str:
        if self.type == "code":
            out: list[str] = []
            for prev, cur in zip([None, *self.lines], self.lines, strict=False):
                if prev is not None and self.pitch > 0:
                    blanks = max(0, round((cur.top - prev.top) / self.pitch) - 1)
                    out.extend([""] * blanks)
                out.append(code_line_text(cur, self.margin, self.cw, self.cfg.code_min_gap_chars))
            return "\n".join(out)
        if self.type == "callout":
            body = "\n\n".join(join_prose(p) for p in self.paragraphs if p)
            return f"{self.title}\n{body}" if self.title is not None else body
        return join_prose([ln.text for ln in self.lines])


def build_page_blocks(page: PageInput, cfg: StructureConfig) -> list[RawBlock]:
    """Typed blocks for one page, in reading order. Labels are dropped."""
    lines = group_lines(page.words, cfg.line_tolerance_pt)
    if not lines:
        return []
    typed = [(ln, line_type(ln, page.rects, cfg)) for ln in lines]
    margin, cw = code_metrics([w for ln, t in typed if t == "code" for w in ln.words], cfg)
    b = _Builder(cfg, margin, cw, _line_pitch(lines))

    for line, kind in typed:
        if kind == "label":
            continue
        if kind == "code":
            rect = _rect_for(line, page.rects, ("code", "highlight"))
            if b.type == "code" and _code_continues(b, line, rect, page.rects, cfg):
                b.extend(line)
            else:
                b.start("code", line, in_rect=rect is not None)
            continue
        if kind == "callout":
            font, _ = line.dominant()
            is_title = _is_heading_font(font, cfg)
            if is_title or b.type != "callout":
                b.start("callout", line, title=is_title, in_rect=True)
            else:
                b.extend(line, new_paragraph=b.gap_breaks(line))
            continue
        # paragraph, heading, chapter_title: same type and a small gap continue the block
        if b.type == kind and not b.gap_breaks(line):
            b.extend(line)
        else:
            b.start(kind, line)
    b.flush()
    return b.blocks


def _code_continues(
    b: _Builder, line: Line, rect: Rect | None, rects: Sequence[Rect], cfg: StructureConfig
) -> bool:
    """Code stays one block while the chain of shaded rectangles from the previous line to this
    one is unbroken. Empty rectangles in between are blank code lines. Without rectangles on
    either side, code lines always continue."""
    prev_rect = _rect_for(b.lines[-1], rects, ("code", "highlight"))
    if prev_rect is None or rect is None:
        return True
    chain = sorted((r for r in rects if r.kind in ("code", "highlight")), key=lambda r: r.top)
    last_bottom = prev_rect.bottom
    for r in chain:
        if r.top < prev_rect.top + cfg.line_tolerance_pt or r is prev_rect:
            continue
        if r.top > last_bottom + cfg.line_tolerance_pt:
            return False
        last_bottom = max(last_bottom, r.bottom)
        if r is rect or (abs(r.top - rect.top) < 0.01 and abs(r.bottom - rect.bottom) < 0.01):
            return True
    return False


# --------------------------------------------------------------------------- across pages


def merge_across_pages(blocks: Sequence[RawBlock]) -> tuple[list[RawBlock], int, int]:
    """Merge a code block that ends a page with a code block that starts the very next page, and
    a callout with a title-less callout continuation. Returns (blocks, n_code, n_callout)."""
    out: list[RawBlock] = []
    n_code = n_callout = 0
    for blk in blocks:
        prev = out[-1] if out else None
        # Only across one page boundary. A skipped part title page or a blank filler page
        # between the two means they are not one block.
        if prev is not None and blk.page_pdf == prev.page_pdf + 1:
            if prev.type == "code" and blk.type == "code":
                out[-1] = _merged(prev, blk, "\n")
                n_code += 1
                continue
            if prev.type == "callout" and blk.type == "callout" and not blk.starts_with_title:
                out[-1] = _merged(prev, blk, "\n\n")
                n_callout += 1
                continue
        out.append(blk)
    return out, n_code, n_callout


def _merged(a: RawBlock, b: RawBlock, sep: str) -> RawBlock:
    return RawBlock(
        type=a.type,
        text=a.text + sep + b.text,
        page_pdf=a.page_pdf,
        top=a.top,
        bottom=b.bottom,
        in_rect=a.in_rect or b.in_rect,
        starts_with_title=a.starts_with_title,
    )


# --------------------------------------------------------------------------- hierarchy


def attach_hierarchy(
    blocks: Sequence[RawBlock],
    chapters: Sequence[Chapter],
    *,
    doc_id: str,
    printed_page_offset: int,
) -> list[Block]:
    """Turn raw blocks into `Block` records with part, chapter, section, ids, and printed pages.
    Chapter titles take the bookmark title. Labels are dropped."""
    section_by_chapter: dict[int, str] = {}
    per_page_counter: dict[int, int] = {}
    out: list[Block] = []
    for raw in blocks:
        if raw.type == "label":
            continue
        chapter = _chapter_for(raw.page_pdf, chapters)
        key = chapter.order
        if raw.type == "heading":
            section_by_chapter[key] = raw.text
        text = chapter.title if raw.type == "chapter_title" else raw.text
        n = per_page_counter.get(raw.page_pdf, 0)
        per_page_counter[raw.page_pdf] = n + 1
        out.append(
            Block(
                block_id=f"{doc_id}:{raw.page_pdf:03d}:{n:03d}",
                type=raw.type,  # type: ignore[arg-type]
                text=text,
                part=chapter.part,
                chapter_no=chapter.chapter_no,
                chapter_title=chapter.title,
                section_title=section_by_chapter.get(key, ""),
                page_pdf=raw.page_pdf,
                page_printed=raw.page_pdf - printed_page_offset,
                block_index=len(out),
                in_rect=raw.in_rect,
            )
        )
    return out


def _chapter_for(page_pdf: int, chapters: Sequence[Chapter]) -> Chapter:
    for c in chapters:
        if c.page_pdf_start <= page_pdf <= c.page_pdf_end:
            return c
    raise ValueError(f"PDF page {page_pdf} lies outside every chapter in the chapter table")


# --------------------------------------------------------------------------- the stage


@stage("ingest.structure", kind="CHAIN")
def structure(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/parsed"),
    out_dir: Path = Path("data/blocks"),
) -> StructureResult:
    """Run Stage 2 on the Stage 1 output of one document."""
    cfg = settings.structure
    pages = sorted(read_jsonl(in_dir / f"{doc_id}.jsonl", ParsedPage), key=lambda p: p.page_pdf)
    chapters = list(read_jsonl(in_dir / f"{doc_id}.chapters.jsonl", Chapter))

    raw: list[RawBlock] = []
    labels = 0
    for page in pages:
        lines = group_lines(page.words, cfg.line_tolerance_pt)
        labels += sum(1 for ln in lines if line_type(ln, page.rects, cfg) == "label")
        raw.extend(build_page_blocks(PageInput(page.page_pdf, page.words, page.rects), cfg))

    merged, n_code, n_callout = merge_across_pages(raw)
    blocks = attach_hierarchy(
        merged, chapters, doc_id=doc_id, printed_page_offset=settings.parse.printed_page_offset
    )

    output_path = out_dir / f"{doc_id}.jsonl"
    write_jsonl(output_path, blocks)
    by_type: dict[str, int] = {}
    for b in blocks:
        by_type[b.type] = by_type.get(b.type, 0) + 1
    return StructureResult(
        doc_id=doc_id,
        blocks_total=len(blocks),
        blocks_by_type=by_type,
        chapters_found=len({(b.chapter_no, b.chapter_title) for b in blocks}),
        code_blocks_merged_across_pages=n_code,
        callouts_merged_across_pages=n_callout,
        labels_dropped=labels,
        output_path=output_path,
    )
