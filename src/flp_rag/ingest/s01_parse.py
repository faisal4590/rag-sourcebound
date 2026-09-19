# Stage 1 - Load and parse. Spec Section 5, Stage 1.
"""Stage 1: turn PDF pages into words with a font name, a size, and a position.

Input: the PDF path. Output: `data/parsed/{doc_id}.jsonl` (one `ParsedPage` per line) and
`data/parsed/{doc_id}.chapters.jsonl` (one `Chapter` per line, the chapter table).

Rules (spec Section 5, Stage 1):
1. `doc_id = sha256(file bytes)[:16]`.
2. If a manifest on disk holds this `doc_id` with the same `chunk_config_hash` and a passed
   verification, stop and report "already indexed". `force=True` skips the check.
3. Build the chapter table from the PDF bookmarks. Chapter numbers come from bookmark order,
   never from the printed `CHAPTER NN` label.
4. Extract words with font name and size. Strip the subset prefix from font names.
5. Delete running-header words: `top < parse.header_max_top_pt` or size equal to
   `parse.header_font_size_pt`.
6. Skip `parse.skip_pages` (cover, table of contents, part title pages).
7. Record pages with zero deleted header words. Warn, never fail.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
import structlog
from pdfminer.pdfdocument import PDFDestinationNotFound
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import resolve1

from flp_rag.contracts import Attrs, Chapter, ParsedPage, Word, write_jsonl
from flp_rag.settings import ParseConfig, Settings
from flp_rag.tracing import current_span, stage

log = structlog.get_logger()

BookmarkRow = tuple[int, str, int]  # (level, title, pdf page)

FRONT_MATTER = "Front matter"
_FONT_PREFIX = re.compile(r"^[A-Za-z0-9]+\+")


@dataclass(frozen=True)
class ParseResult:
    """What Stage 1 reports. `already_indexed` names the index version when the run was skipped."""

    doc_id: str
    pages_total: int
    pages_parsed: int
    words_total: int
    header_words_deleted: int
    pages_without_header: list[int]
    chapters_found: int
    output_path: Path
    chapters_path: Path
    already_indexed: str | None = None

    def to_attrs(self) -> Attrs:
        attrs: Attrs = {
            "rag.doc_id": self.doc_id,
            "rag.pages_total": self.pages_total,
            "rag.pages_parsed": self.pages_parsed,
            "rag.words_total": self.words_total,
            "rag.header_words_deleted": self.header_words_deleted,
            "rag.pages_without_header": json.dumps(self.pages_without_header),
            "rag.chapters_found": self.chapters_found,
        }
        if self.already_indexed is not None:
            attrs["rag.already_indexed"] = self.already_indexed
        return attrs


# --------------------------------------------------------------------------- pure functions


def compute_doc_id(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def strip_font_prefix(fontname: str) -> str:
    """`MBPBAO+Inter-Regular` -> `Inter-Regular`. The tag before `+` marks an embedded subset."""
    return _FONT_PREFIX.sub("", fontname, count=1)


def is_header_word(word: dict[str, Any], cfg: ParseConfig) -> bool:
    """Both spec rules: above the header line, or set in the header font size."""
    return (
        word["top"] < cfg.header_max_top_pt
        or abs(float(word["size"]) - cfg.header_font_size_pt) < 0.01
    )


def split_header_words(
    raw_words: Sequence[dict[str, Any]], cfg: ParseConfig, page_pdf: int
) -> tuple[list[Word], int]:
    """Return the body words as `Word` records and the number of header words deleted."""
    kept: list[Word] = []
    deleted = 0
    for w in raw_words:
        if is_header_word(w, cfg):
            deleted += 1
            continue
        kept.append(
            Word(
                text=w["text"],
                font=strip_font_prefix(w["fontname"]),
                size=round(float(w["size"]), 2),
                x0=round(float(w["x0"]), 2),
                x1=round(float(w["x1"]), 2),
                top=round(float(w["top"]), 2),
                bottom=round(float(w["bottom"]), 2),
                page_pdf=page_pdf,
            )
        )
    return kept, deleted


def build_chapter_table(
    rows: Sequence[BookmarkRow], last_page: int, skip_pages: Sequence[int]
) -> list[Chapter]:
    """Chapter table from bookmark rows.

    A level-1 row whose page is a skipped page is a part title. Rows before the first part are
    front matter with chapter number 0. Rows after it are numbered 1, 2, ... in order, which
    gives the 30 chapters and In Closing as 31. End pages stop before the next chapter and never
    land on a skipped page.
    """
    skipped = set(skip_pages)
    parts = [(title, page) for level, title, page in rows if level == 1 and page in skipped]
    if not parts:
        raise ValueError("bookmarks hold no part title rows; cannot assign chapters to parts")

    chapters: list[tuple[str, str, str, int]] = []  # title, part, part_title, start
    part_label, part_title = FRONT_MATTER, ""
    part_index = 0
    for level, title, page in rows:
        if level == 1 and page in skipped:
            part_index += 1
            part_label, part_title = f"Part {part_index}", title
            continue
        chapters.append((title, part_label, part_title, page))

    table: list[Chapter] = []
    number = 0
    for order, (title, part, ptitle, start) in enumerate(chapters):
        if part != FRONT_MATTER:
            number += 1
        next_start = chapters[order + 1][3] if order + 1 < len(chapters) else last_page + 1
        end = next_start - 1
        while end in skipped and end > start:
            end -= 1
        if end < start:
            raise ValueError(
                f"chapter {title!r} starts on page {start} but the next chapter starts on "
                f"page {next_start}; bookmark pages must increase"
            )
        table.append(
            Chapter(
                order=order,
                chapter_no=0 if part == FRONT_MATTER else number,
                title=title,
                part=part,
                part_title=ptitle,
                page_pdf_start=start,
                page_pdf_end=end,
            )
        )
    return table


def find_existing_index(index_dir: Path, doc_id: str, chunk_config_hash: str) -> str | None:
    """Index version of a passed ingestion with the same document and chunking, or None."""
    if not index_dir.is_dir():
        return None
    for manifest_path in sorted(index_dir.glob("*/manifest.json")):
        try:
            m = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("manifest unreadable", path=str(manifest_path), error=str(exc))
            continue
        if not isinstance(m, dict):
            log.warning("manifest is not an object", path=str(manifest_path))
            continue
        if (
            m.get("doc_id") == doc_id
            and m.get("chunk_config_hash") == chunk_config_hash
            and (m.get("verify") or {}).get("passed") is True
        ):
            return str(m.get("index_version") or manifest_path.parent.name)
    return None


# --------------------------------------------------------------------------- PDF access


def read_bookmarks(pdf: pdfplumber.PDF) -> list[BookmarkRow]:
    """Bookmark rows as (level, title, pdf page). Destinations are named; resolve via the name
    tree and map the page object id to its 1-based index."""
    doc = pdf.doc
    page_ids = [p.pageid for p in PDFPage.create_pages(doc)]
    rows: list[BookmarkRow] = []
    for level, title, dest, action, _se in doc.get_outlines():
        try:
            target = resolve1(dest) if dest is not None else _action_destination(action)
            if isinstance(target, bytes | str):
                target = resolve1(doc.get_dest(target))
            if isinstance(target, dict):
                target = target.get("D")
        except (PDFDestinationNotFound, AttributeError, TypeError, KeyError):
            target = None
        if not isinstance(target, list) or not target or not hasattr(target[0], "objid"):
            raise ValueError(f"bookmark {title!r} has no resolvable page destination")
        rows.append((level, str(title), page_ids.index(target[0].objid) + 1))
    return rows


def _action_destination(action: Any) -> Any:
    resolved = resolve1(action)
    if not isinstance(resolved, dict):
        raise TypeError("bookmark action is not a dictionary")
    return resolve1(resolved.get("D"))


def _extract_words(page: pdfplumber.page.Page) -> list[dict[str, Any]]:
    return page.extract_words(extra_attrs=["fontname", "size"])


# --------------------------------------------------------------------------- the stage


@stage("ingest.parse", kind="CHAIN")
def parse(
    pdf_path: Path,
    settings: Settings,
    *,
    out_dir: Path = Path("data/parsed"),
    index_dir: Path = Path("data/index"),
    force: bool = False,
) -> ParseResult:
    """Run Stage 1 on one PDF. Writes two JSONL files unless the document is already indexed."""
    cfg = settings.parse
    doc_id = compute_doc_id(pdf_path)
    output_path = out_dir / f"{doc_id}.jsonl"
    chapters_path = out_dir / f"{doc_id}.chapters.jsonl"

    if not force:
        existing = find_existing_index(index_dir, doc_id, settings.chunk_config_hash)
        if existing is not None:
            log.info("already indexed", doc_id=doc_id, index_version=existing)
            return ParseResult(
                doc_id=doc_id, pages_total=0, pages_parsed=0, words_total=0,
                header_words_deleted=0, pages_without_header=[], chapters_found=0,
                output_path=output_path, chapters_path=chapters_path, already_indexed=existing,
            )

    skipped = set(cfg.skip_pages)
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages_total = len(pdf.pages)
        chapters = build_chapter_table(read_bookmarks(pdf), pages_total, cfg.skip_pages)

        parsed: list[ParsedPage] = []
        pages_without_header: list[int] = []
        for page in pdf.pages:
            page_pdf = page.page_number
            if page_pdf in skipped:
                continue
            words, deleted = split_header_words(_extract_words(page), cfg, page_pdf)
            if deleted == 0:
                pages_without_header.append(page_pdf)
                log.warning("page has no header words", page_pdf=page_pdf, words=len(words))
            parsed.append(
                ParsedPage(
                    doc_id=doc_id,
                    page_pdf=page_pdf,
                    page_printed=page_pdf - cfg.printed_page_offset,
                    header_words_deleted=deleted,
                    words=words,
                )
            )

    write_jsonl(output_path, parsed)
    write_jsonl(chapters_path, chapters)
    current_span().set_attribute("rag.output_path", str(output_path))
    return ParseResult(
        doc_id=doc_id,
        pages_total=pages_total,
        pages_parsed=len(parsed),
        words_total=sum(len(p.words) for p in parsed),
        header_words_deleted=sum(p.header_words_deleted for p in parsed),
        pages_without_header=pages_without_header,
        chapters_found=len(chapters),
        output_path=output_path,
        chapters_path=chapters_path,
    )
