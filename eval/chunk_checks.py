# Level 1 structural gate for the chunk set. Issue #62. Exit code 1 when any check fails.
"""Structural checks on `data/chunks` against `data/clean`, issue #62 Level 1.

Run: `uv run python eval/chunk_checks.py [--doc-id ID] [--report PATH]`

Checks, each PASS or FAIL:
1. size: every chunk has between `chunk.min_tokens` and `chunk.code_max_tokens` tokens.
2. header_noise: no whole line of `display_text` matches a running header that Stage 1
   deleted (patterns built from the header words of every page, digits generalized), and no
   word in `data/parsed` sits in the header font size or above the header line.
3. glyph_bug: none of the ligature residues `=_construct`, `$this=>`, `=[` appear.
4. sections: `payload.section_title` is set, every heading block inside the chunk is listed in
   `payload.section_titles`, and no chunk spans two chapters.
5. code_integrity: in chunks without a `code_group_id`, `{` and `}` balance. Mismatches are
   listed for review; some book examples are partial on purpose, so this check never fails
   on its own, it reports.
6. coverage: tokens of covered blocks / tokens of all blocks is at least the target. A block is
   covered when its normalized text (lowercase, whitespace collapsed) is a substring of a
   normalized chunk `display_text`. Chapter titles are not chunk text and do not count.
7. chapters: every chapter in the chapter table has at least one chunk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pdfplumber
import tiktoken

from flp_rag.contracts import Block, Chapter, Chunk, ParsedPage, read_jsonl
from flp_rag.ingest.s01_parse import is_header_word
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.settings import ChunkConfig, ParseConfig, Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
GLYPH_PATTERNS = ("=_construct", "$this=>", "=[")  # pdfplumber ligature residues, issue #62


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Lowercase with whitespace runs collapsed. Same definition as the Stage 8 coverage check."""
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------- checks


def check_sizes(chunks: Sequence[Chunk], cfg: ChunkConfig, enc: tiktoken.Encoding) -> Check:
    counts = {c.chunk_id: len(enc.encode(c.display_text)) for c in chunks}
    small = {cid: n for cid, n in counts.items() if n < cfg.min_tokens}
    big = {cid: n for cid, n in counts.items() if n > cfg.code_max_tokens}
    return Check(
        "size", not small and not big,
        f"{len(small)} under {cfg.min_tokens}, {len(big)} over {cfg.code_max_tokens} "
        f"(min {min(counts.values())}, max {max(counts.values())})",
        {"under_min": small, "over_max": big},
    )


def header_patterns(pdf_path: Path, cfg: ParseConfig) -> list[re.Pattern[str]]:
    """One regex per distinct running header line that Stage 1 deletes, digits generalized:
    `\\d+ front line php` on even pages, `chapter \\d+ - <title> \\d+` on odd pages."""
    seen: dict[str, re.Pattern[str]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            words = [w for w in page.extract_words(extra_attrs=["size"]) if is_header_word(w, cfg)]
            if not words:
                continue
            line = normalize_text(" ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"])))
            pattern = re.sub(r"\d+", r"\\d+", re.escape(line).replace(r"\ ", " "))
            seen.setdefault(pattern, re.compile(pattern))
    return list(seen.values())


def check_header_noise(
    chunks: Sequence[Chunk], patterns: Sequence[re.Pattern[str]], pages: Sequence[ParsedPage], cfg: ParseConfig
) -> Check:
    """A whole line of a chunk that reads like a deleted running header is noise. So is any word
    left in `data/parsed` that Stage 1 should have deleted (header font size or above the line)."""
    lines: dict[str, list[str]] = {}
    for c in chunks:
        bad = [ln for ln in c.display_text.splitlines() if any(p.fullmatch(normalize_text(ln)) for p in patterns)]
        if bad:
            lines[c.chunk_id] = bad
    leftovers = [
        {"page_pdf": pg.page_pdf, "text": w.text, "size": w.size, "top": w.top}
        for pg in pages for w in pg.words
        if is_header_word({"top": w.top, "size": w.size}, cfg)
    ]
    return Check(
        "header_noise", not lines and not leftovers and bool(patterns),
        f"{len(patterns)} header patterns, {len(lines)} chunks with a header line, "
        f"{len(leftovers)} header words left in data/parsed",
        {"patterns": [p.pattern for p in patterns], "chunks": lines, "parsed_leftovers": leftovers[:50]},
    )


def check_glyph_bug(chunks: Sequence[Chunk], patterns: Sequence[str] = GLYPH_PATTERNS) -> Check:
    counts = {p: sum(c.display_text.count(p) for c in chunks) for p in patterns}
    where = {p: [c.chunk_id for c in chunks if p in c.display_text] for p in patterns}
    return Check("glyph_bug", not any(counts.values()), f"counts {counts}", {"chunks": where})


def check_sections(chunks: Sequence[Chunk], blocks: Sequence[Block]) -> Check:
    by_id = {b.block_id: b for b in blocks}
    unset: list[str] = []
    missing_field: list[str] = []
    headings_not_listed: dict[str, list[str]] = {}
    two_chapters: list[str] = []
    for c in chunks:
        p = c.payload
        if not p.get("section_title"):
            unset.append(c.chunk_id)
        titles = p.get("section_titles")
        if titles is None:
            missing_field.append(c.chunk_id)
        own = [by_id[b] for b in p.get("block_ids", []) if b in by_id]
        headings = [b.text for b in own if b.type == "heading"]
        unlisted = [h for h in headings if h not in (titles or [])]
        if unlisted:
            headings_not_listed[c.chunk_id] = unlisted
        if {b.chapter_no for b in own} - {int(p["chapter_no"])}:
            two_chapters.append(c.chunk_id)
    passed = not (unset or missing_field or headings_not_listed or two_chapters)
    return Check(
        "sections", passed,
        f"{len(unset)} without section_title, {len(missing_field)} without section_titles, "
        f"{len(headings_not_listed)} with an unlisted heading, {len(two_chapters)} spanning two chapters",
        {"section_title_unset": unset, "section_titles_missing": missing_field,
         "headings_not_listed": headings_not_listed, "two_chapters": two_chapters},
    )


def check_code_integrity(chunks: Sequence[Chunk]) -> Check:
    """Reports brace mismatches for review. It does not fail: the book shows partial classes."""
    mismatches: list[dict[str, Any]] = []
    for c in chunks:
        if c.payload.get("code_group_id") is not None or not c.payload.get("has_code"):
            continue
        opens, closes = c.display_text.count("{"), c.display_text.count("}")
        if opens != closes:
            mismatches.append({"chunk_id": c.chunk_id, "open": opens, "close": closes,
                               "section": c.payload.get("section_title"),
                               "pages": [c.payload.get("page_printed_start"), c.payload.get("page_printed_end")]})
    return Check("code_integrity", True, f"{len(mismatches)} chunks with unbalanced braces (review list)",
                 {"mismatches": mismatches})


def check_coverage(chunks: Sequence[Chunk], blocks: Sequence[Block], enc: tiktoken.Encoding, target: float) -> Check:
    texts = [normalize_text(c.display_text) for c in chunks]
    by_chapter: dict[int, list[str]] = {}
    for c, t in zip(chunks, texts, strict=True):
        by_chapter.setdefault(int(c.payload["chapter_no"]), []).append(t)
    total = covered = 0
    per_chapter: dict[int, list[int]] = {}
    uncovered: list[dict[str, Any]] = []
    for b in blocks:
        if b.type == "chapter_title":
            continue
        n = len(enc.encode(b.text))
        total += n
        hit = any(normalize_text(b.text) in t for t in by_chapter.get(b.chapter_no, []))
        tot_cov = per_chapter.setdefault(b.chapter_no, [0, 0])
        tot_cov[0] += n
        if hit:
            covered += n
            tot_cov[1] += n
        else:
            uncovered.append({"block_id": b.block_id, "type": b.type, "tokens": n,
                              "page_pdf": b.page_pdf, "text": b.text[:120]})
    ratio = covered / total if total else 1.0
    chapters = {ch: round(cov / tot, 4) if tot else 1.0 for ch, (tot, cov) in sorted(per_chapter.items())}
    low = {ch: r for ch, r in chapters.items() if r < target}
    return Check(
        "coverage", ratio >= target and not low,
        f"{ratio:.4f} of {total} clean tokens covered, {len(uncovered)} blocks uncovered, "
        f"{len(low)} chapters under {target}",
        {"coverage": round(ratio, 4), "tokens_total": total, "tokens_covered": covered,
         "per_chapter": chapters, "chapters_under_target": low, "uncovered_blocks": uncovered},
    )


def check_chapters(chunks: Sequence[Chunk], chapters: Sequence[Chapter]) -> Check:
    have = {int(c.payload["chapter_no"]) for c in chunks}
    missing = [f"{ch.chapter_no:02d} {ch.title}" for ch in chapters if ch.chapter_no not in have]
    return Check("chapters", not missing, f"{len(chapters) - len(missing)} of {len(chapters)} chapters have a chunk",
                 {"missing": missing})


def run_checks(
    chunks: Sequence[Chunk],
    blocks: Sequence[Block],
    chapters: Sequence[Chapter],
    pages: Sequence[ParsedPage],
    settings: Settings,
    *,
    pdf_path: Path = PDF,
) -> list[Check]:
    enc = tokenizer(settings.chunk)
    patterns = header_patterns(pdf_path, settings.parse) if pdf_path.exists() else []
    return [
        check_sizes(chunks, settings.chunk, enc),
        check_header_noise(chunks, patterns, pages, settings.parse),
        check_glyph_bug(chunks),
        check_sections(chunks, blocks),
        check_code_integrity(chunks),
        check_coverage(chunks, blocks, enc, settings.verify.min_chapter_coverage),
        check_chapters(chunks, chapters),
    ]


# --------------------------------------------------------------------------- cli


def _discover_doc_id(data_dir: Path) -> str:
    files = sorted((data_dir / "chunks").glob("*.jsonl"))
    if len(files) != 1:
        raise SystemExit(f"expected one file in {data_dir / 'chunks'}, found {len(files)}; pass --doc-id")
    return files[0].stem


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--report", type=Path, default=None, help="JSON report path (default eval/runs/)")
    ap.add_argument("--pdf", type=Path, default=PDF, help="the book, for the running header patterns")
    args = ap.parse_args(argv)
    settings = load_settings()
    doc_id = args.doc_id or _discover_doc_id(args.data_dir)
    chunks = list(read_jsonl(args.data_dir / "chunks" / f"{doc_id}.jsonl", Chunk))
    blocks = list(read_jsonl(args.data_dir / "clean" / f"{doc_id}.jsonl", Block))
    chapters = list(read_jsonl(args.data_dir / "parsed" / f"{doc_id}.chapters.jsonl", Chapter))
    pages = list(read_jsonl(args.data_dir / "parsed" / f"{doc_id}.jsonl", ParsedPage))
    checks = run_checks(chunks, blocks, chapters, pages, settings, pdf_path=args.pdf)

    print(f"chunk_checks: {len(chunks)} chunks, {len(blocks)} clean blocks, chunk_config_hash {settings.chunk_config_hash}")
    for c in checks:
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:15s} {c.summary}")
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    report = args.report or (ROOT / "eval" / "runs" / f"chunk_checks_{stamp}_{settings.chunk_config_hash[:8]}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "doc_id": doc_id, "chunk_config_hash": settings.chunk_config_hash, "chunks": len(chunks),
        "checks": [asdict(c) for c in checks],
    }, indent=2))
    print(f"  report -> {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
