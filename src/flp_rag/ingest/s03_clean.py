# Stage 3 - Cleaning and normalization. Spec Section 5, Stage 3.
"""Stage 3: delete extraction artifacts without any change to the meaning of the text.

Input: `data/blocks/{doc_id}.jsonl`. Output: `data/clean/{doc_id}.jsonl`, the same blocks with
clean text.

Rules, each switchable through `clean.rules_enabled`:
1. `nfc`: normalize to Unicode NFC.
2. `nbsp`: non-breaking spaces become normal spaces.
3. `zero_width`: zero-width characters and backspace (`\\x08`) are deleted.
4. `whitespace`: in prose blocks, runs of spaces and tabs collapse to one space and line ends
   are stripped. Newlines survive: a callout keeps its title line and paragraph breaks.
5. `page_number_lines`: in prose blocks, a line that holds only a page number is deleted.

Code blocks get rules 1-3 only. They are never re-wrapped, re-spaced, or re-quoted.
"""

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from flp_rag.contracts import Attrs, Block, read_jsonl, write_jsonl
from flp_rag.settings import CleanRule, Settings
from flp_rag.tracing import stage

PROSE_TYPES = frozenset({"paragraph", "callout", "heading", "chapter_title"})

_NBSP = re.compile("[\N{NO-BREAK SPACE}\N{NARROW NO-BREAK SPACE}\N{FIGURE SPACE}]")
_ZERO_WIDTH = re.compile(
    "[\N{ZERO WIDTH SPACE}\N{ZERO WIDTH NON-JOINER}\N{ZERO WIDTH JOINER}"
    "\N{WORD JOINER}\N{ZERO WIDTH NO-BREAK SPACE}\x08]"
)
_HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")

RuleCounts = dict[str, int]


@dataclass(frozen=True)
class CleanResult:
    doc_id: str
    blocks_total: int
    blocks_changed: int
    rule_counts: dict[str, dict[str, int]]  # rule -> block type -> fires
    chars_removed: dict[str, int]  # block type -> characters deleted
    output_path: Path

    def to_attrs(self) -> Attrs:
        return {
            "rag.doc_id": self.doc_id,
            "rag.blocks_total": self.blocks_total,
            "rag.blocks_changed": self.blocks_changed,
            "rag.rule_counts": json.dumps(self.rule_counts, sort_keys=True),
            "rag.chars_removed": json.dumps(self.chars_removed, sort_keys=True),
        }


# --------------------------------------------------------------------------- rules


def _rule_nfc(text: str) -> tuple[str, int]:
    """Count is the number of characters whose form changed, not blocks."""
    out = unicodedata.normalize("NFC", text)
    if out == text:
        return out, 0
    # A composition shortens the text by one per character; a reordering keeps the length.
    changed = abs(len(out) - len(text)) or sum(1 for a, b in zip(out, text, strict=True) if a != b)
    return out, max(1, changed)


def _rule_nbsp(text: str) -> tuple[str, int]:
    return _NBSP.subn(" ", text)


def _rule_zero_width(text: str) -> tuple[str, int]:
    return _ZERO_WIDTH.subn("", text)


def _rule_whitespace(text: str) -> tuple[str, int]:
    """Trailing spaces go first so a trailing run is counted once, then inner runs collapse."""
    out, trailing = _TRAILING_SPACE.subn("", text)
    out, runs = _HORIZONTAL_RUN.subn(" ", out)
    return out, trailing + runs


def _rule_page_number_lines(text: str) -> tuple[str, int]:
    """A bare 1-3 digit line inside a prose block is a page number that bled through. A block
    that would become empty is left alone: a number on its own is content, not an artifact."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not _PAGE_NUMBER_LINE.fullmatch(ln)]
    if len(kept) == len(lines) or not any(ln.strip() for ln in kept):
        return text, 0
    return "\n".join(kept), len(lines) - len(kept)


_ALL_TYPES = {
    "nfc": _rule_nfc,
    "nbsp": _rule_nbsp,
    "zero_width": _rule_zero_width,
}
_PROSE_ONLY = {
    "whitespace": _rule_whitespace,
    "page_number_lines": _rule_page_number_lines,
}


def clean_text(text: str, block_type: str, rules: Sequence[CleanRule]) -> tuple[str, RuleCounts]:
    """Apply the enabled rules in spec order. Returns the text and how often each rule fired."""
    counts: RuleCounts = {}
    for rule in rules:
        fn = _ALL_TYPES.get(rule) or (_PROSE_ONLY.get(rule) if block_type in PROSE_TYPES else None)
        if fn is None:
            continue
        text, fired = fn(text)
        if fired:
            counts[rule] = counts.get(rule, 0) + fired
    return text, counts


def clean_block(block: Block, rules: Sequence[CleanRule]) -> tuple[Block, RuleCounts]:
    text, counts = clean_text(block.text, block.type, rules)
    return (replace(block, text=text) if text != block.text else block), counts


def clean_blocks(
    blocks: Sequence[Block], rules: Sequence[CleanRule]
) -> tuple[list[Block], dict[str, dict[str, int]], int, dict[str, int]]:
    """Clean every block. Returns (blocks, rule counts by type, blocks changed, chars removed)."""
    out: list[Block] = []
    rule_counts: dict[str, dict[str, int]] = {}
    chars_removed: dict[str, int] = {}
    changed = 0
    for b in blocks:
        cleaned, counts = clean_block(b, rules)
        out.append(cleaned)
        chars_removed[b.type] = chars_removed.get(b.type, 0) + (len(b.text) - len(cleaned.text))
        if cleaned is not b:
            changed += 1
        for rule, n in counts.items():
            by_type = rule_counts.setdefault(rule, {})
            by_type[b.type] = by_type.get(b.type, 0) + n
    return out, rule_counts, changed, chars_removed


# --------------------------------------------------------------------------- the stage


@stage("ingest.clean", kind="CHAIN")
def clean(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/blocks"),
    out_dir: Path = Path("data/clean"),
) -> CleanResult:
    """Run Stage 3 on the Stage 2 output of one document."""
    blocks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Block))
    cleaned, rule_counts, changed, chars_removed = clean_blocks(blocks, settings.clean.rules_enabled)
    output_path = out_dir / f"{doc_id}.jsonl"
    write_jsonl(output_path, cleaned)
    return CleanResult(
        doc_id=doc_id,
        blocks_total=len(cleaned),
        blocks_changed=changed,
        rule_counts=rule_counts,
        chars_removed=chars_removed,
        output_path=output_path,
    )
