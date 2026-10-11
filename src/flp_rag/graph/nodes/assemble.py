# Stage 15 - Context assembly. Parent expansion, labels, token budget. Spec Section 6, Stage 15.
"""Stage 15: turn the candidates that passed the gate into labeled context blocks.

Plain functions of the candidates, the parent store, and the configuration (spec Section 6,
Stage 15; issue #20). No model call.

1. Select: candidates with `rerank_score >= gate.tau_pass`, at most `context.max_blocks`.
2. Group: one group per `parent_id`, scored by its best chunk, chunks in book order.
3. Budget, in score order; the first option that fits `context.max_tokens` wins:
   the whole parent; the passing chunks of the group; the best chunk; nothing (skipped).
4. Order and label: sort by chapter and printed page, then `[S1]`, `[S2]`, ... The first line
   of a block is its label line. Code blocks stay inside fenced `php` blocks.
5. Output: `context` (the blocks), `label_map` (label -> chunk ids, chapter, section, pages),
   and the span `context.assemble`.

Token counts use the Stage 4 tokenizer and include the label line and the fences. The budget
step reserves the widest label (`[S{max_blocks}]`); the exact count is set after labeling.
`tokens_used` is the sum of the block counts; Stage 16 adds its own separators between blocks.

A store without stored blocks (an index built before the `blocks` table) still serves plain
parent and chunk text; the span then carries `rag.context_plain_text = true` and a warning is
logged, because code in that context is not fenced. Rebuild the index to get fences.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import tiktoken

from flp_rag.contracts import Candidate, ContextBlock
from flp_rag.graph.state import RagState
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.settings import Settings
from flp_rag.stores.parent_store import ParentStore, StoredBlock
from flp_rag.tracing import current_span

PARENT, CHUNKS, CHUNK = "parent", "chunks", "chunk"
WINDOW_SUFFIX = ":w"  # a windowed parent (section over parent_max_tokens), spec Stage 4 rule 6
log = structlog.get_logger()


@dataclass(frozen=True)
class Group:
    parent_id: str
    score: float
    chunks: list[Candidate]  # book order


@dataclass(frozen=True)
class Draft:
    """A block before ordering and labeling."""

    parent_id: str
    kind: str  # parent | chunks | chunk
    body: str
    source_ids: list[str]
    chapter_no: int
    chapter_title: str
    section_title: str
    page_printed_start: int
    page_printed_end: int


@dataclass(frozen=True)
class Assembly:
    blocks: list[ContextBlock]
    label_map: dict[str, dict[str, Any]]
    n_parents_used: int
    n_chunk_fallbacks: int
    n_skipped: int
    tokens_used: int
    truncated: bool
    plain_text: bool  # True when the store had no blocks and code is not fenced
    source_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- selection and grouping


def select_candidates(
    candidates: Sequence[Candidate], tau_pass: float, max_blocks: int
) -> list[Candidate]:
    passing = [c for c in candidates if c.rerank_score is not None and c.rerank_score >= tau_pass]
    passing.sort(key=lambda c: -float(c.rerank_score or 0.0))
    return passing[:max_blocks]


def _book_key(c: Candidate) -> tuple[int, int, str]:
    p = c.payload
    return int(p.get("chapter_no", 0)), int(p.get("page_printed_start", 0)), c.chunk_id


def _parent_of(c: Candidate) -> str:
    pid = c.payload.get("parent_id")
    if not pid:
        raise KeyError(f"candidate {c.chunk_id} has no parent_id in its payload")
    return str(pid)


def group_by_parent(candidates: Sequence[Candidate]) -> list[Group]:
    by_parent: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_parent.setdefault(_parent_of(c), []).append(c)
    groups = [
        Group(pid, max(float(c.rerank_score or 0.0) for c in cs), sorted(cs, key=_book_key))
        for pid, cs in by_parent.items()
    ]
    groups.sort(key=lambda g: -g.score)
    return groups


# --------------------------------------------------------------------------- text


def render_blocks(blocks: Sequence[StoredBlock]) -> str:
    """Blocks joined by blank lines; code inside a fenced php block, indentation untouched."""
    parts = [f"```php\n{b.text}\n```" if b.type == "code" else b.text for b in blocks]
    return "\n\n".join(parts)


def label_line(
    label: str, chapter_no: int, chapter_title: str, section_title: str, start: int, end: int
) -> str:
    pages = f"p. {start}" if start == end else f"p. {start}-{end}"
    where = f"Chapter {chapter_no:02d}: {chapter_title}"
    if section_title and section_title != chapter_title:
        where = f"{where} > {section_title}"
    return f"[{label}] {where}, {pages}"


def block_text(label: str, draft: Draft) -> str:
    head = label_line(
        label,
        draft.chapter_no,
        draft.chapter_title,
        draft.section_title,
        draft.page_printed_start,
        draft.page_printed_end,
    )
    return f"{head}\n{draft.body}"


# --------------------------------------------------------------------------- budget


def _chunk_pages(chunks: Sequence[Candidate]) -> tuple[int, int]:
    starts = [int(c.payload["page_printed_start"]) for c in chunks]
    ends = [int(c.payload["page_printed_end"]) for c in chunks]
    return min(starts), max(ends)


def _chunk_body(store: ParentStore, c: Candidate) -> str:
    blocks = store.get_chunk_blocks(c.chunk_id)
    if blocks:
        return render_blocks(blocks)
    stored = store.get_chunk(c.chunk_id)
    if stored is None:
        raise KeyError(f"candidate {c.chunk_id} is not in the parent store")
    return stored.display_text


def _options(group: Group, store: ParentStore) -> list[Draft]:
    """The four options of the budget step for one group, in order. Option 4 is the empty list."""
    best = group.chunks[0].payload
    chapter_no, chapter_title = int(best["chapter_no"]), str(best["chapter_title"])
    section_title = str(best.get("section_title", ""))
    parent = store.get_parent(group.parent_id)
    if parent is None:
        raise KeyError(
            f"parent {group.parent_id} of candidate {group.chunks[0].chunk_id} is not in the store"
        )
    # A windowed parent is the chunk plus its neighbors, which the blocks table does not hold.
    # The book has none (largest section 1,642 tokens); the plain text is correct if one appears.
    parent_blocks = (
        [] if group.parent_id.endswith(WINDOW_SUFFIX) else store.get_blocks(group.parent_id)
    )
    parent_body = render_blocks(parent_blocks) if parent_blocks else parent.text
    all_ids = [c.chunk_id for c in group.chunks]
    best_chunk = max(group.chunks, key=lambda c: float(c.rerank_score or 0.0))
    chunks_body = "\n\n".join(_chunk_body(store, c) for c in group.chunks)
    start, end = _chunk_pages(group.chunks)
    options = [
        Draft(
            group.parent_id,
            PARENT,
            parent_body,
            all_ids,
            chapter_no,
            chapter_title,
            parent.section_title,
            parent.page_printed_start,
            parent.page_printed_end,
        ),
    ]
    if chunks_body != parent_body:  # the passing chunks may already be the whole parent
        options.append(
            Draft(
                group.parent_id,
                CHUNKS,
                chunks_body,
                all_ids,
                chapter_no,
                chapter_title,
                section_title,
                start,
                end,
            )
        )
    if len(group.chunks) > 1:
        bs, be = _chunk_pages([best_chunk])
        options.append(
            Draft(
                group.parent_id,
                CHUNK,
                _chunk_body(store, best_chunk),
                [best_chunk.chunk_id],
                chapter_no,
                chapter_title,
                section_title,
                bs,
                be,
            )
        )
    return options


def fill_budget(
    groups: Sequence[Group],
    store: ParentStore,
    enc: tiktoken.Encoding,
    max_tokens: int,
    max_blocks: int,
) -> tuple[list[Draft], int, int, int]:
    """Drafts that fit, plus (parents used, chunk fallbacks, skipped). Counts reserve the
    widest label so the final labeled text never exceeds the budget."""
    widest = f"S{max_blocks}"
    remaining = max_tokens
    chosen: list[Draft] = []
    parents = fallbacks = skipped = 0
    for group in groups:
        for draft in _options(group, store):
            cost = len(enc.encode(block_text(widest, draft)))
            if cost <= remaining:
                remaining -= cost
                chosen.append(draft)
                parents += draft.kind == PARENT
                fallbacks += draft.kind != PARENT
                break
        else:
            skipped += 1
    return chosen, parents, fallbacks, skipped


# --------------------------------------------------------------------------- assembly


def assemble(
    candidates: Sequence[Candidate], store: ParentStore, settings: Settings, enc: tiktoken.Encoding
) -> Assembly:
    ctx = settings.context
    selected = select_candidates(candidates, settings.gate.tau_pass, ctx.max_blocks)
    groups = group_by_parent(selected)
    drafts, parents, fallbacks, skipped = fill_budget(
        groups, store, enc, ctx.max_tokens, ctx.max_blocks
    )
    drafts.sort(key=lambda d: (d.chapter_no, d.page_printed_start, d.parent_id))
    blocks: list[ContextBlock] = []
    label_map: dict[str, dict[str, Any]] = {}
    for i, d in enumerate(drafts, start=1):
        label = f"S{i}"
        text = block_text(label, d)
        blocks.append(
            ContextBlock(
                label,
                list(d.source_ids),
                text,
                d.chapter_no,
                d.chapter_title,
                d.page_printed_start,
                d.page_printed_end,
                len(enc.encode(text)),
            )
        )
        label_map[label] = {
            "chunk_ids": list(d.source_ids),
            "parent_id": d.parent_id,
            "kind": d.kind,
            "chapter_no": d.chapter_no,
            "chapter_title": d.chapter_title,
            "section_title": d.section_title,
            "page_printed_start": d.page_printed_start,
            "page_printed_end": d.page_printed_end,
        }
    return Assembly(
        blocks=blocks,
        label_map=label_map,
        n_parents_used=parents,
        n_chunk_fallbacks=fallbacks,
        n_skipped=skipped,
        tokens_used=sum(b.token_count for b in blocks),
        truncated=fallbacks > 0 or skipped > 0,
        plain_text=store.count("blocks") == 0,
        source_ids=[cid for b in blocks for cid in b.source_ids],
    )


def make_assemble(
    settings: Settings, store_path: Path | None
) -> Callable[[RagState], dict[str, Any]]:
    """The graph node. Opens the parent store per request: SQLite connections do not cross
    threads, and LangGraph runs sync nodes in a worker pool."""
    enc = tokenizer(settings.chunk)

    def assemble_node(state: RagState) -> dict[str, Any]:
        if store_path is None:
            raise RuntimeError(
                "context assembly needs the parent store; Deps.store_path is not set"
            )
        with ParentStore.open_readonly(store_path) as store:
            result = assemble(list(state.get("candidates", [])), store, settings, enc)
        span = current_span()
        if result.plain_text:
            log.warning(
                "parent store has no blocks; context code is not fenced", store=str(store_path)
            )
            span.set_attribute("rag.context_plain_text", True)
        for key, value in (
            ("rag.n_blocks", len(result.blocks)),
            ("rag.n_parents_used", result.n_parents_used),
            ("rag.n_chunk_fallbacks", result.n_chunk_fallbacks),
            ("rag.n_skipped", result.n_skipped),
            ("rag.tokens_used", result.tokens_used),
            ("rag.truncated", result.truncated),
            ("rag.source_ids", json.dumps(result.source_ids)),
        ):
            span.set_attribute(key, value)
        return {"context": result.blocks, "label_map": result.label_map}

    return assemble_node
