# SQLite tables parents and chunks. Spec Section 5, Stage 7.
"""Parent store.

One SQLite file per index version, `data/index/{index_version}/store.sqlite`, with three tables:
- `parents(parent_id, chapter_no, section_title, text, page_printed_start, page_printed_end)`
- `chunks(chunk_id, parent_id, display_text, payload_json)`
- `blocks(block_id, parent_id, chunk_id, position, type, text, page_printed)`: the clean blocks
  of every chunk, in book order, so Stage 15 can keep code inside fenced blocks (issue #20).

Stage 15 reads parents and their blocks to build the generator's context. Stage 8 checks that
every `parent_id` in `chunks` exists in `parents`.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from flp_rag.contracts import Block, Chunk, Parent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    parent_id TEXT PRIMARY KEY,
    chapter_no INTEGER NOT NULL,
    section_title TEXT NOT NULL,
    text TEXT NOT NULL,
    page_printed_start INTEGER NOT NULL,
    page_printed_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    display_text TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_parent_id ON chunks(parent_id);
CREATE TABLE IF NOT EXISTS blocks (
    block_id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    type TEXT NOT NULL,
    text TEXT NOT NULL,
    page_printed INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS blocks_parent_id ON blocks(parent_id);
CREATE INDEX IF NOT EXISTS blocks_chunk_id ON blocks(chunk_id);
"""


@dataclass(frozen=True)
class StoredBlock:
    """One clean block as Stage 15 reads it back: typed text with its place in the parent."""

    block_id: str
    parent_id: str
    chunk_id: str
    position: int
    type: str
    text: str
    page_printed: int


_BLOCK_COLUMNS = "block_id, parent_id, chunk_id, position, type, text, page_printed"


class ParentStore:
    """Open with `with ParentStore(path) as store:`."""

    def __init__(self, path: Path | str, *, readonly: bool = False) -> None:
        """Writers create the file and the schema. The query side opens read-only: a missing
        file is an error, not a new empty store, and no DDL runs under the worker pool."""
        self.path = Path(path)
        if readonly:
            if not self.path.exists():
                raise FileNotFoundError(f"parent store {self.path} does not exist; run flp ingest")
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)

    @classmethod
    def open_readonly(cls, path: Path | str) -> Self:
        return cls(path, readonly=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ---- writes

    def put_parents(self, parents: Iterable[Parent]) -> int:
        rows = [
            (
                p.parent_id,
                p.chapter_no,
                p.section_title,
                p.text,
                p.page_printed_start,
                p.page_printed_end,
                p.token_count,
            )
            for p in parents
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO parents VALUES (?, ?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def put_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows = [
            (c.chunk_id, c.parent_id, c.display_text, json.dumps(c.payload, ensure_ascii=False))
            for c in chunks
        ]
        with self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)", rows)
        return len(rows)

    def put_blocks(self, blocks: Iterable[Block], chunks: Iterable[Chunk]) -> int:
        """Store the clean blocks that the chunks list in `payload.block_ids`, in book order.
        A block that no chunk lists (a chapter title) is not stored."""
        owner: dict[str, tuple[str, str]] = {}
        for c in chunks:
            for bid in c.payload.get("block_ids", []):
                if str(bid) in owner:
                    raise ValueError(
                        f"block {bid} is listed by chunks {owner[str(bid)][1]} and {c.chunk_id}"
                    )
                owner[str(bid)] = (c.parent_id, c.chunk_id)
        ordered = sorted(
            (b for b in blocks if b.block_id in owner), key=lambda b: (b.page_pdf, b.block_index)
        )
        rows = [
            (
                b.block_id,
                owner[b.block_id][0],
                owner[b.block_id][1],
                i,
                b.type,
                b.text,
                b.page_printed,
            )
            for i, b in enumerate(ordered)
        ]
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO blocks ({_BLOCK_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ---- reads

    def get_blocks(self, parent_id: str) -> list[StoredBlock]:
        rows = self._conn.execute(
            f"SELECT {_BLOCK_COLUMNS} FROM blocks WHERE parent_id = ? ORDER BY position",
            (parent_id,),
        ).fetchall()
        return [StoredBlock(*r) for r in rows]

    def get_chunk_blocks(self, chunk_id: str) -> list[StoredBlock]:
        rows = self._conn.execute(
            f"SELECT {_BLOCK_COLUMNS} FROM blocks WHERE chunk_id = ? ORDER BY position", (chunk_id,)
        ).fetchall()
        return [StoredBlock(*r) for r in rows]

    def get_parent(self, parent_id: str) -> Parent | None:
        row = self._conn.execute(
            "SELECT parent_id, chapter_no, section_title, text, page_printed_start, "
            "page_printed_end, token_count FROM parents WHERE parent_id = ?",
            (parent_id,),
        ).fetchone()
        return Parent(*row) if row else None

    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, Parent]:
        out: dict[str, Parent] = {}
        for pid in parent_ids:
            parent = self.get_parent(pid)
            if parent is not None:
                out[pid] = parent
        return out

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """The stored chunk. `embedding_text` is not persisted and comes back empty; the store
        serves display text and payload for citations, not text for re-embedding."""
        row = self._conn.execute(
            "SELECT chunk_id, parent_id, display_text, payload_json FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = json.loads(row[3])
        return Chunk(
            chunk_id=row[0],
            parent_id=row[1],
            display_text=row[2],
            embedding_text="",
            token_count=int(payload.get("token_count", 0)),
            payload=payload,
        )

    # ---- index facts (Stage 11, issue #27)

    def chapters(self) -> list[tuple[int, str]]:
        """The distinct (chapter_no, chapter_title) pairs of the indexed chunks, in book order.
        Stage 11 puts this list into the query prompt, so it describes the indexed book, never a
        typed copy of it."""
        rows = self._conn.execute(
            "SELECT json_extract(payload_json, '$.chapter_no') AS no, "
            "json_extract(payload_json, '$.chapter_title') AS title "
            "FROM chunks WHERE json_extract(payload_json, '$.chapter_no') IS NOT NULL "
            "GROUP BY no, title ORDER BY MIN(rowid)"
        ).fetchall()
        return [(int(no), str(title)) for no, title in rows]

    def php_symbols(self) -> set[str]:
        """Every `php_symbols` entry of the indexed chunks, lower-cased."""
        rows = self._conn.execute(
            "SELECT DISTINCT lower(value) FROM chunks, "
            "json_each(json_extract(payload_json, '$.php_symbols'))"
        ).fetchall()
        return {str(row[0]) for row in rows}

    # ---- integrity

    def count(self, table: str) -> int:
        if table not in ("parents", "chunks", "blocks"):
            raise ValueError(f"unknown table {table!r}")
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def orphan_chunk_ids(self) -> list[str]:
        """Chunks whose parent_id has no row in parents. Stage 8 wants this empty."""
        rows = self._conn.execute(
            "SELECT c.chunk_id FROM chunks c LEFT JOIN parents p ON p.parent_id = c.parent_id "
            "WHERE p.parent_id IS NULL ORDER BY c.chunk_id"
        ).fetchall()
        return [r[0] for r in rows]
