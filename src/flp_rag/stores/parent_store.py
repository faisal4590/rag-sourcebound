# SQLite tables parents and chunks. Spec Section 5, Stage 7.
"""Parent store.

One SQLite file per index version, `data/index/{index_version}/store.sqlite`, with two tables:
- `parents(parent_id, chapter_no, section_title, text, page_printed_start, page_printed_end)`
- `chunks(chunk_id, parent_id, display_text, payload_json)`

Stage 15 reads parents by id to build the generator's context. Stage 8 checks that every
`parent_id` in `chunks` exists in `parents`.
"""

import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

from flp_rag.contracts import Chunk, Parent

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
"""


class ParentStore:
    """Open with `with ParentStore(path) as store:`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # ---- writes

    def put_parents(self, parents: Iterable[Parent]) -> int:
        rows = [
            (p.parent_id, p.chapter_no, p.section_title, p.text, p.page_printed_start,
             p.page_printed_end, p.token_count)
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

    # ---- reads

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
            chunk_id=row[0], parent_id=row[1], display_text=row[2],
            embedding_text="", token_count=int(payload.get("token_count", 0)), payload=payload,
        )

    # ---- integrity

    def count(self, table: str) -> int:
        if table not in ("parents", "chunks"):
            raise ValueError(f"unknown table {table!r}")
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def orphan_chunk_ids(self) -> list[str]:
        """Chunks whose parent_id has no row in parents. Stage 8 wants this empty."""
        rows = self._conn.execute(
            "SELECT c.chunk_id FROM chunks c LEFT JOIN parents p ON p.parent_id = c.parent_id "
            "WHERE p.parent_id IS NULL ORDER BY c.chunk_id"
        ).fetchall()
        return [r[0] for r in rows]
