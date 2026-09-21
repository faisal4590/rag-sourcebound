# SQLite cache keyed by sha256(embed_model + embedding_text). Spec Section 5, Stage 6.
"""Embedding cache.

One SQLite file with two tables:
- `dense`: passage vectors keyed by `sha256(embed_model + text)`. A rerun of Stage 6 with
  unchanged text costs zero model calls.
- `queries`: query vectors keyed by the exact question text (spec Stage 19, rule 1).

Vectors are stored as little-endian float32 bytes.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dense (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    question TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (question, model)
);
"""


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256((model + text).encode("utf-8")).hexdigest()


def _pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class EmbeddingCache:
    """Open with `with EmbeddingCache(path) as cache:`."""

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

    # ---- passages

    def get_many(self, model: str, texts: Iterable[str]) -> dict[str, list[float]]:
        """Cached vectors by text. Texts that are not cached are absent from the result."""
        wanted = {cache_key(model, t): t for t in texts}
        found: dict[str, list[float]] = {}
        keys = list(wanted)
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, vector FROM dense WHERE key IN ({marks})", batch
            ).fetchall()
            for key, blob in rows:
                found[wanted[key]] = _unpack(blob)
        return found

    def put_many(self, model: str, pairs: Iterable[tuple[str, Sequence[float]]]) -> int:
        rows = [
            (cache_key(model, text), model, len(vector), _pack(vector), _now())
            for text, vector in pairs
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO dense (key, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ---- queries

    def get_query(self, model: str, question: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM queries WHERE question = ? AND model = ?", (question, model)
        ).fetchone()
        return _unpack(row[0]) if row else None

    def put_query(self, model: str, question: str, vector: Sequence[float]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO queries (question, model, dim, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, model, len(vector), _pack(vector), _now()),
            )

    # ---- stats

    def count(self, table: str = "dense") -> int:
        if table not in ("dense", "queries"):
            raise ValueError(f"unknown table {table!r}")
        return int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
