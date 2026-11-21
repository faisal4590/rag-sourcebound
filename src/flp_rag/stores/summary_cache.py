# SQLite cache of the Stage 5 summaries, keyed by sha256(prompt_version + chunk_id + text). #37.
"""Summary cache.

One table, `summaries`, keyed by `sha256(prompt_version + chunk_id + display_text)` (spec Stage 5,
Milestone 7 rule 5, issue #37 rule 3). A rerun of Stage 5 with the same prompt and the same
chunks makes no model call. A new prompt version or a changed chunk text is a new key, and a
row written by another summary model is a miss (the key follows the issue; the model is checked
on the row).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    key TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    summary TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SummaryCache:
    """Open with `with SummaryCache(path) as cache:`. One connection, one thread (Stage 5 is
    sequential)."""

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

    def get(self, key: str, *, model: str) -> str | None:
        """The cached summary, when it was written by `model`; another model's row is a miss."""
        row = self._conn.execute(
            "SELECT summary FROM summaries WHERE key = ? AND model = ?", (key, model)
        ).fetchone()
        return None if row is None else str(row[0])

    def put(
        self,
        key: str,
        *,
        chunk_id: str,
        prompt_version: str,
        model: str,
        summary: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    chunk_id,
                    prompt_version,
                    model,
                    summary,
                    input_tokens,
                    output_tokens,
                    _now(),
                ),
            )

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])
