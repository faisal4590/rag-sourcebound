# SQLite semantic answer cache keyed by the question vector. Spec Section 6, Stage 19, rules 2-6.
"""The answer cache of Stage 19 (issue #40).

One SQLite file next to the embedding cache. A row holds the question, its dense vector, the
answer with its sources, and the four facts a hit must share with the request: the index
version, the prompt version, the `config_hash` (models, gate, reranker, context budget: a
changed pipeline must not serve old answers), and the embedding model that made the vector.
A lookup reads the live rows of those values and returns the closest one when its cosine
reaches the threshold. Rows expire after `cache.ttl_hours` (every `put` drops the expired
ones); `purge_other_versions` is spec rule 5, run at API start. The abstain text never
reaches this file: Stage 18 stores answered, judged responses only (`graph/cache.py`).

The cache is small (one row per distinct answered question), so the cosine scan runs in
Python over the rows of one version. Vectors are little-endian float32 bytes, like the
embedding cache.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

from flp_rag.contracts import Source, from_dict, to_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    key TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    index_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS answers_versions
    ON answers (index_version, prompt_version, config_hash, embed_model, expires_at);
"""
_FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class CachedAnswer:
    """What a hit hands back to the graph: the stored response and how close the question was."""

    question: str
    answer: str
    sources: list[Source]
    prompt_version: str
    similarity: float
    created_at: str


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector has no length or the lengths differ."""
    if len(a) != len(b):
        return 0.0
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    return dot / norm if norm else 0.0


def _key(embed_model: str, question: str) -> str:
    return hashlib.sha256((embed_model + "\x00" + question).encode("utf-8")).hexdigest()


def _pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).strftime(_FMT)


def _usable(vector: Sequence[float]) -> bool:
    """A key needs a direction: an empty or all-zero vector matches nothing and is not stored."""
    return len(vector) > 0 and any(vector)


class AnswerCache:
    """Open with `with AnswerCache(path, ttl_hours=168) as cache:`. One connection, a lock, and
    `check_same_thread` off, like the embedding cache: the API opens it once and LangGraph
    runs the nodes in a worker pool."""

    def __init__(self, path: Path | str, *, ttl_hours: float) -> None:
        self.path = Path(path)
        self.ttl = timedelta(hours=ttl_hours)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- writes

    def put(
        self,
        vector: Sequence[float],
        *,
        question: str,
        answer: str,
        sources: Sequence[Source],
        index_version: str,
        prompt_version: str,
        config_hash: str,
        embed_model: str,
        now: datetime | None = None,
    ) -> bool:
        """Store one answered response; the same question for the same model replaces its row.
        Expired rows go in the same transaction. False when nothing was written: a TTL of 0
        (rows could never hit) or a vector without direction."""
        if self.ttl <= timedelta(0) or not _usable(vector):
            return False
        moment = now or datetime.now(UTC)
        row = (
            _key(embed_model, question),
            question,
            index_version,
            prompt_version,
            config_hash,
            embed_model,
            len(vector),
            _pack(vector),
            answer,
            json.dumps([to_dict(s) for s in sources], ensure_ascii=False),
            _stamp(moment),
            _stamp(moment + self.ttl),
        )
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM answers WHERE expires_at <= ?", (_stamp(moment),))
            self._conn.execute(
                "INSERT OR REPLACE INTO answers (key, question, index_version, prompt_version, "
                "config_hash, embed_model, dim, vector, answer, sources, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        return True

    def purge_other_versions(
        self, *, index_version: str, prompt_version: str, config_hash: str
    ) -> int:
        """Spec rule 5: drop every row of another index, prompt, or configuration. Returns the
        count."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM answers WHERE index_version != ? OR prompt_version != ? "
                "OR config_hash != ?",
                (index_version, prompt_version, config_hash),
            )
        return int(cur.rowcount)

    def purge_expired(self, now: datetime | None = None) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM answers WHERE expires_at <= ?", (_stamp(now or datetime.now(UTC)),)
            )
        return int(cur.rowcount)

    # ---- reads

    def lookup(
        self,
        vector: Sequence[float],
        *,
        index_version: str,
        prompt_version: str,
        config_hash: str,
        embed_model: str,
        threshold: float,
        now: datetime | None = None,
    ) -> CachedAnswer | None:
        """The closest live row of the same versions, configuration, and model with cosine >=
        threshold; None for a vector without direction."""
        if not _usable(vector):
            return None
        with self._lock:
            rows = self._conn.execute(
                "SELECT question, vector, answer, sources, prompt_version, created_at "
                "FROM answers WHERE index_version = ? AND prompt_version = ? AND config_hash = ? "
                "AND embed_model = ? AND dim = ? AND expires_at > ?",
                (
                    index_version,
                    prompt_version,
                    config_hash,
                    embed_model,
                    len(vector),
                    _stamp(now or datetime.now(UTC)),
                ),
            ).fetchall()
        best: tuple[float, tuple[Any, ...]] | None = None
        for row in rows:
            similarity = cosine(vector, _unpack(row[1]))
            if similarity >= threshold and (best is None or similarity > best[0]):
                best = (similarity, row)
        if best is None:
            return None
        similarity, (question, _, answer, sources, version, created_at) = best
        return CachedAnswer(
            question=question,
            answer=answer,
            sources=[from_dict(Source, s) for s in json.loads(sources)],
            prompt_version=version,
            similarity=similarity,
            created_at=created_at,
        )

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0])
