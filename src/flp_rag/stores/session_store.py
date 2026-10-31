# Session store: one SQLite table of finished turns per session, read by Stage 11, written by Stage 18. Issue #27, Part 1.
"""Session store.

Neither the spec nor an earlier issue says where a conversation's history lives. Issue #27
defines it: one SQLite file (`session.store_path`), one row per finished turn with the session
id, the turn number, the standalone question, the status, the first
`session.answer_excerpt_chars` of the answer, and the time. Stage 18 writes the row after
every response, also for an abstain. Stage 11 reads the last `query.history_turns` rows of the
session for the rewrite. A session expires `session.ttl_hours` after its last turn; expired
rows are deleted on the next read or write. The rows are user content (spec Section 7.12).

One connection, usable from any thread: the API opens the store at startup and LangGraph runs
sync nodes in a worker pool, so `check_same_thread` is off and a lock serialises access.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    answer_excerpt TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_no)
);
CREATE INDEX IF NOT EXISTS turns_created_at ON turns(created_at);
"""
_COLUMNS = "session_id, turn_no, question, status, answer_excerpt, created_at"


def _stamp(now: datetime | None) -> str:
    """ISO 8601 UTC to the second, so the strings sort in time order."""
    moment = now or datetime.now(UTC)
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SessionTurn:
    """One finished request of a session, as Stage 11 reads it back."""

    session_id: str
    turn_no: int
    question: str  # the standalone question of that turn
    status: str  # answered | no_information
    answer_excerpt: str
    created_at: str


def history_from(turns: Sequence[SessionTurn]) -> list[tuple[str, str]]:
    """The `history` of the graph state: a user line and an assistant line per turn, oldest
    first."""
    out: list[tuple[str, str]] = []
    for turn in turns:
        out.append(("user", turn.question))
        out.append(("assistant", turn.answer_excerpt))
    return out


class SessionStore:
    """Open with `with SessionStore(path, ttl_hours=, excerpt_chars=) as sessions:`."""

    def __init__(self, path: Path | str, *, ttl_hours: float, excerpt_chars: int) -> None:
        self.path = Path(path)
        self.ttl = timedelta(hours=ttl_hours)
        self.excerpt_chars = excerpt_chars
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

    # ---- reads

    def last_turns(
        self, session_id: str, n: int, *, now: datetime | None = None
    ) -> list[SessionTurn]:
        """The last `n` turns of the session, oldest first. Expired sessions are purged first,
        so an old session reads as no history."""
        if n <= 0:
            return []
        with self._lock:
            self._purge(now)
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM turns WHERE session_id = ? ORDER BY turn_no DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [SessionTurn(*row) for row in reversed(rows)]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0])

    # ---- writes

    def add_turn(
        self,
        session_id: str,
        *,
        question: str,
        status: str,
        answer: str,
        now: datetime | None = None,
    ) -> SessionTurn:
        """Append the finished turn. The turn number continues the session's count."""
        with self._lock:
            self._purge(now)
            (last,) = self._conn.execute(
                "SELECT COALESCE(MAX(turn_no), 0) FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()
            turn = SessionTurn(
                session_id,
                int(last) + 1,
                question,
                status,
                answer[: self.excerpt_chars],
                _stamp(now),
            )
            self._conn.execute(
                f"INSERT INTO turns ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    turn.session_id,
                    turn.turn_no,
                    turn.question,
                    turn.status,
                    turn.answer_excerpt,
                    turn.created_at,
                ),
            )
            self._conn.commit()
        return turn

    def purge_expired(self, now: datetime | None = None) -> int:
        """Delete every session whose last turn is older than the TTL. Returns the rows removed."""
        with self._lock:
            return self._purge(now)

    def _purge(self, now: datetime | None) -> int:
        cutoff = _stamp((now or datetime.now(UTC)) - self.ttl)
        cursor = self._conn.execute(
            "DELETE FROM turns WHERE session_id IN "
            "(SELECT session_id FROM turns GROUP BY session_id HAVING MAX(created_at) < ?)",
            (cutoff,),
        )
        self._conn.commit()
        return int(cursor.rowcount)
