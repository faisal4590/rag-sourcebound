# Session store (issue #27, Part 1): the last turns of a session, expiry after the TTL.
"""Unit tests for `stores/session_store.py`, no model and no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from flp_rag.stores.session_store import SessionStore, history_from

T0 = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


def open_store(tmp_path: Path, **kw: object) -> SessionStore:
    opts = {"ttl_hours": 24, "excerpt_chars": 300, **kw}
    return SessionStore(tmp_path / "sessions.sqlite", **opts)  # type: ignore[arg-type]


def test_fourth_read_gives_the_last_three_turns_newest_last(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        for i in range(1, 5):
            store.add_turn("s1", question=f"q{i}", status="answered", answer=f"a{i} [S1]", now=T0)
        turns = store.last_turns("s1", 3, now=T0)
    assert [t.question for t in turns] == ["q2", "q3", "q4"]
    assert [t.turn_no for t in turns] == [2, 3, 4]
    assert turns[-1].answer_excerpt == "a4 [S1]"
    assert turns[-1].status == "answered"


def test_turn_numbers_start_at_one_per_session(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        first = store.add_turn("s1", question="q1", status="answered", answer="a1", now=T0)
        other = store.add_turn("s2", question="q1", status="no_information", answer="No", now=T0)
        second = store.add_turn("s1", question="q2", status="answered", answer="a2", now=T0)
        assert (first.turn_no, second.turn_no, other.turn_no) == (1, 2, 1)
        assert [t.question for t in store.last_turns("s2", 3, now=T0)] == ["q1"]
        assert store.last_turns("unknown", 3, now=T0) == []


def test_session_expires_24_hours_after_its_last_turn(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        store.add_turn("s1", question="q1", status="answered", answer="a1", now=T0)
        store.add_turn(
            "s1", question="q2", status="answered", answer="a2", now=T0 + timedelta(hours=2)
        )
        store.add_turn(
            "s2", question="q1", status="answered", answer="a1", now=T0 + timedelta(hours=20)
        )
        # 23 hours after s1's last turn: still there
        assert len(store.last_turns("s1", 3, now=T0 + timedelta(hours=25))) == 2
        # 25 hours after s1's last turn: gone, rows deleted; s2 is younger and stays
        assert store.last_turns("s1", 3, now=T0 + timedelta(hours=27, minutes=1)) == []
        assert store.count() == 1
        assert len(store.last_turns("s2", 3, now=T0 + timedelta(hours=27, minutes=1))) == 1


def test_answer_is_cut_to_the_excerpt_length(tmp_path: Path) -> None:
    with open_store(tmp_path, excerpt_chars=10) as store:
        turn = store.add_turn("s1", question="q", status="answered", answer="x" * 50, now=T0)
    assert turn.answer_excerpt == "x" * 10


def test_history_from_turns_is_user_then_assistant(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        store.add_turn(
            "s1",
            question="Tell me about readonly properties",
            status="answered",
            answer="Readonly properties are set once [S1].",
            now=T0,
        )
        store.add_turn(
            "s1",
            question="What is a backed enum?",
            status="no_information",
            answer="No information found",
            now=T0,
        )
        history = history_from(store.last_turns("s1", 3, now=T0))
    assert history == [
        ("user", "Tell me about readonly properties"),
        ("assistant", "Readonly properties are set once [S1]."),
        ("user", "What is a backed enum?"),
        ("assistant", "No information found"),
    ]


def test_zero_turns_requested_reads_nothing(tmp_path: Path) -> None:
    with open_store(tmp_path) as store:
        store.add_turn("s1", question="q", status="answered", answer="a", now=T0)
        assert store.last_turns("s1", 0, now=T0) == []
