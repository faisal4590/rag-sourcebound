# Stage 19 (issue #40): the SQLite semantic answer cache, no model, no network.
"""Put and look up by cosine over the question vector, the version and TTL rules of spec Stage
19 rules 2 and 5-6, and the purge that runs when the index or the prompt changes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from flp_rag.contracts import Source
from flp_rag.stores.answer_cache import AnswerCache, cosine

SOURCES = [Source("S1", ["d:06:0001"], 6, "Readonly Properties", "Readonly", 75, 77, 0.9)]
VERSIONS = {
    "index_version": "v3",
    "prompt_version": "abcd1234",
    "config_hash": "c0ffee00",
    "embed_model": "fake",
}
T0 = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def put(cache: AnswerCache, vector: list[float], question: str = "q", **kw: object) -> bool:
    return cache.put(
        vector,
        question=question,
        answer=f"A {question} [S1]",
        sources=SOURCES,
        now=T0,
        **{**VERSIONS, **kw},  # type: ignore[arg-type]
    )


def test_cosine_is_scale_free_and_zero_on_a_zero_vector() -> None:
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_lookup_returns_the_closest_live_row_at_or_above_the_threshold(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as cache:
        put(cache, [1.0, 0.0, 0.0], "readonly")
        put(cache, [0.0, 1.0, 0.0], "enums")
        hit = cache.lookup([0.99, 0.1, 0.0], threshold=0.97, now=T0, **VERSIONS)
        assert hit is not None and hit.question == "readonly" and hit.answer == "A readonly [S1]"
        assert hit.sources == SOURCES and hit.prompt_version == "abcd1234"
        assert hit.similarity == pytest.approx(0.9949, abs=1e-3)
        # the threshold is inclusive: exactly 0.97 hits, a little under misses
        assert cache.lookup([0.97, (1 - 0.97**2) ** 0.5, 0.0], threshold=0.97, now=T0, **VERSIONS)
        assert cache.lookup([0.7, 0.7, 0.0], threshold=0.97, now=T0, **VERSIONS) is None
        assert cache.count() == 2


def test_same_question_replaces_its_row(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as cache:
        put(cache, [1.0, 0.0], "q")
        put(cache, [1.0, 0.0], "q")
        assert cache.count() == 1


def test_other_index_prompt_config_or_model_is_a_miss_and_is_purged(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as cache:
        put(cache, [1.0, 0.0], "a")
        put(cache, [1.0, 0.0], "b", index_version="v2")
        put(cache, [1.0, 0.0], "c", prompt_version="00000000")
        put(cache, [1.0, 0.0], "d", embed_model="other")
        put(cache, [1.0, 0.0], "e", config_hash="deadbeef")
        hit = cache.lookup([1.0, 0.0], threshold=0.97, now=T0, **VERSIONS)
        assert hit is not None and hit.question == "a"
        other = {**VERSIONS, "index_version": "v9"}
        assert cache.lookup([1.0, 0.0], threshold=0.97, now=T0, **other) is None
        # spec rule 5: a new index, prompt, or configuration empties the cache of the old rows
        dropped = cache.purge_other_versions(
            index_version="v3", prompt_version="abcd1234", config_hash="c0ffee00"
        )
        assert dropped == 3
        assert cache.count() == 2  # "a" and "d" (same versions, another embed model) stay


def test_rows_expire_after_the_ttl(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=1) as cache:
        put(cache, [1.0, 0.0], "q")
        edge = T0 + timedelta(hours=1)
        assert cache.lookup([1.0, 0.0], threshold=0.97, now=T0 + timedelta(minutes=59), **VERSIONS)
        assert cache.lookup([1.0, 0.0], threshold=0.97, now=edge, **VERSIONS) is None  # <= expires
        assert cache.purge_expired(now=edge) == 1 and cache.count() == 0
        # a put drops the expired rows of the file on its way in
        put(cache, [0.0, 1.0], "old")
        cache.put(
            [1.0, 0.0],
            question="new",
            answer="A",
            sources=SOURCES,
            now=T0 + timedelta(hours=2),
            **VERSIONS,  # type: ignore[arg-type]
        )
        assert cache.count() == 1


def test_ttl_zero_and_directionless_vectors_are_never_stored_or_matched(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=0) as cache:
        assert put(cache, [1.0, 0.0], "q") is False and cache.count() == 0
    with AnswerCache(tmp_path / "answers2.sqlite", ttl_hours=168) as cache:
        assert put(cache, [0.0, 0.0], "zero") is False and put(cache, [], "empty") is False
        put(cache, [1.0, 0.0], "q")
        assert cache.lookup([0.0, 0.0], threshold=0.0, now=T0, **VERSIONS) is None
        assert cache.lookup([], threshold=0.0, now=T0, **VERSIONS) is None


def test_a_vector_of_another_dimension_never_matches(tmp_path: Path) -> None:
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as cache:
        put(cache, [1.0, 0.0, 0.0], "q")
        assert cache.lookup([1.0, 0.0], threshold=0.5, now=T0, **VERSIONS) is None
