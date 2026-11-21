# Stage 19 - Caching. Milestone 7, issue #40. Spec Section 6, Stage 19.
"""The semantic answer cache in the query graph.

    guard_input -> cache_lookup -> understand ...        (miss)
                        |
                        +-> respond                      (hit: Stages 11-17 skipped)

The lookup node embeds the raw question (through the query embedding cache of rule 1, so the
second ask of the same text costs no model call), asks `AnswerCache` for the closest stored
answer of the same `index_version` and `prompt_version`, and on a hit puts the stored answer
and sources into the state with `cache_hit = true`. Stage 18 (`respond.py`) calls
`store_answer` after it built the response: only an answered response whose faithfulness judge
ran and said `supported` is stored (rule 3), never the abstain text.

Four kinds of request are neither looked up nor stored (`cache_skipped` names the reason):
one with conversation history (the raw follow-up "What about enums?" is not the question that
was answered, and the cache key must be), one with retrieval `filters` (a chapter-restricted
answer is not the unrestricted one), one from the evaluation harness (`judge_all`: a run must
measure the pipeline, not the cache), and one without an index version (no alias: nothing to
key on). A store error is recorded on the span and logged, and the request goes on as a miss
or as an unstored answer, like the session turn of Stage 18. The span `cache.lookup` records
`cache.hit`, `cache.similarity`, `cache.skipped`, and `cache.error`; the API copies `cache.hit`
and `cache.similarity` onto `rag.request` (the spec's trace line).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from typing import Any

import structlog

from flp_rag.contracts import Response
from flp_rag.graph.nodes.generate import prompt_version as file_prompt_version
from flp_rag.graph.state import RagState
from flp_rag.models import Embedder
from flp_rag.settings import Settings
from flp_rag.stores.answer_cache import AnswerCache
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.stores.session_store import SessionStore
from flp_rag.tracing import current_span

CACHE_NODE = "cache_lookup"
SKIP_HISTORY = "history"
SKIP_FILTERS = "filters"
SKIP_JUDGE_ALL = "judge_all"
SKIP_NO_INDEX = "no_index"
STORED, NOT_STORED = "stored", "skipped"
_STORE_ERRORS = (sqlite3.Error, ValueError, KeyError, TypeError)  # a bad file or a bad row
log = structlog.get_logger(__name__)


def has_history(sessions: SessionStore | None, session_id: str | None) -> bool:
    """True when the session already holds a turn: Stage 11 will rewrite the question."""
    if sessions is None or not session_id:
        return False
    return bool(sessions.last_turns(str(session_id), 1))


def question_vector(
    question: str, embedder: Embedder, query_cache: EmbeddingCache | None
) -> list[float]:
    """The dense vector of the raw question, from the query cache when it has it (rule 1)."""
    if query_cache is not None:
        cached = query_cache.get_query(embedder.model_name, question)
        if cached is not None:
            return cached
    vector = embedder.embed_queries([question])[0]
    if query_cache is not None:
        query_cache.put_query(embedder.model_name, question, vector)
    return vector


def skip_reason(state: Mapping[str, Any], sessions: SessionStore | None) -> str | None:
    """Why this request stays out of the cache, or None. Checked before any embedding."""
    if state.get("judge_all"):
        return SKIP_JUDGE_ALL
    if state.get("filters") is not None:
        return SKIP_FILTERS
    if not state.get("index_version"):
        return SKIP_NO_INDEX
    if has_history(sessions, state.get("session_id")):
        return SKIP_HISTORY
    return None


def route_after_cache(state: Mapping[str, Any]) -> str:
    """A hit goes straight to the response; everything else continues with Stage 11."""
    return "respond" if state.get("cache_hit") else "understand"


def make_cache_lookup(
    settings: Settings,
    answers: AnswerCache,
    embedder_factory: Callable[[], Embedder],
    query_cache: EmbeddingCache | None,
    sessions: SessionStore | None,
) -> Callable[[RagState], dict[str, Any]]:
    cfg = settings.cache

    def cache_lookup(state: RagState) -> dict[str, Any]:
        span = current_span()
        skipped = skip_reason(state, sessions)
        if skipped is not None:
            span.set_attribute("cache.hit", False)
            span.set_attribute("cache.skipped", skipped)
            return {"cache_hit": False, "cache_skipped": skipped}
        embedder = embedder_factory()
        vector = question_vector(state["question"], embedder, query_cache)
        out: dict[str, Any] = {"cache_hit": False, "question_vector": vector}
        try:
            hit = answers.lookup(
                vector,
                index_version=str(state["index_version"]),
                prompt_version=_prompt_version(settings),
                config_hash=settings.config_hash,
                embed_model=settings.embed.model,  # the pinned name, same on the store side
                threshold=cfg.similarity_threshold,
            )
        except _STORE_ERRORS as exc:  # a broken cache file is a miss, never a failed answer
            span.record_exception(exc)
            span.set_attribute("cache.hit", False)
            span.set_attribute("cache.error", type(exc).__name__)
            log.warning(
                "answer cache lookup failed", error=type(exc).__name__, path=str(answers.path)
            )
            return out
        span.set_attribute("cache.hit", hit is not None)
        if hit is None:
            return out
        out["cache_hit"] = True
        span.set_attribute("cache.similarity", hit.similarity)
        out |= {
            "cache_similarity": hit.similarity,
            "status": "answered",
            "answer": hit.answer,
            "sources": list(hit.sources),
            "prompt_version": hit.prompt_version,
        }
        if state.get("debug"):
            out["debug_info"] = {
                **state.get("debug_info", {}),
                "cache": {"hit": True, "similarity": hit.similarity, "question": hit.question},
            }
        return out

    return cache_lookup


def store_answer(
    state: Mapping[str, Any], response: Response, answers: AnswerCache | None, settings: Settings
) -> str:
    """Stage 18's side of rule 3. Stored when: a cache is configured, the request was neither a
    hit nor skipped, the status is `answered`, the faithfulness judge ran and said supported,
    and the lookup node left the question vector. Returns `stored`, `skipped`, or the error
    class name (recorded on the span and logged; the answer is still delivered)."""
    if answers is None or state.get("cache_hit") or state.get("cache_skipped"):
        return NOT_STORED
    vector = state.get("question_vector")
    if response.status != "answered" or state.get("faithful") is not True or not vector:
        return NOT_STORED
    try:
        written = answers.put(
            vector,
            question=str(state.get("question", "")),
            answer=response.answer,
            sources=response.sources,
            index_version=response.index_version,
            prompt_version=_prompt_version(settings),
            config_hash=settings.config_hash,
            embed_model=settings.embed.model,
        )
    except _STORE_ERRORS as exc:
        current_span().record_exception(exc)
        log.warning("answer not cached", error=type(exc).__name__, path=str(answers.path))
        return type(exc).__name__
    return STORED if written else NOT_STORED


def _prompt_version(settings: Settings) -> str:
    """The configured answer prompt (`gen.prompt_file`), the `prompt_version` of the root span.
    Both the lookup and the store key on it, so a borderline answer written with the strict
    prompt is cached too, and a new answer prompt empties the cache (rule 5). A hit reports
    this version."""
    path = settings.config_path.parent / settings.gen.prompt_file
    return file_prompt_version(path) if path.exists() else ""
