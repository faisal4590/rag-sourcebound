# Stage 13 - Reranking. Cross-encoder scores on question + chunk pairs, sigmoid once, top 8. Spec Section 6, Stage 13.
"""Stage 13: score every candidate against the question with a cross-encoder.

Plain functions of the question, the candidates, the parent store, and `rerank.*` in
`config.yaml` (spec Section 6, Stage 13; issue #26):

1. Pairs: the standalone question and the candidate text, which is the chunk header rebuilt
   from the payload plus the display text from the parent store, the text Stage 6 embedded.
2. Scores: the reranker returns one value from 0 to 1 per pair (`BgeReranker` applies the
   sigmoid to the logit exactly once; Cohere already returns 0..1). Pairs over
   `rerank.max_length` tokens are cut by the model and counted as `rag.n_truncated`.
3. Order: by score, highest first, ties by chunk id. Then one pass over the neighbors: two
   that differ by less than `rerank.hint_margin` swap when only the lower one matches the
   `chapter_hint` of Stage 11. Scores never change; the gate reads the highest score.
4. Cut: the first `rerank.top_n` go on. Every score goes on the span as `rag.scores_all`, the
   data the calibration of spec 8.4 reads.
5. Edge cases: zero candidates make no model call; a model failure raises (status `error` at
   the API, never the abstain text); a candidate missing from the store is an error.

The model loads once (`Deps.reranker`, warmed up at API start) and the span `rerank` (kind
RERANKER) carries `gen_ai.request.model`, `reranker.model_name`, `reranker.top_k`, `rag.n_in`,
`rag.n_out`, `rag.score_top1`, `rag.score_top3_mean`, `rag.scores_all`, `rag.n_truncated`,
`rag.hint_swaps`, `rag.device`.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import structlog

from flp_rag.contracts import Candidate
from flp_rag.graph.state import RagState
from flp_rag.ingest.s04_chunk import BOOK_TITLE, page_label
from flp_rag.models import Reranker
from flp_rag.settings import Settings
from flp_rag.stores.parent_store import ParentStore
from flp_rag.tracing import current_span

log = structlog.get_logger()

RERANK_SCORE_KEY = "rerank"
NUMBERED_CHAPTERS = range(1, 31)


# --------------------------------------------------------------------------- pairs


def chapter_label(chapter_no: int, chapter_title: str) -> str:
    if chapter_no in NUMBERED_CHAPTERS:
        return f"Chapter {chapter_no:02d}: {chapter_title}"
    return chapter_title


def pair_text(payload: Mapping[str, Any], display_text: str) -> str:
    """The chunk header from the payload, then the display text. The header follows Stage 4's
    form; the payload holds the part label but not the printed part title, so that one word
    group is left out."""
    chapter_no, chapter_title = int(payload["chapter_no"]), str(payload["chapter_title"])
    section = str(payload.get("section_title") or chapter_title)
    pages = page_label(int(payload["page_printed_start"]), int(payload["page_printed_end"]))
    head = (
        f"{BOOK_TITLE} > {payload.get('part', '')} > {chapter_label(chapter_no, chapter_title)} "
        f"> {section} ({pages})"
    )
    return f"{head}\n{display_text}"


def pair_texts(candidates: Sequence[Candidate], store: ParentStore) -> list[str]:
    texts: list[str] = []
    for c in candidates:
        chunk = store.get_chunk(c.chunk_id)
        if chunk is None:
            raise KeyError(f"candidate {c.chunk_id} is not in the parent store")
        texts.append(pair_text(c.payload, chunk.display_text))
    return texts


# --------------------------------------------------------------------------- order


def apply_hint(
    ordered: Sequence[Candidate], *, hint: int | None, margin: float
) -> tuple[list[Candidate], int]:
    """One pass over the neighbors of a list sorted by score (Part 4). Two neighbors closer
    than `margin` swap when only the lower one is in the hinted chapter. Scores stay."""
    out = list(ordered)
    if hint is None:
        return out, 0
    swaps = 0
    for i in range(len(out) - 1):
        upper, lower = out[i], out[i + 1]
        gap = _score(upper) - _score(lower)
        if gap < margin and _chapter(lower) == hint and _chapter(upper) != hint:
            out[i], out[i + 1] = lower, upper
            swaps += 1
    return out, swaps


def _chapter(c: Candidate) -> int | None:
    value = c.payload.get("chapter_no")
    return int(value) if value is not None else None


def _score(c: Candidate) -> float:
    if c.rerank_score is None:
        raise ValueError(f"candidate {c.chunk_id} has no rerank_score")
    return float(c.rerank_score)


@dataclass(frozen=True)
class RerankResult:
    kept: list[Candidate]  # the top_n, after the hint pass
    scored: list[Candidate]  # every candidate with its score, best first
    n_truncated: int
    hint_swaps: int

    @property
    def scores_all(self) -> list[tuple[str, float]]:
        return [(c.chunk_id, _score(c)) for c in self.scored]


def rerank(
    question: str,
    candidates: Sequence[Candidate],
    *,
    reranker: Reranker,
    store: ParentStore,
    settings: Settings,
    chapter_hint: int | None = None,
) -> RerankResult:
    """Score, sort, apply the hint, cut. No model call for an empty list."""
    cfg = settings.rerank
    if not candidates:
        return RerankResult([], [], 0, 0)
    texts = pair_texts(candidates, store)
    scores = reranker.score(question, texts)
    if len(scores) != len(candidates):
        raise RuntimeError(
            f"reranker {reranker.model_name} returned {len(scores)} scores for "
            f"{len(candidates)} candidates"
        )
    scored = [
        replace(c, rerank_score=float(s), scores={**c.scores, RERANK_SCORE_KEY: float(s)})
        for c, s in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda c: (-_score(c), c.chunk_id))
    ordered, swaps = apply_hint(scored, hint=chapter_hint, margin=cfg.hint_margin)
    try:
        truncated = reranker.truncated_pairs(question, texts)
    except Exception as exc:  # noqa: BLE001 - a span attribute must not fail the request
        log.warning("truncation count failed", model=reranker.model_name, error=str(exc))
        truncated = -1
    return RerankResult(ordered[: cfg.top_n], scored, truncated, swaps)


# --------------------------------------------------------------------------- Milestone 3 stub


def stub_scores(candidates: Sequence[Candidate], top_n: int) -> list[Candidate]:
    """`rerank.enabled: false` (issue #28): the dense cosine clamped to 0..1 stands in for the
    reranker score, as the Milestone 3 stub did; a candidate without a dense score (sparse
    only) falls back to its fused score. Sorted by that score, then chunk id; the first
    `top_n` go on."""
    scored = [
        replace(
            c,
            rerank_score=min(1.0, max(0.0, float(c.scores.get("dense", c.fused_score)))),
        )
        for c in candidates
    ]
    scored.sort(key=lambda c: (-_score(c), c.chunk_id))
    return scored[:top_n]


def debug_top(kept: Sequence[Candidate]) -> list[dict[str, Any]]:
    """The kept list for the response debug field: the rank of a chapter is read from it."""
    return [{"chunk_id": c.chunk_id, "chapter_no": _chapter(c), "score": _score(c)} for c in kept]


# --------------------------------------------------------------------------- node


def make_rerank(
    settings: Settings, reranker_factory: Callable[[], Reranker], store_path: Path | None
) -> Callable[[RagState], dict[str, Any]]:
    """The graph node. The reranker comes from `Deps` (loaded and warmed at API start) or from
    the factory on first use; the parent store opens read-only per request."""
    model: Reranker | None = None
    lock = threading.Lock()

    def reranker() -> Reranker:
        nonlocal model
        with lock:
            if model is None:
                model = reranker_factory()
            return model

    def rerank_node(state: RagState) -> dict[str, Any]:
        candidates = list(state.get("candidates", []))
        standalone = state.get("standalone")
        question = standalone.standalone if standalone else state["question"]
        hint = standalone.chapter_hint if standalone else None
        span = current_span()
        span.set_attribute("rag.n_in", len(candidates))
        span.set_attribute("rag.rerank_enabled", settings.rerank.enabled)
        if not candidates:  # no model call; the gate abstains with gate.no_candidates
            span.set_attribute("rag.n_out", 0)
            span.set_attribute("rag.score_top1", 0.0)
            return {"candidates": []}
        if not settings.rerank.enabled:
            kept = stub_scores(candidates, settings.rerank.top_n)
            top = [_score(c) for c in kept]
            for key, value in (
                ("rag.n_out", len(kept)),
                ("rag.score_top1", top[0]),
                ("rag.score_top3_mean", sum(top[:3]) / len(top[:3])),
                ("rag.scores_all", json.dumps([(c.chunk_id, s) for c, s in zip(kept, top)])),
                ("rag.device", "none"),
            ):
                span.set_attribute(key, value)
            return with_debug(state, {"candidates": kept}, kept)
        if store_path is None:
            raise RuntimeError("reranking needs the parent store; Deps.store_path is not set")
        rr = reranker()
        with ParentStore.open_readonly(store_path) as store:
            result = rerank(
                question,
                candidates,
                reranker=rr,
                store=store,
                settings=settings,
                chapter_hint=hint,
            )
        top = [s for _, s in result.scores_all]
        for key, value in (
            ("gen_ai.request.model", rr.model_name),
            ("reranker.model_name", rr.model_name),
            ("reranker.top_k", settings.rerank.top_n),
            ("rag.n_out", len(result.kept)),
            ("rag.score_top1", top[0]),
            ("rag.score_top3_mean", sum(top[:3]) / len(top[:3])),
            ("rag.scores_all", json.dumps(result.scores_all)),
            ("rag.n_truncated", result.n_truncated),
            ("rag.hint_swaps", result.hint_swaps),
            ("rag.device", getattr(rr, "device", "unknown")),
        ):
            span.set_attribute(key, value)
        return with_debug(state, {"candidates": result.kept}, result.kept)

    def with_debug(
        state: RagState, out: dict[str, Any], kept: Sequence[Candidate]
    ) -> dict[str, Any]:
        if state.get("debug"):
            out["debug_info"] = {**state.get("debug_info", {}), "rerank": {"top": debug_top(kept)}}
        return out

    return rerank_node
