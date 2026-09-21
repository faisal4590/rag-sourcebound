# Stage 12 - Hybrid retrieval. Dense now (Milestone 2); sparse, variants, and RRF in Milestone 4. Spec Section 6, Stage 12.
"""Stage 12: find the most likely chunks.

Milestone 2 scope: dense retrieval only.
1. Embed the question with `Embedder.embed_query`, through the query cache.
2. Query the `dense` named vector of the alias `index.alias`. Top `retrieval.top_k_dense` by cosine.
3. Apply payload filters only from the explicit `filters` argument (`chapter_no`, `has_code`).
4. Return `Candidate` records with `fused_score = cosine` and `provenance = ["dense"]`.

Span `retrieval.dense`, kind RETRIEVER: `gen_ai.request.model`, `rag.top_k`, `rag.n_hits`,
`rag.collection`, and each hit as `retrieval.documents.{i}.document.id`, `.score`, `.metadata`.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import QdrantClient, models

from flp_rag.contracts import Attrs, Candidate
from flp_rag.ingest.s06_s07_index import DENSE_VECTOR, STORE_FILE, qdrant_client
from flp_rag.models import Embedder, get_embeddings
from flp_rag.settings import Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.stores.parent_store import ParentStore
from flp_rag.tracing import current_span, stage

log = structlog.get_logger()

METADATA_KEYS = ("chunk_id", "chapter_no", "chapter_title", "section_title", "page_printed_start",
                 "page_printed_end", "has_code", "index_version")


@dataclass(frozen=True)
class RetrievalFilters:
    """Only what the request states explicitly. Never inferred (spec Stage 11 rule 6)."""

    chapter_no: tuple[int, ...] = ()
    has_code: bool | None = None

    def to_attrs(self) -> Attrs:
        return {
            "rag.filter_chapters": json.dumps(list(self.chapter_no)),
            **({"rag.filter_has_code": self.has_code} if self.has_code is not None else {}),
        }


def build_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[Any] = []
    if filters.chapter_no:
        must.append(models.FieldCondition(key="chapter_no", match=models.MatchAny(any=list(filters.chapter_no))))
    if filters.has_code is not None:
        must.append(models.FieldCondition(key="has_code", match=models.MatchValue(value=filters.has_code)))
    return models.Filter(must=must) if must else None


def record_documents(candidates: Sequence[Candidate]) -> None:
    """OpenInference document attributes on the current span. Phoenix renders them as a list."""
    span = current_span()
    for i, c in enumerate(candidates):
        prefix = f"retrieval.documents.{i}.document"
        span.set_attribute(f"{prefix}.id", c.chunk_id)
        span.set_attribute(f"{prefix}.score", float(c.fused_score))
        metadata = {k: c.payload.get(k) for k in METADATA_KEYS if k in c.payload}
        span.set_attribute(f"{prefix}.metadata", json.dumps(metadata, ensure_ascii=False))


def query_vector(question: str, embedder: Embedder, cache: EmbeddingCache | None) -> list[float]:
    """The question's dense vector, from the query cache when it has been asked before."""
    if cache is not None:
        hit = cache.get_query(embedder.model_name, question)
        if hit is not None:
            return hit
    vector = embedder.embed_query(question)
    if cache is not None:
        cache.put_query(embedder.model_name, question, vector)
    return vector


@stage("retrieval.dense", kind="RETRIEVER")
def retrieve_dense(
    question: str,
    settings: Settings,
    *,
    filters: RetrievalFilters | None = None,
    top_k: int | None = None,
    collection: str | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    cache: EmbeddingCache | None = None,
) -> list[Candidate]:
    """Dense candidates for one question, best first."""
    client = client or qdrant_client(settings)
    embedder = embedder or get_embeddings(settings)
    k = top_k or settings.retrieval.top_k_dense
    target = collection or settings.index.alias
    vector = query_vector(question, embedder, cache)

    points = client.query_points(
        target, query=vector, using=DENSE_VECTOR, limit=k,
        query_filter=build_filter(filters), with_payload=True,
    ).points
    for p in points:
        if not p.payload or "chunk_id" not in p.payload:
            raise RuntimeError(f"point {p.id} in {target!r} has no payload chunk_id; not an flp-rag collection")
    candidates = [
        Candidate(
            chunk_id=str(p.payload["chunk_id"]),
            fused_score=float(p.score),
            rerank_score=None,
            provenance=["dense"],
            payload=dict(p.payload),
        )
        for p in points
    ]
    span = current_span()
    span.set_attribute("gen_ai.request.model", embedder.model_name)
    span.set_attribute("rag.top_k", k)
    span.set_attribute("rag.n_hits", len(candidates))
    span.set_attribute("rag.collection", target)
    record_documents(candidates)
    return candidates


def display_texts(candidates: Sequence[Candidate], index_dir: Path = Path("data/index")) -> dict[str, str]:
    """Display text per chunk from the parent store of each candidate's index version. Qdrant
    holds vectors and payload only; the text lives in SQLite (spec Stage 7)."""
    out: dict[str, str] = {}
    by_version: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_version.setdefault(str(c.payload.get("index_version", "")), []).append(c)
    for version, group in by_version.items():
        store_path = index_dir / version / STORE_FILE
        if not store_path.exists():
            log.warning("parent store missing; no display text for these chunks",
                        index_version=version, store_path=str(store_path), chunks=len(group))
            continue
        with ParentStore(store_path) as store:
            for c in group:
                chunk = store.get_chunk(c.chunk_id)
                if chunk is not None:
                    out[c.chunk_id] = chunk.display_text
    return out
