# Stage 12 - Hybrid retrieval. Dense + sparse BM25 for every variant, fused with RRF. Spec Section 6, Stage 12.
"""Stage 12: find the 30 most likely chunks with two retrievers and one fusion step.

Plain functions of the query variants, the request filters, and `retrieval.*` in `config.yaml`
(spec Section 6, Stage 12; issue #25):

1. Variants: the standalone question and the paraphrases go to the dense retriever; those plus
   the keyword variant go to the sparse retriever (open question 1). Without Stage 11 the only
   variant is the raw question, so one dense and one sparse list.
2. Dense: all variants in one embedding batch (through the query cache), one Qdrant batch over
   the `dense` vector. Sparse: the query mode of the BM25 model (a weight of 1 per term; Qdrant
   adds the IDF), one Qdrant batch over `sparse_bm25`. Top 30 per list. An empty sparse list is
   valid.
3. Fusion: Reciprocal Rank Fusion with `retrieval.rrf_k`. Every list gives each chunk
   1 / (k + rank) points; the sum is `fused_score`; ties break by the best single rank, then
   by chunk id. Each candidate keeps its provenance (`dense:v0:r3`) and its best raw score per
   retriever (`scores`), which the rerank stub and the trace read.
4. Siblings: a candidate with a `code_group_id` brings the other pieces of its group, with its
   score and the provenance `code_group`, after the cut (open question 4).
5. Filters: only the request's `filters`, on every list. The `chapter_hint` of Stage 11 never
   filters.
6. Failures: a Qdrant or model error is raised (status `error` at the API, never the abstain
   text). A failing sparse side is an error too (open question 2).

Spans: `retrieval.hybrid` (RETRIEVER) with children `retrieval.dense`, `retrieval.sparse`, and
`retrieval.fuse`. `retrieve_dense` keeps the Milestone 2 single-question form for `flp retrieve`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
from qdrant_client import QdrantClient, models

from flp_rag.contracts import Attrs, Candidate, StandaloneQuery
from flp_rag.ingest.s06_s07_index import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    STORE_FILE,
    qdrant_client,
)
from flp_rag.models import Embedder, SparseEmbedder, get_embeddings, get_sparse_embeddings
from flp_rag.settings import Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.stores.parent_store import ParentStore
from flp_rag.tracing import current_span, stage

log = structlog.get_logger()

Retriever = Literal["dense", "sparse"]
CODE_GROUP_PROVENANCE = "code_group"
METADATA_KEYS = (
    "chunk_id",
    "chapter_no",
    "chapter_title",
    "section_title",
    "page_printed_start",
    "page_printed_end",
    "has_code",
    "index_version",
)


# --------------------------------------------------------------------------- inputs


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


@dataclass(frozen=True)
class Variants:
    """The search variants of one request (spec Stage 11; issue #25, Part 1)."""

    standalone: str
    paraphrases: list[str] = field(default_factory=list)
    keyword: str | None = None

    @property
    def dense_queries(self) -> list[str]:
        return [self.standalone, *self.paraphrases]

    @property
    def sparse_queries(self) -> list[str]:
        dense = self.dense_queries
        extra = [self.keyword] if self.keyword and self.keyword not in dense else []
        return [*dense, *extra]

    @property
    def count(self) -> int:
        return len(self.sparse_queries)

    def to_attrs(self) -> Attrs:
        return {"rag.n_variants": self.count}


def variants_from(standalone: StandaloneQuery | None, question: str) -> Variants:
    """Stage 11's output, or the raw question alone. Duplicate phrasings are dropped."""
    if standalone is None:
        return Variants(question)
    seen = {standalone.standalone}
    paraphrases = [v for v in standalone.variants if v not in seen and not seen.add(v)]
    return Variants(standalone.standalone, paraphrases, standalone.keyword_variant or None)


def build_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    must: list[Any] = []
    if filters.chapter_no:
        must.append(
            models.FieldCondition(
                key="chapter_no", match=models.MatchAny(any=list(filters.chapter_no))
            )
        )
    if filters.has_code is not None:
        must.append(
            models.FieldCondition(key="has_code", match=models.MatchValue(value=filters.has_code))
        )
    return models.Filter(must=must) if must else None


# --------------------------------------------------------------------------- result lists


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class ResultList:
    retriever: Retriever
    variant: int  # index into the retriever's query list
    hits: list[Hit]


def _hits(points: Sequence[models.ScoredPoint], target: str) -> list[Hit]:
    out: list[Hit] = []
    for p in points:
        if not p.payload or "chunk_id" not in p.payload:
            raise RuntimeError(
                f"point {p.id} in {target!r} has no payload chunk_id; not an flp-rag collection"
            )
        out.append(Hit(str(p.payload["chunk_id"]), float(p.score), dict(p.payload)))
    return out


def _batch(
    client: QdrantClient,
    target: str,
    retriever: Retriever,
    vectors: Sequence[Any],
    *,
    using: str,
    limit: int,
    filters: RetrievalFilters | None,
) -> list[ResultList]:
    """One Qdrant batch request: one query per variant, the same filter on every list."""
    query_filter = build_filter(filters)
    requests = [
        models.QueryRequest(
            query=vector, using=using, limit=limit, filter=query_filter, with_payload=True
        )
        for vector in vectors
    ]
    responses = client.query_batch_points(target, requests=requests)
    return [
        ResultList(retriever, i, _hits(resp.points, target)) for i, resp in enumerate(responses)
    ]


def _record_lists(span: Any, lists: Sequence[ResultList]) -> None:
    span.set_attribute("rag.n_queries", len(lists))
    span.set_attribute("rag.hits_per_list", json.dumps([len(lst.hits) for lst in lists]))
    span.set_attribute(
        "rag.top_score_per_list",
        json.dumps([lst.hits[0].score if lst.hits else None for lst in lists]),
    )
    span.set_attribute("rag.n_hits", sum(len(lst.hits) for lst in lists))


def query_vector(question: str, embedder: Embedder, cache: EmbeddingCache | None) -> list[float]:
    """The question's dense vector, from the query cache when it has been asked before."""
    return query_vectors([question], embedder, cache)[0]


def query_vectors(
    questions: Sequence[str], embedder: Embedder, cache: EmbeddingCache | None
) -> list[list[float]]:
    """Dense vectors for every variant: cache hits first, the misses in one embedding batch."""
    vectors: list[list[float] | None] = [None] * len(questions)
    if cache is not None:
        for i, q in enumerate(questions):
            vectors[i] = cache.get_query(embedder.model_name, q)
    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        fresh = embedder.embed_queries([questions[i] for i in missing])
        for i, vector in zip(missing, fresh, strict=True):
            vectors[i] = vector
            if cache is not None:
                cache.put_query(embedder.model_name, questions[i], vector)
    return [v for v in vectors if v is not None]


def _dense_lists(
    queries: Sequence[str],
    settings: Settings,
    *,
    filters: RetrievalFilters | None,
    client: QdrantClient,
    embedder: Embedder,
    cache: EmbeddingCache | None,
    target: str,
) -> list[ResultList]:
    vectors = query_vectors(queries, embedder, cache)
    lists = _batch(
        client,
        target,
        "dense",
        vectors,
        using=DENSE_VECTOR,
        limit=settings.retrieval.top_k_dense,
        filters=filters,
    )
    span = current_span()
    span.set_attribute("gen_ai.request.model", embedder.model_name)
    span.set_attribute("rag.top_k", settings.retrieval.top_k_dense)
    span.set_attribute("rag.collection", target)
    _record_lists(span, lists)
    return lists


@stage("retrieval.dense", kind="RETRIEVER")
def retrieve_dense_lists(
    queries: Sequence[str],
    settings: Settings,
    *,
    filters: RetrievalFilters | None,
    client: QdrantClient,
    embedder: Embedder,
    cache: EmbeddingCache | None,
    collection: str,
) -> list[ResultList]:
    """One dense list per variant: one embedding batch, one Qdrant batch."""
    return _dense_lists(
        queries,
        settings,
        filters=filters,
        client=client,
        embedder=embedder,
        cache=cache,
        target=collection,
    )


@stage("retrieval.sparse", kind="RETRIEVER")
def retrieve_sparse_lists(
    queries: Sequence[str],
    settings: Settings,
    *,
    filters: RetrievalFilters | None,
    client: QdrantClient,
    sparse: SparseEmbedder,
    collection: str,
) -> list[ResultList]:
    """One BM25 list per variant, with the query mode of the sparse model. A variant made of
    stop words only has no terms: its list is empty without a request."""
    encoded = sparse.embed_query(queries)
    with_terms = [i for i, v in enumerate(encoded) if v.indices]
    vectors = [
        models.SparseVector(indices=encoded[i].indices, values=encoded[i].values)
        for i in with_terms
    ]
    fetched = (
        _batch(
            client,
            collection,
            "sparse",
            vectors,
            using=SPARSE_VECTOR,
            limit=settings.retrieval.top_k_sparse,
            filters=filters,
        )
        if vectors
        else []
    )
    by_variant = {i: lst.hits for i, lst in zip(with_terms, fetched, strict=True)}
    lists = [ResultList("sparse", i, by_variant.get(i, [])) for i in range(len(queries))]
    span = current_span()
    span.set_attribute("rag.sparse_model", sparse.model_name)
    span.set_attribute("rag.top_k", settings.retrieval.top_k_sparse)
    span.set_attribute("rag.collection", collection)
    _record_lists(span, lists)
    return lists


# --------------------------------------------------------------------------- fusion


@dataclass
class _Tally:
    points: float = 0.0
    best_rank: int = 10**9
    provenance: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@stage("retrieval.fuse", kind="CHAIN")
def fuse(lists: Sequence[ResultList], *, rrf_k: int, top_k: int) -> list[Candidate]:
    """Reciprocal Rank Fusion: 1 / (k + rank) per list, summed per chunk; the top `top_k`
    unique chunks, ties by best single rank then chunk id (Part 3)."""
    tallies: dict[str, _Tally] = {}
    for lst in lists:
        for rank, hit in enumerate(lst.hits, start=1):
            t = tallies.setdefault(hit.chunk_id, _Tally(payload=hit.payload))
            t.points += 1.0 / (rrf_k + rank)
            t.best_rank = min(t.best_rank, rank)
            t.provenance.append(f"{lst.retriever}:v{lst.variant}:r{rank}")
            t.scores[lst.retriever] = max(t.scores.get(lst.retriever, float("-inf")), hit.score)
    # Rounded points, so two chunks with the same ranks in a different list order still tie.
    ordered = sorted(
        tallies.items(), key=lambda kv: (-round(kv[1].points, 12), kv[1].best_rank, kv[0])
    )
    span = current_span()
    span.set_attribute("rag.rrf_k", rrf_k)
    span.set_attribute("rag.n_in", sum(len(lst.hits) for lst in lists))
    span.set_attribute("rag.n_out", min(len(ordered), top_k))
    return [
        Candidate(
            chunk_id=cid,
            fused_score=t.points,
            rerank_score=None,
            provenance=t.provenance,
            payload=t.payload,
            scores=dict(t.scores),
        )
        for cid, t in ordered[:top_k]
    ]


def overlap_ratio(candidates: Sequence[Candidate]) -> float:
    """Of the fused chunks, the share that a dense list and a sparse list both found."""
    if not candidates:
        return 0.0
    both = sum(
        1
        for c in candidates
        if any(p.startswith("dense:") for p in c.provenance)
        and any(p.startswith("sparse:") for p in c.provenance)
    )
    return both / len(candidates)


def add_code_group_siblings(
    candidates: Sequence[Candidate], client: QdrantClient, collection: str
) -> tuple[list[Candidate], int]:
    """Every other piece of a split code example joins with the score, the raw scores, and
    the provenance mark `code_group` of the member that brought it in (Part 4). The list can
    grow past the cut. One scroll per group, not per member."""
    out = list(candidates)
    present = {c.chunk_id for c in out}
    seen_groups: set[str] = set()
    added = 0
    for c in candidates:
        group = c.payload.get("code_group_id")
        if not group or group in seen_groups:
            continue
        seen_groups.add(str(group))
        records, _ = client.scroll(
            collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="code_group_id", match=models.MatchValue(value=group))
                ]
            ),
            limit=100,
            with_payload=True,
        )
        for rec in sorted(records, key=lambda rec: int((rec.payload or {}).get("part_index") or 0)):
            payload = dict(rec.payload or {})
            cid = str(payload.get("chunk_id", ""))
            if not cid or cid in present:
                continue
            out.append(
                Candidate(
                    cid, c.fused_score, None, [CODE_GROUP_PROVENANCE], payload, dict(c.scores)
                )
            )
            present.add(cid)
            added += 1
    return out, added


# --------------------------------------------------------------------------- the stage


def record_documents(candidates: Sequence[Candidate]) -> None:
    """OpenInference document attributes on the current span. Phoenix renders them as a list."""
    span = current_span()
    for i, c in enumerate(candidates):
        prefix = f"retrieval.documents.{i}.document"
        span.set_attribute(f"{prefix}.id", c.chunk_id)
        span.set_attribute(f"{prefix}.score", float(c.fused_score))
        metadata = {k: c.payload.get(k) for k in METADATA_KEYS if k in c.payload}
        span.set_attribute(f"{prefix}.metadata", json.dumps(metadata, ensure_ascii=False))


@stage("retrieval.hybrid", kind="RETRIEVER")
def retrieve_hybrid(
    variants: Variants,
    settings: Settings,
    *,
    filters: RetrievalFilters | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
    sparse: SparseEmbedder | None = None,
    cache: EmbeddingCache | None = None,
    collection: str | None = None,
) -> list[Candidate]:
    """The fused top `retrieval.top_k_fused` candidates for one request, plus code siblings."""
    client = client or qdrant_client(settings)
    embedder = embedder or get_embeddings(settings)
    sparse = sparse or get_sparse_embeddings(settings)
    target = collection or settings.index.alias
    dense = retrieve_dense_lists(
        variants.dense_queries,
        settings,
        filters=filters,
        client=client,
        embedder=embedder,
        cache=cache,
        collection=target,
    )
    sparse_lists = retrieve_sparse_lists(
        variants.sparse_queries,
        settings,
        filters=filters,
        client=client,
        sparse=sparse,
        collection=target,
    )
    fused = fuse(
        [*dense, *sparse_lists],
        rrf_k=settings.retrieval.rrf_k,
        top_k=settings.retrieval.top_k_fused,
    )
    candidates, added = add_code_group_siblings(fused, client, target)
    span = current_span()
    span.set_attribute("rag.n_variants", variants.count)
    span.set_attribute("rag.n_dense_hits", sum(len(lst.hits) for lst in dense))
    span.set_attribute("rag.n_sparse_hits", sum(len(lst.hits) for lst in sparse_lists))
    span.set_attribute("rag.n_fused", len(candidates))
    span.set_attribute("rag.n_siblings_added", added)
    span.set_attribute("rag.overlap_ratio", overlap_ratio(fused))
    span.set_attribute("rag.top_fused_score", fused[0].fused_score if fused else 0.0)
    span.set_attribute("rag.collection", target)
    record_documents(candidates)
    return candidates


# --------------------------------------------------------------------------- Milestone 2 form


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
    """Dense candidates for one question, best first (`flp retrieve`, the eval scripts)."""
    client = client or qdrant_client(settings)
    embedder = embedder or get_embeddings(settings)
    target = collection or settings.index.alias
    k = top_k or settings.retrieval.top_k_dense
    bounded = settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"top_k_dense": k})}
    )
    (lst,) = _dense_lists(
        [question],
        bounded,
        filters=filters,
        client=client,
        embedder=embedder,
        cache=cache,
        target=target,
    )
    candidates = [
        Candidate(h.chunk_id, h.score, None, ["dense"], h.payload, {"dense": h.score})
        for h in lst.hits
    ]
    record_documents(candidates)
    return candidates


def display_texts(
    candidates: Sequence[Candidate], index_dir: Path = Path("data/index")
) -> dict[str, str]:
    """Display text per chunk from the parent store of each candidate's index version. Qdrant
    holds vectors and payload only; the text lives in SQLite (spec Stage 7)."""
    out: dict[str, str] = {}
    by_version: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_version.setdefault(str(c.payload.get("index_version", "")), []).append(c)
    for version, group in by_version.items():
        store_path = index_dir / version / STORE_FILE
        if not store_path.exists():
            log.warning(
                "parent store missing; no display text for these chunks",
                index_version=version,
                store_path=str(store_path),
                chunks=len(group),
            )
            continue
        with ParentStore(store_path) as store:
            for c in group:
                chunk = store.get_chunk(c.chunk_id)
                if chunk is not None:
                    out[c.chunk_id] = chunk.display_text
    return out
