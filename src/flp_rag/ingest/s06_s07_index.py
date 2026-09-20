# Stage 6 and Stage 7 - Embedding and indexing with QdrantVectorStore (hybrid), parent store, manifest. Spec Section 5.
"""Stage 6: one dense vector and one sparse vector per chunk.

Input: `data/enriched/{doc_id}.jsonl`. Output: `data/vectors/{doc_id}.parquet` with columns
`chunk_id`, `dense`, `sparse_indices`, `sparse_values`, and file metadata `embed_model`, `dim`,
`sparse_model`, `doc_id`.

Rules (spec Section 5, Stage 6):
1. Embed `embedding_text` in batches of `embed.batch_size`. Dense vectors are unit length.
2. The sparse BM25 vector covers `display_text` plus the joined `php_symbols`.
3. Dense vectors are cached in SQLite by `sha256(embed_model + embedding_text)`. A rerun with
   unchanged text costs zero model calls.
4. A failing batch is retried `embed.retry_attempts` times with exponential backoff. If any
   chunk still has no vector, the run fails. Nothing partial is written.
5. `embed_model` and `dim` go into the output file. Stage 7 copies them into the manifest.

Stage 7 (indexing into Qdrant, the parent store, the manifest) lands with issue #13.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from flp_rag.contracts import Attrs, Chunk, read_jsonl
from flp_rag.models import (
    Embedder,
    SparseEmbedder,
    SparseVector,
    get_embeddings,
    get_sparse_embeddings,
)
from flp_rag.settings import Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.tracing import current_span, stage

log = structlog.get_logger()


@dataclass(frozen=True)
class VectorRow:
    chunk_id: str
    dense: list[float]
    sparse: SparseVector


@dataclass(frozen=True)
class EmbedResult:
    doc_id: str
    vectors_total: int
    dense_dim: int
    embed_model: str
    sparse_model: str
    cache_hits: int
    cache_misses: int
    batches: int
    self_retrieval_failures: int
    output_path: Path

    def to_attrs(self) -> Attrs:
        return {
            "rag.doc_id": self.doc_id,
            "gen_ai.request.model": self.embed_model,
            "rag.vectors_total": self.vectors_total,
            "rag.dense_dim": self.dense_dim,
            "rag.sparse_model": self.sparse_model,
            "rag.cache_hits": self.cache_hits,
            "rag.cache_misses": self.cache_misses,
            "rag.batches": self.batches,
            "rag.self_retrieval_failures": self.self_retrieval_failures,
        }


# --------------------------------------------------------------------------- helpers


def sparse_text(chunk: Chunk) -> str:
    """Display text plus the symbols, so the symbols count as BM25 terms."""
    symbols = chunk.payload.get("php_symbols") or []
    return chunk.display_text + ("\n" + " ".join(symbols) if symbols else "")


def with_retries[T](fn: Callable[[], T], *, attempts: int, backoff_s: float, what: str) -> T:
    """Call `fn` up to `attempts` times. The wait doubles after each failure."""
    wait = backoff_s
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f"{what} failed after {attempts} attempts: {exc}") from exc
            log.warning("retrying", what=what, attempt=attempt, error=str(exc), wait_s=wait)
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def self_retrieval_failures(vectors: Sequence[Sequence[float]]) -> list[int]:
    """Indexes of vectors whose nearest neighbor is not themselves, or whose self-score is under
    0.99. Duplicate vectors fail here, which is the point of the check."""
    import numpy as np

    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=np.float32)
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -1.0)
    best_other = scores.max(axis=1)
    self_scores = np.einsum("ij,ij->i", matrix, matrix)
    return [int(i) for i in range(len(vectors)) if best_other[i] >= self_scores[i] - 1e-6 or self_scores[i] < 0.99]


# --------------------------------------------------------------------------- batches


@stage("embed.batch", kind="EMBEDDING")
def embed_batch(
    texts: Sequence[str], tokens: int, embedder: Embedder, *, retry_attempts: int, backoff_s: float
) -> list[list[float]]:
    """One batch, one span. Takes scalars, not Settings: the decorator records inputs on the
    span, and this runs once per batch."""
    span = current_span()
    span.set_attribute("gen_ai.request.model", embedder.model_name)
    span.set_attribute("rag.batch_size", len(texts))
    span.set_attribute("rag.batch_tokens", tokens)
    vectors = with_retries(
        lambda: embedder.embed_passages(texts),
        attempts=retry_attempts,
        backoff_s=backoff_s,
        what=f"embedding batch of {len(texts)}",
    )
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
    return [normalize(v) for v in vectors]


def embed_dense(
    chunks: Sequence[Chunk], embedder: Embedder, cache: EmbeddingCache, settings: Settings
) -> tuple[dict[str, list[float]], int, int, int]:
    """Dense vectors by chunk_id. Returns (vectors, cache hits, cache misses, batches run)."""
    texts = [c.embedding_text for c in chunks]
    cached = cache.get_many(embedder.model_name, texts)
    vectors: dict[str, list[float]] = {c.chunk_id: cached[c.embedding_text] for c in chunks if c.embedding_text in cached}
    missing = [c for c in chunks if c.embedding_text not in cached]
    batches = 0
    size = settings.embed.batch_size
    for start in range(0, len(missing), size):
        batch = missing[start : start + size]
        fresh = embed_batch(
            [c.embedding_text for c in batch], sum(c.token_count for c in batch), embedder,
            retry_attempts=settings.embed.retry_attempts, backoff_s=settings.embed.retry_backoff_s,
        )
        cache.put_many(embedder.model_name, [(c.embedding_text, v) for c, v in zip(batch, fresh, strict=True)])
        for c, v in zip(batch, fresh, strict=True):
            vectors[c.chunk_id] = v
        batches += 1
    if len(vectors) != len(chunks):
        raise RuntimeError(f"{len(chunks) - len(vectors)} chunks have no dense vector; refusing a partial index")
    # Hits count chunks served from the cache, so hits + misses == chunks even with repeated text.
    return vectors, len(chunks) - len(missing), len(missing), batches


# --------------------------------------------------------------------------- parquet


def write_vectors(path: Path, rows: Sequence[VectorRow], *, doc_id: str, embed_model: str, dim: int, sparse_model: str) -> None:
    table = pa.table(
        {
            "chunk_id": pa.array([r.chunk_id for r in rows], pa.string()),
            "dense": pa.array([r.dense for r in rows], pa.list_(pa.float32())),
            "sparse_indices": pa.array([r.sparse.indices for r in rows], pa.list_(pa.int32())),
            "sparse_values": pa.array([r.sparse.values for r in rows], pa.list_(pa.float32())),
        }
    )
    metadata = {"doc_id": doc_id, "embed_model": embed_model, "dim": str(dim), "sparse_model": sparse_model}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **{k.encode(): v.encode() for k, v in metadata.items()}})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def read_vectors(path: Path) -> tuple[list[VectorRow], dict[str, str]]:
    table = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    rows = [
        VectorRow(chunk_id=cid, dense=list(dense), sparse=SparseVector(indices=list(idx), values=list(val)))
        for cid, dense, idx, val in zip(
            table.column("chunk_id").to_pylist(),
            table.column("dense").to_pylist(),
            table.column("sparse_indices").to_pylist(),
            table.column("sparse_values").to_pylist(),
            strict=True,
        )
    ]
    return rows, meta


# --------------------------------------------------------------------------- the stage


@stage("ingest.embed", kind="CHAIN")
def embed(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/enriched"),
    out_dir: Path = Path("data/vectors"),
    cache_path: Path | None = None,
    embedder: Embedder | None = None,
    sparse: SparseEmbedder | None = None,
) -> EmbedResult:
    """Run Stage 6 on the Stage 5 output of one document."""
    chunks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Chunk))
    embedder = embedder or get_embeddings(settings)
    sparse = sparse or get_sparse_embeddings(settings)
    cache_file = cache_path or (settings.config_path.parent / settings.embed.cache_path)

    with EmbeddingCache(cache_file) as cache:
        dense, hits, misses, batches = embed_dense(chunks, embedder, cache, settings)

    sparse_vectors = sparse.embed([sparse_text(c) for c in chunks])
    if len(sparse_vectors) != len(chunks):
        raise RuntimeError("sparse embedder returned a different number of vectors than chunks")

    rows = [VectorRow(c.chunk_id, dense[c.chunk_id], sv) for c, sv in zip(chunks, sparse_vectors, strict=True)]
    dim = embedder.dim
    bad = [r.chunk_id for r in rows if len(r.dense) != dim]
    if bad:
        raise RuntimeError(f"{len(bad)} dense vectors do not have {dim} dimensions, e.g. {bad[:3]}")
    failures = self_retrieval_failures([r.dense for r in rows])
    if failures:
        log.warning("self-retrieval check failed for some chunks", chunk_ids=[rows[i].chunk_id for i in failures][:10])

    output_path = out_dir / f"{doc_id}.parquet"
    write_vectors(output_path, rows, doc_id=doc_id, embed_model=embedder.model_name, dim=dim, sparse_model=sparse.model_name)
    return EmbedResult(
        doc_id=doc_id,
        vectors_total=len(rows),
        dense_dim=dim,
        embed_model=embedder.model_name,
        sparse_model=sparse.model_name,
        cache_hits=hits,
        cache_misses=misses,
        batches=batches,
        self_retrieval_failures=len(failures),
        output_path=output_path,
    )

