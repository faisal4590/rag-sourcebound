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

Stage 7: store vectors and payloads in a versioned Qdrant collection, parents and display text
in SQLite, and write the manifest.

Rules (spec Section 5, Stage 7):
1. `index_version` comes from the Stage 5 payload. The collection is `flp_chunks_{index_version}`
   with named vectors `dense` (cosine) and `sparse_bm25`, and payload indexes on `chapter_no`,
   `has_code`, `section_title`, `php_symbols`.
2. Points go into the new collection only. The collection behind the alias is never touched;
   re-indexing the collection that the alias points at is refused.
3. Parent store: `data/index/{index_version}/store.sqlite` with tables `parents` and `chunks`.
4. `manifest.json`: doc_id, counts, model names, chunk_config_hash, config_hash, git commit,
   created_at, composition, cost. No `verify` block yet: Stage 8 adds it and switches the alias.
"""

import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from qdrant_client import QdrantClient, models

from flp_rag.contracts import Attrs, Chunk, Parent, read_jsonl
from flp_rag.models import (
    Embedder,
    SparseEmbedder,
    SparseVector,
    get_embeddings,
    get_sparse_embeddings,
)
from flp_rag.settings import Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.stores.parent_store import ParentStore
from flp_rag.tracing import current_span, git_sha, stage

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



# =========================================================================== Stage 7


DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse_bm25"
COLLECTION_PREFIX = "flp_chunks_"
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "chapter_no": models.PayloadSchemaType.INTEGER,
    "has_code": models.PayloadSchemaType.BOOL,
    "section_title": models.PayloadSchemaType.KEYWORD,
    "php_symbols": models.PayloadSchemaType.KEYWORD,
}
STORE_FILE = "store.sqlite"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class IndexResult:
    doc_id: str
    index_version: str
    collection: str
    points_upserted: int
    parents_stored: int
    chunks_stored: int
    alias_switched: bool
    manifest_path: Path
    store_path: Path

    def to_attrs(self) -> Attrs:
        return {
            "rag.doc_id": self.doc_id,
            "rag.index_version": self.index_version,
            "rag.collection": self.collection,
            "rag.points_upserted": self.points_upserted,
            "rag.parents_stored": self.parents_stored,
            "rag.chunks_stored": self.chunks_stored,
            "rag.alias_switched": self.alias_switched,
        }


def point_id(chunk_id: str) -> str:
    """Qdrant ids are UUIDs or integers. A UUID5 of the chunk id is stable across runs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def collection_name(index_version: str) -> str:
    return f"{COLLECTION_PREFIX}{index_version}"


def qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.index.qdrant_url)


def alias_target(client: QdrantClient, alias: str) -> str | None:
    for a in client.get_aliases().aliases:
        if a.alias_name == alias:
            return a.collection_name
    return None


def create_collection(client: QdrantClient, name: str, dim: int) -> None:
    """A fresh collection with the two named vectors and the payload indexes. An existing
    collection of the same name is replaced; the caller guarantees it is not behind the alias."""
    if client.collection_exists(name):
        log.warning("replacing an existing collection that is not behind the alias", collection=name)
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def upsert_points(
    client: QdrantClient, name: str, chunks: Sequence[Chunk], rows: Sequence[VectorRow], batch_size: int = 64
) -> int:
    by_id = {r.chunk_id: r for r in rows}
    missing = [c.chunk_id for c in chunks if c.chunk_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} chunks have no vector row, e.g. {missing[:3]}")
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={
                    DENSE_VECTOR: by_id[c.chunk_id].dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=by_id[c.chunk_id].sparse.indices, values=by_id[c.chunk_id].sparse.values
                    ),
                },
                payload={**c.payload, "chunk_id": c.chunk_id},
            )
            for c in batch
        ]
        client.upsert(name, points=points, wait=True)
        total += len(points)
    return total


def build_manifest(
    *,
    settings: Settings,
    doc_id: str,
    index_version: str,
    collection: str,
    chunks: Sequence[Chunk],
    parents: Sequence[Parent],
    vector_meta: dict[str, str],
    points: int,
    created_at: str,
) -> dict[str, Any]:
    composition: dict[str, int] = {}
    for c in chunks:
        kind = str(c.payload.get("chunk_kind", "unknown"))
        composition[kind] = composition.get(kind, 0) + 1
    return {
        "doc_id": doc_id,
        "index_version": index_version,
        "collection": collection,
        "alias": settings.index.alias,
        "created_at": created_at,
        "git_sha": git_sha(),
        "config_hash": settings.config_hash,
        # From the payload, like created_at: `--only index` may run under edited settings.
        "chunk_config_hash": str(chunks[0].payload.get("chunk_config_hash", settings.chunk_config_hash)),
        "models": {
            "embed_model": vector_meta.get("embed_model", settings.embed.model),
            "embed_dim": int(vector_meta.get("dim", settings.embed.dim)),
            "sparse_model": vector_meta.get("sparse_model", settings.embed.sparse_model),
            "rerank_model": settings.rerank.model,
        },
        "counts": {
            "chunks": len(chunks),
            "parents": len(parents),
            "points": points,
            "chunks_with_code": sum(1 for c in chunks if c.payload.get("has_code")),
        },
        "composition": composition,
        "cost_usd": 0.0,
    }


def _single_index_version(chunks: Sequence[Chunk]) -> str:
    versions = {str(c.payload.get("index_version", "")) for c in chunks}
    if len(versions) != 1 or "" in versions:
        raise RuntimeError(f"enriched chunks carry {len(versions)} index versions: {sorted(versions)[:3]}")
    return versions.pop()


@stage("ingest.index", kind="CHAIN")
def index(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/enriched"),
    vectors_dir: Path = Path("data/vectors"),
    parents_dir: Path = Path("data/parents"),
    index_dir: Path = Path("data/index"),
    client: QdrantClient | None = None,
) -> IndexResult:
    """Run Stage 7: a new collection, the parent store, and the manifest. The alias is untouched."""
    chunks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Chunk))
    parents = list(read_jsonl(parents_dir / f"{doc_id}.jsonl", Parent))
    rows, vector_meta = read_vectors(vectors_dir / f"{doc_id}.parquet")
    index_version = _single_index_version(chunks)
    collection = collection_name(index_version)
    dim = int(vector_meta.get("dim", settings.embed.dim))

    # Validate before writing anything. A failure here leaves no collection, no directory under
    # data/index (which would burn the next index version), and no store file.
    orphans = sorted({c.parent_id for c in chunks} - {p.parent_id for p in parents})
    if orphans:
        raise RuntimeError(f"{len(orphans)} chunks point at parents that do not exist, e.g. {orphans[:3]}")
    missing_rows = [c.chunk_id for c in chunks if c.chunk_id not in {r.chunk_id for r in rows}]
    if missing_rows:
        raise RuntimeError(f"{len(missing_rows)} chunks have no vector row, e.g. {missing_rows[:3]}")

    client = client or qdrant_client(settings)
    current = alias_target(client, settings.index.alias)
    if current == collection:
        raise RuntimeError(
            f"{collection} is the collection behind alias {settings.index.alias}; "
            "a new ingestion run must get a new index version"
        )
    create_collection(client, collection, dim)
    try:
        points = upsert_points(client, collection, chunks, rows)
        counted = client.count(collection, exact=True).count
        if counted != len(chunks):
            raise RuntimeError(f"collection {collection} holds {counted} points for {len(chunks)} chunks")
    except Exception:
        # Never leave a half-filled collection behind. It is not aliased, but it would linger.
        client.delete_collection(collection)
        raise

    version_dir = index_dir / index_version
    store_path = version_dir / STORE_FILE
    if store_path.exists():
        store_path.unlink()
    with ParentStore(store_path) as store:
        parents_stored = store.put_parents(parents)
        chunks_stored = store.put_chunks(chunks)
        if store.orphan_chunk_ids():
            raise AssertionError("orphans after the pre-check; the store write is inconsistent")

    manifest = build_manifest(
        settings=settings, doc_id=doc_id, index_version=index_version, collection=collection,
        chunks=chunks, parents=parents, vector_meta=vector_meta, points=points,
        created_at=str(chunks[0].payload.get("created_at", "")),
    )
    manifest_path = version_dir / MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return IndexResult(
        doc_id=doc_id,
        index_version=index_version,
        collection=collection,
        points_upserted=points,
        parents_stored=parents_stored,
        chunks_stored=chunks_stored,
        alias_switched=False,
        manifest_path=manifest_path,
        store_path=store_path,
    )
