"""Done-when tests for Stage 6, embedding. Spec Section 5, Stage 6. Issue #12.

Synthetic tests use a fake dense embedder and the real, tiny BM25 sparse model. The done-when
test embeds the whole book with bge-m3 and skips when the PDF or the model is absent.
"""

import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Chunk, read_jsonl, write_jsonl
from flp_rag.ingest import s06_s07_index as s6
from flp_rag.settings import load_settings
from flp_rag.stores.embedding_cache import EmbeddingCache, cache_key

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "front-line-php-revised-for-php-82.pdf"
ENRICHED = ROOT / "data" / "enriched" / "bca146b9df2d0a2b.jsonl"


def _cached(repo: str) -> bool:
    return os.path.isdir(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}"))


needs_book = pytest.mark.skipif(
    not (PDF.exists() and ENRICHED.exists() and _cached("BAAI/bge-m3")),
    reason="needs the PDF, a Stage 5 run in data/enriched/, and bge-m3 in the HF cache",
)


class FakeEmbedder:
    """Deterministic unit vectors from the text hash. Counts calls. Can fail on demand."""

    model_name = "fake/embedder"
    dim = 8

    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("simulated outage")
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            raw = [b / 255.0 - 0.5 for b in digest[: self.dim]]
            norm = math.sqrt(sum(x * x for x in raw))
            out.append([x / norm * 3.0 for x in raw])  # not unit length on purpose
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]


def _chunk(i: int, text: str, symbols: list[str] | None = None) -> Chunk:
    return Chunk(chunk_id=f"d:05:{i:04d}", parent_id="p", display_text=text,
                 embedding_text=f"hdr {i}\n{text}", token_count=len(text.split()),
                 payload={"chapter_no": 5, "has_code": bool(symbols), "php_symbols": symbols or []})


@pytest.fixture
def fast_settings():
    s = load_settings()
    return s.model_copy(update={"embed": s.embed.model_copy(update={"batch_size": 2, "retry_backoff_s": 0.0})})


@pytest.fixture
def chunks_dir(tmp_path: Path) -> Path:
    chunks = [
        _chunk(0, "Readonly properties can be set once.", ["readonly"]),
        _chunk(1, "Enums are backed by int or string.", ["enum"]),
        _chunk(2, "Match is an expression, switch is a statement.", ["match"]),
        _chunk(3, "The JIT compiles hot code paths."),
        _chunk(4, "Fibers give cooperative multitasking."),
    ]
    write_jsonl(tmp_path / "enriched" / "d.jsonl", chunks)
    return tmp_path


# --------------------------------------------------------------------------- cache


def test_cache_round_trip_and_key_scoping(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "c.sqlite") as cache:
        assert cache.get_many("m", ["a", "b"]) == {}
        assert cache.put_many("m", [("a", [1.0, 2.0]), ("b", [3.0, 4.0])]) == 2
        got = cache.get_many("m", ["a", "b", "zzz"])
        assert got == {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        assert cache.get_many("other-model", ["a"]) == {}  # key includes the model name
        assert cache.count() == 2
        cache.put_query("m", "what is readonly?", [0.5, 0.5])
        assert cache.get_query("m", "what is readonly?") == [0.5, 0.5]
        assert cache.get_query("m", "something else") is None
    with EmbeddingCache(tmp_path / "c.sqlite") as cache:  # persists across reopen
        assert cache.count() == 2 and cache.count("queries") == 1
    assert cache_key("m", "a") != cache_key("m", "b") and len(cache_key("m", "a")) == 64


# --------------------------------------------------------------------------- helpers


def test_with_retries_recovers_then_gives_up() -> None:
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("down")
        return "ok"

    assert s6.with_retries(flaky, attempts=3, backoff_s=0.0, what="x") == "ok"
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        s6.with_retries(lambda: (_ for _ in ()).throw(ValueError("no")), attempts=2, backoff_s=0.0, what="x")


def test_self_retrieval_check_flags_duplicates_only() -> None:
    a, b, c = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    assert s6.self_retrieval_failures([a, b, c]) == []
    assert s6.self_retrieval_failures([a, b, a]) == [0, 2]
    assert s6.self_retrieval_failures([]) == []


def test_sparse_text_appends_symbols() -> None:
    assert s6.sparse_text(_chunk(0, "text", ["readonly", "enum"])) == "text\nreadonly enum"
    assert s6.sparse_text(_chunk(0, "text")) == "text"


# --------------------------------------------------------------------------- the stage


def test_embed_writes_parquet_normalizes_and_uses_the_cache(chunks_dir: Path, fast_settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(fast_settings, exporter=exporter, force=True)
    fake = FakeEmbedder()

    first = s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
                     cache_path=chunks_dir / "cache.sqlite", embedder=fake)

    assert (first.vectors_total, first.cache_hits, first.cache_misses, first.batches) == (5, 0, 5, 3)
    assert fake.calls == 3 and first.dense_dim == 8 and first.embed_model == "fake/embedder"
    assert first.sparse_model == "Qdrant/bm25" and first.self_retrieval_failures == 0
    rows, meta = s6.read_vectors(first.output_path)
    assert [r.chunk_id for r in rows] == [f"d:05:{i:04d}" for i in range(5)]
    assert all(abs(math.sqrt(sum(x * x for x in r.dense)) - 1.0) < 1e-5 for r in rows)
    assert all(len(r.sparse.indices) == len(r.sparse.values) > 0 for r in rows)
    assert meta["embed_model"] == "fake/embedder" and meta["dim"] == "8" and meta["doc_id"] == "d"
    batch_spans = [s for s in exporter.get_finished_spans() if s.name == "embed.batch"]
    assert len(batch_spans) == 3
    assert batch_spans[0].attributes["rag.batch_size"] == 2 and "rag.batch_tokens" in batch_spans[0].attributes
    parent = next(s for s in exporter.get_finished_spans() if s.name == "ingest.embed")
    assert parent.attributes["rag.cache_hits"] == 0 and parent.attributes["rag.vectors_total"] == 5

    second = s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
                      cache_path=chunks_dir / "cache.sqlite", embedder=fake)
    assert (second.cache_hits, second.cache_misses, second.batches) == (5, 0, 0)
    assert fake.calls == 3  # zero new model calls
    rows2, _ = s6.read_vectors(second.output_path)
    assert [r.dense for r in rows2] == [r.dense for r in rows]


def test_cache_hits_count_chunks_not_distinct_texts(tmp_path: Path, fast_settings) -> None:
    tracing.configure_tracing(fast_settings, exporter=InMemorySpanExporter(), force=True)
    twins = [_chunk(0, "same text"), _chunk(1, "same text"), _chunk(2, "other")]
    write_jsonl(tmp_path / "enriched" / "d.jsonl", twins)
    kwargs = {"in_dir": tmp_path / "enriched", "out_dir": tmp_path / "vectors", "cache_path": tmp_path / "c.sqlite"}
    fake = FakeEmbedder()
    first = s6.embed("d", fast_settings, embedder=fake, **kwargs)
    second = s6.embed("d", fast_settings, embedder=fake, **kwargs)
    assert first.cache_hits + first.cache_misses == 3
    assert (second.cache_hits, second.cache_misses) == (3, 0)


def test_batch_spans_do_not_carry_the_settings_tree(chunks_dir: Path, fast_settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(fast_settings, exporter=exporter, force=True)
    s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
             cache_path=chunks_dir / "cache.sqlite", embedder=FakeEmbedder())
    batch = next(s for s in exporter.get_finished_spans() if s.name == "embed.batch")
    assert "tau_pass" not in str(batch.attributes.get("input.value", ""))
    assert len(str(batch.attributes.get("input.value", ""))) < 2000


def test_wrong_dimension_from_a_stale_cache_row_fails(chunks_dir: Path, fast_settings) -> None:
    tracing.configure_tracing(fast_settings, exporter=InMemorySpanExporter(), force=True)
    chunks = list(read_jsonl(chunks_dir / "enriched" / "d.jsonl", Chunk))
    with EmbeddingCache(chunks_dir / "cache.sqlite") as cache:
        cache.put_many("fake/embedder", [(chunks[2].embedding_text, [1.0, 0.0, 0.0])])  # 3 dims, not 8
    with pytest.raises(RuntimeError, match="do not have 8 dimensions"):
        s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
                 cache_path=chunks_dir / "cache.sqlite", embedder=FakeEmbedder())


def test_embed_retries_then_succeeds(chunks_dir: Path, fast_settings) -> None:
    tracing.configure_tracing(fast_settings, exporter=InMemorySpanExporter(), force=True)
    fake = FakeEmbedder(fail_times=2)
    result = s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
                      cache_path=chunks_dir / "cache.sqlite", embedder=fake)
    assert result.vectors_total == 5 and fake.calls == 5  # 2 failures + 3 batches


def test_embed_fails_fast_and_writes_nothing_partial(chunks_dir: Path, fast_settings) -> None:
    tracing.configure_tracing(fast_settings, exporter=InMemorySpanExporter(), force=True)
    fake = FakeEmbedder(fail_times=99)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        s6.embed("d", fast_settings, in_dir=chunks_dir / "enriched", out_dir=chunks_dir / "vectors",
                 cache_path=chunks_dir / "cache.sqlite", embedder=fake)
    assert not (chunks_dir / "vectors" / "d.parquet").exists()
    with EmbeddingCache(chunks_dir / "cache.sqlite") as cache:
        assert cache.count() == 0


# --------------------------------------------------------------------------- real book


@needs_book
def test_every_chunk_finds_itself_with_bge_m3(tmp_path: Path) -> None:
    settings = load_settings()
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    doc_id = "bca146b9df2d0a2b"
    chunks = list(read_jsonl(ENRICHED, Chunk))

    result = s6.embed(doc_id, settings, in_dir=ENRICHED.parent, out_dir=tmp_path / "vectors",
                      cache_path=tmp_path / "cache.sqlite")

    assert result.vectors_total == len(chunks) == 279
    assert result.dense_dim == 1024 and result.embed_model == "BAAI/bge-m3"
    assert result.self_retrieval_failures == 0
    rows, meta = s6.read_vectors(result.output_path)
    assert meta["dim"] == "1024" and len(rows) == 279
    again = s6.embed(doc_id, settings, in_dir=ENRICHED.parent, out_dir=tmp_path / "vectors",
                     cache_path=tmp_path / "cache.sqlite")
    assert again.cache_hits == 279 and again.cache_misses == 0
