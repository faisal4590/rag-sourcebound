"""Done-when tests for Stage 12, dense retrieval, and the M2 `flp ask`. Spec Section 6, Stage 12. Issue #15.

Synthetic tests index five chunks into QdrantClient(":memory:") with a fake embedder. The live
test asks the real question against the alias and skips without a server or an alias.
"""

import json
import re
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from flp_rag import cli, tracing
from flp_rag.contracts import Candidate, Chunk, read_jsonl, write_jsonl
from flp_rag.graph.nodes import retrieve as r
from flp_rag.ingest import s06_s07_index as s7
from flp_rag.settings import load_settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from tests.test_s06_s07_index import FakeEmbedder, _qdrant_up
from tests.test_s08_verify import CHAPTERS, _chunk, _parent

ROOT = Path(__file__).resolve().parents[1]


def _alias_up() -> bool:
    if not _qdrant_up():
        return False
    settings = load_settings()
    return s7.alias_target(s7.qdrant_client(settings), settings.index.alias) is not None


needs_live_alias = pytest.mark.skipif(not _alias_up(), reason="needs Qdrant with the alias switched")

CHUNKS = [
    _chunk(0, "Constructor promotion removes boilerplate.", CHAPTERS[0]),
    _chunk(1, "Promoted properties may have defaults.", CHAPTERS[0]),
    _chunk(2, "Readonly properties are set once.", CHAPTERS[1], parent="d:06:s001"),
    _chunk(3, "A readonly class makes every property readonly.", CHAPTERS[1], parent="d:06:s001"),
    _chunk(4, "Clone with readonly is an error before PHP 8.3.", CHAPTERS[1], parent="d:06:s001"),
]
CHUNKS = [Chunk(c.chunk_id, c.parent_id, c.display_text, c.embedding_text, c.token_count,
                {**c.payload, "has_code": c.chunk_id.endswith(("1", "3"))}) for c in CHUNKS]
PARENTS = [_parent("d:05:s000", CHAPTERS[0]), _parent("d:06:s001", CHAPTERS[1])]


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory):
    """An in-memory Qdrant with the five chunks indexed under a fake embedder."""
    tmp = tmp_path_factory.mktemp("data")
    settings = load_settings()
    settings = settings.model_copy(update={"embed": settings.embed.model_copy(update={"dim": 8, "batch_size": 2})})
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    write_jsonl(tmp / "enriched" / "d.jsonl", CHUNKS)
    write_jsonl(tmp / "parents" / "d.jsonl", PARENTS)
    embedder = FakeEmbedder()
    s7.embed("d", settings, in_dir=tmp / "enriched", out_dir=tmp / "vectors", cache_path=tmp / "cache.sqlite", embedder=embedder)
    client = QdrantClient(":memory:")
    result = s7.index("d", settings, in_dir=tmp / "enriched", vectors_dir=tmp / "vectors",
                      parents_dir=tmp / "parents", index_dir=tmp / "index", client=client)
    return settings, client, embedder, result.collection, tmp


# --------------------------------------------------------------------------- filters


def test_build_filter() -> None:
    assert r.build_filter(None) is None
    assert r.build_filter(r.RetrievalFilters()) is None
    f = r.build_filter(r.RetrievalFilters(chapter_no=(5, 6), has_code=True))
    assert f is not None and len(f.must) == 2
    assert f.must[0].key == "chapter_no" and f.must[0].match.any == [5, 6]
    assert f.must[1].key == "has_code" and f.must[1].match.value is True
    assert r.RetrievalFilters(chapter_no=(6,)).to_attrs() == {"rag.filter_chapters": "[6]"}


# --------------------------------------------------------------------------- retrieval


def test_retrieve_dense_returns_ranked_candidates_with_span_documents(indexed) -> None:
    settings, client, embedder, collection, _ = indexed
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)

    hits = r.retrieve_dense(CHUNKS[2].embedding_text, settings, client=client, embedder=embedder,
                            collection=collection)

    assert hits and hits[0].chunk_id == CHUNKS[2].chunk_id and hits[0].fused_score == pytest.approx(1.0, abs=1e-5)
    assert all(isinstance(h, Candidate) and h.provenance == ["dense"] and h.rerank_score is None for h in hits)
    assert [h.fused_score for h in hits] == sorted((h.fused_score for h in hits), reverse=True)
    assert len(hits) == 5 <= settings.retrieval.top_k_dense
    assert hits[0].payload["chapter_no"] == 6 and hits[0].payload["chunk_id"] == CHUNKS[2].chunk_id
    span = next(s for s in exporter.get_finished_spans() if s.name == "retrieval.dense")
    a = span.attributes
    assert a["openinference.span.kind"] == "RETRIEVER" and a["gen_ai.request.model"] == "fake/embedder"
    assert a["rag.top_k"] == settings.retrieval.top_k_dense and a["rag.n_hits"] == 5
    assert a["rag.collection"] == collection
    assert a["retrieval.documents.0.document.id"] == CHUNKS[2].chunk_id
    assert a["retrieval.documents.0.document.score"] == pytest.approx(1.0, abs=1e-5)
    meta = json.loads(a["retrieval.documents.0.document.metadata"])
    assert meta["chapter_no"] == 6 and meta["page_printed_start"] == 75


def test_filters_narrow_the_candidates(indexed) -> None:
    settings, client, embedder, collection, _ = indexed
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    only5 = r.retrieve_dense("anything", settings, client=client, embedder=embedder, collection=collection,
                             filters=r.RetrievalFilters(chapter_no=(5,)))
    assert {h.payload["chapter_no"] for h in only5} == {5} and len(only5) == 2
    code = r.retrieve_dense("anything", settings, client=client, embedder=embedder, collection=collection,
                            filters=r.RetrievalFilters(has_code=True))
    assert {h.chunk_id for h in code} == {CHUNKS[1].chunk_id, CHUNKS[3].chunk_id}
    top1 = r.retrieve_dense("anything", settings, client=client, embedder=embedder, collection=collection, top_k=1)
    assert len(top1) == 1


def test_query_vector_uses_and_fills_the_cache(indexed, tmp_path: Path) -> None:
    _, _, embedder, _, _ = indexed
    with EmbeddingCache(tmp_path / "q.sqlite") as cache:
        first = r.query_vector("What is readonly?", embedder, cache)
        stored = cache.get_query(embedder.model_name, "What is readonly?")
        assert stored == pytest.approx(first, abs=1e-6)  # the cache holds float32
        assert r.query_vector("What is readonly?", embedder, cache) == stored  # served from the cache
    assert r.query_vector("x", embedder, None) == embedder.embed_query("x")


def test_cli_paths_resolve_against_the_config_file(tmp_path: Path) -> None:
    settings = load_settings().model_copy(update={"config_path": tmp_path / "elsewhere" / "config.yaml"})
    assert cli.index_dir(settings) == tmp_path / "elsewhere" / "data" / "index"
    assert cli.query_cache_path(settings) == tmp_path / "elsewhere" / "data" / "cache" / "embeddings.sqlite"


def test_display_texts_come_from_the_parent_store(indexed) -> None:
    settings, client, embedder, collection, tmp = indexed
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    hits = r.retrieve_dense("readonly", settings, client=client, embedder=embedder, collection=collection, top_k=3)
    texts = r.display_texts(hits, index_dir=tmp / "index")
    assert set(texts) == {h.chunk_id for h in hits}
    assert all(texts[h.chunk_id] == next(c.display_text for c in CHUNKS if c.chunk_id == h.chunk_id) for h in hits)
    assert r.display_texts(hits, index_dir=tmp / "nowhere") == {}


# --------------------------------------------------------------------------- flp ask


def test_flp_ask_prints_chunks_with_chapter_pages_and_score(indexed, monkeypatch: pytest.MonkeyPatch) -> None:
    settings, client, embedder, collection, tmp = indexed
    monkeypatch.setattr(cli, "load_settings", lambda: settings.model_copy(
        update={"index": settings.index.model_copy(update={"alias": collection})}))
    monkeypatch.setattr(r, "qdrant_client", lambda _s: client)
    monkeypatch.setattr(r, "get_embeddings", lambda _s: embedder)
    monkeypatch.setattr(cli, "index_dir", lambda _s: tmp / "index")
    monkeypatch.setattr(cli, "query_cache_path", lambda _s: tmp / "qcache.sqlite")

    result = CliRunner().invoke(cli.app, ["ask", CHUNKS[2].embedding_text, "--top-k", "3"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("1.") and CHUNKS[2].chunk_id in lines[0]
    assert "Chapter 06" in lines[0] and "Readonly Properties" in lines[0] and "p. 75" in lines[0]
    assert "score=1.000" in lines[0]
    assert "Readonly properties are set once." in result.output
    assert "trace_id=" in result.output

    filtered = CliRunner().invoke(cli.app, ["ask", "anything", "--chapter", "5", "--code"])
    assert filtered.exit_code == 0 and "Chapter 05" in filtered.output and "Chapter 06" not in filtered.output

    empty = CliRunner().invoke(cli.app, ["ask", "anything", "--chapter", "99"])
    assert empty.exit_code == 2 and "no chunks" in empty.output


# --------------------------------------------------------------------------- live


@needs_live_alias
def test_flp_ask_readonly_finds_chapter_6_on_the_real_index() -> None:
    result = CliRunner().invoke(cli.app, ["ask", "What is a readonly property?", "--top-k", "5"])
    assert result.exit_code == 0, result.output
    # A model-loading progress bar can share the first line, so parse with a pattern.
    rows = re.findall(r"(\d+)\. (\S+) Chapter (\d+): .*? score=(\d\.\d+)", result.output)
    assert [int(n) for n, *_ in rows] == [1, 2, 3, 4, 5], result.output
    assert any(ch == "06" for _, _, ch, _ in rows[:3]), result.output
    chunk_ids = [cid for _, cid, _, _ in rows]
    enriched = {c.chunk_id for c in read_jsonl(ROOT / "data" / "enriched" / "bca146b9df2d0a2b.jsonl", Chunk)}
    assert set(chunk_ids) <= enriched
