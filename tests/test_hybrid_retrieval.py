# "Done when" test for Stage 12 hybrid retrieval: sparse BM25, variants, RRF. Issue #25.
"""Fusion and the sibling rule are plain functions of result lists. The retrievers run against
a fake Qdrant client that records every batch request, and a fake sparse model that records
which mode it was asked for. One test uses the in-memory Qdrant of `tests/test_retrieve.py`
with the real tiny BM25 model for the spans; one live test needs the alias."""

import json
from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import models
from qdrant_client.http import models as http_models

from flp_rag import tracing
from flp_rag.contracts import Candidate, StandaloneQuery
from flp_rag.graph import build
from flp_rag.graph.nodes import retrieve as r
from flp_rag.graph.state import initial_state
from flp_rag.models import SparseVector
from flp_rag.settings import load_settings
from tests.test_retrieve import CHUNKS, indexed, needs_live_alias  # noqa: F401 - fixture
from tests.test_s06_s07_index import FakeEmbedder

Hit = tuple[str, float, dict[str, Any]]


def hits(*rows: tuple[str, float]) -> list[r.Hit]:
    return [r.Hit(cid, score, {"chunk_id": cid, "chapter_no": 6}) for cid, score in rows]


def lst(retriever: str, variant: int, *rows: tuple[str, float]) -> r.ResultList:
    return r.ResultList(retriever, variant, hits(*rows))  # type: ignore[arg-type]


class FakeSparse:
    """Records the mode. Query vectors carry a weight of 1 per term, like fastembed's query mode."""

    model_name = "fake/bm25"

    def __init__(self) -> None:
        self.query_calls: list[list[str]] = []
        self.doc_calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        self.doc_calls.append(list(texts))
        return [SparseVector([1, 2], [0.5, 0.25]) for _ in texts]

    def embed_query(self, texts: Sequence[str]) -> list[SparseVector]:
        self.query_calls.append(list(texts))
        return [SparseVector([abs(hash(t)) % 1000], [1.0]) for t in texts]


class FakeClient:
    """A Qdrant stand-in: scripted hit lists per retriever, served in request order."""

    def __init__(
        self,
        dense: Sequence[Sequence[Hit]] = (),
        sparse: Sequence[Sequence[Hit]] = (),
        groups: dict[str, Sequence[Hit]] | None = None,
        fail: Exception | None = None,
    ) -> None:
        self.dense, self.sparse = [list(x) for x in dense], [list(x) for x in sparse]
        self.groups = groups or {}
        self.fail = fail
        self.calls: list[tuple[str, list[models.QueryRequest]]] = []
        self.scrolls: list[models.Filter] = []

    def query_batch_points(self, collection: str, requests: list[models.QueryRequest]) -> list[Any]:
        self.calls.append((collection, list(requests)))
        if self.fail is not None:
            raise self.fail
        out = []
        for req in requests:
            source = self.dense if req.using == "dense" else self.sparse
            rows = source.pop(0) if source else []
            out.append(
                http_models.QueryResponse(
                    points=[
                        models.ScoredPoint(id=i, version=0, score=s, payload={"chunk_id": cid, **p})
                        for i, (cid, s, p) in enumerate(rows)
                    ]
                )
            )
        return out

    def scroll(
        self, collection: str, *, scroll_filter: models.Filter, limit: int, with_payload: bool
    ):
        self.scrolls.append(scroll_filter)
        group = scroll_filter.must[0].match.value  # type: ignore[union-attr]
        rows = self.groups.get(group, [])
        records = [
            models.Record(id=i, payload={"chunk_id": cid, **p})
            for i, (cid, _s, p) in enumerate(rows)
        ]
        return records, None


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def run_hybrid(client: FakeClient, settings, variants: r.Variants | None = None, **kw: Any):
    sparse = kw.pop("sparse", FakeSparse())
    embedder = kw.pop("embedder", FakeEmbedder())
    out = r.retrieve_hybrid(
        variants or r.Variants("How do readonly properties work?"),
        settings,
        client=client,
        embedder=embedder,
        sparse=sparse,
        collection="c",
        **kw,
    )
    return out, sparse, embedder


# --------------------------------------------------------------------------- fusion


def test_rrf_arithmetic_of_example_4(settings) -> None:
    dense = lst("dense", 0, ("A", 0.9), ("B", 0.8), ("E", 0.7))
    sparse = lst("sparse", 0, ("C", 12.0), ("D", 11.0), ("A", 10.0))
    fused = r.fuse([dense, sparse], rrf_k=60, top_k=30)
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["A"].fused_score == pytest.approx(1 / 61 + 1 / 63)
    assert by_id["B"].fused_score == pytest.approx(1 / 62)
    assert fused[0].chunk_id == "A" and fused.index(by_id["B"]) > fused.index(by_id["A"])
    assert [c.fused_score for c in fused] == sorted((c.fused_score for c in fused), reverse=True)
    assert len(fused) == 5 and len({c.chunk_id for c in fused}) == 5  # unique chunk ids


def test_ties_break_by_best_rank_then_chunk_id() -> None:
    # Y and X each sit at rank 1 in one list: equal points, equal best rank, id order wins.
    fused = r.fuse(
        [lst("dense", 0, ("Y", 0.9), ("Z", 0.8)), lst("sparse", 0, ("X", 5.0), ("Z", 4.0))],
        rrf_k=60,
        top_k=30,
    )
    assert [c.chunk_id for c in fused] == ["Z", "X", "Y"]  # Z: 2/62 > 1/61
    # Same points, different best rank: the better single rank wins.
    a_lists = [
        lst("dense", 0, ("P", 0.9), ("Q", 0.8)),
        lst("sparse", 0, ("Q", 5.0), ("R", 4.0), ("P", 3.0)),
    ]
    fused = r.fuse(a_lists, rrf_k=60, top_k=30)
    assert fused[0].chunk_id == "Q"  # 1/62 + 1/61 vs P: 1/61 + 1/63
    assert fused[0].fused_score > fused[1].fused_score


def test_provenance_and_scores_record_every_source() -> None:
    fused = r.fuse(
        [lst("dense", 0, ("B", 0.9), ("A", 0.8)), lst("sparse", 2, ("A", 12.0))], rrf_k=60, top_k=30
    )
    a = next(c for c in fused if c.chunk_id == "A")
    assert a.provenance == ["dense:v0:r2", "sparse:v2:r1"]
    assert a.scores == {"dense": 0.8, "sparse": 12.0} and a.rerank_score is None
    assert json.loads(a.to_attrs()["rag.scores_all"]) == {"dense": 0.8, "sparse": 12.0}


def test_top_k_cuts_the_fused_list() -> None:
    fused = r.fuse(
        [lst("dense", 0, *((f"c{i}", 1 - i / 100) for i in range(40)))], rrf_k=60, top_k=30
    )
    assert len(fused) == 30 and fused[0].chunk_id == "c0"


def test_overlap_ratio_counts_chunks_both_retrievers_found() -> None:
    fused = r.fuse(
        [lst("dense", 0, ("A", 0.9), ("B", 0.8)), lst("sparse", 0, ("A", 5.0), ("C", 4.0))],
        rrf_k=60,
        top_k=30,
    )
    assert r.overlap_ratio(fused) == pytest.approx(1 / 3)
    assert r.overlap_ratio([]) == 0.0


# --------------------------------------------------------------------------- variants


def test_variants_come_from_stage_11_or_the_raw_question() -> None:
    v = r.variants_from(None, "How do readonly properties work?")
    assert (
        v.dense_queries == ["How do readonly properties work?"]
        and v.sparse_queries == v.dense_queries
    )
    sq = StandaloneQuery(
        raw="raw",
        standalone="standalone",
        intent="book_question",
        wants_code=False,
        chapter_hint=9,
        variants=["standalone", "para one", "para two", "standalone"],
        keyword_variant="readonly clone",
    )
    v = r.variants_from(sq, "raw")
    assert v.standalone == "standalone" and v.paraphrases == ["para one", "para two"]
    assert v.keyword == "readonly clone"
    assert v.dense_queries == ["standalone", "para one", "para two"]  # no keyword variant for dense
    assert v.sparse_queries == ["standalone", "para one", "para two", "readonly clone"]
    assert v.count == 4
    assert r.Variants("q", ["p"], "p").sparse_queries == [
        "q",
        "p",
    ]  # a duplicate keyword is dropped


def test_one_variant_gives_one_dense_and_one_sparse_list_in_two_batches(settings) -> None:
    client = FakeClient(
        dense=[[("A", 0.9, {}), ("B", 0.8, {})]], sparse=[[("B", 9.0, {}), ("C", 8.0, {})]]
    )
    out, sparse, embedder = run_hybrid(client, settings)
    assert [c.chunk_id for c in out][:1] == ["B"] and len(out) == 3
    assert len(client.calls) == 2 and [len(reqs) for _, reqs in client.calls] == [1, 1]
    dense_req, sparse_req = client.calls[0][1][0], client.calls[1][1][0]
    assert dense_req.using == "dense" and dense_req.limit == settings.retrieval.top_k_dense
    assert sparse_req.using == "sparse_bm25" and sparse_req.limit == settings.retrieval.top_k_sparse
    assert isinstance(sparse_req.query, models.SparseVector) and sparse_req.query.values == [1.0]
    assert sparse.query_calls == [["How do readonly properties work?"]] and sparse.doc_calls == []
    assert embedder.query_batches == [["How do readonly properties work?"]]


def test_five_variants_use_one_embedding_batch_and_one_qdrant_batch_per_retriever(settings) -> None:
    v = r.Variants("q", ["p1", "p2", "p3"], "kw")
    client = FakeClient(dense=[[("A", 0.9, {})]] * 4, sparse=[[("A", 1.0, {})]] * 5)
    out, sparse, embedder = run_hybrid(client, settings, variants=v)
    assert [len(reqs) for _, reqs in client.calls] == [4, 5]
    assert embedder.query_batches == [["q", "p1", "p2", "p3"]]
    assert sparse.query_calls == [["q", "p1", "p2", "p3", "kw"]]
    (a,) = out
    assert a.fused_score == pytest.approx(9 / 61) and len(a.provenance) == 9


def test_an_empty_sparse_list_is_not_an_error(settings) -> None:
    client = FakeClient(dense=[[("A", 0.9, {}), ("B", 0.8, {})]], sparse=[[]])
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    out, _, _ = run_hybrid(client, settings)
    assert [c.chunk_id for c in out] == ["A", "B"] and all(":v0:" in c.provenance[0] for c in out)
    span = next(s for s in exporter.get_finished_spans() if s.name == "retrieval.hybrid")
    assert span.attributes["rag.n_sparse_hits"] == 0 and span.attributes["rag.n_dense_hits"] == 2
    assert span.attributes["rag.overlap_ratio"] == 0.0


def test_code_group_siblings_join_with_the_members_score(settings) -> None:
    piece2 = ("g1:p2", 0.9, {"code_group_id": "g1", "part_index": 2})
    client = FakeClient(
        dense=[[piece2, ("B", 0.5, {})]],
        sparse=[[]],
        groups={"g1": [("g1:p1", 0.0, {"code_group_id": "g1", "part_index": 1}), piece2]},
    )
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    out, _, _ = run_hybrid(client, settings)
    by_id = {c.chunk_id: c for c in out}
    assert set(by_id) == {"g1:p2", "B", "g1:p1"}
    assert by_id["g1:p1"].fused_score == by_id["g1:p2"].fused_score
    assert by_id["g1:p1"].provenance == ["code_group"]
    assert by_id["g1:p1"].scores == by_id["g1:p2"].scores  # inherits, so the gate sees it
    assert len(client.scrolls) == 1
    span = next(s for s in exporter.get_finished_spans() if s.name == "retrieval.hybrid")
    assert span.attributes["rag.n_siblings_added"] == 1 and span.attributes["rag.n_fused"] == 3


def test_sibling_rule_can_exceed_top_k() -> None:
    members = [
        Candidate(f"c{i}", 1.0 - i / 100, None, ["dense:v0:r1"], {"chunk_id": f"c{i}"})
        for i in range(30)
    ]
    members[-1] = Candidate(
        "g:p2", 0.5, None, ["dense:v0:r30"], {"chunk_id": "g:p2", "code_group_id": "g"}
    )
    client = FakeClient(
        groups={"g": [("g:p1", 0, {"code_group_id": "g"}), ("g:p2", 0, {"code_group_id": "g"})]}
    )
    out, added = r.add_code_group_siblings(members, client, "c")  # type: ignore[arg-type]
    assert added == 1 and len(out) == 31 and out[-1].chunk_id == "g:p1"


def test_request_filters_reach_every_list_and_the_chapter_hint_never(settings) -> None:
    v = r.Variants("q", ["p1"], "kw")
    client = FakeClient(dense=[[]] * 2, sparse=[[]] * 3)
    run_hybrid(
        client, settings, variants=v, filters=r.RetrievalFilters(chapter_no=(6, 30), has_code=True)
    )
    for _, reqs in client.calls:
        for req in reqs:
            assert req.filter is not None and req.filter.must[0].match.any == [6, 30]
            assert req.filter.must[1].match.value is True
    client = FakeClient(dense=[[]], sparse=[[]])
    sq = StandaloneQuery("q", "q", "book_question", False, 9, ["q"])
    run_hybrid(client, settings, variants=r.variants_from(sq, "q"), filters=None)
    assert all(req.filter is None for _, reqs in client.calls for req in reqs)


def test_a_stop_word_variant_sends_no_sparse_request(settings) -> None:
    class StopWordSparse(FakeSparse):
        def embed_query(self, texts):  # type: ignore[no-untyped-def]
            self.query_calls.append(list(texts))
            return [SparseVector([], []) if t == "the" else SparseVector([7], [1.0]) for t in texts]

    client = FakeClient(dense=[[("A", 0.9, {})]] * 2, sparse=[[("B", 3.0, {})]])
    out, _, _ = run_hybrid(
        client, settings, variants=r.Variants("q", ["the"]), sparse=StopWordSparse()
    )
    assert [len(reqs) for _, reqs in client.calls] == [2, 1]  # one sparse request, not two
    b = next(c for c in out if c.chunk_id == "B")
    assert b.provenance == ["sparse:v0:r1"]


def test_one_scroll_per_code_group(settings) -> None:
    rows = [(f"g:p{i}", 0.9 - i / 10, {"code_group_id": "g", "part_index": i}) for i in (1, 2, 3)]
    client = FakeClient(dense=[rows], sparse=[[]], groups={"g": rows})
    run_hybrid(client, settings)
    assert len(client.scrolls) == 1


def test_a_failing_qdrant_raises_and_never_abstains(settings) -> None:
    client = FakeClient(fail=ConnectionError("qdrant down"))
    with pytest.raises(ConnectionError, match="qdrant down"):
        run_hybrid(client, settings)


# --------------------------------------------------------------------------- query mode


def test_fastembed_query_mode_has_a_weight_of_one_per_term() -> None:
    from flp_rag.models import FastembedSparse

    sparse = FastembedSparse("Qdrant/bm25")
    try:
        (query,) = sparse.embed_query(["readonly properties readonly"])
    except Exception as exc:  # noqa: BLE001 - no model cache, no network: skip
        pytest.skip(f"Qdrant/bm25 is not available: {exc}")
    assert set(query.values) == {1.0} and len(query.indices) == len(set(query.indices))
    (doc,) = sparse.embed(["readonly properties readonly"])
    assert max(doc.values) != 1.0 or len(doc.values) != len(
        query.values
    )  # document mode weighs terms


# --------------------------------------------------------------------------- spans with real BM25


def test_spans_and_attributes_on_the_in_memory_index(indexed) -> None:  # noqa: F811
    settings, client, embedder, collection, _ = indexed
    from flp_rag.models import get_sparse_embeddings

    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    out = r.retrieve_hybrid(
        r.Variants("readonly properties"),
        settings,
        client=client,
        embedder=embedder,
        sparse=get_sparse_embeddings(settings),
        collection=collection,
    )
    assert out and {c.chunk_id for c in out} <= {c.chunk_id for c in CHUNKS}
    spans = {s.name: s for s in exporter.get_finished_spans()}
    parent = spans["retrieval.hybrid"]
    assert parent.attributes["openinference.span.kind"] == "RETRIEVER"
    for child in ("retrieval.dense", "retrieval.sparse", "retrieval.fuse"):
        assert spans[child].parent.span_id == parent.context.span_id, child
    a = parent.attributes
    assert a["rag.n_variants"] == 1 and a["rag.n_dense_hits"] == 5 and a["rag.n_sparse_hits"] >= 2
    assert a["rag.n_fused"] == len(out) and 0.0 <= a["rag.overlap_ratio"] <= 1.0
    assert a["rag.top_fused_score"] == pytest.approx(out[0].fused_score)
    assert a["retrieval.documents.0.document.id"] == out[0].chunk_id
    assert json.loads(a["retrieval.documents.0.document.metadata"])["chapter_no"] in (5, 6)
    d = spans["retrieval.dense"].attributes
    assert d["gen_ai.request.model"] == "fake/embedder" and d["rag.n_queries"] == 1
    assert (
        json.loads(d["rag.hits_per_list"]) == [5]
        and len(json.loads(d["rag.top_score_per_list"])) == 1
    )
    s = spans["retrieval.sparse"].attributes
    assert s["rag.sparse_model"] == settings.embed.sparse_model and s["rag.n_queries"] == 1
    f = spans["retrieval.fuse"].attributes
    assert f["rag.rrf_k"] == settings.retrieval.rrf_k and f["rag.n_in"] >= f["rag.n_out"] == len(
        out
    )
    # BM25 ranks a readonly chunk first: the rare word outweighs "properties" in Chapter 5.
    first_sparse = next(c for c in out if "sparse:v0:r1" in c.provenance)
    assert first_sparse.payload["chapter_no"] == 6


# --------------------------------------------------------------------------- graph


def test_rerank_stub_reads_the_dense_score_until_the_reranker_lands(settings) -> None:
    nodes = build.default_nodes(build.Deps(settings=settings))
    state = initial_state("q", request_id="r1")
    state["candidates"] = [
        Candidate(
            "A", 0.03, None, ["dense:v0:r1", "sparse:v0:r2"], {}, {"dense": 0.72, "sparse": 9.0}
        ),
        Candidate("B", 0.02, None, ["sparse:v0:r1"], {}, {"sparse": 12.0}),
        Candidate("C", 0.01, None, ["dense:v0:r2"], {}, {"dense": -0.1}),
    ]
    out = nodes.rerank(state)["candidates"]
    assert [c.rerank_score for c in out] == [0.72, 0.0, 0.0]


# --------------------------------------------------------------------------- live


@pytest.mark.live
@needs_live_alias
def test_live_readonly_question_hits_chapter_6_in_both_lists() -> None:
    from flp_rag.models import get_embeddings, get_sparse_embeddings

    settings = load_settings()
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    out = r.retrieve_hybrid(
        r.Variants("How do readonly properties work?"),
        settings,
        embedder=get_embeddings(settings),
        sparse=get_sparse_embeddings(settings),
    )
    assert len(out) == settings.retrieval.top_k_fused
    dense_top10 = [
        c
        for c in out
        if any(p.startswith("dense:v0:r") and int(p.rsplit("r", 1)[1]) <= 10 for p in c.provenance)
    ]
    sparse_top10 = [
        c
        for c in out
        if any(p.startswith("sparse:v0:r") and int(p.rsplit("r", 1)[1]) <= 10 for p in c.provenance)
    ]
    assert any(c.payload["chapter_no"] == 6 for c in dense_top10)
    assert any(c.payload["chapter_no"] == 6 for c in sparse_top10)
    assert 0.0 < r.overlap_ratio(out) < 1.0
