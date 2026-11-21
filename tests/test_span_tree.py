# Issue #29, Parts 3-7: every request path and the ingestion run against the span registry.
"""Real nodes, fake models, an in-memory a fake Qdrant. Each request goes through the
API so the root span is the real `rag.request`. The checker of `span_registry.py` compares the
trace with the registry: the exact span set of the path, kind and parent of every span, every
required attribute, no null, the size limits, the content switch, and the cost sum."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from flp_rag import span_registry as reg
from flp_rag import tracing
from flp_rag.api import app as api
from flp_rag.contracts import Chapter, Chunk, Parent, ParsedPage, write_jsonl
from flp_rag.graph import build
from flp_rag.ingest import run as ingest_run
from flp_rag.ingest import s01_parse
from flp_rag.ingest import s06_s07_index as s7
from flp_rag.ingest import s08_verify as s8
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.answer_cache import AnswerCache
from flp_rag.stores.parent_store import ParentStore
from tests.test_check_output import FakeJudge
from tests.test_generate import FakeChat
from tests.test_hybrid_retrieval import FakeClient, FakeSparse
from tests.test_rerank import FakeReranker, payload
from tests.test_s02_structure import STA, W, prose_line
from tests.test_s06_s07_index import FakeEmbedder
from tests.test_understand import FULL, FakeQueryModel

REG = reg.load_registry()
SUPPORTED = {"supported": True, "unsupported_claims": []}
SMALLTALK = {
    **FULL,
    "standalone": "hi there",
    "intent": "smalltalk",
    "chapter_hint": None,
    "paraphrases": [],
    "keywords": "",
}
ANSWER = "Readonly properties are set once [S1]."


class LoopingClient(FakeClient):
    """The fake Qdrant of the retrieval tests, serving the same lists on every request."""

    def query_batch_points(self, collection: str, requests: list[Any]) -> list[Any]:
        dense, sparse = [list(x) for x in self.dense], [list(x) for x in self.sparse]
        out = super().query_batch_points(collection, requests)
        self.dense, self.sparse = dense, sparse
        return out


def hits(*ids: str) -> list[tuple[str, float, dict[str, Any]]]:
    return [
        (cid, 0.9 - 0.1 * i, payload(cid, 6, "Readonly", f"Sec {cid}", 70, 71))
        for i, cid in enumerate(ids)
    ]


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tree") / "store.sqlite"
    chunks = [
        Chunk(
            f"c{i}",
            "p6",
            f"Display text of chunk {i}.",
            f"chunk {i}",
            5,
            payload(f"c{i}", 6, "Readonly", f"Sec c{i}", 70, 71),
        )
        for i in range(6)
    ]
    with ParentStore(path) as store:
        store.put_parents(
            [Parent("p6", 6, "Readonly", "Readonly properties are set once.", 70, 71, 5)]
        )
        store.put_chunks(chunks)
    return path


def settings_for(*, capture_content: bool = True, cache: bool = False) -> Settings:
    s = load_settings()
    return s.model_copy(
        update={
            "gen": s.gen.model_copy(update={"retry_backoff_s": 0.0}),
            "query": s.query.model_copy(update={"retry_backoff_s": 0.0}),
            "output": s.output.model_copy(update={"faithfulness_sample_rate": 1.0}),
            "trace": s.trace.model_copy(update={"capture_content": capture_content}),
            "cache": s.cache.model_copy(update={"semantic_enabled": cache}),
        }
    )


def make_deps(
    settings: Settings,
    store_path: Path,
    *,
    chat: FakeChat,
    judge: FakeJudge,
    query: FakeQueryModel,
    reranker: FakeReranker,
    answers: AnswerCache | None = None,
) -> build.Deps:
    return build.Deps(
        settings=settings,
        client=LoopingClient(
            dense=[hits("c0", "c1", "c2", "c3")] * 5, sparse=[hits("c1", "c4"), hits("c2")]
        ),
        embedder=FakeEmbedder(),
        sparse=FakeSparse(),
        reranker=reranker,
        store_path=store_path,
        chat=chat,
        judge=judge,
        query_chat=query,
        answers=answers,
        collection="flp_chunks_v-test",
    )


def ask(deps: build.Deps, question: str, **body: Any) -> tuple[int, list[reg.SpanView]]:
    """One request through the API; the views of its trace. A fresh exporter each time: a
    forced reconfigure shuts the old provider down, and with it the exporter it had."""
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(deps.settings, exporter=exporter, force=True)
    app = api.create_app(
        deps.settings, deps=deps, graph_factory=build.compile_default, configure_tracing=False
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/ask", json={"question": question, **body})
    trace_id = r.headers["X-Trace-Id"]
    views = [v for v in reg.from_readable(exporter.get_finished_spans()) if v.trace_id == trace_id]
    return r.status_code, views


def good(settings: Settings, store_path: Path, **over: Any) -> build.Deps:
    kw: dict[str, Any] = {
        "chat": FakeChat(ANSWER),
        "judge": FakeJudge(SUPPORTED),
        "query": FakeQueryModel(FULL),
        "reranker": FakeReranker({"chunk 0": 0.95, "chunk 1": 0.6}),
    }
    kw.update(over)
    return make_deps(settings, store_path, **kw)


def names(views: Sequence[reg.SpanView]) -> list[str]:
    return [v.name for v in views]


# --------------------------------------------------------------------------- the five paths


def test_answered_path_matches_the_registry(store_path: Path) -> None:
    status, views = ask(good(settings_for(), store_path), "How do readonly properties work?")
    assert status == 200
    assert reg.check_trace(REG, views, "answered", capture_content=True) == []
    assert reg.cost_mismatch(views) is None
    assert names(views).count("retrieval.hybrid") == 1  # one span, not the node plus the stage
    root = next(v for v in views if v.parent is None)
    assert (
        root.attributes["rag.status"] == "answered"
        and root.attributes["rag.abstain_reason"] == "none"
    )
    assert root.attributes["cache.hit"] is False and root.attributes["rag.session_id"] == "none"
    rerank = next(v for v in views if v.name == "rerank")
    assert rerank.attributes["reranker.output_documents.0.document.id"] == "c0"
    assert "reranker.output_documents.0.document.content" in rerank.attributes
    dense = next(v for v in views if v.name == "retrieval.dense")
    assert dense.attributes["retrieval.documents.0.document.id"] == "c0"


def test_gate_abstain_path_has_no_context_generation_or_output_spans(store_path: Path) -> None:
    status, views = ask(
        good(settings_for(), store_path, reranker=FakeReranker({"chunk 0": 0.05})),
        "What is the boiling point of water?",
    )
    assert status == 200
    assert reg.check_trace(REG, views, "gate_abstain", capture_content=True) == []
    assert {"context.assemble", "llm.generate", "guardrails.output"}.isdisjoint(names(views))
    gate = next(v for v in views if v.name == "relevance.gate")
    assert gate.attributes["rag.abstain_reason"] == "gate.low_top_score"
    root = next(v for v in views if v.parent is None)
    assert root.attributes["rag.abstain_reason"] == "gate.low_top_score"


def test_smalltalk_path_stops_before_retrieval(store_path: Path) -> None:
    status, views = ask(
        good(settings_for(), store_path, query=FakeQueryModel(SMALLTALK)), "hi there"
    )
    assert status == 200
    assert reg.check_trace(REG, views, "smalltalk", capture_content=True) == []
    understand = next(v for v in views if v.name == "query.understand")
    assert understand.attributes["rag.chapter_hint"] == -1  # the sentinel, never a lost key


def test_repair_path_has_two_generation_spans_and_the_cost_sum(store_path: Path) -> None:
    status, views = ask(
        good(settings_for(), store_path, chat=FakeChat("bad [S9].", ANSWER)),
        "How do readonly properties work?",
    )
    assert status == 200
    assert reg.check_trace(REG, views, "repair", capture_content=True) == []
    assert names(views).count("llm.generate") == 2
    assert reg.cost_mismatch(views) is None
    root = next(v for v in views if v.parent is None)
    llm = [v for v in views if v.kind == "LLM"]
    assert len(llm) == 4  # classify, two generations, the judge
    assert root.attributes["rag.cost_usd"] == pytest.approx(
        sum(v.attributes["rag.cost_usd"] for v in llm)
    )


def test_unfaithful_path_withdraws_the_answer(store_path: Path) -> None:
    verdict = {"supported": False, "unsupported_claims": ["readonly properties can be unset"]}
    status, views = ask(
        good(settings_for(), store_path, judge=FakeJudge(verdict)),
        "How do readonly properties work?",
    )
    assert status == 200
    assert reg.check_trace(REG, views, "unfaithful", capture_content=True) == []
    out = next(v for v in views if v.name == "guardrails.output")
    assert (
        out.attributes["rag.faithful"] is False
        and out.attributes["rag.abstain_reason"] == "output.unfaithful"
    )
    assert "rag.rejected_answer" in out.attributes  # content, the switch is on


def test_generate_error_path_ends_at_the_failing_span_with_status_error(store_path: Path) -> None:
    status, views = ask(
        good(settings_for(), store_path, chat=FakeChat(TimeoutError("t"), TimeoutError("t"))),
        "How do readonly properties work?",
    )
    assert status == 500
    assert reg.check_trace(REG, views, "generate_error", capture_content=True) == []
    generate = next(v for v in views if v.name == "llm.generate")
    assert generate.status == "ERROR"
    assert "guardrails.output" not in names(views) and "response.build" not in names(views)
    root = next(v for v in views if v.parent is None)
    assert (
        root.attributes["rag.status"] == "error" and root.attributes["rag.abstain_reason"] == "none"
    )


def test_guard_reject_path_is_root_and_the_guard_span(store_path: Path) -> None:
    status, views = ask(good(settings_for(), store_path), "x" * 2001)
    assert status == 400
    assert reg.check_trace(REG, views, "guard_reject", capture_content=True) == []
    assert names(views) == ["rag.request", "guardrails.input"]
    assert views[1].status == "ERROR"
    assert views[0].attributes["rag.abstain_reason"] == "guard.rejected"


def test_cache_hit_path_answers_after_the_lookup(store_path: Path, tmp_path: Path) -> None:
    settings = settings_for(cache=True)
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers:
        deps = good(settings, store_path, answers=answers)
        status, first = ask(deps, "How do readonly properties work?")
        assert status == 200 and answers.count() == 1
        assert (
            reg.check_trace(REG, first, "answered", capture_content=True, cache_enabled=True) == []
        )
        deps = good(settings, store_path, answers=answers)
        status, hit = ask(deps, "How do readonly properties work?")
    assert status == 200
    assert reg.check_trace(REG, hit, "cache_hit", capture_content=True) == []
    root = next(v for v in hit if v.parent is None)
    assert root.attributes["cache.hit"] is True and root.attributes["cache.similarity"] > 0.97
    assert reg.cost_mismatch(hit) is None and root.attributes["rag.cost_usd"] == 0.0


# --------------------------------------------------------------------------- content off


def test_content_switch_off_leaves_no_content_attribute_on_any_span(store_path: Path) -> None:
    status, views = ask(
        good(settings_for(capture_content=False), store_path),
        "How do readonly properties work?",
    )
    assert status == 200
    assert reg.check_trace(REG, views, "answered", capture_content=False) == []
    keys = {k for v in views for k in v.attributes}
    assert not any(REG.is_content(k) for k in keys), sorted(k for k in keys if REG.is_content(k))
    rerank = next(v for v in views if v.name == "rerank")
    assert rerank.attributes["rag.score_top1"] == pytest.approx(0.95)  # scores and counts stay
    tracing.configure_tracing(load_settings(), exporter=InMemorySpanExporter(), force=True)


# --------------------------------------------------------------------------- the ingestion tree

CHAPTERS = [
    Chapter(0, 5, "Property Promotion", "Part 1", "PHP, the Language", 65, 66),
    Chapter(1, 6, "Readonly Properties", "Part 1", "PHP, the Language", 67, 68),
]
SENTENCES = [
    "Constructor promotion removes the boilerplate of declaring and assigning properties by hand.",
    "A promoted property is declared in the constructor signature with a visibility keyword.",
    "Readonly properties can be assigned once and any later write raises an error at runtime.",
    "The readonly modifier requires a typed property and works with promoted properties too.",
]


def synthetic_page(doc_id: str, page_pdf: int, chapter: Chapter, first: bool) -> ParsedPage:
    words, top = [], 80.0
    if first:
        words += [
            W(t, 65.2 + i * 90, top, font=STA, size=34.0, page=page_pdf)
            for i, t in enumerate(chapter.title.split())
        ]
        top += 60
    for i in range(10):
        words += prose_line(SENTENCES[(i + page_pdf) % len(SENTENCES)], top, page=page_pdf)
        top += 14.0
    return ParsedPage(doc_id, page_pdf, page_pdf - 2, 2, words)


def test_ingestion_run_matches_the_registry_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stages 2-8 are real; Stage 1 is a fake under the same span that writes synthetic pages
    (no PDF in the test suite); the embedder, the sparse model, and Qdrant are fakes."""
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    doc_id = s01_parse.compute_doc_id(pdf)

    @tracing.stage("ingest.parse", kind="CHAIN")
    def fake_parse(
        pdf_path: Path, settings: Settings, *, out_dir: Path, index_dir: Path, force: bool = False
    ) -> s01_parse.ParseResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        pages = [
            synthetic_page(doc_id, p, ch, p == ch.page_pdf_start)
            for ch in CHAPTERS
            for p in range(ch.page_pdf_start, ch.page_pdf_end + 1)
        ]
        write_jsonl(out_dir / f"{doc_id}.jsonl", pages)
        write_jsonl(out_dir / f"{doc_id}.chapters.jsonl", CHAPTERS)
        return s01_parse.ParseResult(
            doc_id,
            len(pages),
            len(pages),
            sum(len(p.words) for p in pages),
            8,
            [],
            2,
            out_dir / f"{doc_id}.jsonl",
            out_dir / f"{doc_id}.chapters.jsonl",
        )

    smoke = tmp_path / "smoke.jsonl"
    smoke.write_text(
        '{"query": "constructor promotion boilerplate", "expected_chapter": 5}\n'
        '{"query": "readonly modifier typed property", "expected_chapter": 6}\n'
    )
    s = load_settings()
    s = s.model_copy(
        update={
            "embed": s.embed.model_copy(
                update={
                    "dim": 8,
                    "batch_size": 2,
                    "retry_backoff_s": 0.0,
                    "cache_path": str(tmp_path / "cache.sqlite"),
                }
            ),
            "verify": s.verify.model_copy(update={"smoke_queries": smoke}),
        }
    )
    client = QdrantClient(":memory:")
    monkeypatch.setattr(s01_parse, "parse", fake_parse)
    monkeypatch.setattr(s7, "get_embeddings", lambda settings: FakeEmbedder())
    monkeypatch.setattr(s7, "get_sparse_embeddings", lambda settings: FakeSparse())
    monkeypatch.setattr(s7, "qdrant_client", lambda settings: client)
    monkeypatch.setattr(s8, "qdrant_client", lambda settings: client)
    monkeypatch.setattr(s8, "get_embeddings", lambda settings: FakeEmbedder())
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(s, exporter=exporter, force=True)

    results = ingest_run.run(s, pdf_path=pdf, data_dir=tmp_path / "data")

    views = reg.from_readable(exporter.get_finished_spans())
    assert {v.name for v in views} == set(REG.ingest)
    problems = [p for v in views for p in reg.check_span(REG, v, capture_content=True)]
    assert problems == []
    root = next(v for v in views if v.parent is None)
    assert (
        root.attributes["rag.result"] in ("ok", "verify_failed")
        and root.attributes["rag.cost_usd"] == 0.0
    )
    batches = [v for v in views if v.name == "embed.batch"]
    assert batches and all(
        v.attributes["rag.cost_usd"] == 0.0 for v in batches
    )  # local model: 0, not missing
    assert results["chunk"].chunks_total >= 2
    tracing.configure_tracing(load_settings(), exporter=InMemorySpanExporter(), force=True)


# --------------------------------------------------------------------------- the two rerank branches


def test_rerank_disabled_and_no_candidates_keep_every_rerank_attribute(store_path: Path) -> None:
    """The Milestone 3 stub (issue #28) and the empty list set the same keys as a model call."""
    s = load_settings()
    disabled = settings_for().model_copy(
        update={"rerank": s.rerank.model_copy(update={"enabled": False})}
    )
    status, views = ask(good(disabled, store_path), "How do readonly properties work?")
    assert status == 200 and reg.check_trace(REG, views, "answered", capture_content=True) == []
    rerank = next(v for v in views if v.name == "rerank")
    assert rerank.attributes["rag.rerank_enabled"] is False
    assert rerank.attributes["rag.device"] == "none"
    assert rerank.attributes["gen_ai.request.model"] == "none"
    assert "reranker.input_documents.0.document.id" in rerank.attributes

    deps = good(settings_for(), store_path)
    deps.client = LoopingClient(dense=[[]] * 5, sparse=[[], []])  # nothing found
    status, views = ask(deps, "How do readonly properties work?")
    assert status == 200 and reg.check_trace(REG, views, "gate_abstain", capture_content=True) == []
    rerank = next(v for v in views if v.name == "rerank")
    assert rerank.attributes["rag.n_in"] == 0 and rerank.attributes["rag.scores_all"] == "[]"
    gate = next(v for v in views if v.name == "relevance.gate")
    assert gate.attributes["rag.abstain_reason"] == "gate.no_candidates"
