"""Done-when tests for the query graph wiring. Issue #16.

A graph compiled with stub nodes must reach `respond` on every route: pass, abstain at the gate,
smalltalk at understand, and unfaithful at the output check. Node spans must be children of the
caller's `rag.request` span with the spec's names.
"""

import json
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient

from flp_rag import tracing
from flp_rag.contracts import Candidate
from flp_rag.graph import build
from flp_rag.graph.state import RagState, initial_state
from flp_rag.settings import ABSTAIN_TEXT, load_settings

# --------------------------------------------------------------------------- stub nodes


def stub_nodes(*, gate: str = "pass", smalltalk: bool = False, unfaithful: bool = False) -> build.GraphNodes:
    def guard_input(state: RagState) -> dict[str, Any]:
        return {"injection_suspected": False, "language": "en"}

    def understand(state: RagState) -> dict[str, Any]:
        if smalltalk:
            return {"status": "no_information", "answer": ABSTAIN_TEXT, "abstain_reason": "query.smalltalk"}
        return {}

    def retrieve(state: RagState) -> dict[str, Any]:
        return {"candidates": [Candidate("d:06:0001", 0.7, None, ["dense"], {"chapter_no": 6})]}

    def rerank(state: RagState) -> dict[str, Any]:
        return {"candidates": [Candidate(c.chunk_id, c.fused_score, c.fused_score, c.provenance, c.payload)
                               for c in state["candidates"]]}

    def gate_node(state: RagState) -> dict[str, Any]:
        if gate == "abstain":
            return {"gate_decision": "abstain", "status": "no_information", "answer": ABSTAIN_TEXT,
                    "abstain_reason": "gate.low_top_score", "top_score": 0.1}
        return {"gate_decision": "pass", "borderline": gate == "borderline", "top_score": 0.7}

    def assemble(state: RagState) -> dict[str, Any]:
        label_map = {"S1": {"chunk_ids": ["d:06:0001"], "parent_id": "p", "kind": "chunk",
                            "chapter_no": 6, "chapter_title": "Readonly Properties",
                            "section_title": "Readonly", "page_printed_start": 75,
                            "page_printed_end": 77}}
        return {"context": [], "label_map": label_map}

    def generate(state: RagState) -> dict[str, Any]:
        return {"answer": "Readonly properties are set once [S1]."}

    def check_output(state: RagState) -> dict[str, Any]:
        if unfaithful:
            return {"check_result": "abstain", "status": "no_information", "answer": ABSTAIN_TEXT,
                    "abstain_reason": "output.unfaithful"}
        return {"check_result": "ok", "status": "answered"}

    def respond(state: RagState) -> dict[str, Any]:
        return {"status": state.get("status") or "answered"}

    return build.GraphNodes(guard_input, understand, retrieve, rerank, gate_node, assemble, generate, check_output, respond)


def run(nodes: build.GraphNodes, question: str = "How do readonly properties work?", **kw: Any) -> RagState:
    return build.build_graph(nodes).invoke(initial_state(question, request_id="r1", **kw))


# --------------------------------------------------------------------------- routes


def test_pass_route_visits_every_node_in_order() -> None:
    state = run(stub_nodes())
    assert state["visited"] == list(build.NODE_ORDER)
    assert state["status"] == "answered" and state["answer"].endswith("[S1].")


def test_borderline_also_continues_to_generation() -> None:
    state = run(stub_nodes(gate="borderline"))
    assert state["borderline"] is True and "generate" in state["visited"]


def test_abstain_at_the_gate_skips_generation() -> None:
    state = run(stub_nodes(gate="abstain"))
    assert state["visited"] == ["guard_input", "understand", "retrieve", "rerank", "gate", "respond"]
    assert state["status"] == "no_information" and state["answer"] == ABSTAIN_TEXT
    assert state["abstain_reason"] == "gate.low_top_score"


def test_smalltalk_abstains_before_retrieval() -> None:
    state = run(stub_nodes(smalltalk=True), question="hi there")
    assert state["visited"] == ["guard_input", "understand", "respond"]
    assert state["answer"] == ABSTAIN_TEXT and state["abstain_reason"] == "query.smalltalk"


def test_unfaithful_answer_abstains_at_the_output_check() -> None:
    state = run(stub_nodes(unfaithful=True))
    assert state["visited"][-2:] == ["check_output", "respond"]
    assert state["status"] == "no_information" and state["abstain_reason"] == "output.unfaithful"


def test_compiled_graph_has_the_nine_nodes_and_conditional_edges() -> None:
    g = build.build_graph(stub_nodes()).get_graph()
    assert set(g.nodes) == {"__start__", "__end__", *build.NODE_ORDER}
    edges = {(e.source, e.target) for e in g.edges}
    assert {("understand", "retrieve"), ("understand", "respond"), ("gate", "assemble"), ("gate", "respond"),
            ("check_output", "respond"), ("check_output", "generate"),
            ("respond", "__end__")} <= edges
    assert ("understand", "assemble") not in edges


def test_routing_functions() -> None:
    assert build.route_after_understand({"status": "no_information"}) == "respond"
    assert build.route_after_understand({}) == "retrieve"
    assert build.route_after_gate({"gate_decision": "pass"}) == "assemble"
    assert build.route_after_gate({"gate_decision": "pass", "borderline": True}) == "assemble"
    assert build.route_after_gate({"gate_decision": "abstain"}) == "respond"


# --------------------------------------------------------------------------- spans


def test_node_spans_are_children_of_the_request_span_with_spec_names() -> None:
    settings = load_settings()
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    app = build.build_graph(stub_nodes())

    @tracing.stage("rag.request", kind="CHAIN")
    def request() -> RagState:
        return app.invoke(initial_state("q", request_id="r1"))

    request()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["rag.request"]
    expected = [build.SPANS[n][0] for n in build.NODE_ORDER]
    assert set(expected) <= set(spans)
    for name in expected:
        assert spans[name].parent is not None and spans[name].parent.span_id == root.context.span_id
    assert spans["retrieval.hybrid"].attributes["openinference.span.kind"] == "RETRIEVER"
    assert spans["llm.generate"].attributes["openinference.span.kind"] == "LLM"
    starts = [spans[n].start_time for n in expected]
    assert starts == sorted(starts)


def test_abstain_route_has_no_generation_spans() -> None:
    settings = load_settings()
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    build.build_graph(stub_nodes(gate="abstain")).invoke(initial_state("q", request_id="r1"))
    names = {s.name for s in exporter.get_finished_spans()}
    assert {"relevance.gate", "response.build"} <= names
    assert not {"context.assemble", "llm.generate", "guardrails.output"} & names


def test_async_nodes_are_awaited_under_their_span() -> None:
    import asyncio

    nodes = stub_nodes()

    async def async_generate(state: RagState) -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"answer": "streamed answer [S1]."}

    app = build.build_graph(build.GraphNodes(**{**nodes.as_dict(), "generate": async_generate}))
    state = asyncio.run(app.ainvoke(initial_state("q", request_id="r1")))
    assert state["answer"] == "streamed answer [S1]." and "generate" in state["visited"]


def test_live_stub_nodes_write_spec_attributes() -> None:
    settings = load_settings()
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    deps = build.Deps(settings=settings)
    nodes = build.default_nodes(deps)
    stubs = stub_nodes()
    # Real guard, gate, respond from default_nodes; the rest from the stubs (no Qdrant needed).
    mixed = build.GraphNodes(**{**stubs.as_dict(), "guard_input": nodes.guard_input, "gate": nodes.gate,
                                "respond": nodes.respond})
    build.build_graph(mixed).invoke(initial_state("How do readonly properties work?", request_id="r1"))
    spans = {s.name: s.attributes for s in exporter.get_finished_spans()}
    assert spans["guardrails.input"]["guard.chars"] == len("How do readonly properties work?")
    g = spans["relevance.gate"]
    assert g["rag.gate_decision"] == "pass" and g["rag.score_top1"] == pytest.approx(0.7)
    assert (g["rag.tau_low"], g["rag.tau_pass"], g["rag.tau_high"]) == (0.2, 0.35, 0.6) and g["rag.n_pass"] == 1
    r = spans["response.build"]
    assert r["rag.status"] == "answered" and r["rag.n_sources"] == 1 and r["rag.answer_chars"] > 0


def test_instrument_langchain_reattaches_to_a_new_provider() -> None:
    pytest.importorskip("openinference.instrumentation.langchain")
    from langchain_core.runnables import RunnableLambda

    settings = load_settings()
    first, second = InMemorySpanExporter(), InMemorySpanExporter()
    p1 = tracing.configure_tracing(settings, exporter=first, force=True)
    assert tracing.instrument_langchain(p1)
    RunnableLambda(lambda x: x).invoke(1)
    p2 = tracing.configure_tracing(settings, exporter=second, force=True)
    assert tracing.instrument_langchain(p2)
    RunnableLambda(lambda x: x).invoke(2)
    assert second.get_finished_spans(), "LangChain spans must follow the new provider"
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().uninstrument()


# --------------------------------------------------------------------------- default nodes


def test_default_nodes_run_real_retrieval_and_abstain_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The M3 skeleton: dense retrieval and the stub gate work end to end on the abstain route,
    which needs none of the placeholder nodes."""
    import tempfile
    from pathlib import Path

    from flp_rag.contracts import write_jsonl
    from flp_rag.ingest import s06_s07_index as s7
    from tests.test_retrieve import CHUNKS, PARENTS
    from tests.test_s06_s07_index import FakeEmbedder

    tmp = Path(tempfile.mkdtemp())
    settings = load_settings()
    settings = settings.model_copy(update={
        "embed": settings.embed.model_copy(update={"dim": 8, "batch_size": 2}),
        # The fake reranker gives 1.0 to the exact-text chunk and 0.1 to the rest.
        "gate": settings.gate.model_copy(update={"tau_low": 0.5, "tau_pass": 0.9, "tau_high": 0.95}),
    })
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    write_jsonl(tmp / "enriched" / "d.jsonl", CHUNKS)
    write_jsonl(tmp / "parents" / "d.jsonl", PARENTS)
    embedder = FakeEmbedder()
    s7.embed("d", settings, in_dir=tmp / "enriched", out_dir=tmp / "vectors", cache_path=tmp / "c.sqlite", embedder=embedder)
    client = QdrantClient(":memory:")
    indexed = s7.index("d", settings, in_dir=tmp / "enriched", vectors_dir=tmp / "vectors",
                       parents_dir=tmp / "parents", index_dir=tmp / "index", client=client)
    from tests.test_generate import FakeChat
    from tests.test_rerank import FakeReranker

    class QuoteReranker(FakeReranker):
        """1.0 when the question quotes the chunk, else 0.1: only such a question passes the gate."""

        def score(self, query, texts):  # type: ignore[no-untyped-def]
            self.calls.append((query, list(texts)))
            return [1.0 if query.strip() and query.strip() in t else 0.1 for t in texts]

    reranker = QuoteReranker()
    # Stage 11 on a scripted small model: the classification and the variants, no network.
    understanding = json.dumps({"standalone": "", "intent": "book_question", "wants_code": False,
                                "chapter_hint": None, "paraphrases": [], "keywords": "readonly"})
    deps = build.Deps(settings=settings, client=client, embedder=embedder, collection=indexed.collection,
                      store_path=indexed.store_path, chat=FakeChat("Readonly is set once [S1]."),
                      query_chat=FakeChat(understanding, understanding), reranker=reranker)
    app = build.compile_default(deps)

    # Any question that does not quote a chunk scores 0.1 at the fake reranker: the gate abstains.
    state = app.invoke(initial_state("completely unrelated question about cooking", request_id="r1", debug=True))
    assert state["visited"] == ["guard_input", "understand", "retrieve", "rerank", "gate", "respond"]
    # Every candidate that reached the gate carries a reranker score in 0..1.
    assert state["candidates"] and all(0.0 <= c.rerank_score <= 1.0 for c in state["candidates"])
    assert len(state["candidates"]) <= settings.rerank.top_n
    assert state["status"] == "no_information" and state["answer"] == ABSTAIN_TEXT
    assert state["abstain_reason"].startswith("gate.")
    response = state["response"]
    assert response.status == "no_information" and response.debug["visited"][-1] == "respond"

    # The exact text of a chunk scores 1.0 at the reranker and passes the gate; Stage 15 runs on
    # the store (plain text, the fixture has no clean blocks), Stage 16 on the fake chat, and
    # Stage 17 on the label map (the judge is not sampled at rate 0).
    deps.settings = settings.model_copy(update={
        "output": settings.output.model_copy(update={"faithfulness_sample_rate": 0.0})})
    app = build.compile_default(deps)
    state = app.invoke(initial_state(CHUNKS[2].display_text, request_id="r2"))
    assert state["visited"][-4:] == ["assemble", "generate", "check_output", "respond"]
    assert state["check_result"] == "ok" and state["status"] == "answered"
    assert state["response"].answer == "Readonly is set once [S1]." and "S1" in state["label_map"]
