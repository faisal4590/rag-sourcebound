# StateGraph wiring for Stages 9-18. The gate node returns "pass" or "abstain" as a conditional edge. compile() lives here.
"""The query graph.

    guard_input -> understand -> retrieve -> rerank -> gate -+-> assemble -> generate -> check_output -> respond
                        |                                     +-> respond (abstain)
                        +-> respond (smalltalk)

Each node is a plain function `RagState -> partial state`. `build_graph` wraps every node in
one `@stage` span with the name from spec Section 7.4, so the nodes themselves stay free of
tracing code. The root span `rag.request` is opened by the caller (Stage 9, or `flp ask`)
before `invoke`, so the node spans become its children.

`GraphNodes` names the nine node functions. Tests pass stubs. `default_nodes` wires the real
implementations that exist and marks the rest as Milestone 3 placeholders by issue number.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient

from flp_rag.contracts import Candidate, Response
from flp_rag.graph.nodes import assemble as assemble_mod
from flp_rag.graph.nodes import gate as gate_mod
from flp_rag.graph.nodes import generate as gen_mod
from flp_rag.graph.nodes import guard_input as guard_mod
from flp_rag.graph.nodes import retrieve as retrieve_mod
from flp_rag.graph.state import RagState
from flp_rag.models import ChatModel, Embedder, get_chat_model
from flp_rag.settings import ABSTAIN_TEXT, Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.tracing import current_span, stage

Node = Callable[[RagState], dict[str, Any]]

# node name -> (span name, openinference kind), spec Section 7.4
SPANS: dict[str, tuple[str, str]] = {
    "guard_input": ("guardrails.input", "CHAIN"),
    "understand": ("query.understand", "CHAIN"),
    "retrieve": ("retrieval.hybrid", "RETRIEVER"),
    "rerank": ("rerank", "RERANKER"),
    "gate": ("relevance.gate", "CHAIN"),
    "assemble": ("context.assemble", "CHAIN"),
    "generate": ("llm.generate", "LLM"),
    "check_output": ("guardrails.output", "CHAIN"),
    "respond": ("response.build", "CHAIN"),
}
NODE_ORDER = tuple(SPANS)


@dataclass(frozen=True)
class GraphNodes:
    guard_input: Node
    understand: Node
    retrieve: Node
    rerank: Node
    gate: Node
    assemble: Node
    generate: Node
    check_output: Node
    respond: Node

    def as_dict(self) -> dict[str, Node]:
        return {name: getattr(self, name) for name in NODE_ORDER}


# --------------------------------------------------------------------------- routing


def route_after_understand(state: RagState) -> str:
    """Smalltalk abstains before retrieval (spec Stage 11 rule 3)."""
    return "respond" if state.get("status") == "no_information" else "retrieve"


def route_after_gate(state: RagState) -> str:
    """`pass` continues (borderline is a flag on that path); `abstain` skips generation."""
    return "assemble" if state.get("gate_decision") == "pass" else "respond"


# --------------------------------------------------------------------------- build


def _traced(name: str, fn: Node) -> Node:
    """One spec span around a node. Sync and async nodes both work; Stage 16 streams async."""
    span_name, kind = SPANS[name]
    if inspect.iscoroutinefunction(fn):

        @stage(span_name, kind=kind)  # type: ignore[arg-type]
        async def async_node(state: RagState) -> dict[str, Any]:
            update = await fn(state) or {}
            return {**update, "visited": [name]}

        async_node.__name__ = name
        return async_node

    @stage(span_name, kind=kind)  # type: ignore[arg-type]
    def node(state: RagState) -> dict[str, Any]:
        update = fn(state) or {}
        return {**update, "visited": [name]}

    node.__name__ = name
    return node


def build_graph(nodes: GraphNodes) -> CompiledStateGraph:
    """Compile the query graph. Every node runs under its spec span."""
    graph: StateGraph = StateGraph(RagState)
    for name, fn in nodes.as_dict().items():
        graph.add_node(name, _traced(name, fn))
    graph.add_edge(START, "guard_input")
    graph.add_edge("guard_input", "understand")
    graph.add_conditional_edges(
        "understand", route_after_understand, {"retrieve": "retrieve", "respond": "respond"}
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "gate")
    graph.add_conditional_edges(
        "gate", route_after_gate, {"assemble": "assemble", "respond": "respond"}
    )
    graph.add_edge("assemble", "generate")
    graph.add_edge("generate", "check_output")
    graph.add_edge("check_output", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


# --------------------------------------------------------------------------- default nodes


@dataclass
class Deps:
    """Clients the nodes need. Tests inject fakes; production builds them from settings."""

    settings: Settings
    client: QdrantClient | None = None
    embedder: Embedder | None = None
    cache: EmbeddingCache | None = None
    collection: str | None = None
    store_path: Path | None = None  # data/index/{index_version}/store.sqlite, for Stage 15
    chat: ChatModel | None = None  # the answer model; built from gen.model on first use when None
    extra: dict[str, Any] = field(default_factory=dict)


def default_nodes(deps: Deps) -> GraphNodes:
    """Milestone 3 skeleton. Real: guard_input (#18), retrieve (dense, #15), gate (#19),
    assemble (#20), generate (#21). Stubs that later issues replace: understand passthrough
    (#27), rerank_score = fused_score (#26, so the gate runs on dense cosine until the reranker
    lands), check_output (#22), respond minimal (#23)."""
    s = deps.settings

    guard_input = guard_mod.make_guard_input(s)
    gate = gate_mod.make_gate(s)
    assemble = assemble_mod.make_assemble(s, deps.store_path)
    generate = gen_mod.make_generate(s, lambda: deps.chat or get_chat_model(s, "gen"))

    def understand(state: RagState) -> dict[str, Any]:
        from flp_rag.contracts import StandaloneQuery

        q = state["question"]
        return {
            "standalone": StandaloneQuery(
                raw=q,
                standalone=q,
                intent="book_question",
                wants_code=False,
                chapter_hint=None,
                variants=[q],
            )
        }

    def retrieve(state: RagState) -> dict[str, Any]:
        standalone = state.get("standalone")
        question = standalone.standalone if standalone else state["question"]
        candidates = retrieve_mod.retrieve_dense(
            question,
            s,
            filters=state.get("filters"),
            client=deps.client,
            embedder=deps.embedder,
            cache=deps.cache,
            collection=deps.collection,
        )
        return {"candidates": candidates}

    def rerank(state: RagState) -> dict[str, Any]:
        # TODO(#26): remove with the real reranker. The dense cosine stands in for the rerank
        # score; it can be negative for off-topic questions and the gate needs 0 to 1, so the
        # stub clamps. The thresholds belong to the reranker's scale: recalibrate at #26.
        stubbed = [
            Candidate(
                c.chunk_id,
                c.fused_score,
                min(1.0, max(0.0, c.fused_score)),
                c.provenance,
                c.payload,
            )
            for c in state.get("candidates", [])
        ]
        return {"candidates": stubbed[: s.rerank.top_n]}

    def check_output(state: RagState) -> dict[str, Any]:
        raise NotImplementedError("Stage 17 output checks is issue #22")

    def respond(state: RagState) -> dict[str, Any]:
        status = state.get("status") or ("answered" if state.get("answer") else "error")
        answer = state.get("answer") or ABSTAIN_TEXT
        response = Response(
            answer=answer,
            status=status,
            sources=list(state.get("sources", [])),  # type: ignore[arg-type]
            trace_id=state.get("trace_id", ""),
            timings_ms=dict(state.get("timings_ms", {})),
            model={"embed": s.embed.model},
            index_version="",
            prompt_version=state.get("prompt_version", ""),
            debug={
                "visited": [*state.get("visited", []), "respond"],
                "abstain_reason": state.get("abstain_reason"),
                **state.get("debug_info", {}),
            }
            if state.get("debug")
            else None,
        )
        span = current_span()
        span.set_attribute("rag.status", status)
        span.set_attribute("rag.n_sources", len(response.sources))
        span.set_attribute("rag.answer_chars", len(answer))
        return {"status": status, "answer": answer, "response": response}

    return GraphNodes(
        guard_input, understand, retrieve, rerank, gate, assemble, generate, check_output, respond
    )


def compile_default(deps: Deps) -> CompiledStateGraph:
    """The graph the API and the CLI run. Built once at startup."""
    return build_graph(default_nodes(deps))
