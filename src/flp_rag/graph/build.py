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
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient

from flp_rag.contracts import Candidate, Response
from flp_rag.graph.nodes import guard_input as guard_mod
from flp_rag.graph.nodes import retrieve as retrieve_mod
from flp_rag.graph.state import RagState
from flp_rag.models import Embedder
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
    """`pass` and `borderline` continue; `abstain` skips generation (spec Stage 14)."""
    return "assemble" if state.get("gate_decision") in ("pass", "borderline") else "respond"


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
    extra: dict[str, Any] = field(default_factory=dict)


def default_nodes(deps: Deps) -> GraphNodes:
    """Milestone 3 skeleton. Real: guard_input (#18), retrieve (dense, #15). Stubs that later
    issues replace: understand passthrough (#27), rerank_score = fused_score (#26), a gate on
    the stub score with the configured thresholds (#19), assemble (#20), generate (#21),
    check_output (#22), respond minimal (#23)."""
    s = deps.settings

    guard_input = guard_mod.make_guard_input(s)

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
        stubbed = [
            Candidate(c.chunk_id, c.fused_score, c.fused_score, c.provenance, c.payload)
            for c in state.get("candidates", [])
        ]
        return {"candidates": stubbed[: s.rerank.top_n]}

    def gate(state: RagState) -> dict[str, Any]:
        # Placeholder for #19: the spec's decision table on the stub score.
        scores = [c.rerank_score or 0.0 for c in state.get("candidates", [])]
        top = max(scores, default=0.0)
        g = s.gate
        n_pass = sum(1 for x in scores if x >= g.tau_pass)
        if top < g.tau_low:
            decision, reason = "abstain", "gate.low_top_score"
        elif n_pass == 0:
            decision, reason = "abstain", "gate.no_passing_chunk"
        elif top < g.tau_high:
            decision, reason = "borderline", None
        else:
            decision, reason = "pass", None
        span = current_span()
        for key, value in (
            ("rag.gate_decision", decision),
            ("rag.score_top1", top),
            ("rag.n_pass", n_pass),
            ("rag.tau_low", g.tau_low),
            ("rag.tau_pass", g.tau_pass),
            ("rag.tau_high", g.tau_high),
            ("rag.borderline", decision == "borderline"),
        ):
            span.set_attribute(key, value)
        if reason:
            span.set_attribute("rag.abstain_reason", reason)
            return {
                "gate_decision": "abstain",
                "borderline": False,
                "top_score": top,
                "status": "no_information",
                "answer": ABSTAIN_TEXT,
                "abstain_reason": reason,
            }
        return {"gate_decision": decision, "borderline": decision == "borderline", "top_score": top}

    def assemble(state: RagState) -> dict[str, Any]:
        raise NotImplementedError("Stage 15 context assembly is issue #20")

    def generate(state: RagState) -> dict[str, Any]:
        raise NotImplementedError("Stage 16 generation is issue #21")

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
            prompt_version="",
            debug={
                "visited": [*state.get("visited", []), "respond"],
                "abstain_reason": state.get("abstain_reason"),
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
