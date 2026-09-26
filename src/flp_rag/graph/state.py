# RagState TypedDict: question, standalone, intent, variants, candidates, context, answer, status, abstain_reason, trace_id.
"""The state that flows through the query graph (Stages 10-18).

Every node receives the whole state and returns only the keys it changed. `visited` uses an
append reducer, so the list records the path a request took; the debug field of the response
and the tests read it.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from flp_rag.contracts import Candidate, ContextBlock, Response, Source, StandaloneQuery
from flp_rag.graph.nodes.retrieve import RetrievalFilters

Status = Literal["answered", "no_information", "error"]
GateDecision = Literal["pass", "borderline", "abstain"]

Turn = tuple[Literal["user", "assistant"], str]


class RagState(TypedDict, total=False):
    # request (Stage 9)
    request_id: str
    session_id: str | None
    trace_id: str
    question: str
    history: list[Turn]
    filters: RetrievalFilters | None
    stream: bool
    debug: bool
    # guardrails (Stage 10)
    injection_suspected: bool
    language: str
    # query understanding (Stage 11)
    standalone: StandaloneQuery | None
    # retrieval and reranking (Stages 12-13)
    candidates: list[Candidate]
    # gate (Stage 14)
    gate_decision: GateDecision
    borderline: bool
    top_score: float
    # context, generation, checks (Stages 15-17)
    context: list[ContextBlock]
    answer: str
    # response (Stage 18)
    status: Status
    abstain_reason: str | None
    sources: list[Source]
    timings_ms: dict[str, int]
    cost_usd: float
    response: Response | None  # what Stage 18 hands to the API and the CLI
    debug_info: dict[str, Any]  # only filled when the request has debug=true
    # path taken, appended by every node
    visited: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    *,
    request_id: str,
    trace_id: str = "",
    session_id: str | None = None,
    history: list[Turn] | None = None,
    filters: RetrievalFilters | None = None,
    stream: bool = False,
    debug: bool = False,
) -> RagState:
    return RagState(
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        question=question,
        history=list(history or []),
        filters=filters,
        stream=stream,
        debug=debug,
        candidates=[],
        context=[],
        answer="",
        abstain_reason=None,
        sources=[],
        timings_ms={},
        cost_usd=0.0,
        response=None,
        debug_info={},
        visited=[],
    )
