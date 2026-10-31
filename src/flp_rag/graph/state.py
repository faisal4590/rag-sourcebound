# RagState TypedDict: question, standalone, intent, variants, candidates, context, answer, status, abstain_reason, trace_id.
"""The state that flows through the query graph (Stages 10-18).

Every node receives the whole state and returns only the keys it changed. `visited` uses an
append reducer, so the list records the path a request took; the debug field of the response
and the tests read it.
"""

import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from flp_rag.contracts import Candidate, ContextBlock, Response, Source, StandaloneQuery
from flp_rag.graph.nodes.retrieve import RetrievalFilters

Status = Literal["answered", "no_information", "error"]
GateDecision = Literal["pass", "abstain"]  # borderline is the flag below
CheckResult = Literal["ok", "repair", "abstain"]  # Stage 17's edge value

Turn = tuple[Literal["user", "assistant"], str]


def sum_dicts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Reducer for `timings_ms`: a node that runs twice (the repair loop) adds up."""
    return {key: left.get(key, 0) + right.get(key, 0) for key in [*left, *right]}


class RagState(TypedDict, total=False):
    # request (Stage 9)
    request_id: str
    session_id: str | None
    trace_id: str
    index_version: str  # the index behind the alias when the request started (Stage 9)
    started_at: float  # perf_counter at request start, for the end-to-end latency
    question: str
    history: list[Turn]
    filters: RetrievalFilters | None
    stream: bool
    debug: bool
    judge_all: bool  # the evaluation harness sets it: the faithfulness judge reads every answer
    run_tag: str | None  # a label for a batch of debug requests (issue #28); on the root span
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
    n_pass: int  # candidates at or above gate.tau_pass, the ones Stage 15 may use
    # context, generation, checks (Stages 15-17)
    context: list[ContextBlock]
    label_map: dict[str, dict[str, Any]]  # label -> chunk ids, chapter, section, pages (Stage 15)
    answer: str
    prompt_version: str  # sha256(prompt file)[:8] of the prompt that produced the answer
    repair_calls: int  # model repair calls spent so far (budget gen.max_repair_calls)
    check_result: CheckResult
    repair_failure: str | None  # Stage 17 -> 16: the line that names the failed check
    rejected_answer: str | None  # the answer Stage 17 sent back or replaced
    # response (Stage 18)
    status: Status
    abstain_reason: str | None
    sources: list[Source]
    timings_ms: Annotated[dict[str, int], sum_dicts]  # span name -> ms, summed per node that ran
    input_tokens: int  # totals over every model call of the request
    output_tokens: int
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
    judge_all: bool = False,
    index_version: str = "",
    run_tag: str | None = None,
) -> RagState:
    return RagState(
        request_id=request_id,
        session_id=session_id,
        trace_id=trace_id,
        index_version=index_version,
        started_at=time.perf_counter(),
        question=question,
        history=list(history or []),
        filters=filters,
        stream=stream,
        debug=debug,
        judge_all=judge_all,
        run_tag=run_tag,
        candidates=[],
        context=[],
        answer="",
        repair_failure=None,
        rejected_answer=None,
        abstain_reason=None,
        sources=[],
        timings_ms={},
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        response=None,
        debug_info={},
        visited=[],
    )
