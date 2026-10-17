# Stage 18 - Response and feedback. Response record and the one-line request log. Spec Section 6, Stage 18.
"""Stage 18: the final state of a request becomes one `Response` record and one log line.

Plain functions of the final graph state and the settings (spec Section 6, Stage 18;
issue #23). No model call.

1. One builder for all three statuses: `build_response(state, settings)` at the end of a
   normal request, `build_response(state, settings, error=...)` from the API layer when a
   stage raised and the request left the graph early. The error answer is
   `api.error_text`, never the abstain text and never an internal detail.
2. Labels to sources: the one marker parser of Stages 16-18 reads the cited labels; each
   distinct label gives one `Source` in citation order, with every chunk id of the block, the
   chapter, the section titles of the block's chunks joined with " + ", the printed pages, and
   the highest rerank score among the block's chunks. Labels are not renumbered; `[S1, S3]` in
   the answer becomes `[S1][S3]`. A cited label outside the label map is a bug (Stage 17
   checked them) and raises `UnknownLabel`.
3. Totals from the state: `timings_ms` per span (the graph wrapper fills it), token totals and
   `cost_usd` (the model-calling nodes add theirs), model names from the settings, the index
   version from Stage 9, the prompt version from Stage 16 or the configured answer prompt.
4. Exactly one `INFO` line per request, through structlog, without the question or the answer.
5. The debug field only with `debug: true`: the abstain reason, the gate decision with the
   thresholds, the uncited context blocks, the repair count and the failed check, the timings,
   and the path taken.
6. `format_response` is the CLI view of the same record.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import structlog

from flp_rag.contracts import Candidate, Response, ResponseStatus, Source
from flp_rag.graph.markers import canonical_markers, find_markers
from flp_rag.graph.nodes.generate import prompt_version
from flp_rag.graph.state import RagState
from flp_rag.settings import ABSTAIN_TEXT, Settings
from flp_rag.tracing import current_span

log = structlog.get_logger()

SECTION_JOIN = " + "
RESPOND_NODE = "respond"


class UnknownLabel(ValueError):
    """The answer cites a label that the context does not have. Stage 17 should have caught it."""


# --------------------------------------------------------------------------- sources


def cited_labels(answer: str) -> list[str]:
    """Distinct labels in the order of their first citation."""
    return list(dict.fromkeys(find_markers(answer)))


def section_titles(chunk_ids: Sequence[str], candidates: Sequence[Candidate], fallback: str) -> str:
    """The section titles of the block's chunks, in block order, each once. Stage 15 builds a
    block's `chunk_ids` from the candidates that passed the gate, the same list the state
    carries, so every id is found in production; the label map's section stands in for a test
    fixture without candidates."""
    by_id = {c.chunk_id: c for c in candidates}
    titles: list[str] = []
    for cid in chunk_ids:
        candidate = by_id.get(cid)
        if candidate is None:
            continue
        for title in candidate.payload.get("section_titles") or [
            candidate.payload.get("section_title")
        ]:
            if title and title not in titles:
                titles.append(str(title))
    return SECTION_JOIN.join(titles) if titles else fallback


def block_score(chunk_ids: Sequence[str], candidates: Sequence[Candidate]) -> float:
    scores = [
        float(c.rerank_score if c.rerank_score is not None else c.fused_score)
        for c in candidates
        if c.chunk_id in set(chunk_ids)
    ]
    return max(scores, default=0.0)


def sources_for(
    answer: str, label_map: Mapping[str, Mapping[str, Any]], candidates: Sequence[Candidate]
) -> list[Source]:
    sources: list[Source] = []
    for label in cited_labels(answer):
        entry = label_map.get(label)
        if entry is None:
            raise UnknownLabel(f"the answer cites [{label}] but the context has no such block")
        chunk_ids = [str(cid) for cid in entry.get("chunk_ids", [])]
        sources.append(
            Source(
                label=label,
                chunk_ids=chunk_ids,
                chapter_no=int(entry["chapter_no"]),
                chapter_title=str(entry["chapter_title"]),
                section_title=section_titles(
                    chunk_ids, candidates, str(entry.get("section_title", ""))
                ),
                page_printed_start=int(entry["page_printed_start"]),
                page_printed_end=int(entry["page_printed_end"]),
                score=block_score(chunk_ids, candidates),
            )
        )
    return sources


# --------------------------------------------------------------------------- the record


def model_names(settings: Settings) -> dict[str, str]:
    return {
        "gen": settings.gen.model,
        "query": settings.query.model,
        "judge": settings.output.judge_model,
        "embed": settings.embed.model,
        "rerank": settings.rerank.model,
    }


def default_prompt_version(settings: Settings) -> str:
    """The configured answer prompt, for requests that never reached Stage 16."""
    path = settings.config_path.parent / settings.gen.prompt_file
    return prompt_version(path) if path.exists() else ""


def latency_ms(state: Mapping[str, Any]) -> int:
    started = state.get("started_at")
    return int((time.perf_counter() - float(started)) * 1000) if started else 0


def uncited_blocks(state: Mapping[str, Any], cited: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "chapter_no": entry.get("chapter_no"),
            "chapter_title": entry.get("chapter_title"),
            "section_title": entry.get("section_title"),
            "page_printed_start": entry.get("page_printed_start"),
            "page_printed_end": entry.get("page_printed_end"),
        }
        for label, entry in state.get("label_map", {}).items()
        if label not in cited
    ]


def debug_field(
    state: Mapping[str, Any], settings: Settings, sources: Sequence[Source], error: str | None
) -> dict[str, Any]:
    info = dict(state.get("debug_info", {}))
    gate = {
        "score_top1": state.get("top_score"),
        **info.pop("gate", {}),
        "tau_low": settings.gate.tau_low,
        "tau_pass": settings.gate.tau_pass,
        "tau_high": settings.gate.tau_high,
    }
    debug: dict[str, Any] = {
        "abstain_reason": state.get("abstain_reason"),
        "gate": gate,
        "uncited_blocks": uncited_blocks(state, [s.label for s in sources]),
        "repair_calls": int(state.get("repair_calls", 0)),
        "timings_ms": dict(state.get("timings_ms", {})),
        "visited": [*state.get("visited", []), RESPOND_NODE],
        **info,
    }
    if error is not None:
        debug["error"] = error
    return debug


def build_response(
    state: Mapping[str, Any], settings: Settings, *, error: str | None = None
) -> Response:
    """The one builder. With `error`, the request left the graph early: status `error`, the
    fixed text, no sources; `error` itself goes only into the debug field."""
    if error is not None:
        status: ResponseStatus = "error"
        answer, sources = settings.api.error_text, []
    else:
        status = state.get("status")
        if status not in ("answered", "no_information", "error"):
            raise ValueError(f"the final state has no status to respond with: {status!r}")
        if status == "answered":
            answer = canonical_markers(str(state.get("answer", "")))
            sources = sources_for(answer, state.get("label_map", {}), state.get("candidates", []))
        else:
            answer, sources = (
                ABSTAIN_TEXT if status == "no_information" else settings.api.error_text,
                [],
            )
    return Response(
        answer=answer,
        status=status,
        sources=sources,
        trace_id=str(state.get("trace_id", "")),
        timings_ms=dict(state.get("timings_ms", {})),
        model=model_names(settings),
        index_version=str(state.get("index_version", "")),
        prompt_version=str(state.get("prompt_version") or default_prompt_version(settings)),
        debug=debug_field(state, settings, sources, error) if state.get("debug") else None,
    )


# --------------------------------------------------------------------------- the log line


@dataclass(frozen=True)
class RequestSummary:
    """The fields of the one summary line (spec Stage 18 rule 3, plus the four of issue #23)."""

    request_id: str
    trace_id: str
    status: ResponseStatus
    abstain_reason: str | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    index_version: str
    prompt_version: str
    config_hash: str
    n_sources: int
    repair_calls: int
    borderline: bool

    def to_log(self) -> dict[str, Any]:
        return asdict(self)


def summarize(state: Mapping[str, Any], response: Response, settings: Settings) -> RequestSummary:
    return RequestSummary(
        request_id=str(state.get("request_id", "")),
        trace_id=response.trace_id,
        status=response.status,
        abstain_reason=state.get("abstain_reason") if response.status != "answered" else None,
        latency_ms=latency_ms(state),
        input_tokens=int(state.get("input_tokens", 0)),
        output_tokens=int(state.get("output_tokens", 0)),
        cost_usd=float(state.get("cost_usd", 0.0)),
        index_version=response.index_version,
        prompt_version=response.prompt_version,
        config_hash=settings.config_hash,
        n_sources=len(response.sources),
        repair_calls=int(state.get("repair_calls", 0)),
        borderline=bool(state.get("borderline")),
    )


def log_request(summary: RequestSummary) -> None:
    """Exactly one line per request. Never the question or the answer."""
    log.info("request", **summary.to_log())


# --------------------------------------------------------------------------- CLI view


def format_response(response: Response, *, debug: bool = False) -> str:
    """The same record as text: the answer, one line per source, the footer, the debug field."""
    lines = [response.answer]
    for s in response.sources:
        pages = f"p. {s.page_printed_start}" + (
            f"-{s.page_printed_end}" if s.page_printed_end != s.page_printed_start else ""
        )
        where = f"Chapter {s.chapter_no:02d} {s.chapter_title}"
        lines.append(f"{s.label} · {where} · {s.section_title} · {pages}")
    if debug and response.debug is not None:
        import json

        lines.append(json.dumps(response.debug, indent=2, ensure_ascii=False, default=str))
    lines.append(f"status={response.status} trace_id={response.trace_id}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- node


def make_respond(settings: Settings) -> Callable[[RagState], dict[str, Any]]:
    def respond_node(state: RagState) -> dict[str, Any]:
        response = build_response(state, settings)
        span = current_span()
        span.set_attribute("rag.status", response.status)
        span.set_attribute("rag.n_sources", len(response.sources))
        span.set_attribute("rag.answer_chars", len(response.answer))
        log_request(summarize(state, response, settings))
        return {
            "status": response.status,
            "answer": response.answer,
            "sources": response.sources,
            "response": response,
        }

    return respond_node
