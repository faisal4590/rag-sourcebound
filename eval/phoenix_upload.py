# Spec 8.6 rule 3 (issue #34): per-case scores as span annotations on each request's root span in Phoenix.
"""Root span ids by GraphQL from the trace ids, then `POST /v1/span_annotations`. The
`identifier` `<run_tag>:<case_id>:<metric>` makes a rerun upsert instead of duplicating."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

import httpx
from metrics import RECALL_K, CaseResult

TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
PROJECT_PAGE = 50  # projects per GraphQL page; the lookup filters by name client side


def root_span_ids(
    phoenix_url: str,
    project: str,
    trace_ids: Iterable[str],
    *,
    timeout_s: float,
    transport: httpx.BaseTransport | None,
) -> dict[str, str]:
    """trace id -> root span id. Only well-formed trace ids (32 hex characters) are looked up,
    because the id lands inside the GraphQL filter string."""
    out: dict[str, str] = {}
    with httpx.Client(transport=transport, timeout=timeout_s) as client:
        for trace_id in trace_ids:
            if not TRACE_ID.match(trace_id or ""):
                continue
            condition = f"trace_id == '{trace_id}' and parent_id is None"
            query = (
                f"{{ projects(first: {PROJECT_PAGE}) {{ edges {{ node {{ name "
                f'spans(first: 1, filterCondition: "{condition}") '
                "{ edges { node { context { spanId traceId } } } } } } } }"
            )
            r = client.post(f"{phoenix_url.rstrip('/')}/graphql", json={"query": query})
            r.raise_for_status()
            for edge in (r.json().get("data") or {}).get("projects", {}).get("edges", []):
                node = edge["node"]
                if node["name"] != project:
                    continue
                for span in node["spans"]["edges"]:
                    out[trace_id] = span["node"]["context"]["spanId"]
    return out


def annotations_for(result: CaseResult, span_id: str, *, tag: str) -> list[dict[str, Any]]:
    scores: list[tuple[str, str, float, str | None]] = []  # name, annotator kind, score, label
    if result.group == "answerable":
        hit = result.rank_rerank is not None and result.rank_rerank <= RECALL_K
        scores.append(("retrieval_hit_at_5", "CODE", float(hit), "hit" if hit else "miss"))
    if result.faithful is not None:
        label = "faithful" if result.faithful else "unfaithful"
        scores.append(("faithfulness", "LLM", float(result.faithful), label))
    if result.correctness is not None:
        scores.append(("correctness", "LLM", result.correctness / 2, str(result.correctness)))
    if result.cited_total:
        scores.append(("citation_accuracy", "CODE", result.cited_hits / result.cited_total, None))
    if result.code_present is not None:
        scores.append(("code_presence", "CODE", float(result.code_present), None))
    if not result.failed:
        right = result.abstained == result.expected_abstain
        scores.append(("abstain_correct", "CODE", float(right), "right" if right else "wrong"))
    return [
        {
            "name": name,
            "span_id": span_id,
            "annotator_kind": kind,
            "result": {"score": score, **({"label": label} if label is not None else {})},
            "identifier": f"{tag}:{result.case_id}:{name}",
            "metadata": {"case_id": result.case_id, "run_tag": tag},
        }
        for name, kind, score, label in scores
    ]


def upload_scores(
    results: Sequence[CaseResult],
    phoenix_url: str,
    project: str,
    *,
    tag: str,
    timeout_s: float,
    transport: httpx.BaseTransport | None = None,
) -> tuple[int, int]:
    """(annotations sent, cases whose trace was not found). Raises `httpx.HTTPError` when
    Phoenix does not answer; the caller turns that into a warning."""
    traces = {r.trace_id for r in results}
    spans = root_span_ids(phoenix_url, project, traces, timeout_s=timeout_s, transport=transport)
    data = [
        a
        for r in results
        if r.trace_id in spans
        for a in annotations_for(r, spans[r.trace_id], tag=tag)
    ]
    missing = sum(1 for r in results if r.trace_id not in spans)
    if data:
        with httpx.Client(transport=transport, timeout=timeout_s) as client:
            url = f"{phoenix_url.rstrip('/')}/v1/span_annotations?sync=false"
            client.post(url, json={"data": data}).raise_for_status()
    return len(data), missing
