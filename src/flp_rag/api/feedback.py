# Stage 18 rule 4 (issue #31): a user rating becomes the `user_feedback` score on the trace.
"""Feedback to the trace backend.

`POST /feedback` carries a `trace_id` and a rating. `write_feedback` finds the root span of
that trace in Phoenix (GraphQL, the same lookup the evaluation harness uses) and posts one
span annotation named `user_feedback`: score 1 for `up`, -1 for `down`, the rating as the
label, the comment as the explanation, annotator kind HUMAN, and the identifier
`user_feedback:<trace_id>`, so a second rating of the same answer replaces the first. The
write is synchronous (`?sync=true`) with `feedback.timeout_s`, so the score is visible in
Phoenix when the endpoint answers. Every failure is a status word, never an exception: a
rating is never lost to a 500 because the backend was slow.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
import structlog

from flp_rag.settings import Settings

ANNOTATION = "user_feedback"
SCORES = {"up": 1, "down": -1}
TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
WRITTEN, MISSING, INVALID = "written", "missing_trace", "invalid_input"
log = structlog.get_logger(__name__)


def phoenix_base_url(settings: Settings) -> str:
    endpoint = settings.trace.exporter_endpoint
    return endpoint.split("/v1/")[0] if "/v1/" in endpoint else endpoint


def root_span_id(client: httpx.Client, phoenix_url: str, project: str, trace_id: str) -> str | None:
    """The root span of the trace, or None when Phoenix has no such trace (yet)."""
    condition = f"trace_id == '{trace_id}' and parent_id is None"
    query = (
        "{ projects(first: 50) { edges { node { name "
        f'spans(first: 1, filterCondition: "{condition}") '
        "{ edges { node { context { spanId } } } } } } } }"
    )
    r = client.post(f"{phoenix_url.rstrip('/')}/graphql", json={"query": query})
    r.raise_for_status()
    for edge in (r.json().get("data") or {}).get("projects", {}).get("edges", []):
        node = edge["node"]
        if node["name"] != project:
            continue
        for span in node["spans"]["edges"]:
            return str(span["node"]["context"]["spanId"])
    return None


def annotation(span_id: str, trace_id: str, rating: str, comment: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"score": SCORES[rating], "label": rating}
    if comment:
        result["explanation"] = comment
    return {
        "name": ANNOTATION,
        "span_id": span_id,
        "annotator_kind": "HUMAN",
        "result": result,
        "identifier": f"{ANNOTATION}:{trace_id}",
    }


def write_feedback(
    trace_id: str,
    rating: str,
    comment: str | None,
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> str:
    """`written`, `missing_trace`, `invalid_input`, or the error class name. Never raises.
    `feedback.timeout_s` bounds the whole write (two calls, half each); a trace Phoenix has
    not received yet (it exports in batches) is looked up once more after
    `feedback.missing_retry_s`."""
    if not TRACE_ID.match(trace_id or "") or rating not in SCORES:
        return INVALID
    cfg = settings.feedback
    url, project = phoenix_base_url(settings), settings.eval.phoenix_project
    if comment is not None:
        comment = comment[: cfg.max_comment_chars]
    try:
        with httpx.Client(transport=transport, timeout=cfg.timeout_s / 2) as client:
            span_id = root_span_id(client, url, project, trace_id)
            if span_id is None and cfg.missing_retry_s > 0:
                time.sleep(cfg.missing_retry_s)
                span_id = root_span_id(client, url, project, trace_id)
            if span_id is None:
                return MISSING
            r = client.post(
                f"{url.rstrip('/')}/v1/span_annotations",
                params={"sync": "true"},
                json={"data": [annotation(span_id, trace_id, rating, comment)]},
            )
            r.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a rating never fails the request
        log.warning("feedback not written to the trace", error=type(exc).__name__, trace=trace_id)
        return type(exc).__name__
    return WRITTEN
