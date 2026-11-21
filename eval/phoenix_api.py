# The Phoenix REST calls the observability scripts share (#32): spans by window or trace, annotations.
"""Phoenix REST helpers.

`fetch_spans` reads every span of a project in a time window, page by page. `trace_spans`
reads the spans of one trace. `annotations` reads the span annotations of a list of span ids
in batches, following the cursor (Phoenix answers 404 when none of the ids exist, which means
none). The rows keep the REST shape; `span_registry.from_phoenix` turns them into `SpanView`s.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from flp_rag.settings import Settings

PAGE = 1000
BATCH = 100


def phoenix_base_url(settings: Settings) -> str:
    """The Phoenix host from the OTLP endpoint (`.../v1/traces`)."""
    endpoint = settings.trace.exporter_endpoint
    return endpoint.split("/v1/")[0] if "/v1/" in endpoint else endpoint


def _paged(client: httpx.Client, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        query = {**params, "limit": PAGE, **({"cursor": cursor} if cursor else {})}
        r = client.get(url, params=query)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("data") or [])
        cursor = body.get("next_cursor")
        if not cursor:
            return out


def spans_url(phoenix_url: str, project: str) -> str:
    return f"{phoenix_url.rstrip('/')}/v1/projects/{quote(project, safe='')}/spans"


def fetch_spans(
    client: httpx.Client, phoenix_url: str, project: str, *, since: datetime, until: datetime
) -> list[dict[str, Any]]:
    """Every span row of the window."""
    params = {"start_time": since.isoformat(), "end_time": until.isoformat()}
    return _paged(client, spans_url(phoenix_url, project), params)


def trace_spans(
    client: httpx.Client, phoenix_url: str, project: str, trace_id: str
) -> list[dict[str, Any]]:
    """The spans of one trace."""
    return _paged(client, spans_url(phoenix_url, project), {"trace_id": [trace_id]})


def annotations(
    client: httpx.Client,
    phoenix_url: str,
    project: str,
    span_ids: Iterable[str],
    *,
    names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """The span annotations of these spans, `BATCH` ids per request."""
    ids = sorted(set(span_ids))
    url = f"{phoenix_url.rstrip('/')}/v1/projects/{quote(project, safe='')}/span_annotations"
    out: list[dict[str, Any]] = []
    for start in range(0, len(ids), BATCH):
        params: dict[str, Any] = {"span_ids": ids[start : start + BATCH]}
        if names:
            params["include_annotation_names"] = list(names)
        try:
            out.extend(_paged(client, url, params))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
    return out
