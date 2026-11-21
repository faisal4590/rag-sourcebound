# Issue #29, Part 8: trace coverage of a tagged run, read from Phoenix and checked against the registry.
"""Trace coverage on the smoke queries.

    uv run python eval/smoke_run.py --tag cov-<date> --repeats 1 --pause 55
    uv run python eval/trace_coverage.py --tag cov-<date> [--hours 2] [--json eval/runs/coverage.json]

Coverage is the share of requests whose trace has a root span and every span and attribute
the registry requires for its path. The script reads every span of the last `--hours` from the
Phoenix REST API (`GET /v1/projects/<project>/spans`, paged), keeps the traces whose root carries
`rag.run_tag == --tag`, reads the path of each trace from its root (`classify_path`), and runs
`check_trace` plus the cost sum on it. One line per request: trace id, path, PASS or FAIL, the
first problem. Exit 1 under 100% or when no trace carries the tag.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from flp_rag.settings import Settings, load_settings
from flp_rag.span_registry import (
    ROOT_REQUEST,
    Registry,
    SpanView,
    check_trace,
    classify_path,
    cost_mismatch,
    from_phoenix,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE = 1000


def phoenix_base_url(settings: Settings) -> str:
    endpoint = settings.trace.exporter_endpoint
    return endpoint.split("/v1/")[0] if "/v1/" in endpoint else endpoint


def fetch_spans(
    client: httpx.Client, phoenix_url: str, project: str, *, since: datetime, until: datetime
) -> list[dict[str, Any]]:
    """Every span row of the window, all pages."""
    url = f"{phoenix_url.rstrip('/')}/v1/projects/{quote(project, safe='')}/spans"
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "start_time": since.isoformat(),
            "end_time": until.isoformat(),
            "limit": PAGE,
        }
        if cursor:
            params["cursor"] = cursor
        r = client.get(url, params=params)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("data") or [])
        cursor = body.get("next_cursor")
        if not cursor:
            return out


def traces_with_tag(views: Sequence[SpanView], tag: str) -> dict[str, list[SpanView]]:
    """trace id -> its spans, for the traces whose root span carries the run tag."""
    by_trace: dict[str, list[SpanView]] = defaultdict(list)
    for v in views:
        by_trace[v.trace_id].append(v)
    return {
        tid: spans
        for tid, spans in by_trace.items()
        if any(
            s.parent is None and s.name == ROOT_REQUEST and s.attributes.get("rag.run_tag") == tag
            for s in spans
        )
    }


def check_one(registry: Registry, spans: Sequence[SpanView], settings: Settings) -> dict[str, Any]:
    """The report row of one trace: path, pass, problems."""
    root = next(s for s in spans if s.parent is None and s.name == ROOT_REQUEST)
    path = classify_path(root.attributes, [s.name for s in spans])
    if path is None:
        return {
            "trace_id": root.trace_id,
            "path": None,
            "pass": None,  # SKIP: not a path the registry describes (a timeout, say)
            "problems": [
                (
                    f"no registry path for status {root.attributes.get('rag.status')!r} "
                    f"and reason {root.attributes.get('rag.abstain_reason')!r}"
                )
            ],
        }
    problems = check_trace(
        registry,
        spans,
        path,
        capture_content=settings.trace.capture_content,
        cache_enabled=settings.cache.semantic_enabled,
        sparse_enabled=settings.retrieval.sparse_enabled,
    )
    gap = cost_mismatch(spans)
    if gap is not None:
        problems.append(f"root rag.cost_usd differs from the LLM span sum by {gap:.6f}")
    return {"trace_id": root.trace_id, "path": path, "pass": not problems, "problems": problems}


def report(rows: Sequence[dict[str, Any]], tag: str) -> dict[str, Any]:
    """Coverage = passed / checked. A trace without a registry path is listed as SKIP and
    stays out of the denominator; `skipped` counts them so they never pass unseen."""
    checked = [r for r in rows if r["pass"] is not None]
    passed = sum(1 for r in checked if r["pass"])
    return {
        "tag": tag,
        "traces": len(checked),
        "passed": passed,
        "skipped": len(rows) - len(checked),
        "coverage": passed / len(checked) if checked else 0.0,
        "rows": list(rows),
    }


def print_report(rep: dict[str, Any]) -> None:
    for r in rep["rows"]:
        first = r["problems"][0] if r["problems"] else ""
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[r["pass"]]
        print(f"{r['trace_id']}  {r['path']!s:16}  {mark}  {first}")
    print(
        f"coverage {rep['passed']}/{rep['traces']} = {rep['coverage']:.0%} for tag "
        f"{rep['tag']!r}, {rep['skipped']} skipped"
    )


def run(
    settings: Settings,
    *,
    tag: str,
    hours: float,
    out: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> int:
    registry = load_registry()
    until = now or datetime.now(UTC)
    with httpx.Client(timeout=settings.eval.timeout_s, transport=transport) as client:
        rows = fetch_spans(
            client,
            phoenix_base_url(settings),
            settings.eval.phoenix_project,
            since=until - timedelta(hours=hours),
            until=until,
        )
    views = from_phoenix(rows)
    traces = traces_with_tag(views, tag)
    if not traces:
        print(f"no trace of the last {hours:g} h carries rag.run_tag = {tag!r}")
        return 1
    rep = report([check_one(registry, spans, settings) for spans in traces.values()], tag)
    print_report(rep)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
    return 0 if rep["traces"] and rep["passed"] == rep["traces"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="the run_tag of the requests to check")
    parser.add_argument("--hours", type=float, default=2.0, help="look back this far in Phoenix")
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)
    return run(load_settings(), tag=args.tag, hours=args.hours, out=args.json)


if __name__ == "__main__":
    sys.exit(main())
