# Issue #28: the 10 smoke queries through the API with a run tag, repeated, written to one run file per side.
"""Run the smoke queries against the running API and write one run file.

Protocol (issue #28, Part 3): one warm-up request (the first query), dropped; then every
query of `eval/smoke_queries.jsonl`, `--repeats` times, with `debug: true` and the tag in
`run_tag` (never `session_id`: that would turn the batch into one conversation). Each row
holds the status, the rank of the expected chapter in the reranked list (`debug.rerank.top`),
whether the answer cited that chapter, the stage timings, the cost, and the trace id.

    uv run python eval/smoke_run.py --tag m4-before          # API running on the "before" config
    uv run python eval/smoke_run.py --tag m4-after           # API running on the "after" config

The file lands in `eval/runs/smoke_<tag>_<date>.json`. `docs/notes/m4-before-after.md` reads
both files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "eval" / "smoke_queries.jsonl"
RUNS = ROOT / "eval" / "runs"
TOP_N = 5  # Part 7: the expected chapter must be in the reranked top 5

http_transport: httpx.BaseTransport | None = None  # tests inject a MockTransport


@dataclass(frozen=True)
class SmokeQuery:
    query: str
    expected_chapter: int


def load_queries(path: Path) -> list[SmokeQuery]:
    out: list[SmokeQuery] = []
    for line in path.read_text().splitlines():
        if line.strip():
            data = json.loads(line)
            out.append(SmokeQuery(str(data["query"]), int(data["expected_chapter"])))
    return out


def default_out(tag: str, *, date: str | None = None) -> Path:
    stamp = date or datetime.now(UTC).strftime("%Y%m%d")
    return RUNS / f"smoke_{tag}_{stamp}.json"


# --------------------------------------------------------------------------- one request


def ask(
    client: httpx.Client, api_url: str, question: str, *, tag: str
) -> tuple[dict[str, Any], int]:
    """POST /ask with debug and the tag. A 504 or 500 body is returned as it is: status
    `error` is a result of the run, not a failure of the script. A transport error or a body
    that is not JSON becomes an error row too, so one bad response does not end the run."""
    started = time.perf_counter()
    try:
        response = client.post(
            f"{api_url.rstrip('/')}/ask",
            json={"question": question, "debug": True, "run_tag": tag},
        )
        body = response.json()
    except httpx.HTTPError as exc:
        detail = f"transport: {type(exc).__name__}: {exc}"
        return {"status": "error", "detail": detail}, _ms(started)
    except ValueError:
        detail = f"http {response.status_code}: {response.text[:200]}"
        return {"status": "error", "detail": detail}, _ms(started)
    if not isinstance(body, dict):
        body = {"status": "error", "detail": f"unexpected body: {str(body)[:200]}"}
    return body, _ms(started)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def rank_of_chapter(body: dict[str, Any], chapter: int) -> int | None:
    """1-based rank of the first reranked candidate in the expected chapter, or None."""
    debug = body.get("debug") or {}
    top = (debug.get("rerank") or {}).get("top") or []
    for rank, entry in enumerate(top, start=1):
        if entry.get("chapter_no") == chapter:
            return rank
    return None


def cites_chapter(body: dict[str, Any], chapter: int) -> bool:
    return any(s.get("chapter_no") == chapter for s in body.get("sources") or [])


def row_from(body: dict[str, Any], q: SmokeQuery, repeat: int, elapsed_ms: int) -> dict[str, Any]:
    debug = body.get("debug") or {}
    return {
        "question": q.query,
        "expected_chapter": q.expected_chapter,
        "repeat": repeat,
        "status": body.get("status") or "error",
        "abstain_reason": debug.get("abstain_reason"),
        "rank": rank_of_chapter(body, q.expected_chapter),
        "cited": cites_chapter(body, q.expected_chapter),
        "timings_ms": dict(debug.get("timings_ms") or body.get("timings_ms") or {}),
        "elapsed_ms": elapsed_ms,
        "cost_usd": float(debug.get("cost_usd") or 0.0),
        "trace_id": body.get("trace_id"),
        "index_version": body.get("index_version", ""),
        "detail": body.get("detail"),
    }


# --------------------------------------------------------------------------- the protocol


def run_protocol(
    api_url: str,
    queries: Sequence[SmokeQuery],
    *,
    tag: str,
    repeats: int,
    timeout_s: float,
    pause_s: float = 0.0,
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """One warm-up (dropped), then `repeats` passes over the queries, in order. `pause_s`
    waits before every recorded request: the Groq free tier allows 8,000 tokens per minute
    for the answer model and one answered request uses about 6,000, so back-to-back requests
    hit 429s. `on_row` receives the rows so far after every request (the partial run file)."""
    if not queries:
        raise ValueError("no smoke queries to run")
    rows: list[dict[str, Any]] = []
    with httpx.Client(transport=http_transport, timeout=timeout_s) as client:
        ask(client, api_url, queries[0].query, tag=tag)  # loads the models; not recorded
        for repeat in range(1, repeats + 1):
            for q in queries:
                if pause_s:
                    time.sleep(pause_s)
                body, elapsed_ms = ask(client, api_url, q.query, tag=tag)
                row = row_from(body, q, repeat, elapsed_ms)
                rows.append(row)
                if on_row is not None:
                    on_row(rows)
                print(
                    f"{tag} {repeat} {row['status']:>14} {elapsed_ms:>6} ms  "
                    f"rank {row['rank']}  {q.query}",
                    flush=True,
                )
    return rows


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per query: the ranks per repeat, the best and median rank, the statuses, the citations.
    Over all rows: median wall time, median cost, median time per stage."""
    per_query: list[dict[str, Any]] = []
    for question in dict.fromkeys(r["question"] for r in rows):
        mine = [r for r in rows if r["question"] == question]
        ranks = [r["rank"] for r in mine]
        found = [rank for rank in ranks if rank is not None]
        per_query.append(
            {
                "question": question,
                "expected_chapter": mine[0]["expected_chapter"],
                "ranks": ranks,
                "best_rank": min(found) if found else None,
                "median_rank": _median(found) if found else None,
                "in_top5": sum(1 for rank in found if rank <= TOP_N),
                "statuses": dict(Counter(r["status"] for r in mine)),
                "cited": sum(1 for r in mine if r["cited"]),
                "trace_ids": [r["trace_id"] for r in mine],
            }
        )
    stages = sorted({name for r in rows for name in r["timings_ms"]})
    return {
        "n_requests": len(rows),
        "n_answered": sum(1 for r in rows if r["status"] == "answered"),
        "median_elapsed_ms": _median([r["elapsed_ms"] for r in rows]),
        "median_cost_usd": _median([r["cost_usd"] for r in rows]),
        "median_timings_ms": {
            name: _median([r["timings_ms"][name] for r in rows if name in r["timings_ms"]])
            for name in stages
        },
        "per_query": per_query,
    }


def print_table(summary: dict[str, Any]) -> None:
    print(f"{'expected':>8}  {'best':>4}  {'median':>6}  {'top5':>4}  {'cited':>5}  question")
    for q in summary["per_query"]:
        best = q["best_rank"] if q["best_rank"] is not None else "-"
        med = f"{q['median_rank']:.1f}" if q["median_rank"] is not None else "-"
        print(
            f"{q['expected_chapter']:>8}  {best!s:>4}  {med:>6}  {q['in_top5']:>4}  "
            f"{q['cited']:>5}  {q['question']}"
        )
    print(
        f"requests {summary['n_requests']}, answered {summary['n_answered']}, "
        f"median wall {summary['median_elapsed_ms']:.0f} ms, "
        f"median cost {summary['median_cost_usd']:.5f} USD"
    )
    print("median ms per stage:", json.dumps(summary["median_timings_ms"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help='for example "m4-before" or "m4-after"')
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--api-url", default=None, help="default: cli.api_url in config.yaml")
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=90.0, help="client timeout per request, s")
    parser.add_argument(
        "--pause", type=float, default=0.0, help="seconds between requests (free-tier token limits)"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing run file")
    args = parser.parse_args(argv)
    api_url = args.api_url or load_settings().cli.api_url
    queries = load_queries(args.queries)
    if not queries:
        parser.error(f"{args.queries} holds no queries")
    out = args.out or default_out(args.tag)
    if out.exists() and not args.force:
        parser.error(f"{out} exists; pass --force to overwrite it or --out for another name")
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(rows: Sequence[dict[str, Any]], *, partial: bool) -> None:
        record = {
            "tag": args.tag,
            "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "api_url": api_url,
            "repeats": args.repeats,
            "pause_s": args.pause,
            "partial": partial,
            "index_version": rows[0].get("index_version", "") if rows else "",
            "summary": summarize(rows),
            "rows": list(rows),
        }
        out.write_text(json.dumps(record, indent=2))

    rows = run_protocol(
        api_url,
        queries,
        tag=args.tag,
        repeats=args.repeats,
        timeout_s=args.timeout,
        pause_s=args.pause,
        on_row=lambda so_far: write(so_far, partial=True),
    )
    write(rows, partial=False)
    print_table(summarize(rows))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
