# The five dashboard panels of spec 7.9 and the five alert rules of 7.10 from Phoenix spans. #32.
"""Dashboard and alerts without a metrics backend.

Phoenix stores traces and span annotations and has no dashboard builder or alerting (20.9),
so the panels of spec 7.9 are computed here from the spans of a time window, read over the
REST API, and the alert rules of 7.10 are evaluated on the same spans. The Grafana version of
the same panels and rules, for a deployment that exports the OTel metrics
(`trace.metrics_endpoint`), is under `docs/dashboards/`.

    uv run python eval/dashboard.py --hours 24 [--json eval/runs/dashboard.json]
    uv run python eval/dashboard.py --alerts        # exit 1 when a rule fires, 2 without Phoenix

Panels: 1 requests per hour by status and abstain reason; 2 latency p50 and p95 per span;
3 quality (the rerank top-1 score histogram, the faithfulness fail rate by borderline, the
retriever overlap histogram); 4 cost per hour by model, tokens per request, cache hit rate;
5 user feedback by day (the `user_feedback` annotations) and the golden-set scores of every
evaluation run in `eval/runs/` by index and prompt version.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from flp_rag.settings import Settings, load_settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import percentile
from phoenix_api import annotations, fetch_spans, phoenix_base_url

ROOT = Path(__file__).resolve().parents[1]
ROOT_SPAN, ASK_ROUTE = "rag.request", "/ask"
LLM_KIND, EMBED_BATCH = "LLM", "embed.batch"
COST_WINDOW_HOURS = 24  # spec 7.10: "cost per day"
EXIT_ALERT, EXIT_UNREACHABLE = 1, 2
BUCKETS = [round(0.05 * i, 2) for i in range(1, 21)]  # 0.05 .. 1.00
EVAL_COLUMNS = (
    "recall_at_5",
    "mrr_at_10",
    "faithfulness",
    "correctness",
    "citation_accuracy",
    "abstain_f1",
    "p95_latency_s",
)

Row = dict[str, Any]


# --------------------------------------------------------------------------- row helpers


def parse_time(value: Any) -> datetime | None:
    """An ISO timestamp as an aware datetime; a naive one is UTC; anything else is None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)  # Python 3.11+ reads the Z suffix
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def when(row: Row, key: str = "start_time") -> datetime | None:
    return parse_time(row.get(key))


def duration_ms(row: Row) -> float | None:
    """The span duration, or None for a span still open or a malformed row."""
    start, end = when(row), when(row, "end_time")
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def attrs(row: Row) -> dict[str, Any]:
    value = row.get("attributes")
    return value if isinstance(value, dict) else {}


def name_of(row: Row) -> str:
    return str(row.get("name") or "")


def trace_of(row: Row) -> str:
    context = row.get("context")
    return str(context.get("trace_id") or "") if isinstance(context, dict) else ""


def span_id_of(row: Row) -> str:
    context = row.get("context")
    return str(context.get("span_id") or "") if isinstance(context, dict) else ""


def ask_roots(rows: Iterable[Row]) -> list[Row]:
    """The root spans of `/ask` requests (health and page requests have roots too)."""
    return [
        r
        for r in rows
        if name_of(r) == ROOT_SPAN and attrs(r).get("http.route", ASK_ROUTE) == ASK_ROUTE
    ]


def in_window(rows: Iterable[Row], *, since: datetime) -> list[Row]:
    """The rows that start at or after `since`; a row without a start time is out."""
    return [r for r in rows if (start := when(r)) is not None and start >= since]


def hour_of(row: Row) -> str:
    start = when(row)
    return start.astimezone(UTC).strftime("%Y-%m-%dT%H:00Z") if start else "unknown"


def day_of(stamp: str) -> str:
    parsed = parse_time(stamp)
    return parsed.astimezone(UTC).date().isoformat() if parsed else "unknown"


def number(value: Any) -> float | None:
    """A finite number, or None (a bool, a string, NaN, and infinity are not scores)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def histogram(values: Iterable[float]) -> dict[str, int]:
    """Counts per 0.05 bucket, labelled by the bucket's upper edge (Prometheus `le`)."""
    counts = Counter()
    for v in values:
        edge = next((b for b in BUCKETS if v <= b + 1e-9), BUCKETS[-1])
        counts[f"{edge:.2f}"] += 1
    return {f"{b:.2f}": counts.get(f"{b:.2f}", 0) for b in BUCKETS}


def durations(rows: Iterable[Row]) -> list[float]:
    return [d for r in rows if (d := duration_ms(r)) is not None]


# --------------------------------------------------------------------------- the panels


def panel_requests(rows: Sequence[Row]) -> dict[str, Any]:
    """Panel 1: requests per hour by status and abstain reason."""
    per_hour: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for r in ask_roots(rows):
        a = attrs(r)
        key = str(a.get("rag.status", "none"))
        reason = str(a.get("rag.abstain_reason", "none"))
        if key == "no_information":
            key = f"no_information:{reason}"
        per_hour[hour_of(r)][key] += 1
        totals[key] += 1
    return {"per_hour": {h: dict(c) for h, c in sorted(per_hour.items())}, "totals": dict(totals)}


def panel_latency(rows: Sequence[Row]) -> dict[str, Any]:
    """Panel 2: p50 and p95 per span name, in ms."""
    by_name: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if (d := duration_ms(r)) is not None:
            by_name[name_of(r)].append(d)
    table = {
        name: {"p50": round(percentile(v, 0.5)), "p95": round(percentile(v, 0.95)), "n": len(v)}
        for name, v in by_name.items()
    }
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["p50"]))


def panel_quality(rows: Sequence[Row]) -> dict[str, Any]:
    """Panel 3: the rerank top-1 histogram, the faithfulness fail rate (all and borderline),
    the dense/sparse overlap histogram."""
    rerank = [
        v
        for r in rows
        if name_of(r) == "rerank" and (v := number(attrs(r).get("rag.score_top1"))) is not None
    ]
    overlap = [
        v
        for r in rows
        if name_of(r) == "retrieval.hybrid"
        and (v := number(attrs(r).get("rag.overlap_ratio"))) is not None
    ]
    borderline = {
        trace_of(r)
        for r in rows
        if name_of(r) == "relevance.gate" and attrs(r).get("rag.borderline") is True
    }
    judged = judged_rows(rows)

    def fail_rate(group: Sequence[Row]) -> dict[str, Any]:
        failed = sum(1 for r in group if attrs(r).get("rag.faithful") is False)
        return {
            "judged": len(group),
            "failed": failed,
            "rate": failed / len(group) if group else None,
        }

    return {
        "rerank_score_top1": {"histogram": histogram(rerank), "n": len(rerank)},
        "faithfulness_fail_rate": {
            "all": fail_rate(judged),
            "borderline": fail_rate([r for r in judged if trace_of(r) in borderline]),
            "not_borderline": fail_rate([r for r in judged if trace_of(r) not in borderline]),
        },
        "overlap_ratio": {"histogram": histogram(overlap), "n": len(overlap)},
    }


def judged_rows(rows: Iterable[Row]) -> list[Row]:
    """The faithfulness judge spans that carry a verdict."""
    return [
        r
        for r in rows
        if name_of(r) == "check.faithfulness" and isinstance(attrs(r).get("rag.faithful"), bool)
    ]


def cost_rows(rows: Iterable[Row]) -> list[Row]:
    """The spans that carry a model cost: LLM spans and embedding batches."""
    return [
        r
        for r in rows
        if (
            attrs(r).get("openinference.span.kind") == LLM_KIND
            or r.get("span_kind") == LLM_KIND
            or name_of(r) == EMBED_BATCH
        )
        and number(attrs(r).get("rag.cost_usd")) is not None
    ]


def panel_cost(rows: Sequence[Row]) -> dict[str, Any]:
    """Panel 4: cost per hour by model, tokens per request, cache hit rate."""
    per_hour: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total = 0.0
    for r in cost_rows(rows):
        model = str(attrs(r).get("gen_ai.request.model", "none"))
        cost = float(attrs(r)["rag.cost_usd"])
        per_hour[hour_of(r)][model] += cost
        total += cost
    tokens_by_trace: Counter = Counter()
    for r in rows:
        a = attrs(r)
        if a.get("openinference.span.kind") == LLM_KIND or r.get("span_kind") == LLM_KIND:
            for key in ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"):
                if (count := number(a.get(key))) is not None:
                    tokens_by_trace[trace_of(r)] += int(count)
    roots = ask_roots(rows)
    hits = sum(1 for r in roots if attrs(r).get("cache.hit") is True)
    tokens = list(tokens_by_trace.values())
    return {
        "cost_usd_per_hour": {h: dict(m) for h, m in sorted(per_hour.items())},
        "cost_usd_total": round(total, 6),
        "tokens_per_request": {
            "mean": round(sum(tokens) / len(tokens)) if tokens else None,
            "p50": round(percentile(tokens, 0.5)) if tokens else None,
            "p95": round(percentile(tokens, 0.95)) if tokens else None,
            "n": len(tokens),
        },
        "cache_hit_rate": {
            "hits": hits,
            "requests": len(roots),
            "rate": hits / len(roots) if roots else None,
        },
    }


def panel_feedback(annotation_rows: Iterable[Row], run_files: Iterable[Path]) -> dict[str, Any]:
    """Panel 5: user feedback by day and the golden-set scores of every evaluation run."""
    by_day: dict[str, Counter] = defaultdict(Counter)
    for a in annotation_rows:
        if a.get("name") != "user_feedback":
            continue
        label = str((a.get("result") or {}).get("label") or "none")
        by_day[day_of(str(a.get("created_at") or "1970-01-01T00:00:00Z"))][label] += 1
    runs = []
    for path in sorted(run_files):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        metrics = data.get("metrics")
        if not isinstance(metrics, dict) or "n_cases" not in data:
            continue  # a calibration, smoke, or coverage file
        runs.append(
            {
                "file": path.name,
                "timestamp": data.get("timestamp"),
                "index_version": data.get("index_version"),
                "prompt_version": data.get("prompt_version"),
                "golden_version": data.get("golden_version"),
                "n_cases": data.get("n_cases"),
                **{k: (metrics.get(k) or {}).get("value") for k in EVAL_COLUMNS},
            }
        )
    return {"feedback_by_day": {d: dict(c) for d, c in sorted(by_day.items())}, "golden_runs": runs}


# --------------------------------------------------------------------------- the alerts


def evaluate_alerts(
    rows: Sequence[Row], settings: Settings, *, now: datetime
) -> list[dict[str, Any]]:
    """Spec 7.10. Each rule: its window, the measured value, the threshold, OK / ALERT / NO DATA."""
    cfg = settings.alerts
    out: list[dict[str, Any]] = []

    def rule(
        name: str, window: str, value: float | None, threshold: float, fired: bool | None
    ) -> None:
        state = "NO DATA" if fired is None else ("ALERT" if fired else "OK")
        out.append(
            {"rule": name, "window": window, "value": value, "threshold": threshold, "state": state}
        )

    roots_1h = in_window(ask_roots(rows), since=now - timedelta(hours=cfg.abstain_window_hours))
    abstains = sum(1 for r in roots_1h if attrs(r).get("rag.status") == "no_information")
    rate = abstains / len(roots_1h) if roots_1h else None
    rule(
        "abstain_rate",
        f"{cfg.abstain_window_hours:g} h",
        rate,
        cfg.abstain_rate_max,
        None
        if rate is None
        else (len(roots_1h) > cfg.abstain_min_requests and rate > cfg.abstain_rate_max),
    )

    roots_10m = in_window(ask_roots(rows), since=now - timedelta(minutes=cfg.p95_window_minutes))
    timed = durations(roots_10m)
    p95 = percentile(timed, 0.95) / 1000 if timed else None
    rule(
        "p95_latency_s",
        f"{cfg.p95_window_minutes:g} min",
        p95,
        cfg.p95_latency_max_s,
        None if p95 is None else p95 > cfg.p95_latency_max_s,
    )

    errors = [
        r
        for r in in_window(rows, since=now - timedelta(minutes=cfg.errors_window_minutes))
        if r.get("status_code") == "ERROR"
    ]
    rule(
        "errors",
        f"{cfg.errors_window_minutes:g} min",
        len(errors),
        cfg.errors_max,
        len(errors) > cfg.errors_max,
    )

    judged = judged_rows(
        in_window(rows, since=now - timedelta(hours=cfg.faithfulness_window_hours))
    )
    failed = sum(1 for r in judged if attrs(r).get("rag.faithful") is False)
    fail_rate = failed / len(judged) if judged else None
    rule(
        "faithfulness_fail_rate",
        f"{cfg.faithfulness_window_hours:g} h",
        fail_rate,
        cfg.faithfulness_fail_rate_max,
        None if fail_rate is None else fail_rate > cfg.faithfulness_fail_rate_max,
    )

    day = in_window(cost_rows(rows), since=now - timedelta(hours=COST_WINDOW_HOURS))
    cost = sum(float(attrs(r)["rag.cost_usd"]) for r in day)
    rule(
        "cost_usd_per_day",
        f"{COST_WINDOW_HOURS:g} h",
        round(cost, 6),
        cfg.cost_max_usd_per_day,
        cost > cfg.cost_max_usd_per_day,
    )
    return out


# --------------------------------------------------------------------------- report


def alert_window_hours(settings: Settings) -> float:
    """The longest window any rule of 7.10 reads: the fetch must cover it."""
    cfg = settings.alerts
    return max(
        cfg.abstain_window_hours,
        cfg.p95_window_minutes / 60,
        cfg.errors_window_minutes / 60,
        cfg.faithfulness_window_hours,
        COST_WINDOW_HOURS,
    )


def build(
    rows: Sequence[Row],
    annotation_rows: Sequence[Row],
    run_files: Sequence[Path],
    settings: Settings,
    *,
    now: datetime,
    hours: float,
    alert_rows: Sequence[Row] | None = None,
) -> dict[str, Any]:
    """The report: the panels over `rows` (the `--hours` window) and the rules of 7.10 over
    `alert_rows`, which must cover every rule's own window (default: the same rows)."""
    return {
        "window_hours": hours,
        "until": now.isoformat(),
        "spans": len(rows),
        "requests": len(ask_roots(rows)),
        "panels": {
            "requests": panel_requests(rows),
            "latency": panel_latency(rows),
            "quality": panel_quality(rows),
            "cost": panel_cost(rows),
            "feedback_and_evaluation": panel_feedback(annotation_rows, run_files),
        },
        "alerts": evaluate_alerts(
            alert_rows if alert_rows is not None else rows, settings, now=now
        ),
    }


def print_report(rep: dict[str, Any], *, alerts_only: bool = False) -> None:
    if not alerts_only:
        p = rep["panels"]
        print(
            f"window {rep['window_hours']:g} h until {rep['until']}: "
            f"{rep['spans']} spans, {rep['requests']} requests"
        )
        print("\n1. requests by status (total over the window)")
        for key, n in sorted(p["requests"]["totals"].items()):
            print(f"   {key:40} {n}")
        print("\n2. latency per span, ms (p50 / p95 / n)")
        for name, v in p["latency"].items():
            print(f"   {name:22} {v['p50']:7} {v['p95']:7} {v['n']:5}")
        q = p["quality"]
        print("\n3. quality")
        print(
            f"   rerank top-1 score, n={q['rerank_score_top1']['n']}: "
            + " ".join(f"{b}:{n}" for b, n in q["rerank_score_top1"]["histogram"].items() if n)
        )
        for group, f in q["faithfulness_fail_rate"].items():
            rate = "-" if f["rate"] is None else f"{f['rate']:.2f}"
            print(f"   faithfulness fail rate {group:14} {f['failed']}/{f['judged']} = {rate}")
        print(
            f"   overlap ratio, n={q['overlap_ratio']['n']}: "
            + " ".join(f"{b}:{n}" for b, n in q["overlap_ratio"]["histogram"].items() if n)
        )
        c = p["cost"]
        print("\n4. cost")
        print(
            f"   total USD {c['cost_usd_total']}; tokens per request "
            f"mean {c['tokens_per_request']['mean']} p50 {c['tokens_per_request']['p50']} "
            f"p95 {c['tokens_per_request']['p95']}"
        )
        hit = c["cache_hit_rate"]
        print(
            f"   cache hit rate {hit['hits']}/{hit['requests']} = "
            + ("-" if hit["rate"] is None else f"{hit['rate']:.2f}")
        )
        f = p["feedback_and_evaluation"]
        print("\n5. feedback by day")
        for day, counts in f["feedback_by_day"].items():
            print(f"   {day} {dict(counts)}")
        print(
            "   golden-set runs "
            "(index / prompt / n / recall@5 / mrr / faithful / correct / F1 / p95 s)"
        )
        for run in f["golden_runs"]:
            cells = " ".join(
                f"{v:.3f}" if (v := number(run[k])) is not None else "-"
                for k in EVAL_COLUMNS
                if k != "citation_accuracy"
            )
            print(
                f"   {str(run['timestamp'])[:10]} {run['index_version']} {run['prompt_version']} "
                f"{run['n_cases']!s:>4} {cells}"
            )
    print("\nalerts (spec 7.10)")
    for a in rep["alerts"]:
        value = (
            "-"
            if a["value"] is None
            else (f"{a['value']:.3f}" if isinstance(a["value"], float) else str(a["value"]))
        )
        print(
            f"   {a['state']:7} {a['rule']:24} {a['window']:8} value {value:>8}  "
            f"threshold {a['threshold']}"
        )


def run(
    settings: Settings,
    *,
    hours: float,
    out: Path | None,
    alerts_only: bool,
    runs_dir: Path,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> int:
    until = now or datetime.now(UTC)
    fetch_hours = max(hours, alert_window_hours(settings))  # the rules read their own windows
    url, project = phoenix_base_url(settings), settings.eval.phoenix_project
    try:
        with httpx.Client(timeout=settings.eval.timeout_s, transport=transport) as client:
            since = until - timedelta(hours=fetch_hours)
            rows = fetch_spans(client, url, project, since=since, until=until)
            root_ids = [sid for r in ask_roots(rows) if (sid := span_id_of(r))]
            annotation_rows = (
                annotations(client, url, project, root_ids, names=["user_feedback"])
                if root_ids
                else []
            )
    except httpx.HTTPError as exc:
        print(f"phoenix unreachable at {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    panel_rows = in_window(rows, since=until - timedelta(hours=hours))
    rep = build(
        panel_rows,
        annotation_rows,
        sorted(runs_dir.glob("*.json")),
        settings,
        now=until,
        hours=hours,
        alert_rows=rows,
    )
    print_report(rep, alerts_only=alerts_only)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=str))
    return EXIT_ALERT if any(a["state"] == "ALERT" for a in rep["alerts"]) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument("--alerts", action="store_true", help="print the alert rules only")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "eval" / "runs")
    args = parser.parse_args(argv)
    return run(
        load_settings(),
        hours=args.hours,
        out=args.json,
        alerts_only=args.alerts,
        runs_dir=args.runs_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
