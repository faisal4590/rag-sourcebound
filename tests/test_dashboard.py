# Issue #32: the five panels and the five alert rules from Phoenix rows; the runbook on one trace.
"""Fake Phoenix rows in the REST shape drive `eval/dashboard.py` and `eval/trace_show.py`; the
numbers of every panel and the state of every alert rule are checked against the rows."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import dashboard as db
import trace_show

NOW = datetime(2026, 11, 21, 18, 0, tzinfo=UTC)


def row(
    name: str,
    trace: str,
    span: str,
    *,
    minutes_ago: float,
    ms: float = 100,
    parent: str | None = None,
    status: str = "OK",
    kind: str = "CHAIN",
    **attrs: Any,
) -> dict[str, Any]:
    start = NOW - timedelta(minutes=minutes_ago)
    return {
        "name": name,
        "context": {"trace_id": trace, "span_id": span},
        "parent_id": parent,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(milliseconds=ms)).isoformat(),
        "status_code": status,
        "span_kind": kind,
        "attributes": {"openinference.span.kind": kind, **attrs},
    }


def request(
    trace: str,
    *,
    minutes_ago: float,
    status: str = "answered",
    reason: str = "none",
    ms: float = 3000,
    cache_hit: bool = False,
    top1: float = 0.9,
    faithful: bool | None = True,
    borderline: bool = False,
    cost: float = 0.001,
    error: bool = False,
) -> list[dict[str, Any]]:
    root = f"{trace}r"
    rows = [
        row(
            "rag.request",
            trace,
            root,
            minutes_ago=minutes_ago,
            ms=ms,
            **{
                "rag.status": status,
                "rag.abstain_reason": reason,
                "cache.hit": cache_hit,
                "http.route": "/ask",
                "rag.latency_ms": int(ms),
                "rag.cost_usd": cost,
                "rag.index_version": "v3",
                "rag.prompt_version": "abcd1234",
                "rag.config_hash": "c0ffee00",
            },
        ),
        row(
            "retrieval.hybrid",
            trace,
            f"{trace}h",
            minutes_ago=minutes_ago,
            ms=300,
            parent=root,
            kind="RETRIEVER",
            **{
                "rag.overlap_ratio": 0.45,
                "rag.n_fused": 2,
                "rag.n_dense_hits": 10,
                "rag.n_sparse_hits": 8,
                "retrieval.documents.0.document.id": "c0",
                "retrieval.documents.0.document.score": 0.03,
                "retrieval.documents.0.document.metadata.chapter_no": 6,
                "retrieval.documents.0.document.metadata.section_title": "Readonly",
                "retrieval.documents.0.document.metadata.page_printed_start": 75,
                "retrieval.documents.0.document.metadata.page_printed_end": 77,
            },
        ),
        row(
            "rerank",
            trace,
            f"{trace}k",
            minutes_ago=minutes_ago,
            ms=1200,
            parent=root,
            kind="RERANKER",
            **{
                "rag.score_top1": top1,
                "rag.n_in": 2,
                "rag.n_out": 2,
                "gen_ai.request.model": "BAAI/bge-reranker-base",
            },
        ),
        row(
            "relevance.gate",
            trace,
            f"{trace}g",
            minutes_ago=minutes_ago,
            ms=1,
            parent=root,
            **{
                "rag.gate_decision": "abstain" if status == "no_information" else "pass",
                "rag.gate_reason": reason if status == "no_information" else "gate.pass",
                "rag.score_top1": top1,
                "rag.tau_low": 0.3,
                "rag.tau_pass": 0.35,
                "rag.tau_high": 0.654,
                "rag.borderline": borderline,
                "rag.n_pass": 2,
            },
        ),
    ]
    if status == "answered":
        rows.append(
            row(
                "llm.generate",
                trace,
                f"{trace}l",
                minutes_ago=minutes_ago,
                ms=1500,
                parent=root,
                kind="LLM",
                status="ERROR" if error else "OK",
                **{
                    "gen_ai.request.model": "groq:openai/gpt-oss-120b",
                    "gen_ai.usage.input_tokens": 5000,
                    "gen_ai.usage.output_tokens": 400,
                    "rag.cost_usd": cost,
                    "rag.prompt_version": "abcd1234",
                    "rag.repair_calls": 0,
                    "output.value": "Readonly properties are set once [S1].",
                },
            )
        )
        if faithful is not None:
            rows.append(
                row(
                    "check.faithfulness",
                    trace,
                    f"{trace}f",
                    minutes_ago=minutes_ago,
                    ms=800,
                    parent=f"{trace}o",
                    kind="LLM",
                    **{
                        "gen_ai.request.model": "groq:openai/gpt-oss-20b",
                        "gen_ai.usage.input_tokens": 6000,
                        "gen_ai.usage.output_tokens": 100,
                        "rag.cost_usd": 0.0,
                        "rag.faithful": faithful,
                    },
                )
            )
    return rows


def fake_phoenix(
    rows: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/spans"):
            wanted = request.url.params.get_list("trace_id")
            data = [r for r in rows if not wanted or r["context"]["trace_id"] in wanted]
            return httpx.Response(200, json={"data": data, "next_cursor": None})
        assert request.url.path.endswith("/span_annotations")
        ids = set(request.url.params.get_list("span_ids"))
        return httpx.Response(200, json={"data": [a for a in annotations if a["span_id"] in ids]})

    return httpx.MockTransport(handle)


def sample_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += request("t1", minutes_ago=5, ms=2000, top1=0.95)
    rows += request("t2", minutes_ago=7, ms=12000, top1=0.80, faithful=False, borderline=True)
    rows += request(
        "t3",
        minutes_ago=90,
        status="no_information",
        reason="gate.low_top_score",
        ms=1500,
        top1=0.1,
        faithful=None,
    )
    rows += request("t4", minutes_ago=200, cache_hit=True, ms=25, top1=0.9, faithful=None)
    rows += request("t5", minutes_ago=3, ms=9000, error=True, faithful=None)
    rows.append(
        row(
            "rag.request",
            "h1",
            "h1r",
            minutes_ago=1,
            ms=2,
            **{"http.route": "/health", "rag.status": "none"},
        )
    )
    return rows


def run_file(tmp_path: Path) -> Path:
    path = tmp_path / "20261108_baseline_v1.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-11-08T10:00:00+00:00",
                "index_version": "v3",
                "prompt_version": "abcd1234",
                "golden_version": "v1",
                "n_cases": 100,
                "metrics": {
                    "recall_at_5": {"value": 1.0},
                    "mrr_at_10": {"value": 0.983},
                    "faithfulness": {"value": 1.0},
                    "correctness": {"value": 0.883},
                    "citation_accuracy": {"value": 0.966},
                    "abstain_f1": {"value": 0.92},
                    "p95_latency_s": {"value": 7.3},
                },
            }
        )
    )
    (tmp_path / "smoke_x.json").write_text(json.dumps({"tag": "x", "rows": []}))  # not an eval run
    return path


def test_the_five_panels_count_what_the_rows_hold(tmp_path: Path) -> None:
    rows = sample_rows()
    annotations = [
        {
            "name": "user_feedback",
            "span_id": "t2r",
            "created_at": "2026-11-21T17:55:00+00:00",
            "result": {"label": "down", "score": -1.0},
        },
        {
            "name": "user_feedback",
            "span_id": "t1r",
            "created_at": "2026-11-20T09:00:00+00:00",
            "result": {"label": "up", "score": 1.0},
        },
    ]
    run_file(tmp_path)
    rep = db.build(
        rows, annotations, sorted(tmp_path.glob("*.json")), load_settings(), now=NOW, hours=24
    )
    assert rep["requests"] == 5  # the /health root is not a request
    p = rep["panels"]
    assert p["requests"]["totals"] == {"answered": 4, "no_information:gate.low_top_score": 1}
    assert sum(sum(c.values()) for c in p["requests"]["per_hour"].values()) == 5
    assert p["latency"]["rag.request"]["n"] == 6 and p["latency"]["rerank"]["p50"] == 1200
    q = p["quality"]
    assert q["rerank_score_top1"]["n"] == 5 and q["rerank_score_top1"]["histogram"]["0.95"] == 1
    assert q["faithfulness_fail_rate"]["all"] == {"judged": 2, "failed": 1, "rate": 0.5}
    assert q["faithfulness_fail_rate"]["borderline"] == {"judged": 1, "failed": 1, "rate": 1.0}
    assert q["overlap_ratio"]["histogram"]["0.45"] == 5
    c = p["cost"]
    assert c["cost_usd_total"] == pytest.approx(
        0.004
    )  # four generations at 0.001; the judge is free
    # t1 and t2: 5,400 + 6,100 judged; t4 and t5: 5,400 -> [11500, 11500, 5400, 5400]
    assert c["tokens_per_request"]["n"] == 4 and c["tokens_per_request"]["mean"] == 8450
    assert c["tokens_per_request"]["p50"] == 8450 and c["tokens_per_request"]["p95"] == 11500
    assert c["cache_hit_rate"] == {"hits": 1, "requests": 5, "rate": 0.2}
    f = p["feedback_and_evaluation"]
    assert f["feedback_by_day"] == {"2026-11-20": {"up": 1}, "2026-11-21": {"down": 1}}
    assert len(f["golden_runs"]) == 1 and f["golden_runs"][0]["recall_at_5"] == 1.0
    assert f["golden_runs"][0]["index_version"] == "v3"


def test_alert_rules_fire_on_the_thresholds_of_config() -> None:
    settings = load_settings()
    quiet = db.evaluate_alerts(sample_rows(), settings, now=NOW)
    states = {a["rule"]: a["state"] for a in quiet}
    # five requests in the hour: under abstain_min_requests, so the 1/4 abstain share is no alert
    assert (
        states["abstain_rate"] == "OK"
        and states["errors"] == "OK"
        and states["cost_usd_per_day"] == "OK"
    )
    # p95 of the three requests in the last 10 min: 12 s, over 8 s
    assert states["p95_latency_s"] == "ALERT"
    assert states["faithfulness_fail_rate"] == "ALERT"  # 1 of 2 judged in the hour
    rows = sample_rows()
    for i in range(25):  # 25 abstains in the last hour
        rows += request(
            f"a{i}",
            minutes_ago=10 + i,
            status="no_information",
            reason="gate.low_top_score",
            ms=200,
            top1=0.1,
            faithful=None,
        )
    loud = {a["rule"]: a for a in db.evaluate_alerts(rows, settings, now=NOW)}
    assert loud["abstain_rate"]["state"] == "ALERT" and loud["abstain_rate"]["value"] > 0.8
    none = {a["rule"]: a["state"] for a in db.evaluate_alerts([], settings, now=NOW)}
    assert (
        none["abstain_rate"] == "NO DATA"
        and none["errors"] == "OK"
        and none["cost_usd_per_day"] == "OK"
    )


def test_run_reads_phoenix_prints_and_exits_1_on_an_alert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_file(tmp_path)
    out = tmp_path / "dash.json"
    code = db.run(
        load_settings(),
        hours=24,
        out=out,
        alerts_only=False,
        runs_dir=tmp_path,
        transport=fake_phoenix(sample_rows(), []),
        now=NOW,
    )
    assert code == 1  # the p95 and the faithfulness rules fire on the sample
    text = capsys.readouterr().out
    assert (
        "1. requests by status" in text
        and "ALERT   p95_latency_s" in text
        and "golden-set runs" in text
    )
    rep = json.loads(out.read_text())
    assert rep["requests"] == 5 and len(rep["alerts"]) == 5


def test_malformed_and_open_spans_do_not_break_the_report() -> None:
    rows = sample_rows()
    rows.append({"name": "rerank", "context": {"trace_id": "x"}})  # no times, no attributes
    rows.append(
        row("rag.request", "o1", "o1r", minutes_ago=2, **{"rag.status": "answered"})
        | {"end_time": None}  # a span still open
    )
    rows.append(row("llm.generate", "z1", "z1l", minutes_ago=2, kind="LLM") | {"start_time": "?"})
    naive = row("rag.request", "n1", "n1r", minutes_ago=4, **{"rag.status": "answered"})
    naive["start_time"] = naive["start_time"].replace("+00:00", "")  # naive = UTC
    naive["end_time"] = naive["end_time"].replace("+00:00", "Z")  # the Z suffix
    rows.append(naive)
    rows.append(row("rerank", "s1", "s1k", minutes_ago=2, **{"rag.score_top1": float("nan")}))
    rep = db.build(rows, [], [], load_settings(), now=NOW, hours=24)
    assert rep["requests"] == 7  # the open and the naive roots count as requests
    assert rep["panels"]["latency"]["rag.request"]["n"] == 7  # the open one has no duration
    assert rep["panels"]["quality"]["rerank_score_top1"]["n"] == 5  # NaN is not a score
    assert {a["rule"] for a in rep["alerts"]} == {
        "abstain_rate",
        "p95_latency_s",
        "errors",
        "faithfulness_fail_rate",
        "cost_usd_per_day",
    }


def test_the_alert_rules_read_their_own_windows_whatever_hours_says(tmp_path: Path) -> None:
    rows = sample_rows()
    rows += request("old", minutes_ago=20 * 60, cost=5.0)  # 20 h ago: out of --hours 1, in 24 h
    calls: list[dict[str, str]] = []
    real = fake_phoenix(rows, [])

    def spy(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        return real.handle_request(request)

    code = db.run(
        load_settings(),
        hours=1,
        out=tmp_path / "d.json",
        alerts_only=True,
        runs_dir=tmp_path,
        transport=httpx.MockTransport(spy),
        now=NOW,
    )
    assert code == db.EXIT_ALERT
    rep = json.loads((tmp_path / "d.json").read_text())
    cost = next(a for a in rep["alerts"] if a["rule"] == "cost_usd_per_day")
    assert cost["state"] == "ALERT" and cost["value"] > 5  # the 20 h old request is in
    assert rep["requests"] == 3  # the panels keep the 1 h window
    since = datetime.fromisoformat(calls[0]["start_time"])
    assert NOW - since >= timedelta(hours=24)


def test_phoenix_down_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    code = db.run(
        load_settings(),
        hours=1,
        out=None,
        alerts_only=True,
        runs_dir=tmp_path,
        transport=httpx.MockTransport(down),
        now=NOW,
    )
    assert code == db.EXIT_UNREACHABLE and "phoenix unreachable" in capsys.readouterr().err


def test_annotations_follow_the_cursor_and_404_means_none() -> None:
    import phoenix_api

    pages = {None: (["a1"], "p2"), "p2": (["a2"], None)}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("span_ids") == "none":
            return httpx.Response(404, json={"detail": "not found"})
        ids, nxt = pages[request.url.params.get("cursor")]
        data = [{"id": i, "name": "user_feedback", "span_id": "s1"} for i in ids]
        return httpx.Response(200, json={"data": data, "next_cursor": nxt})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        got = phoenix_api.annotations(
            client, "http://p", "flp-rag", ["s1"], names=["user_feedback"]
        )
        assert [a["id"] for a in got] == ["a1", "a2"]
        assert phoenix_api.annotations(client, "http://p", "flp-rag", ["none"]) == []


def test_trace_show_reads_both_metadata_shapes() -> None:
    flat = {
        "retrieval.documents.0.document.id": "c0",
        "retrieval.documents.0.document.score": 0.5,
        "retrieval.documents.0.document.metadata.chapter_no": 6,
        "retrieval.documents.0.document.metadata.page_printed_start": 75,
        "retrieval.documents.0.document.metadata.page_printed_end": 77,
    }
    as_json = {  # the shape the app writes: one JSON string
        "retrieval.documents.0.document.id": "c0",
        "retrieval.documents.0.document.score": "n/a",
        "retrieval.documents.0.document.metadata": json.dumps(
            {
                "chapter_no": 6,
                "section_title": "Readonly",
                "page_printed_start": 75,
                "page_printed_end": 77,
            }
        ),
    }
    broken = {
        "retrieval.documents.0.document.id": "c0",
        "retrieval.documents.0.document.metadata": "[1, 2]",  # not a record
    }
    (line,) = trace_show.documents(flat, "retrieval.documents", 10)
    assert "ch 6" in line and "p. 75-77" in line and "score 0.500" in line
    (line,) = trace_show.documents(as_json, "retrieval.documents", 10)
    assert "ch 6" in line and "Readonly" in line and "score ?" in line
    (line,) = trace_show.documents(broken, "retrieval.documents", 10)
    assert "ch ?" in line and "p. ?-?" in line


def test_trace_show_walks_the_runbook_steps(capsys: pytest.CaptureFixture[str]) -> None:
    rows = request(
        "t3",
        minutes_ago=1,
        status="no_information",
        reason="gate.low_top_score",
        top1=0.1,
        faithful=None,
    )
    code = trace_show.run(load_settings(), "t3", transport=fake_phoenix(rows, []))
    assert code == 0
    text = capsys.readouterr().out
    assert (
        "1. root span rag.request" in text
        and "status no_information  abstain_reason gate.low_top_score" in text
    )
    assert "3. retrieval.hybrid" in text and "c0  ch 6  Readonly" in text and "p. 75-77" in text
    assert "the gate stopped the request: top score 0.1 against tau_pass 0.35" in text
    assert "6. context.assemble" in text and "absent: no context was built" in text
    assert "llm.generate absent" in text
    assert trace_show.run(load_settings(), "zz", transport=fake_phoenix(rows, [])) == 1
