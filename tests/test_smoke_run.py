# Issue #28: `eval/smoke_run.py` sends one warm-up and repeats x 10 tagged requests, writes one run file.
"""Unit tests for the smoke run script against a fake API; no model, no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import smoke_run


def body(
    top_chapters: list[int],
    *,
    status: str = "answered",
    cited: list[int] = (),  # type: ignore[assignment]
    timings: dict[str, int] | None = None,
    cost: float = 0.0,
    trace_id: str = "t" * 32,
) -> dict[str, Any]:
    return {
        "answer": "x [S1]" if status == "answered" else "No information found",
        "status": status,
        "sources": [
            {
                "label": f"S{i + 1}",
                "chunk_ids": [f"d:{c}"],
                "chapter_no": c,
                "chapter_title": "T",
                "section_title": "S",
                "page_printed_start": 1,
                "page_printed_end": 2,
                "score": 0.9,
            }
            for i, c in enumerate(cited)
        ],
        "trace_id": trace_id,
        "timings_ms": timings or {},
        "model": {"gen": "fake"},
        "index_version": "v3-test",
        "prompt_version": "abcd1234",
        "debug": {
            "abstain_reason": None if status == "answered" else "gate.low_top_score",
            "timings_ms": timings or {"rerank": 100, "llm.generate": 50},
            "cost_usd": cost,
            "rerank": {
                "top": [
                    {"chunk_id": f"d:{c}:{i}", "chapter_no": c, "score": 0.5}
                    for i, c in enumerate(top_chapters)
                ]
            },
        },
    }


class FakeApi:
    def __init__(self, *bodies: dict[str, Any]) -> None:
        self.bodies = list(bodies)
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            out = self.bodies.pop(0) if self.bodies else body([6], cited=[6])
            code = 200 if out["status"] != "error" else 504
            return httpx.Response(code, json=out)

        return httpx.MockTransport(handler)


def test_rank_and_citation_are_read_from_the_debug_field() -> None:
    b = body([5, 6, 6, 9], cited=[6])
    assert smoke_run.rank_of_chapter(b, 6) == 2
    assert smoke_run.rank_of_chapter(b, 7) is None
    assert smoke_run.cites_chapter(b, 6) and not smoke_run.cites_chapter(b, 5)
    assert smoke_run.rank_of_chapter({"status": "error", "debug": None}, 6) is None


def test_protocol_sends_one_warm_up_and_repeats_times_queries(
    tmp_path: Path, monkeypatch: Any
) -> None:
    queries = [smoke_run.SmokeQuery(f"q{i}", chapter) for i, chapter in enumerate([5, 6, 7])]
    api = FakeApi()
    monkeypatch.setattr(smoke_run, "http_transport", api.transport())
    rows = smoke_run.run_protocol("http://api", queries, tag="m4-after", repeats=2, timeout_s=5)
    assert len(api.requests) == 1 + 2 * 3
    assert all(r["debug"] is True and r["run_tag"] == "m4-after" for r in api.requests)
    assert all("session_id" not in r for r in api.requests)
    assert api.requests[0]["question"] == "q0"  # the warm-up, dropped
    assert [r["question"] for r in rows] == ["q0", "q1", "q2", "q0", "q1", "q2"]
    assert [r["repeat"] for r in rows] == [1, 1, 1, 2, 2, 2]
    assert rows[0]["rank"] is None and rows[0]["cited"] is False  # q0 expects chapter 5
    assert rows[1]["rank"] == 1 and rows[1]["cited"] is True  # q1 expects chapter 6
    assert rows[0]["status"] == "answered" and rows[0]["expected_chapter"] == 5
    assert rows[0]["trace_id"] == "t" * 32 and rows[0]["elapsed_ms"] >= 0


def test_pause_waits_before_every_recorded_request(monkeypatch: Any) -> None:
    waits: list[float] = []
    monkeypatch.setattr(smoke_run.time, "sleep", waits.append)
    api = FakeApi()
    monkeypatch.setattr(smoke_run, "http_transport", api.transport())
    queries = [smoke_run.SmokeQuery("a", 6), smoke_run.SmokeQuery("b", 6)]
    smoke_run.run_protocol("http://api", queries, tag="t", repeats=2, timeout_s=5, pause_s=7.5)
    assert waits == [7.5] * 4 and len(api.requests) == 5


def test_summary_has_per_query_ranks_and_medians() -> None:
    rows = [
        {
            "question": "q",
            "expected_chapter": 6,
            "repeat": 1,
            "status": "answered",
            "rank": 2,
            "cited": True,
            "timings_ms": {"rerank": 100, "llm.generate": 50},
            "elapsed_ms": 300,
            "cost_usd": 0.001,
            "trace_id": "a",
            "abstain_reason": None,
        },
        {
            "question": "q",
            "expected_chapter": 6,
            "repeat": 2,
            "status": "answered",
            "rank": 1,
            "cited": True,
            "timings_ms": {"rerank": 200, "llm.generate": 70},
            "elapsed_ms": 500,
            "cost_usd": 0.003,
            "trace_id": "b",
            "abstain_reason": None,
        },
        {
            "question": "q",
            "expected_chapter": 6,
            "repeat": 3,
            "status": "no_information",
            "rank": None,
            "cited": False,
            "timings_ms": {"rerank": 150},
            "elapsed_ms": 400,
            "cost_usd": 0.002,
            "trace_id": "c",
            "abstain_reason": "gate.low_top_score",
        },
    ]
    s = smoke_run.summarize(rows)
    q = s["per_query"][0]
    assert q["question"] == "q" and q["ranks"] == [2, 1, None]
    assert q["best_rank"] == 1 and q["median_rank"] == 1.5 and q["in_top5"] == 2
    assert q["statuses"] == {"answered": 2, "no_information": 1} and q["cited"] == 2
    assert s["median_elapsed_ms"] == 400 and s["median_cost_usd"] == 0.002
    assert s["median_timings_ms"] == {"rerank": 150, "llm.generate": 60}
    assert s["n_requests"] == 3 and s["n_answered"] == 2


def test_main_writes_the_run_file(tmp_path: Path, monkeypatch: Any) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"query": "a", "expected_chapter": 6}\n{"query": "b", "expected_chapter": 6}\n'
    )
    api = FakeApi()
    monkeypatch.setattr(smoke_run, "http_transport", api.transport())
    out = tmp_path / "run.json"
    code = smoke_run.main(
        [
            "--tag",
            "m4-before",
            "--repeats",
            "1",
            "--api-url",
            "http://api",
            "--queries",
            str(queries),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert data["tag"] == "m4-before" and data["repeats"] == 1 and data["api_url"] == "http://api"
    assert data["index_version"] == "v3-test" and len(data["rows"]) == 2
    assert data["summary"]["n_requests"] == 2 and data["summary"]["per_query"][0]["best_rank"] == 1
    assert len(api.requests) == 3


def test_bad_responses_become_error_rows_and_the_run_continues(monkeypatch: Any) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        n = len(calls)
        if n == 2:
            return httpx.Response(502, text="Bad Gateway")  # not JSON
        if n == 3:
            raise httpx.ReadTimeout("read timed out")
        if n == 4:
            return httpx.Response(422, json={"detail": "x"})  # JSON without a status
        return httpx.Response(200, json=body([6], cited=[6]))

    monkeypatch.setattr(smoke_run, "http_transport", httpx.MockTransport(handler))
    queries = [smoke_run.SmokeQuery(f"q{i}", 6) for i in range(4)]
    rows = smoke_run.run_protocol("http://api", queries, tag="t", repeats=1, timeout_s=5)
    assert [r["status"] for r in rows] == ["error", "error", "error", "answered"]
    assert rows[0]["detail"].startswith("http 502") and "Bad Gateway" in rows[0]["detail"]
    assert rows[1]["detail"].startswith("transport: ReadTimeout")
    assert rows[2]["rank"] is None and rows[3]["rank"] == 1


def test_no_queries_is_an_error() -> None:
    with pytest.raises(ValueError, match="no smoke queries"):
        smoke_run.run_protocol("http://api", [], tag="t", repeats=1, timeout_s=5)


def test_main_writes_partial_files_and_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: Any
) -> None:
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"query": "a", "expected_chapter": 6}\n{"query": "b", "expected_chapter": 6}\n'
    )
    out = tmp_path / "run.json"
    seen: list[tuple[int, bool]] = []
    original = out.write_text

    def spy(text: str, *a: Any, **kw: Any) -> int:
        data = json.loads(text)
        seen.append((len(data["rows"]), data["partial"]))
        return original(text, *a, **kw)

    monkeypatch.setattr(
        type(out),
        "write_text",
        lambda self, text, *a, **kw: spy(text, *a, **kw) if self == out else original(text),
    )
    monkeypatch.setattr(smoke_run, "http_transport", FakeApi().transport())
    args = [
        "--tag",
        "t",
        "--repeats",
        "1",
        "--api-url",
        "http://api",
        "--queries",
        str(queries),
        "--out",
        str(out),
    ]
    assert smoke_run.main(args) == 0
    assert seen == [(1, True), (2, True), (2, False)]
    with pytest.raises(SystemExit):
        smoke_run.main(args)  # the file exists and --force was not given
    monkeypatch.setattr(smoke_run, "http_transport", FakeApi().transport())
    assert smoke_run.main([*args, "--force"]) == 0


def test_default_out_path_names_tag_and_date() -> None:
    path = smoke_run.default_out("m4-after", date="20261031")
    assert path == ROOT / "eval" / "runs" / "smoke_m4-after_20261031.json"


@pytest.mark.live
def test_live_protocol_against_the_running_api() -> None:
    """The full protocol on the running API with the current configuration, one repeat."""
    import socket

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8000)) != 0:
            pytest.skip("no API on :8000")
    rows = smoke_run.run_protocol(
        "http://localhost:8000",
        smoke_run.load_queries(ROOT / "eval" / "smoke_queries.jsonl"),
        tag="m4-live-test",
        repeats=1,
        timeout_s=90,
    )
    assert len(rows) == 10
    assert all(r["status"] in ("answered", "no_information", "error") for r in rows)
