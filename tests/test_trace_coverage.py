# Issue #29, Part 8: the coverage script against a fake Phoenix, no network.
"""Rows in the REST shape (nested attributes), grouped by trace, filtered by the run tag,
classified, checked; one line per trace and exit 1 under 100%."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import trace_coverage as tc


def row(
    name: str,
    span_id: str,
    trace_id: str,
    parent: str | None,
    attrs: dict[str, Any],
    *,
    kind: str = "CHAIN",
    status: str = "OK",
) -> dict[str, Any]:
    nested: dict[str, Any] = {"openinference": {"span": {"kind": kind}}}
    for key, value in attrs.items():
        node = nested
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return {
        "name": name,
        "context": {"trace_id": trace_id, "span_id": span_id},
        "parent_id": parent,
        "attributes": nested,
        "status_code": status,
        "start_time": f"2026-11-21T12:00:0{len(span_id) % 10}Z",
    }


def hit_trace(tag: str, trace_id: str = "t1") -> list[dict[str, Any]]:
    root = {
        "rag.request_id": "r",
        "rag.session_id": "none",
        "rag.index_version": "v3",
        "rag.prompt_version": "abcd1234",
        "rag.config_hash": "c0ffee00",
        "rag.status": "answered",
        "rag.abstain_reason": "none",
        "rag.latency_ms": 20,
        "rag.cost_usd": 0.0,
        "cache.hit": True,
        "cache.similarity": 0.99,
        "http.route": "/ask",
        "http.status_code": 200,
        "rag.run_tag": tag,
    }
    return [
        row("rag.request", "s1", trace_id, None, root),
        row(
            "guardrails.input",
            "s2",
            trace_id,
            "s1",
            {
                "guard.chars": 10,
                "guard.injection_suspected": False,
                "guard.injection_patterns": "[]",
                "guard.language": "en",
                "rag.latency_ms": 1,
            },
        ),
        row(
            "cache.lookup",
            "s3",
            trace_id,
            "s1",
            {"cache.hit": True, "cache.similarity": 0.99, "rag.latency_ms": 2},
        ),
        row(
            "response.build",
            "s4",
            trace_id,
            "s1",
            {
                "rag.status": "answered",
                "rag.n_sources": 1,
                "rag.answer_chars": 40,
                "rag.session_turn": "skipped",
                "cache.stored": "skipped",
                "rag.latency_ms": 1,
            },
        ),
    ]


def fake_phoenix(rows: list[dict[str, Any]]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/projects/flp-rag/spans")
        assert request.url.params["limit"] == str(tc.PAGE)
        return httpx.Response(200, json={"data": rows, "next_cursor": None})

    return httpx.MockTransport(handle)


def test_coverage_passes_a_complete_trace_and_ignores_other_tags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [*hit_trace("cov1"), *hit_trace("other", "t2")]
    out = tmp_path / "coverage.json"
    code = tc.run(load_settings(), tag="cov1", hours=1, out=out, transport=fake_phoenix(rows))
    assert code == 0
    rep = json.loads(out.read_text())
    assert rep["traces"] == 1 and rep["passed"] == 1 and rep["coverage"] == 1.0
    assert rep["rows"][0]["path"] == "cache_hit" and rep["rows"][0]["problems"] == []
    text = capsys.readouterr().out
    assert "t1" in text and "PASS" in text and "coverage 1/1 = 100%" in text and "t2" not in text


def test_coverage_fails_a_trace_with_a_missing_span_and_names_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [r for r in hit_trace("cov1") if r["name"] != "guardrails.input"]
    code = tc.run(load_settings(), tag="cov1", hours=1, transport=fake_phoenix(rows))
    assert code == 1
    text = capsys.readouterr().out
    assert "FAIL" in text and "missing span guardrails.input" in text and "0/1" in text


def test_coverage_fails_an_unclassified_trace_and_an_empty_window(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = hit_trace("cov1")
    rows[0]["attributes"]["rag"]["status"] = "error"  # a timeout: no registry path
    rows[0]["attributes"]["rag"]["abstain_reason"] = "timeout"
    rows[0]["attributes"]["cache"]["hit"] = False
    assert tc.run(load_settings(), tag="cov1", hours=1, transport=fake_phoenix(rows)) == 1
    text = capsys.readouterr().out
    assert "SKIP" in text and "no registry path" in text and "0/0" in text
    assert tc.run(load_settings(), tag="cov1", hours=1, transport=fake_phoenix([])) == 1
    assert "no trace" in capsys.readouterr().out


def test_phoenix_metadata_fields_fold_back_into_the_json_string() -> None:
    from flp_rag.span_registry import from_phoenix

    rows = [
        row(
            "rerank",
            "s1",
            "t1",
            None,
            {
                "reranker.output_documents.0.document.id": "c0",
                "reranker.output_documents.0.document.score": 0.9,
                "reranker.output_documents.0.document.metadata.chapter_no": 6,
                "reranker.output_documents.0.document.metadata.section_title": "Readonly",
            },
            kind="RERANKER",
        )
    ]
    attrs = from_phoenix(rows)[0].attributes
    assert json.loads(attrs["reranker.output_documents.0.document.metadata"]) == {
        "chapter_no": 6,
        "section_title": "Readonly",
    }
    assert "reranker.output_documents.0.document.metadata.chapter_no" not in attrs


def test_cost_mismatch_is_a_failure() -> None:
    rows = hit_trace("cov1")
    rows[0]["attributes"]["rag"]["cost_usd"] = 0.5  # no LLM span ran on a cache hit
    code = tc.run(load_settings(), tag="cov1", hours=1, transport=fake_phoenix(rows))
    assert code == 1
