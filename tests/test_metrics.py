# Issue #30: OTel metrics derived from spans (spec 7.7) and the logging rules of spec 7.8.
"""Every metric of Section 7.7 appears in the exporter output after one request; nothing is
counted twice (one span processor is the only source); the per-stage DEBUG lines appear only
for a `debug: true` request; an ERROR line names the span and the exception without a stack
trace; every line carries trace_id, span_id, request_id, and index_version."""

from __future__ import annotations

import json
from typing import Any

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import metrics, tracing
from flp_rag.settings import load_settings

SPEC_METRICS = {
    "rag.requests",
    "rag.stage.latency_ms",
    "rag.tokens",
    "rag.cost_usd",
    "rag.rerank_score_top1",
    "rag.retrieval.overlap_ratio",
    "rag.errors",
    "rag.user_feedback",
}


def points(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    """metric name -> its data points, from the in-memory reader."""
    out: dict[str, list[Any]] = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out.setdefault(m.name, []).extend(m.data.data_points)
    return out


@pytest.fixture
def readers() -> tuple[InMemorySpanExporter, InMemoryMetricReader]:
    exporter, reader = InMemorySpanExporter(), InMemoryMetricReader()
    tracing.configure_tracing(load_settings(), exporter=exporter, force=True, metric_reader=reader)
    yield exporter, reader
    tracing.configure_tracing(load_settings(), exporter=InMemorySpanExporter(), force=True)


def fake_request(*, error: bool = False) -> None:
    """A request-shaped span tree with the attributes the metrics read."""

    @tracing.stage("rag.request")
    def request() -> None:
        span = tracing.current_span()
        span.set_attribute("rag.status", "answered" if not error else "error")
        span.set_attribute("rag.abstain_reason", None)
        span.set_attribute("cache.hit", False)
        hybrid()
        rerank()
        generate()
        if error:
            failing()

    @tracing.stage("retrieval.hybrid", kind="RETRIEVER")
    def hybrid() -> None:
        tracing.current_span().set_attribute("rag.overlap_ratio", 0.4)

    @tracing.stage("rerank", kind="RERANKER")
    def rerank() -> None:
        tracing.current_span().set_attribute("rag.score_top1", 0.93)

    @tracing.stage("llm.generate", kind="LLM")
    def generate() -> None:
        span = tracing.current_span()
        span.set_attribute("gen_ai.request.model", "fake:gen")
        span.set_attribute("gen_ai.usage.input_tokens", 1000)
        span.set_attribute("gen_ai.usage.output_tokens", 100)
        span.set_attribute("rag.cost_usd", 0.0045)

    @tracing.stage("check.code", kind="TOOL")
    def failing() -> None:
        raise ValueError("boom")

    try:
        request()
    except ValueError:
        pass


def test_every_spec_metric_appears_in_the_exporter_output(readers) -> None:
    exporter, reader = readers
    fake_request()
    fake_request(error=True)
    metrics.record_feedback("up")
    got = points(reader)
    assert SPEC_METRICS <= set(got), sorted(SPEC_METRICS - set(got))
    requests = {tuple(sorted(p.attributes.items())): p.value for p in got["rag.requests"]}
    assert requests[(("abstain_reason", "none"), ("cache.hit", False), ("status", "answered"))] == 1
    assert requests[(("abstain_reason", "none"), ("cache.hit", False), ("status", "error"))] == 1
    tokens = {
        (p.attributes["direction"], p.attributes["gen_ai.request.model"]): p.value
        for p in got["rag.tokens"]
    }
    assert tokens[("input", "fake:gen")] == 2000 and tokens[("output", "fake:gen")] == 200
    cost = next(
        p for p in got["rag.cost_usd"] if p.attributes["gen_ai.request.model"] == "fake:gen"
    )
    assert cost.value == pytest.approx(0.009)
    latency = {p.attributes["span.name"]: p.count for p in got["rag.stage.latency_ms"]}
    assert (
        latency["rag.request"] == 2 and latency["llm.generate"] == 2 and latency["check.code"] == 1
    )
    assert next(p for p in got["rag.rerank_score_top1"]).sum == pytest.approx(0.93 * 2)
    assert next(p for p in got["rag.retrieval.overlap_ratio"]).count == 2
    errors = {
        (p.attributes["span.name"], p.attributes["exception.type"]): p.value
        for p in got["rag.errors"]
    }
    assert errors[("check.code", "ValueError")] == 1 and errors[("rag.request", "ValueError")] == 1
    feedback = next(p for p in got["rag.user_feedback"])
    assert feedback.attributes["rating"] == "up" and feedback.value == 1
    # the spans still reached the trace exporter: metrics are derived, not a second record
    assert len(exporter.get_finished_spans()) == 9


def test_nothing_is_recorded_twice_when_tracing_is_reconfigured(readers) -> None:
    _exporter, reader = readers
    fake_request()
    got = points(reader)
    assert sum(p.value for p in got["rag.requests"]) == 1
    assert (
        sum(
            p.count
            for p in got["rag.stage.latency_ms"]
            if p.attributes["span.name"] == "rag.request"
        )
        == 1
    )


def test_error_lines_name_the_span_and_the_exception_once_without_a_stack_trace(
    readers, capsys: pytest.CaptureFixture[str]
) -> None:
    tracing.configure_logging()
    fake_request(error=True)
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    errors = [line for line in lines if line["level"] == "error"]
    assert len(errors) == 1, errors  # the failing stage logs once; the parents re-raise in silence
    err = errors[0]
    assert err["span"] == "check.code" and err["exception_type"] == "ValueError"
    assert err["message"] == "boom" and "Traceback" not in out and "exception" not in err
    # a group that wraps the logged exception is quiet too; a new one logs
    assert tracing.log_error_once(ExceptionGroup("g", [ValueError("other")]), span="x") is True
    assert len(err["trace_id"]) == 32 and len(err["span_id"]) == 16


def test_debug_lines_appear_only_for_a_debug_request(
    readers, capsys: pytest.CaptureFixture[str]
) -> None:
    tracing.configure_logging()
    log = tracing.get_logger("test")

    @tracing.stage("relevance.gate")
    def stage() -> None:
        log.debug("stage", stage="relevance.gate", latency_ms=1)

    stage()
    with tracing.request_context(request_id="r1", index_version="v3", debug=True):
        stage()
        log.info("inside")
    stage()
    lines = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    debug = [line for line in lines if line["level"] == "debug"]
    assert len(debug) == 1 and debug[0]["request_id"] == "r1" and debug[0]["index_version"] == "v3"
    info = next(line for line in lines if line["event"] == "inside")
    assert info["request_id"] == "r1" and info["index_version"] == "v3" and "debug" not in info
    # after the context, lines carry neither key
    log.info("outside")
    outside = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "request_id" not in outside
