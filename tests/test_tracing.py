"""Done-when tests for the @stage decorator and OTLP setup. Spec Section 7.3. Issue #2."""

import asyncio
from dataclasses import dataclass

import pytest
import structlog
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from flp_rag import tracing
from flp_rag.settings import load_settings


@dataclass(frozen=True)
class Query:
    text: str
    k: int

    def to_attrs(self) -> dict:
        return {"rag.question_raw": self.text, "rag.top_k": self.k}


@dataclass(frozen=True)
class Hits:
    ids: list[str]

    def to_attrs(self) -> dict:
        return {"rag.n_hits": len(self.ids)}


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    tracing.configure_tracing(load_settings(), exporter=exp, force=True)
    yield exp
    exp.clear()


def _only_span(exporter: InMemorySpanExporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def test_stage_sets_name_kind_and_latency(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("retrieval.dense", kind="RETRIEVER")
    def retrieve(q: Query) -> Hits:
        return Hits(ids=["a", "b"])

    retrieve(Query("readonly", 30))

    span = _only_span(exporter)
    assert span.name == "retrieval.dense"
    assert span.attributes["openinference.span.kind"] == "RETRIEVER"
    assert isinstance(span.attributes["rag.latency_ms"], int)
    assert span.status.status_code is StatusCode.OK or span.status.status_code is StatusCode.UNSET


def test_stage_records_input_and_output_attrs(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("retrieval.dense", kind="RETRIEVER")
    def retrieve(q: Query) -> Hits:
        return Hits(ids=["a", "b", "c"])

    retrieve(Query("enums", 10))

    attrs = _only_span(exporter).attributes
    assert attrs["rag.question_raw"] == "enums"
    assert attrs["rag.top_k"] == 10
    assert attrs["rag.n_hits"] == 3


def test_stage_captures_content_when_enabled(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("query.classify", kind="LLM")
    def classify(q: Query) -> str:
        return "book_question"

    classify(Query("what is match?", 1))

    attrs = _only_span(exporter).attributes
    assert "input.value" in attrs
    assert attrs["output.value"] == "book_question"


def test_stage_skips_content_when_disabled() -> None:
    # A forced reconfigure shuts the old provider down, and with it the old exporter.
    exporter = InMemorySpanExporter()
    settings = load_settings()
    quiet = settings.model_copy(
        update={"trace": settings.trace.model_copy(update={"capture_content": False})}
    )
    tracing.configure_tracing(quiet, exporter=exporter, force=True)

    @tracing.stage("query.classify", kind="LLM")
    def classify(q: Query) -> str:
        return "smalltalk"

    classify(Query("hi", 1))

    attrs = _only_span(exporter).attributes
    assert "input.value" not in attrs
    assert "output.value" not in attrs


def test_stage_records_exception_and_reraises(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("ingest.parse", kind="CHAIN")
    def parse(path: str) -> None:
        raise ValueError("no text layer")

    with pytest.raises(ValueError, match="no text layer"):
        parse("book.pdf")

    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert "rag.latency_ms" in span.attributes
    events = [e for e in span.events if e.name == "exception"]
    assert events and events[0].attributes["exception.type"] == "ValueError"


def test_stage_wraps_async_functions(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("retrieval.sparse", kind="RETRIEVER")
    async def retrieve(q: Query) -> Hits:
        await asyncio.sleep(0)
        return Hits(ids=["x"])

    asyncio.run(retrieve(Query("fibers", 5)))

    span = _only_span(exporter)
    assert span.name == "retrieval.sparse"
    assert span.attributes["rag.n_hits"] == 1


def test_current_span_allows_extra_attributes(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("relevance.gate", kind="CHAIN")
    def gate(q: Query) -> str:
        tracing.current_span().set_attribute("rag.gate_decision", "pass")
        return "pass"

    gate(Query("q", 1))

    assert _only_span(exporter).attributes["rag.gate_decision"] == "pass"


def test_nested_stages_form_a_tree(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("retrieval.dense", kind="RETRIEVER")
    def dense(q: Query) -> Hits:
        return Hits(ids=["a"])

    @tracing.stage("retrieval.hybrid", kind="RETRIEVER")
    def hybrid(q: Query) -> Hits:
        return dense(q)

    hybrid(Query("q", 1))

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"retrieval.hybrid", "retrieval.dense"}
    child, parent = spans["retrieval.dense"], spans["retrieval.hybrid"]
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


def test_resource_carries_service_identity(exporter: InMemorySpanExporter) -> None:
    @tracing.stage("hello", kind="CHAIN")
    def hello() -> str:
        return "hi"

    hello()

    res = _only_span(exporter).resource.attributes
    assert res["service.name"] == "flp-rag"
    assert res["deployment.environment"] == "dev"
    assert isinstance(res["service.version"], str) and res["service.version"]


def test_structlog_processor_injects_trace_ids(exporter: InMemorySpanExporter) -> None:
    captured: dict = {}

    @tracing.stage("response.build", kind="CHAIN")
    def build() -> None:
        captured.update(tracing.add_trace_context(None, "info", {"event": "request"}))

    build()

    span = _only_span(exporter)
    assert captured["trace_id"] == format(span.context.trace_id, "032x")
    assert captured["span_id"] == format(span.context.span_id, "016x")

    outside = tracing.add_trace_context(None, "info", {"event": "boot"})
    assert "trace_id" not in outside


def test_configure_logging_binds_processor() -> None:
    tracing.configure_logging()
    processors = structlog.get_config()["processors"]
    assert tracing.add_trace_context in processors
