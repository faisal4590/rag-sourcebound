"""Done-when tests for the @stage decorator and OTLP setup. Spec Section 7.3. Issue #2."""

import asyncio
import json
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
    assert len(events) == 1, "the SDK must not record the exception a second time"
    assert events[0].attributes["exception.type"] == "ValueError"


def test_attribute_and_content_limits_come_from_config() -> None:
    exporter = InMemorySpanExporter()
    settings = load_settings()
    tight = settings.model_copy(
        update={"trace": settings.trace.model_copy(
            update={"content_max_chars": 5, "attr_string_max_chars": 20})}
    )
    tracing.configure_tracing(tight, exporter=exporter, force=True)

    @tracing.stage("query.classify", kind="LLM")
    def classify(q: Query) -> str:
        return "book_question"

    classify(Query("readonly properties are set once", 1))

    attrs = _only_span(exporter).attributes
    assert attrs["rag.question_raw"] == "readonly…(truncated)"  # 20 characters, marked (#29)
    assert attrs["output.value"] == "book_"


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


def test_logger_writes_to_the_current_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    # The stream is looked up per call, so a swapped-and-closed stdout never breaks later logs.
    tracing.configure_logging()
    structlog.get_logger().info("first")
    assert "first" in capsys.readouterr().out
    structlog.get_logger().warning("second")
    assert "second" in capsys.readouterr().out



# --------------------------------------------------------------------------- issue #29 rules


def test_attribute_rules_sentinel_truncation_json_limit_and_lists(
    exporter: InMemorySpanExporter,
) -> None:
    """Part 2: a None becomes the registry sentinel, a long text is cut and marked, a list is
    JSON within the JSON limit, an unknown None is dropped."""

    @tracing.stage("relevance.gate")
    def gate() -> None:
        span = tracing.current_span()
        span.set_attribute("rag.abstain_reason", None)  # text sentinel
        span.set_attribute("rag.chapter_hint", None)  # number sentinel
        span.set_attribute("rag.unknown_thing", None)  # no sentinel known: dropped
        span.set_attribute("rag.gate_reason", "x" * 1500)
        span.set_attribute("rag.scores_all", [(f"c{i}", 0.123456) for i in range(400)])
        span.set_attribute("guard.injection_patterns", ("a", "b"))

    gate()
    a = _only_span(exporter).attributes
    assert a["rag.abstain_reason"] == "none" and a["rag.chapter_hint"] == -1
    assert "rag.unknown_thing" not in a
    assert len(a["rag.gate_reason"]) == 1000 and a["rag.gate_reason"].endswith("…(truncated)")
    scores = json.loads(a["rag.scores_all"])  # a cut list stays valid JSON
    assert len(a["rag.scores_all"]) <= 4000 and scores[-1] == "…(truncated)" and len(scores) > 100
    assert a["guard.injection_patterns"] == '["a", "b"]'


def test_content_switch_gates_every_content_attribute_through_current_span() -> None:
    settings = load_settings()
    off = settings.model_copy(
        update={"trace": settings.trace.model_copy(update={"capture_content": False})}
    )
    exp = InMemorySpanExporter()
    tracing.configure_tracing(off, exporter=exp, force=True)

    @tracing.stage("query.understand")
    def understand(q: Query) -> Hits:
        span = tracing.current_span()
        span.set_attribute("rag.question_standalone", "the rewrite")
        span.set_attribute("retrieval.documents.0.document.content", "book text")
        span.set_attribute("rag.n_variants", 2)
        return Hits(["a"])

    understand(Query("the question", 3))
    a = exp.get_finished_spans()[0].attributes
    assert "rag.question_raw" not in a  # from Query.to_attrs
    assert "rag.question_standalone" not in a and "retrieval.documents.0.document.content" not in a
    assert "input.value" not in a and "output.value" not in a
    assert a["rag.n_variants"] == 2 and a["rag.top_k"] == 3
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)


def test_instrumentor_config_follows_the_content_switch() -> None:
    settings = load_settings()
    off = settings.model_copy(
        update={"trace": settings.trace.model_copy(update={"capture_content": False})}
    )
    config = tracing.instrumentor_config(off)
    assert config.hide_inputs is True and config.hide_outputs is True
    assert tracing.instrumentor_config(settings).hide_inputs is False
