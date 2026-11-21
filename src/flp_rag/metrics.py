# OpenTelemetry metrics derived from spans, spec Section 7.7. Issue #30.
"""Metrics.

Every metric of spec 7.7 is derived from finished spans by one span processor, so nothing is
counted twice and the code never counts what the trace already holds:

| metric | from |
| `rag.requests` (status, abstain_reason, cache.hit, http.route) | the root span `rag.request` |
| `rag.stage.latency_ms` (span.name) | every span |
| `rag.tokens` (gen_ai.request.model, direction, span.name) | `LLM` spans |
| `rag.cost_usd` (gen_ai.request.model) | `LLM` spans and `embed.batch` |
| `rag.rerank_score_top1` | the `rerank` span |
| `rag.retrieval.overlap_ratio` | the `retrieval.hybrid` span |
| `rag.errors` (span.name, exception.type) | spans with status ERROR |
| `rag.user_feedback` (rating) | `POST /feedback`, the one metric not read from a span |

The ratios of the spec (abstain rate, faithfulness fail rate, cache hit rate) are computed by
the backend from these. `configure_metrics` installs the meter provider: an OTLP exporter when
`trace.metrics_endpoint` is set, the given reader in tests, otherwise a provider that records
and exports nowhere.
"""

from __future__ import annotations

from typing import Any

from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode

from flp_rag.settings import Settings

METER_NAME = "flp-rag"
LLM_KIND = "LLM"
UNIT_BUCKETS = [0.05 * i for i in range(1, 21)]  # 0.05 .. 1.00 for the two 0..1 histograms
LATENCY_BUCKETS_MS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000]
EMBED_BATCH, REQUEST_SPAN, RERANK_SPAN, HYBRID_SPAN = (
    "embed.batch",
    "rag.request",
    "rerank",
    "retrieval.hybrid",
)


class _State:
    provider: MeterProvider | None = None
    instruments: Instruments | None = None


class Instruments:
    """The instruments of spec 7.7, created once per meter provider."""

    def __init__(self, provider: MeterProvider) -> None:
        meter = provider.get_meter(METER_NAME)
        self.requests = meter.create_counter(
            "rag.requests", unit="{request}", description="requests"
        )
        self.latency = meter.create_histogram(
            "rag.stage.latency_ms",
            unit="ms",
            description="span duration",
            explicit_bucket_boundaries_advisory=LATENCY_BUCKETS_MS,
        )
        self.tokens = meter.create_counter("rag.tokens", unit="{token}", description="model tokens")
        self.cost = meter.create_counter("rag.cost_usd", unit="USD", description="model cost")
        self.rerank_top1 = meter.create_histogram(
            "rag.rerank_score_top1",
            unit="1",
            description="best reranker score",
            explicit_bucket_boundaries_advisory=UNIT_BUCKETS,
        )
        self.overlap = meter.create_histogram(
            "rag.retrieval.overlap_ratio",
            unit="1",
            description="dense and sparse overlap",
            explicit_bucket_boundaries_advisory=UNIT_BUCKETS,
        )
        self.errors = meter.create_counter(
            "rag.errors", unit="{error}", description="spans in error"
        )
        self.feedback = meter.create_counter(
            "rag.user_feedback", unit="{rating}", description="user feedback"
        )


def configure_metrics(
    settings: Settings, resource: Resource, reader: MetricReader | None = None
) -> MeterProvider:
    """Install the meter provider. Tests pass an `InMemoryMetricReader`; production exports
    over OTLP when `trace.metrics_endpoint` is set."""
    readers: list[MetricReader] = []
    if reader is not None:
        readers.append(reader)
    elif settings.trace.metrics_endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=settings.trace.metrics_endpoint,
                    timeout=settings.trace.export_timeout_s,
                ),
                export_interval_millis=int(settings.trace.metrics_export_interval_s * 1000),
            )
        )
    if _State.provider is not None:
        _State.provider.shutdown()
    provider = MeterProvider(resource=resource, metric_readers=readers)
    if not isinstance(otel_metrics.get_meter_provider(), MeterProvider):
        otel_metrics.set_meter_provider(provider)
    _State.provider = provider
    _State.instruments = Instruments(provider)
    return provider


def shutdown_metrics() -> None:
    if _State.provider is not None:
        _State.provider.force_flush()
        _State.provider.shutdown()
        _State.provider = None
        _State.instruments = None


def instruments() -> Instruments | None:
    return _State.instruments


def record_feedback(rating: str) -> None:
    """`rag.user_feedback` by rating, from `POST /feedback` (the score itself lands on the
    trace in issue #31)."""
    if _State.instruments is not None:
        _State.instruments.feedback.add(1, {"rating": rating})


# --------------------------------------------------------------------------- the span processor


def exception_type(span: ReadableSpan) -> str:
    """The recorded exception's type from the `exception` event; `none` for an ERROR status
    without one (an HTTP error status set by the middleware), so the label set stays small."""
    for event in span.events or ():
        if event.name == "exception":
            kind = (event.attributes or {}).get("exception.type")
            if kind:
                return str(kind).rpartition(".")[2]
    return "none"


_warned = False


class SpanMetricsProcessor(SpanProcessor):
    """Derives every metric from a finished span. Registered on the tracer provider by
    `tracing.configure_tracing`, after the exporter, so it sees what the trace sees."""

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        inst = _State.instruments
        if inst is None:
            return
        try:
            self._record(inst, span)
        except Exception as exc:  # noqa: BLE001 - a metric must never fail a request
            global _warned
            if not _warned:
                _warned = True
                import structlog

                structlog.get_logger(__name__).warning(
                    "metrics skipped", span=span.name, error=type(exc).__name__
                )

    @staticmethod
    def _record(inst: Instruments, span: ReadableSpan) -> None:
        attrs = dict(span.attributes or {})
        name = span.name
        if span.start_time is not None and span.end_time is not None:
            inst.latency.record((span.end_time - span.start_time) / 1e6, {"span.name": name})
        if name == REQUEST_SPAN:
            inst.requests.add(
                1,
                {
                    "status": str(attrs.get("rag.status", "none")),
                    "abstain_reason": str(attrs.get("rag.abstain_reason", "none")),
                    "cache.hit": bool(attrs.get("cache.hit", False)),
                    "http.route": str(attrs.get("http.route", "none")),
                },
            )
        model = str(attrs.get("gen_ai.request.model", "none"))
        if attrs.get("openinference.span.kind") == LLM_KIND:
            for direction in ("input", "output"):
                count = attrs.get(f"gen_ai.usage.{direction}_tokens")
                if isinstance(count, int | float):
                    inst.tokens.add(
                        int(count),
                        {"gen_ai.request.model": model, "direction": direction, "span.name": name},
                    )
        if attrs.get("openinference.span.kind") == LLM_KIND or name == EMBED_BATCH:
            cost = attrs.get("rag.cost_usd")
            if isinstance(cost, int | float):
                inst.cost.add(float(cost), {"gen_ai.request.model": model})
        if name == RERANK_SPAN and isinstance(attrs.get("rag.score_top1"), int | float):
            inst.rerank_top1.record(float(attrs["rag.score_top1"]))
        if name == HYBRID_SPAN and isinstance(attrs.get("rag.overlap_ratio"), int | float):
            inst.overlap.record(float(attrs["rag.overlap_ratio"]))
        if span.status.status_code == StatusCode.ERROR:
            inst.errors.add(1, {"span.name": name, "exception.type": exception_type(span)})

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
