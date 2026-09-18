# OpenTelemetry setup, Phoenix exporter, the @stage decorator, the structlog processor. Spec Section 7.3.
"""Tracing.

One decorator does all span work. Business code never imports the tracer directly.

    @stage("retrieval.dense", kind="RETRIEVER")
    def retrieve_dense(q: StandaloneQuery, cfg: RetrievalConfig) -> list[Candidate]:
        ...

The decorator opens a span, sets `openinference.span.kind`, copies `to_attrs()` from the input
and the output, records `input.value` / `output.value` when `trace.capture_content` is on,
records exceptions with status ERROR and re-raises, and adds `rag.latency_ms`.
"""

import functools
import inspect
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

from flp_rag.settings import Settings

SpanKind = Literal["CHAIN", "RETRIEVER", "RERANKER", "LLM", "TOOL", "EMBEDDING"]

SERVICE_NAME = "flp-rag"
ENVIRONMENT = "dev"

_Primitive = str | bool | int | float


@runtime_checkable
class HasAttrs(Protocol):
    def to_attrs(self) -> Mapping[str, Any]: ...


class _State:
    """Module state set by `configure_tracing`."""

    provider: TracerProvider | None = None
    capture_content: bool = True
    content_max_chars: int = 20_000
    attr_string_max_chars: int = 1_000


# --------------------------------------------------------------------------- setup


def configure_tracing(
    settings: Settings,
    exporter: SpanExporter | None = None,
    *,
    force: bool = False,
) -> TracerProvider:
    """Install the tracer provider once. Pass `exporter` to override OTLP (tests).

    `force=True` replaces a provider installed earlier in the same process.
    """
    if _State.provider is not None and not force:
        return _State.provider

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": git_sha(),
            "deployment.environment": ENVIRONMENT,
            "openinference.project.name": SERVICE_NAME,
        }
    )
    provider = TracerProvider(resource=resource)
    if exporter is None:
        exporter = OTLPSpanExporter(endpoint=settings.trace.exporter_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    if _State.provider is not None:
        _State.provider.shutdown()
    # The global provider can be set only once per process. Later calls go through our own handle.
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(provider)
    _State.provider = provider
    _State.capture_content = settings.trace.capture_content
    _State.content_max_chars = settings.trace.content_max_chars
    _State.attr_string_max_chars = settings.trace.attr_string_max_chars
    return provider


def shutdown_tracing() -> None:
    """Flush and stop the provider. Call at process exit so short scripts export their spans."""
    if _State.provider is not None:
        _State.provider.force_flush()
        _State.provider.shutdown()
        _State.provider = None


def _tracer() -> trace.Tracer:
    provider = _State.provider or trace.get_tracer_provider()
    return provider.get_tracer(SERVICE_NAME)


def current_span() -> Span | trace.Span:
    """The active span, for attributes that are not part of a stage's output type."""
    return trace.get_current_span()


@functools.cache
def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --------------------------------------------------------------------------- decorator

# The decorator records exceptions and sets status itself. Stop the SDK from doing it a second time.
_SPAN_OPTS = {"record_exception": False, "set_status_on_exception": False}


def stage(name: str, kind: SpanKind = "CHAIN") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap one pipeline stage in one span named `name`."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _tracer().start_as_current_span(name, **_SPAN_OPTS) as span:
                    _on_enter(span, kind, args, kwargs)
                    started = time.perf_counter()
                    try:
                        result = await fn(*args, **kwargs)
                    except BaseException as exc:
                        _on_error(span, exc, started)
                        raise
                    _on_exit(span, result, started)
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _tracer().start_as_current_span(name, **_SPAN_OPTS) as span:
                _on_enter(span, kind, args, kwargs)
                started = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    _on_error(span, exc, started)
                    raise
                _on_exit(span, result, started)
                return result

        return wrapper

    return decorate


def _on_enter(span: trace.Span, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    span.set_attribute("openinference.span.kind", kind)
    inputs = [*args, *kwargs.values()]
    for value in inputs:
        _set_attrs(span, value)
    if _State.capture_content:
        span.set_attribute("input.value", _content(inputs[0] if len(inputs) == 1 else inputs))


def _on_exit(span: trace.Span, result: Any, started: float) -> None:
    _set_attrs(span, result)
    if isinstance(result, (list, tuple)):
        span.set_attribute("rag.n_out", len(result))
    if _State.capture_content:
        span.set_attribute("output.value", _content(result))
    span.set_attribute("rag.latency_ms", _elapsed_ms(started))
    span.set_status(Status(StatusCode.OK))


def _on_error(span: trace.Span, exc: BaseException, started: float) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
    span.set_attribute("rag.latency_ms", _elapsed_ms(started))


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# --------------------------------------------------------------------------- attributes


def _set_attrs(span: trace.Span, value: Any) -> None:
    if not isinstance(value, HasAttrs):
        return
    for key, raw in value.to_attrs().items():
        clean = _primitive(raw)
        if clean is not None:
            span.set_attribute(key, clean)


def _primitive(value: Any) -> _Primitive | None:
    """Coerce one attribute value to what OTel accepts. Lists become short JSON. None is dropped."""
    if value is None:
        return None
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[: _State.attr_string_max_chars]
    return json.dumps(value, default=str)[: _State.attr_string_max_chars]


def _content(value: Any) -> str:
    """Serialize a stage input or output for `input.value` / `output.value`."""
    limit = _State.content_max_chars
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(_jsonable(value), default=str)[:limit]
    except (TypeError, ValueError):
        return repr(value)[:limit]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- logging


def add_trace_context(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: inject `trace_id` and `span_id` from the active span."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        return {
            **event_dict,
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    return event_dict


def configure_logging(level: int = logging.INFO) -> None:
    """structlog with JSON output to stdout. Every line carries trace_id and span_id."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_trace_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )
