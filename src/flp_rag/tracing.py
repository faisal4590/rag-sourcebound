# OpenTelemetry setup, Phoenix exporter, the @stage decorator, the structlog processor. Spec Section 7.3.
"""Tracing.

One decorator does all span work. Business code never imports the tracer directly.

    @stage("retrieval.dense", kind="RETRIEVER")
    def retrieve_dense(q: StandaloneQuery, cfg: RetrievalConfig) -> list[Candidate]:
        ...

The decorator opens a span, sets `openinference.span.kind`, copies `to_attrs()` from the input
and the output, records `input.value` / `output.value` when `trace.capture_content` is on,
records exceptions with status ERROR and re-raises, and adds `rag.latency_ms`.

Every attribute, from the decorator or from `current_span().set_attribute(...)`, passes the
rules of issue #29 (`span_registry.yaml`): a None becomes the attribute's sentinel (or is
dropped when the registry knows none), a list or record becomes a JSON string of at most
`trace.attr_json_max_chars`, a text is cut at `trace.attr_string_max_chars` and ends with the
truncated marker, and a content attribute (question text, prompts, book text) is dropped when
`trace.capture_content` is false. `current_span()` returns a thin proxy that applies them.
"""

import contextlib
import contextvars
import functools
import inspect
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode

from flp_rag import metrics as metrics_mod
from flp_rag.settings import Settings
from flp_rag.span_registry import CONTENT_VALUES, Registry, load_registry

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
    attr_json_max_chars: int = 4_000


def _registry() -> Registry:
    return load_registry()


# --------------------------------------------------------------------------- setup


def configure_tracing(
    settings: Settings,
    exporter: SpanExporter | None = None,
    *,
    force: bool = False,
    metric_reader: MetricReader | None = None,
) -> TracerProvider:
    """Install the tracer provider once, and with it the meter provider and the span processor
    that derives the metrics of spec 7.7 from finished spans (issue #30). Pass `exporter` to
    override OTLP (tests), `metric_reader` to read the metrics in a test.

    `force=True` replaces a provider installed earlier in the same process.
    """
    if _State.provider is not None and not force:
        return _State.provider
    load_registry()  # a broken registry file fails here, at start, never inside a request

    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "service.version": git_sha(),
            "deployment.environment": ENVIRONMENT,
            "openinference.project.name": SERVICE_NAME,
        }
    )
    # The SDK keeps the first `max_attributes` attributes and drops the rest (oldest first), so
    # the spec kind set at span start would be the first to go on a span with 30 documents.
    provider = TracerProvider(
        resource=resource,
        span_limits=SpanLimits(max_attributes=settings.trace.max_span_attributes),
    )
    if exporter is None:
        exporter = OTLPSpanExporter(
            endpoint=settings.trace.exporter_endpoint, timeout=settings.trace.export_timeout_s
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter, export_timeout_millis=int(settings.trace.export_timeout_s * 1000)
            )
        )
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    if _State.provider is not None:
        _State.provider.shutdown()  # the old tracer first, so its processor records no more
    metrics_mod.configure_metrics(settings, resource, reader=metric_reader)
    provider.add_span_processor(metrics_mod.SpanMetricsProcessor())
    # The global provider can be set only once per process. Later calls go through our own handle.
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(provider)
    _State.provider = provider
    _State.capture_content = settings.trace.capture_content
    _State.content_max_chars = settings.trace.content_max_chars
    _State.attr_string_max_chars = settings.trace.attr_string_max_chars
    _State.attr_json_max_chars = settings.trace.attr_json_max_chars
    if settings.trace.instrument_langchain:
        instrument_langchain(provider, instrumentor_config(settings))
    return provider


def instrumentor_config(settings: Settings) -> Any:
    """The OpenInference `TraceConfig` that follows `trace.capture_content` (issue #29 Part 6):
    with the switch off the instrumentor hides inputs and outputs too, so prompts never reach
    the backend through the second mechanism."""
    from openinference.instrumentation import TraceConfig

    hide = not settings.trace.capture_content
    return TraceConfig(hide_inputs=hide, hide_outputs=hide)


def instrument_langchain(provider: TracerProvider, config: Any | None = None) -> bool:
    """Opt-in OpenInference spans for LangChain and LangGraph internals, attached to our
    provider so they land in the same trace. Call after `configure_tracing`. Returns whether
    the instrumentor was installed."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError:  # pragma: no cover
        return False
    instrumentor = LangChainInstrumentor()
    # The instrumentor is a process-wide singleton bound to one provider. After a forced
    # reconfigure it must be re-attached, or LangChain spans keep going to the old provider.
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    kwargs: dict[str, Any] = {"tracer_provider": provider}
    if config is not None:
        kwargs["config"] = config
    instrumentor.instrument(**kwargs)
    return True


def shutdown_tracing() -> None:
    """Flush and stop the providers. Call at process exit so short scripts export their spans."""
    if _State.provider is not None:
        _State.provider.force_flush()
        _State.provider.shutdown()
        _State.provider = None
    metrics_mod.shutdown_metrics()


def _tracer() -> trace.Tracer:
    provider = _State.provider or trace.get_tracer_provider()
    return provider.get_tracer(SERVICE_NAME)


def get_tracer() -> trace.Tracer:
    """The project tracer, for the one caller that opens the root span by hand (Stage 9)."""
    return _tracer()


def current_span() -> "RuledSpan":
    """The active span, for attributes that are not part of a stage's output type. The proxy
    applies the attribute rules; everything else goes to the span."""
    return RuledSpan(trace.get_current_span())


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
    ruled = RuledSpan(span)
    ruled.set_attribute("openinference.span.kind", kind)
    inputs = [*args, *kwargs.values()]
    for value in inputs:
        _set_attrs(ruled, value)
    ruled.set_attribute("input.value", _content(inputs[0] if len(inputs) == 1 else inputs))


def _on_exit(span: trace.Span, result: Any, started: float) -> None:
    ruled = RuledSpan(span)
    _set_attrs(ruled, result)
    if isinstance(result, (list, tuple)):
        ruled.set_attribute("rag.n_out", len(result))
    ruled.set_attribute("output.value", _content(result))  # dropped by the rules when off
    ruled.set_attribute("rag.latency_ms", _elapsed_ms(started))
    span.set_status(Status(StatusCode.OK))


def _on_error(span: trace.Span, exc: BaseException, started: float) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
    RuledSpan(span).set_attribute("rag.latency_ms", _elapsed_ms(started))
    log_error_once(exc, span=getattr(span, "name", ""))


_LOGGED = "_flp_logged"  # the mark on an exception that has its ERROR line already


def _members(exc: BaseException) -> list[BaseException]:
    """The exception and, for an exception group, the ones inside it."""
    inner = getattr(exc, "exceptions", None)
    return [exc, *(e for sub in (inner or ()) for e in _members(sub))]


def log_error_once(exc: BaseException, *, span: str) -> bool:
    """Spec 7.8 rule 5: one ERROR line per failure with the exception type, the message, and
    the span name, never the stack trace (the span holds it). The parents a stage error passes
    through on its way out, and the API handler, see the mark and stay quiet. The mark is an
    attribute on the exception object itself (exceptions take no weak references), so it
    crosses the thread and task boundaries a stage error crosses, and a group that wraps a
    logged exception is quiet too."""
    if any(getattr(e, _LOGGED, False) for e in _members(exc)):
        return False
    with contextlib.suppress(AttributeError, TypeError):  # a rare type that refuses attributes
        setattr(exc, _LOGGED, True)
    content = _State.capture_content  # a message can carry the question or a provider body
    get_logger(__name__).error(
        "stage failed",
        span=span,
        exception_type=type(exc).__name__,
        message=str(exc)[: _State.attr_string_max_chars] if content else "(content off)",
    )
    return True


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# --------------------------------------------------------------------------- attributes


class RuledSpan:
    """A span whose `set_attribute` applies the registry rules. Every other call goes to the
    wrapped span. `raw` is the span itself for the few callers that need the SDK object."""

    __slots__ = ("raw",)

    def __init__(self, span: trace.Span) -> None:
        self.raw = span

    def set_attribute(self, key: str, value: Any) -> None:
        clean = clean_attribute(key, value)
        if clean is not None:
            self.raw.set_attribute(key, clean)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def __getattr__(self, name: str) -> Any:
        if name == "raw":  # an unpickled or copied proxy without its span: no recursion
            raise AttributeError(name)
        return getattr(self.raw, name)


def clean_attribute(key: str, value: Any) -> _Primitive | None:
    """One attribute value under the rules of issue #29 Part 2 and Part 6. None means drop."""
    registry = _registry()
    if not _State.capture_content and registry.is_content(key):
        return None
    if value is None:
        return registry.sentinel_for(key)
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        limit = _State.content_max_chars if key in CONTENT_VALUES else _State.attr_string_max_chars
        return _cut(value, limit, registry.truncated_marker)
    return _json_within(value, _State.attr_json_max_chars, registry.truncated_marker)


def _json_within(value: Any, limit: int, marker: str) -> str:
    """A JSON string of at most `limit` characters. A list that does not fit loses its last
    elements and ends with the marker as one more element, so the result stays valid JSON; a
    record that does not fit is cut as text (and is then not parseable, by design)."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return _cut(str(value), limit, marker)
    if len(text) <= limit:
        return text
    if isinstance(value, list | tuple):
        items = list(value)
        while items:
            items.pop()
            text = json.dumps([*items, marker], default=str, ensure_ascii=False)
            if len(text) <= limit:
                return text
    return _cut(text, limit, marker)


def _cut(text: str, limit: int, marker: str) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


def _set_attrs(span: RuledSpan, value: Any) -> None:
    if not isinstance(value, HasAttrs):
        return
    for key, raw in value.to_attrs().items():
        span.set_attribute(key, raw)


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

_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flp_request_id", default=None
)
_INDEX_VERSION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "flp_index_version", default=None
)
_DEBUG: contextvars.ContextVar[bool] = contextvars.ContextVar("flp_debug", default=False)


@contextlib.contextmanager
def request_context(
    *, request_id: str, index_version: str | None, debug: bool = False
) -> Iterator[None]:
    """The request facts every log line of the request carries (spec 7.8 rule 2) and the
    `debug` flag that lets its DEBUG lines through (rule 4). Stage 9 opens it per request."""
    tokens = (
        _REQUEST_ID.set(request_id),
        _INDEX_VERSION.set(index_version or None),
        _DEBUG.set(debug),
    )
    try:
        yield
    finally:
        _REQUEST_ID.reset(tokens[0])
        _INDEX_VERSION.reset(tokens[1])
        _DEBUG.reset(tokens[2])


def debug_requested() -> bool:
    return _DEBUG.get()


def set_debug(flag: bool) -> None:
    """Mark the current request as a debug request once its body is known (the handler runs in
    the request's own task, so the flag never leaks to another request)."""
    _DEBUG.set(flag)


def get_logger(name: str = SERVICE_NAME) -> Any:
    return structlog.get_logger(name)


def add_trace_context(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: `trace_id` and `span_id` from the active span, `request_id` and
    `index_version` from the request context (spec 7.8 rule 2)."""
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict = {
            **event_dict,
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    request_id, index_version = _REQUEST_ID.get(), _INDEX_VERSION.get()
    if request_id is not None and not event_dict.get("request_id"):
        event_dict["request_id"] = request_id
    if index_version is not None and not event_dict.get("index_version"):
        event_dict["index_version"] = index_version  # an empty value yields to the context
    return event_dict


def drop_debug_unless_requested(
    _logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor, spec 7.8 rule 4: DEBUG lines exist only inside a request that asked
    for `debug: true`; every other DEBUG line is dropped before rendering."""
    if method == "debug" and not _DEBUG.get():
        raise structlog.DropEvent
    return event_dict


class _StdoutLogger:
    """Writes to whatever `sys.stdout` is at call time. structlog's PrintLogger binds the stream
    once, which breaks after a test runner or a redirect swaps and closes it."""

    def msg(self, message: str) -> None:
        print(message, file=sys.stdout, flush=True)

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


def configure_logging(level: int = logging.DEBUG) -> None:
    """structlog with JSON output to stdout. Every line carries trace_id, span_id, request_id,
    and index_version; DEBUG lines pass only inside a debug request (`drop_debug_unless_requested`),
    so the level filter stays at DEBUG and the request decides."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            drop_debug_unless_requested,
            add_trace_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=lambda *_args: _StdoutLogger(),
        cache_logger_on_first_use=False,
    )
