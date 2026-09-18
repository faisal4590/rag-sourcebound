# Milestone 0 smoke: one traced function. Spec Section 11, Milestone 0. Issue #3.
"""Run `python -m flp_rag.hello` and find the span named `hello` in Phoenix."""

import argparse

import structlog
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from flp_rag import tracing
from flp_rag.settings import load_settings


@tracing.stage("hello", kind="CHAIN")
def hello(name: str = "world") -> str:
    greeting = f"hello, {name}"
    tracing.current_span().set_attribute("rag.greeting", greeting)
    return greeting


class _LastSpan(SpanProcessor):
    """Remember the last ended span so the script can print its trace id."""

    name = ""
    trace_id = ""

    def on_end(self, span: ReadableSpan) -> None:
        self.name = span.name
        self.trace_id = format(span.context.trace_id, "032x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="world")
    args = parser.parse_args(argv)

    settings = load_settings()
    tracing.configure_logging()
    provider = tracing.configure_tracing(settings)
    recorder = _LastSpan()
    provider.add_span_processor(recorder)
    log = structlog.get_logger()

    greeting = hello(args.name)
    tracing.shutdown_tracing()
    log.info(
        "hello",
        greeting=greeting,
        span_name=recorder.name,
        trace_id=recorder.trace_id,
        exporter=settings.trace.exporter_endpoint,
        config_hash=settings.config_hash,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
