"""Done-when test for Milestone 0: one trace with one span named `hello`. Issue #3."""

import json
import subprocess
import sys

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import hello, tracing
from flp_rag.settings import load_settings


def test_hello_emits_one_span_named_hello() -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(load_settings(), exporter=exporter, force=True)

    greeting = hello.hello("Phoenix")

    spans = exporter.get_finished_spans()
    assert greeting == "hello, Phoenix"
    assert [s.name for s in spans] == ["hello"]
    span = spans[0]
    assert span.attributes["openinference.span.kind"] == "CHAIN"
    assert span.attributes["rag.greeting"] == "hello, Phoenix"
    assert span.resource.attributes["service.name"] == "flp-rag"


def test_module_runs_and_prints_trace_id() -> None:
    # Real process, real exporter config. If Phoenix is down the export fails inside the batch
    # processor within trace.export_timeout_s, and the script still exits 0 with a trace id.
    out = subprocess.run(
        [sys.executable, "-m", "flp_rag.hello"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["event"] == "hello"
    assert len(line["trace_id"]) == 32 and int(line["trace_id"], 16) != 0
    assert line["span_name"] == "hello"
