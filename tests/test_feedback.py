# Issue #31: POST /feedback writes `user_feedback` onto the trace; the CLI asks with one keypress.
# ruff: noqa: F811 - the `settings` and `exporter` fixtures of other modules are parameters here
"""The score lands on the root span of the trace as a Phoenix span annotation (1 for up, -1 for
down), the endpoint never fails on a Phoenix error, `feedback.enabled: false` turns the side
effect off, and the CLI prompt posts the rating of a `u` or `d` keypress and skips on Enter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import cli, tracing
from flp_rag.api import feedback
from flp_rag.settings import Settings, load_settings
from tests.test_api import _client, exporter  # noqa: F401 - fixture
from tests.test_cli_ask import FakeApi, run, settings  # noqa: F401 - fixture
from tests.test_cli_ask import body as record
from tests.test_metrics import points

TRACE = "a" * 32
SPAN = "b" * 16


class FakePhoenix:
    """Answers the root-span GraphQL lookup and records the annotation POSTs."""

    def __init__(self, *, known: set[str] | None = None, down: bool = False) -> None:
        self.known = known if known is not None else {TRACE}
        self.down = down
        self.posts: list[dict[str, Any]] = []
        self.sync: list[str | None] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            if self.down:
                raise httpx.ConnectError("phoenix down")
            if request.url.path == "/graphql":
                query = json.loads(request.content)["query"]
                found = [t for t in self.known if f"trace_id == '{t}'" in query]
                spans = [{"node": {"context": {"spanId": SPAN, "traceId": t}}} for t in found]
                node = {"name": "flp-rag", "spans": {"edges": spans}}
                return httpx.Response(200, json={"data": {"projects": {"edges": [{"node": node}]}}})
            assert request.url.path == "/v1/span_annotations"
            self.sync.append(request.url.params.get("sync"))
            self.posts.append(json.loads(request.content))
            return httpx.Response(200, json={"data": [{"id": "ann1"}]})

        return httpx.MockTransport(handle)


def test_write_feedback_puts_the_score_on_the_root_span() -> None:
    phoenix = FakePhoenix()
    settings = load_settings()
    status = feedback.write_feedback(
        TRACE, "down", "wrong page", settings, transport=phoenix.transport()
    )
    assert status == "written" and len(phoenix.posts) == 1 and phoenix.sync == ["true"]
    (ann,) = phoenix.posts[0]["data"]
    assert ann["name"] == "user_feedback" and ann["span_id"] == SPAN
    assert ann["annotator_kind"] == "HUMAN"
    assert ann["result"] == {"score": -1, "label": "down", "explanation": "wrong page"}
    assert ann["identifier"] == f"user_feedback:{TRACE}"  # a second rating replaces the first
    up = feedback.write_feedback(TRACE, "up", None, settings, transport=phoenix.transport())
    assert up == "written" and phoenix.posts[1]["data"][0]["result"] == {"score": 1, "label": "up"}


def test_write_feedback_reports_a_missing_trace_and_a_phoenix_error_without_raising() -> None:
    settings = load_settings()
    missing = FakePhoenix(known=set())
    assert (
        feedback.write_feedback(TRACE, "up", None, settings, transport=missing.transport())
        == "missing_trace"
    )
    assert missing.posts == []
    down = FakePhoenix(down=True)
    assert (
        feedback.write_feedback(TRACE, "up", None, settings, transport=down.transport())
        == "ConnectError"
    )
    assert (
        feedback.write_feedback("not-a-trace-id", "up", None, settings, transport=down.transport())
        == "invalid_input"
    )
    # a trace Phoenix exports a moment later is looked up once more
    late = FakePhoenix(known=set())
    quick = settings.model_copy(
        update={"feedback": settings.feedback.model_copy(update={"missing_retry_s": 0.01})}
    )
    calls: list[str] = []
    real = late.transport()

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 2:  # the second lookup finds it
            late.known = {TRACE}
        return real.handle_request(request)

    status = feedback.write_feedback(TRACE, "up", None, quick, transport=httpx.MockTransport(flaky))
    assert status == "written" and calls[:2] == ["/graphql", "/graphql"]
    # a long comment is cut to feedback.max_comment_chars
    long = FakePhoenix()
    feedback.write_feedback(TRACE, "down", "x" * 5000, settings, transport=long.transport())
    explanation = long.posts[0]["data"][0]["result"]["explanation"]
    assert len(explanation) == settings.feedback.max_comment_chars


def test_endpoint_writes_the_annotation_counts_the_metric_and_honours_the_switch(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    phoenix = FakePhoenix()
    reader = InMemoryMetricReader()
    exporter = InMemorySpanExporter()  # a fresh one: the forced reconfigure stops the fixture's
    tracing.configure_tracing(load_settings(), exporter=exporter, force=True, metric_reader=reader)
    with _client(tmp_path, exporter, extra={"phoenix_transport": phoenix.transport()}) as (
        client,
        _,
    ):
        r = client.post("/feedback", json={"trace_id": TRACE, "rating": "down", "comment": "wrong"})
    assert r.status_code == 200 and r.json() == {
        "ok": True,
        "trace_id": r.json()["trace_id"],
        "recorded": True,
    }
    assert len(phoenix.posts) == 1 and phoenix.posts[0]["data"][0]["result"]["score"] == -1
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    assert (
        root.attributes["rag.feedback_rating"] == "down"
        and root.attributes["rag.feedback_written"] == "written"
    )
    got = points(reader)
    assert next(p for p in got["rag.user_feedback"]).attributes["rating"] == "down"

    missing = FakePhoenix(known=set())
    (tmp_path / "missing").mkdir()
    with _client(
        tmp_path / "missing", exporter, extra={"phoenix_transport": missing.transport()}
    ) as (client, _):
        r = client.post("/feedback", json={"trace_id": TRACE, "rating": "up"})
    assert r.status_code == 200 and r.json()["recorded"] is False and missing.posts == []

    off = FakePhoenix()
    (tmp_path / "off").mkdir()
    with _client(
        tmp_path / "off",
        exporter,
        extra={"phoenix_transport": off.transport()},
        feedback_enabled=False,
    ) as (client, _):
        r = client.post("/feedback", json={"trace_id": TRACE, "rating": "up"})
    assert r.status_code == 200 and r.json()["recorded"] is False and off.posts == []
    tracing.configure_tracing(load_settings(), exporter=InMemorySpanExporter(), force=True)


# --------------------------------------------------------------------------- the CLI keypress


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "interactive", lambda: True)


def feedback_ok() -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "trace_id": "c" * 32, "recorded": True})


def test_cli_asks_once_and_posts_the_keypress(settings: Settings, monkeypatch, interactive) -> None:
    monkeypatch.setattr(cli, "read_key", lambda: "d")
    api = FakeApi(httpx.Response(200, json=record()), feedback_ok())
    result = run(api, "What is readonly?", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.output
    assert "Helpful?" in result.output and "feedback sent: down" in result.output
    assert len(api.requests) == 2 and api.requests[1].url.path == "/feedback"
    assert api.last_json == {"trace_id": record()["trace_id"], "rating": "down"}


def test_cli_skips_on_enter_and_never_asks_with_json_or_when_disabled(
    settings: Settings, monkeypatch, interactive
) -> None:
    monkeypatch.setattr(cli, "read_key", lambda: "\r")
    api = FakeApi(httpx.Response(200, json=record()))
    result = run(api, "q", monkeypatch=monkeypatch)
    assert result.exit_code == 0 and "Helpful?" in result.output and len(api.requests) == 1

    monkeypatch.setattr(cli, "read_key", lambda: "d")
    api = FakeApi(httpx.Response(200, json=record()))
    result = run(api, "q", "--json", monkeypatch=monkeypatch)
    assert result.exit_code == 0 and "Helpful?" not in result.output and len(api.requests) == 1

    off = settings.model_copy(
        update={"feedback": settings.feedback.model_copy(update={"enabled": False})}
    )
    monkeypatch.setattr(cli, "load_settings", lambda: off)
    api = FakeApi(httpx.Response(200, json=record()))
    result = run(api, "q", monkeypatch=monkeypatch)
    assert result.exit_code == 0 and "Helpful?" not in result.output and len(api.requests) == 1


def test_cli_reports_a_rating_the_server_could_not_record(
    settings: Settings, monkeypatch, interactive
) -> None:
    monkeypatch.setattr(cli, "read_key", lambda: "d")
    not_recorded = httpx.Response(200, json={"ok": True, "trace_id": "c" * 32, "recorded": False})
    api = FakeApi(httpx.Response(200, json=record()), not_recorded)
    result = run(api, "q", monkeypatch=monkeypatch)
    assert result.exit_code == 0 and "feedback not recorded" in result.output


def test_cli_feedback_failure_is_one_note_and_keeps_the_exit_code(
    settings: Settings, monkeypatch, interactive
) -> None:
    monkeypatch.setattr(cli, "read_key", lambda: "u")
    api = FakeApi(httpx.Response(200, json=record()), httpx.ConnectError("gone"))
    result = run(api, "q", monkeypatch=monkeypatch)
    assert result.exit_code == 0 and "feedback not sent" in result.output
    assert "Traceback" not in result.output


def test_key_mapping() -> None:
    assert cli.rating_for("u") == "up" and cli.rating_for("D") == "down"
    for key in ("\r", "\n", "x", "", "\x03", "\x04"):  # Enter, other keys, Ctrl-C, Ctrl-D
        assert cli.rating_for(key) is None
