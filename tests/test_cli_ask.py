# "Done when" test for `flp ask` as a thin client of the API. Issue #24, Milestone 3.
"""`flp ask` sends the question to `POST /ask`, prints the `Response` record by status, and exits
with 0 (answered), 2 (no information), or 1 (error). A fake HTTP transport stands in for the
API, so the tests need no server, no index, and no model. One live test needs the API."""

import json
import os
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from flp_rag import cli
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings

TRACE = "a" * 32
SOURCES = [
    {
        "label": "S1",
        "chunk_ids": ["d:06:0046"],
        "chapter_no": 6,
        "chapter_title": "Readonly Properties",
        "section_title": "Readonly and clone",
        "page_printed_start": 80,
        "page_printed_end": 81,
        "score": 0.87,
    },
    {
        "label": "S3",
        "chunk_ids": ["d:05:0038", "d:05:0039"],
        "chapter_no": 5,
        "chapter_title": "Property Promotion",
        "section_title": "How It Works + Defaults",
        "page_printed_start": 65,
        "page_printed_end": 70,
        "score": 0.6,
    },
]


def body(
    status: str = "answered",
    answer: str = "Readonly is set once [S1]. Promotion too [S3].",
    sources: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "status": status,
        "sources": sources if sources is not None else SOURCES,
        "trace_id": TRACE,
        "timings_ms": {"llm.generate": 900},
        "model": {"gen": "groq:x"},
        "index_version": "v3-test",
        "prompt_version": "ab12cd34",
        "debug": None,
        "detail": None,
        **extra,
    }


class FakeApi:
    """Records every request and answers with a scripted response."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            out = self.responses.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

        return httpx.MockTransport(handler)

    @property
    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def sse(*events: tuple[str, dict[str, Any]]) -> httpx.Response:
    raw = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=raw.encode())


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = load_settings()
    monkeypatch.setattr(cli, "load_settings", lambda: s)
    monkeypatch.delenv("FLP_API_URL", raising=False)
    return s


def run(
    api: FakeApi, *args: str, monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None
):
    monkeypatch.setattr(cli, "http_transport", api.transport())
    return CliRunner().invoke(cli.app, ["ask", *args], env=env)


# --------------------------------------------------------------------------- Part 1: options


def test_every_option_maps_to_its_request_field(settings, monkeypatch) -> None:
    api = FakeApi(httpx.Response(200, json=body()))
    result = run(
        api,
        "What does readonly do?",
        "--session",
        "s1",
        "--chapter",
        "6",
        "--chapter",
        "30",
        "--code",
        "--debug",
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 0, result.output
    sent = api.last_json
    assert sent["question"] == "What does readonly do?" and sent["session_id"] == "s1"
    assert sent["filters"] == {"chapter_no": [6, 30], "has_code": True}
    assert sent["debug"] is True and sent["stream"] is False
    assert api.requests[-1].url.path == "/ask" and api.requests[-1].method == "POST"

    api = FakeApi(httpx.Response(200, json=body()))
    run(api, "plain", monkeypatch=monkeypatch)
    sent = api.last_json
    assert sent["filters"] is None and sent["session_id"] is None and sent["debug"] is False


# --------------------------------------------------------------------------- Part 2: address


def test_api_address_order_flag_then_env_then_config(settings, monkeypatch) -> None:
    assert settings.cli.api_url == "http://localhost:8000"
    api = FakeApi(httpx.Response(200, json=body()))
    run(api, "q", monkeypatch=monkeypatch)
    assert str(api.requests[-1].url).startswith("http://localhost:8000/ask")

    api = FakeApi(httpx.Response(200, json=body()))
    run(api, "q", monkeypatch=monkeypatch, env={"FLP_API_URL": "http://env-host:9000"})
    assert str(api.requests[-1].url).startswith("http://env-host:9000/ask")

    api = FakeApi(httpx.Response(200, json=body()))
    run(
        api,
        "q",
        "--api-url",
        "http://flag-host:7000/",
        monkeypatch=monkeypatch,
        env={"FLP_API_URL": "http://env-host:9000"},
    )
    assert str(api.requests[-1].url).startswith("http://flag-host:7000/ask")


def test_client_timeout_is_longer_than_the_server_timeout(settings) -> None:
    assert settings.cli.timeout_s > settings.api.timeout_s
    with pytest.raises(ValueError, match="cli.timeout_s"):
        type(settings).model_validate(
            {
                **settings.model_dump(),
                "cli": {"api_url": "http://x", "timeout_s": settings.api.timeout_s},
            }
        )


def test_api_not_running_gives_a_clear_message_on_stderr(settings, monkeypatch) -> None:
    api = FakeApi(httpx.ConnectError("refused"))
    result = run(api, "What are enums?", monkeypatch=monkeypatch)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert (
        "The API is not running at http://localhost:8000. Start it with make serve."
        in result.stderr
    )


def test_http_400_and_504_print_the_message_and_exit_1(settings, monkeypatch) -> None:
    reject = body(
        "error",
        answer=settings.api.error_text,
        sources=[],
        detail="question must be 1-2000 characters",
    )
    result = run(FakeApi(httpx.Response(400, json=reject)), "x" * 2001, monkeypatch=monkeypatch)
    assert result.exit_code == 1
    assert "question must be 1-2000 characters" in result.stdout
    assert result.stdout.rstrip().endswith(f"trace_id={TRACE}")

    timeout = body(
        "error", answer=settings.api.error_text, sources=[], detail="request exceeded 20 s"
    )
    result = run(FakeApi(httpx.Response(504, json=timeout)), "slow", monkeypatch=monkeypatch)
    assert result.exit_code == 1
    assert settings.api.error_text in result.stdout and "request exceeded 20 s" in result.stdout
    assert result.stdout.rstrip().endswith(f"trace_id={TRACE}")

    # A 422 from the validation layer has the short body without an answer.
    short = {
        "status": "error",
        "answer": "",
        "trace_id": TRACE,
        "detail": "question: field required",
    }
    result = run(FakeApi(httpx.Response(422, json=short)), "q", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and "question: field required" in result.stdout


def test_odd_bodies_do_not_crash(settings, monkeypatch) -> None:
    # FastAPI's default 422 carries a list in `detail`.
    fastapi_422 = {"detail": [{"loc": ["body", "question"], "msg": "field required"}]}
    result = run(FakeApi(httpx.Response(422, json=fastapi_422)), "q", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and "field required" in result.stdout
    # A proxy may answer with a JSON string or no JSON at all.
    result = run(FakeApi(httpx.Response(502, json="bad gateway")), "q", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and result.stdout == ""
    assert "without a JSON record" in result.stderr
    result = run(FakeApi(httpx.Response(502, text="<html>")), "q", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and result.stdout == ""
    result = run(FakeApi(), "q", "--api-url", "not a url", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and "Bad API address" in result.stderr


def test_sources_print_in_numeric_label_order(settings, monkeypatch) -> None:
    s10 = {**SOURCES[0], "label": "S10"}
    s2 = {**SOURCES[1], "label": "S2"}
    api = FakeApi(httpx.Response(200, json=body(sources=[s10, s2])))
    result = run(api, "q", monkeypatch=monkeypatch)
    rows = [line for line in result.stdout.split("\n") if line.startswith("S")]
    assert [row.split()[0] for row in rows] == ["S2", "S10"]


# --------------------------------------------------------------------------- Part 3: by status


def test_answered_prints_answer_sources_table_and_trace_id_last(settings, monkeypatch) -> None:
    result = run(FakeApi(httpx.Response(200, json=body())), "q", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.output
    lines = result.stdout.rstrip("\n").split("\n")
    assert lines[0] == "Readonly is set once [S1]. Promotion too [S3]."
    rows = [line for line in lines if line.startswith(("S1", "S3"))]
    assert len(rows) == 2 and rows[0].startswith("S1") and rows[1].startswith("S3")
    assert "06 Readonly Properties" in rows[0] and "Readonly and clone" in rows[0]
    assert "80-81" in rows[0] and "0.87" in rows[0]
    assert "05 Property Promotion" in rows[1] and "65-70" in rows[1] and "0.60" in rows[1]
    assert lines[-1] == f"trace_id={TRACE}"
    assert "\x1b" not in result.stdout  # no color codes outside a terminal


def test_abstain_prints_exactly_the_text_and_the_trace_id_with_exit_2(
    settings, monkeypatch
) -> None:
    api = FakeApi(httpx.Response(200, json=body("no_information", answer=ABSTAIN_TEXT, sources=[])))
    result = run(api, "What is the capital of France?", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    assert result.stdout == f"{ABSTAIN_TEXT}\ntrace_id={TRACE}\n"


def test_debug_field_prints_between_the_table_and_the_trace_id(settings, monkeypatch) -> None:
    debug = {"abstain_reason": "gate.no_candidates", "gate": {"score_top1": None}}
    api = FakeApi(
        httpx.Response(
            200, json=body("no_information", answer=ABSTAIN_TEXT, sources=[], debug=debug)
        )
    )
    result = run(api, "What are fibers?", "--chapter", "99", "--debug", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    lines = result.stdout.rstrip("\n").split("\n")
    assert lines[0] == ABSTAIN_TEXT and lines[-1] == f"trace_id={TRACE}"
    assert "gate.no_candidates" in result.stdout
    assert api.last_json["filters"] == {"chapter_no": [99], "has_code": None}


def test_json_prints_the_raw_record(settings, monkeypatch) -> None:
    result = run(FakeApi(httpx.Response(200, json=body())), "q", "--json", monkeypatch=monkeypatch)
    assert result.exit_code == 0
    assert json.loads(result.stdout) == body()
    api = FakeApi(httpx.Response(200, json=body("no_information", answer=ABSTAIN_TEXT, sources=[])))
    result = run(api, "q", "--json", monkeypatch=monkeypatch)
    assert result.exit_code == 2 and json.loads(result.stdout)["status"] == "no_information"


# --------------------------------------------------------------------------- Part 4: streaming


def test_stream_prints_pieces_then_sources_and_trace_id(settings, monkeypatch) -> None:
    api = FakeApi(
        sse(
            ("token", {"text": "Readonly is "}),
            ("token", {"text": "set once [S1]."}),
            (
                "done",
                {"status": "answered", "sources": SOURCES[:1], "trace_id": TRACE, "answer": None},
            ),
        )
    )
    result = run(api, "q", "--stream", monkeypatch=monkeypatch)
    assert result.exit_code == 0, result.output
    assert api.last_json["stream"] is True
    lines = result.stdout.rstrip("\n").split("\n")
    assert lines[0] == "Readonly is set once [S1]."
    assert any(line.startswith("S1") and "80-81" in line for line in lines)
    assert lines[-1] == f"trace_id={TRACE}"


def test_stream_withdrawn_answer_prints_the_note_and_the_abstain(settings, monkeypatch) -> None:
    api = FakeApi(
        sse(
            ("token", {"text": "Readonly can be "}),
            ("token", {"text": "reset with reflection."}),
            (
                "done",
                {
                    "status": "no_information",
                    "sources": [],
                    "trace_id": TRACE,
                    "answer": ABSTAIN_TEXT,
                },
            ),
        )
    )
    result = run(api, "q", "--stream", monkeypatch=monkeypatch)
    assert result.exit_code == 2
    lines = result.stdout.rstrip("\n").split("\n")
    assert lines[0] == "Readonly can be reset with reflection."
    assert "Answer withdrawn after the checks" in lines[1]
    assert lines[-2] == ABSTAIN_TEXT and lines[-1] == f"trace_id={TRACE}"


def test_stream_falls_back_to_the_normal_response_with_a_note(settings, monkeypatch) -> None:
    result = run(
        FakeApi(httpx.Response(200, json=body())), "q", "--stream", monkeypatch=monkeypatch
    )
    assert result.exit_code == 0
    assert "did not stream" in result.stderr
    assert result.stdout.startswith("Readonly is set once [S1].")
    # An error answer is plain JSON by design: no misleading note.
    reject = body("error", answer=settings.api.error_text, sources=[], detail="too long")
    result = run(
        FakeApi(httpx.Response(400, json=reject)), "q", "--stream", monkeypatch=monkeypatch
    )
    assert result.exit_code == 1 and "did not stream" not in result.stderr


def test_stream_guards_malformed_streams_and_the_json_flag(settings, monkeypatch) -> None:
    broken = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=b"event: token\ndata: {no\n\n"
    )
    result = run(FakeApi(broken), "q", "--stream", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and "malformed stream" in result.stderr
    result = run(FakeApi(), "q", "--stream", "--json", monkeypatch=monkeypatch)
    assert result.exit_code == 1 and "cannot be combined" in result.stderr


# --------------------------------------------------------------------------- Parts 5 and 6


def test_retrieve_only_runs_the_local_retriever_and_makes_no_api_call(
    settings, monkeypatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_retrieve(question: str, chapter, code, top_k) -> None:
        calls.append({"question": question, "chapter": chapter, "code": code, "top_k": top_k})

    monkeypatch.setattr(cli, "retrieve", fake_retrieve)
    api = FakeApi()
    result = run(
        api,
        "How do backed enums work?",
        "--retrieve-only",
        "--chapter",
        "9",
        monkeypatch=monkeypatch,
    )
    assert result.exit_code == 0, result.output
    assert api.requests == [] and calls == [
        {"question": "How do backed enums work?", "chapter": [9], "code": None, "top_k": None}
    ]
    assert "does not answer" in result.stderr


def test_eval_and_calibrate_say_available_from_milestone_6() -> None:
    for command in ("eval", "calibrate"):
        result = CliRunner().invoke(cli.app, [command])
        assert result.exit_code == 1 and "Available from Milestone 6" in result.stderr


def test_ingest_dispatches_to_the_ingest_runner(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(settings, *, pdf_path, only, force):
        seen.update(only=only, force=force)
        return {}

    monkeypatch.setattr(cli.run_mod, "run", fake_run)
    monkeypatch.setattr(cli.run_mod, "verify_passed", lambda results: True)
    monkeypatch.setattr(cli.tracing, "configure_tracing", lambda s: None)
    monkeypatch.setattr(cli.tracing, "shutdown_tracing", lambda: None)
    result = CliRunner().invoke(cli.app, ["ingest", "--only", "parse", "--force"])
    assert result.exit_code == 0, result.output
    assert seen == {"only": "parse", "force": True}


# --------------------------------------------------------------------------- live


def _api_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.live
def test_live_done_when_two_questions() -> None:
    url = os.environ.get("FLP_API_URL", "http://localhost:8000")
    if not _api_up(url):
        pytest.skip("the API is not running")
    off = CliRunner().invoke(cli.app, ["ask", "What is the capital of France?", "--api-url", url])
    assert off.exit_code == 2 and off.stdout.startswith(ABSTAIN_TEXT + "\n"), off.output
    on = CliRunner().invoke(cli.app, ["ask", "What does readonly do?", "--api-url", url])
    assert on.exit_code == 0, on.output
    lines = on.stdout.rstrip("\n").split("\n")
    assert "[S" in on.stdout and lines[-1].startswith(
        "trace_id="
    )  # the marker may close the answer
    assert any(line.startswith("S1") and "06 Readonly Properties" in line for line in lines)
