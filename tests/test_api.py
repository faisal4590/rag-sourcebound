"""Done-when tests for Stage 9, the FastAPI app. Spec Section 6, Stage 9. Issue #17.

The app is built with a stub graph and fake deps, so no model and no live Qdrant are needed.
One live test hits /health against the running services and skips without them.
"""

import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient, models

from flp_rag import tracing
from flp_rag.api import app as api
from flp_rag.contracts import Source
from flp_rag.graph import build
from flp_rag.graph.state import RagState
from flp_rag.settings import ABSTAIN_TEXT, load_settings
from tests.test_graph_build import stub_nodes

ROOT = Path(__file__).resolve().parents[1]


def _stub_graph_factory(
    *,
    gate: str = "pass",
    sleep_s: float = 0.0,
    raise_not_implemented: bool = False,
    raise_generic: bool = False,
    ran: list[str] | None = None,
):
    def factory(deps: build.Deps):
        nodes = stub_nodes(gate=gate)

        def generate(state: RagState) -> dict[str, Any]:
            if sleep_s:
                time.sleep(sleep_s)
            if raise_not_implemented:
                raise NotImplementedError("Stage 16 generation is issue #21")
            if raise_generic:
                raise RuntimeError("secret://qdrant-host:6333 refused")
            return {"answer": "Readonly properties can be set once [S1]."}

        def check_output(state: RagState) -> dict[str, Any]:
            if ran is not None:
                ran.append("check_output")
            return {"status": "answered"}

        def respond(state: RagState) -> dict[str, Any]:
            from flp_rag.contracts import Response

            status = state.get("status") or "answered"
            sources = (
                [Source("S1", "d:06:0001", 6, "Readonly Properties", "Readonly", 75, 77, 0.9)]
                if status == "answered"
                else []
            )
            response = Response(
                answer=state.get("answer") or ABSTAIN_TEXT,
                status=status,
                sources=sources,
                trace_id=state.get("trace_id", ""),
                timings_ms={"rerank": 3},
                model={"gen": "stub"},
                index_version="",
                prompt_version="",
                debug={"visited": [*state.get("visited", []), "respond"]}
                if state.get("debug")
                else None,
            )
            return {"status": status, "response": response, "sources": sources}

        return build.build_graph(
            build.GraphNodes(
                **{
                    **nodes.as_dict(),
                    "generate": generate,
                    "check_output": check_output,
                    "respond": respond,
                }
            )
        )

    return factory


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    exp = InMemorySpanExporter()
    tracing.configure_tracing(load_settings(), exporter=exp, force=True)
    return exp


@contextlib.contextmanager
def _client(
    tmp_path: Path, exporter: InMemorySpanExporter, *, alias: bool = True, **factory_kw: Any
) -> Iterator[tuple[TestClient, str]]:
    """A test client over an in-memory Qdrant whose alias points at a collection with a manifest."""
    settings = load_settings().model_copy(update={"config_path": tmp_path / "config.yaml"})
    (tmp_path / "config.yaml").write_text((ROOT / "config.yaml").read_text())
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "answer_v1.md").write_text((ROOT / "prompts" / "answer_v1.md").read_text())
    version = "v7-test-abcdef01"
    manifest_dir = tmp_path / "data" / "index" / version
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps({"index_version": version, "collection": f"flp_chunks_{version}"})
    )
    client = QdrantClient(":memory:")
    client.create_collection(
        f"flp_chunks_{version}",
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    )
    if alias:
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=f"flp_chunks_{version}", alias_name=settings.index.alias
                    )
                )
            ]
        )
    deps = build.Deps(settings=settings, client=client)
    app = api.create_app(
        settings,
        deps=deps,
        graph_factory=_stub_graph_factory(**factory_kw),
        configure_tracing=False,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, version


# --------------------------------------------------------------------------- /ask


def test_ask_returns_a_typed_response_with_trace_id(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter) as (client, version):
        r = client.post("/ask", json={"question": "hello"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered" and body["answer"].endswith("[S1].")
    assert len(body["trace_id"]) == 32 and r.headers["X-Trace-Id"] == body["trace_id"]
    assert body["index_version"] == version and len(body["prompt_version"]) == 8
    assert body["sources"][0]["page_printed_start"] == 75 and body["debug"] is None
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    a = root.attributes
    assert format(root.context.trace_id, "032x") == body["trace_id"]
    assert len(a["rag.request_id"]) == 36 and a["rag.request_id"][14] == "7"  # UUIDv7
    assert a["rag.index_version"] == version and a["rag.prompt_version"] == body["prompt_version"]
    assert a["rag.config_hash"] == load_settings().config_hash
    assert (
        a["rag.status"] == "answered" and a["rag.latency_ms"] >= 0 and a["http.status_code"] == 200
    )
    children = {
        s.name
        for s in exporter.get_finished_spans()
        if s.parent and s.parent.span_id == root.context.span_id
    }
    assert {"guardrails.input", "response.build"} <= children


def test_ask_debug_and_filters_and_abstain(tmp_path: Path, exporter: InMemorySpanExporter) -> None:
    with _client(tmp_path, exporter, gate="abstain") as (client, _):
        r = client.post(
            "/ask",
            json={
                "question": "What is the capital of France?",
                "debug": True,
                "filters": {"chapter_no": [6], "has_code": False},
                "session_id": "s1",
            },
        )
    body = r.json()
    assert (
        r.status_code == 200
        and body["status"] == "no_information"
        and body["answer"] == ABSTAIN_TEXT
    )
    assert body["sources"] == [] and body["debug"]["visited"][-1] == "respond"
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    assert (
        root.attributes["rag.session_id"] == "s1"
        and root.attributes["rag.abstain_reason"] == "gate.low_top_score"
    )


def test_validation_error_carries_a_trace_id(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter) as (client, _):
        r = client.post("/ask", json={"nope": 1})
    assert r.status_code == 422
    body = r.json()
    assert (
        body["status"] == "error" and len(body["trace_id"]) == 32 and "question" in body["detail"]
    )
    assert r.headers["X-Trace-Id"] == body["trace_id"]


def test_unhandled_and_not_implemented_errors_carry_a_trace_id(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter, raise_not_implemented=True) as (client, _):
        r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 501
    body = r.json()
    assert body["status"] == "error" and "#21" in body["detail"] and len(body["trace_id"]) == 32
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    assert root.attributes["rag.status"] == "error"
    assert any(e.name == "exception" for e in root.events)


def test_unexpected_errors_hide_internals_from_the_client(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter, raise_generic=True) as (client, _):
        r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 500
    body = r.json()
    assert body["status"] == "error" and body["trace_id"] in body["detail"]
    assert "secret" not in r.text and "qdrant-host" not in r.text
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    assert root.status.status_code.name == "ERROR"
    assert any("qdrant-host" in str(e.attributes.get("exception.message")) for e in root.events)


def test_timeout_returns_504_with_trace_id(tmp_path: Path, exporter: InMemorySpanExporter) -> None:
    """The slow node finishes on its own (threads cannot be interrupted) but no later node runs."""
    settings_patch = {"api": load_settings().api.model_copy(update={"timeout_s": 0.2})}
    ran: list[str] = []
    with _client(tmp_path, exporter, sleep_s=0.6, ran=ran) as (client, _):
        client.app.state.rag.settings = client.app.state.rag.settings.model_copy(
            update=settings_patch
        )
        r = client.post("/ask", json={"question": "slow"})
        time.sleep(0.8)
    assert r.status_code == 504
    body = r.json()
    assert body["status"] == "error" and "0.2" in body["detail"] and len(body["trace_id"]) == 32
    assert ran == [], "a node after the cancelled one still ran"
    root = next(s for s in exporter.get_finished_spans() if s.name == "rag.request")
    assert root.attributes["rag.status"] == "error"
    assert root.attributes["rag.abstain_reason"] == "timeout"


def test_stream_sends_token_events_then_done(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with (
        _client(tmp_path, exporter) as (client, _),
        client.stream("POST", "/ask", json={"question": "hello", "stream": True}) as r,
    ):
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert len(r.headers["X-Trace-Id"]) == 32
        raw = b"".join(r.iter_bytes()).decode()
    events = [blk for blk in raw.strip().split("\n\n") if blk]
    names = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert names[-1] == "done" and names.count("token") >= 5
    tokens = "".join(json.loads(e.split("\ndata: ")[1])["text"] for e in events[:-1])
    assert tokens == "Readonly properties can be set once [S1]."
    done = json.loads(events[-1].split("\ndata: ")[1])
    assert (
        done["status"] == "answered"
        and done["sources"][0]["chunk_id"] == "d:06:0001"
        and len(done["trace_id"]) == 32
    )


def test_stream_abstain_sends_only_done(tmp_path: Path, exporter: InMemorySpanExporter) -> None:
    with _client(tmp_path, exporter, gate="abstain") as (client, _):
        r = client.post("/ask", json={"question": "hi", "stream": True})
    events = [blk for blk in r.text.strip().split("\n\n") if blk]
    assert len(events) == 1 and events[0].startswith("event: done")
    done = json.loads(events[0].split("\ndata: ")[1])
    assert (
        done["status"] == "no_information"
        and done["answer"] == ABSTAIN_TEXT
        and done["sources"] == []
    )


# --------------------------------------------------------------------------- /feedback and /health


def test_feedback_is_accepted(tmp_path: Path, exporter: InMemorySpanExporter) -> None:
    with _client(tmp_path, exporter) as (client, _):
        ok = client.post(
            "/feedback", json={"trace_id": "0" * 32, "rating": "down", "comment": "wrong page"}
        )
        bad = client.post("/feedback", json={"trace_id": "short", "rating": "meh"})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert bad.status_code == 422 and bad.json()["status"] == "error"


def test_health_reports_index_version_models_and_backends(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter) as (client, version):
        r = client.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert body["index_version"] == version and body["collection"] == f"flp_chunks_{version}"
    assert (
        body["models"]["embed"] == "BAAI/bge-m3"
        and body["models"]["rerank"] == "BAAI/bge-reranker-v2-m3"
    )
    assert body["backends"]["qdrant"] is True and isinstance(body["backends"]["phoenix"], bool)
    expected = "ok" if body["backends"]["phoenix"] else "degraded"
    assert body["status"] == expected and len(body["trace_id"]) == 32
    assert len(body["prompt_version"]) == 8


def test_health_is_degraded_without_an_alias(
    tmp_path: Path, exporter: InMemorySpanExporter
) -> None:
    with _client(tmp_path, exporter, alias=False) as (client, _):
        r = client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "degraded"
    assert body["index_version"] is None and body["collection"] is None
    assert body["backends"]["qdrant"] is True


def test_prompt_version_is_the_file_hash() -> None:
    import hashlib

    settings = load_settings()
    expected = hashlib.sha256((ROOT / "prompts" / "answer_v1.md").read_bytes()).hexdigest()[:8]
    assert api.prompt_version(settings) == expected


# --------------------------------------------------------------------------- live


def _live_up() -> bool:
    try:
        QdrantClient(url=load_settings().index.qdrant_url, timeout=2).get_collections()
        return True
    except Exception:  # noqa: BLE001 - reachability probe
        return False


@pytest.mark.skipif(not _live_up(), reason="needs the live Qdrant")
def test_health_against_the_live_index() -> None:
    from flp_rag.ingest.s06_s07_index import qdrant_client

    settings = load_settings()
    deps = build.Deps(settings=settings, client=qdrant_client(settings))
    app = api.create_app(
        settings, deps=deps, graph_factory=_stub_graph_factory(), configure_tracing=False
    )
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["backends"]["qdrant"] is True
    assert (
        body["index_version"]
        and body["index_version"].startswith("v")
        and "bgem3" in body["index_version"]
    )
    assert body["collection"] == f"flp_chunks_{body['index_version']}"


def test_embedding_cache_is_usable_from_a_worker_thread(tmp_path: Path) -> None:
    """LangGraph runs sync nodes in a thread pool; the cache is opened once at startup."""
    from concurrent.futures import ThreadPoolExecutor

    from flp_rag.stores.embedding_cache import EmbeddingCache

    with EmbeddingCache(tmp_path / "e.sqlite") as cache:
        cache.put_query("m", "q", [0.5, 0.5])
        with ThreadPoolExecutor(max_workers=2) as pool:
            got = pool.submit(cache.get_query, "m", "q").result()
            pool.submit(cache.put_query, "m", "q2", [1.0, 0.0]).result()
        assert got == [0.5, 0.5] and cache.count("queries") == 2
