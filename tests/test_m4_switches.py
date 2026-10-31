# Issue #28: the Milestone 3 / Milestone 4 switches, the run tag, and the debug rerank list.
"""Unit tests: `query.enabled`, `retrieval.sparse_enabled`, `rerank.enabled` reproduce the
Milestone 3 pipeline on the same code; `run_tag` reaches the root span only with `debug`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Candidate
from flp_rag.graph import build
from flp_rag.graph.nodes import rerank as rk
from flp_rag.graph.nodes import respond as respond_mod
from flp_rag.graph.nodes import retrieve as r
from flp_rag.graph.nodes import understand as u
from flp_rag.graph.state import initial_state
from flp_rag.settings import Settings, load_settings
from tests.test_api import _client, exporter  # noqa: F401 - fixture
from tests.test_hybrid_retrieval import FakeClient, FakeEmbedder, FakeSparse
from tests.test_rerank import FakeReranker, cand, store_path  # noqa: F401 - fixture
from tests.test_understand import FACTS, FULL, FakeQueryModel


def switched(**sections: dict[str, Any]) -> Settings:
    s = load_settings()
    update = {name: getattr(s, name).model_copy(update=values) for name, values in sections.items()}
    return s.model_copy(update=update)


def spans(exp: InMemorySpanExporter, name: str) -> list[Any]:
    return [sp for sp in exp.get_finished_spans() if sp.name == name]


# --------------------------------------------------------------------------- defaults


def test_the_three_switches_default_to_the_milestone_4_pipeline(tmp_path: Path) -> None:
    import yaml

    s = load_settings()
    assert s.query.enabled is True
    assert s.retrieval.sparse_enabled is True
    assert s.rerank.enabled is True
    data = yaml.safe_load(s.config_path.read_text())
    data["query"]["enabled"] = False
    data["retrieval"]["sparse_enabled"] = False
    data["rerank"]["enabled"] = False
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data))
    before = load_settings(tmp_path / "config.yaml")
    assert (before.query.enabled, before.retrieval.sparse_enabled, before.rerank.enabled) == (
        False,
        False,
        False,
    )
    assert before.config_hash != s.config_hash  # every trace names its side
    assert before.chunk_config_hash == s.chunk_config_hash  # same index


def test_build_deps_leaves_disabled_components_unbuilt(monkeypatch: Any) -> None:
    from flp_rag.api import app as api

    built: list[str] = []
    monkeypatch.setattr(api, "qdrant_client", lambda s: built.append("client") or object())
    monkeypatch.setattr(api, "get_embeddings", lambda s: built.append("embedder") or object())
    monkeypatch.setattr(api, "get_sparse_embeddings", lambda s: built.append("sparse") or object())
    monkeypatch.setattr(api, "get_reranker", lambda s: built.append("reranker") or object())
    monkeypatch.setattr(api, "EmbeddingCache", lambda path: object())
    monkeypatch.setattr(api, "SessionStore", lambda path, **kw: object())
    before = switched(retrieval={"sparse_enabled": False}, rerank={"enabled": False})
    deps = api.build_deps(before)
    assert deps.sparse is None and deps.reranker is None
    assert built == ["client", "embedder"]
    built.clear()
    after = api.build_deps(load_settings())
    assert after.sparse is not None and after.reranker is not None
    assert built == ["client", "embedder", "sparse", "reranker"]


# --------------------------------------------------------------------------- query.enabled


def test_query_disabled_uses_the_raw_question_without_a_model_call() -> None:
    s = switched(query={"enabled": False})
    exp = InMemorySpanExporter()
    tracing.configure_tracing(s, exporter=exp, force=True)
    chat = FakeQueryModel(FULL)
    built: list[str] = []

    def chat_factory() -> FakeQueryModel:
        built.append("chat")
        return chat

    node = build._traced("understand", u.make_understand(s, chat_factory, lambda: FACTS, None))
    out = node(initial_state("What about enums?", request_id="r1", history=[("user", "x")]))
    q = out["standalone"]
    assert q.standalone == "What about enums?" and q.variants == ["What about enums?"]
    assert q.keyword_variant is None and q.intent == "book_question"
    assert built == [] and chat.calls == []
    assert "status" not in out
    a = spans(exp, "query.understand")[0].attributes
    assert a["rag.understand_enabled"] is False and a["rag.n_variants"] == 1
    assert spans(exp, "query.classify") == []


# --------------------------------------------------------------------------- sparse_enabled


def test_sparse_disabled_runs_dense_only_and_records_no_overlap() -> None:
    s = switched(retrieval={"sparse_enabled": False})
    exp = InMemorySpanExporter()
    tracing.configure_tracing(s, exporter=exp, force=True)
    client = FakeClient(
        dense=[[("A", 0.9, {"chapter_no": 6}), ("B", 0.8, {"chapter_no": 6})]],
        sparse=[[("C", 12.0, {"chapter_no": 7})]],
    )
    sparse = FakeSparse()
    variants = r.Variants("How do readonly properties work?", [], "readonly properties")

    @tracing.stage("retrieval.hybrid", kind="RETRIEVER")
    def run() -> list[Candidate]:
        return r.retrieve_hybrid(
            variants, s, client=client, embedder=FakeEmbedder(), sparse=sparse, collection="c"
        )

    out = run()
    assert [c.chunk_id for c in out] == ["A", "B"]
    assert all(set(c.scores) == {"dense"} for c in out)
    assert len(client.calls) == 1 and all(req.using == "dense" for req in client.calls[0][1])
    assert sparse.query_calls == []
    a = spans(exp, "retrieval.hybrid")[0].attributes
    assert a["rag.sparse_enabled"] is False and a["rag.n_sparse_hits"] == 0
    assert "rag.overlap_ratio" not in a
    assert spans(exp, "retrieval.sparse") == [] and len(spans(exp, "retrieval.dense")) == 1


# --------------------------------------------------------------------------- rerank.enabled


def test_rerank_disabled_uses_the_dense_cosine_as_the_milestone_3_stub_did() -> None:
    s = switched(rerank={"enabled": False, "top_n": 2})
    exp = InMemorySpanExporter()
    tracing.configure_tracing(s, exporter=exp, force=True)
    reranker = FakeReranker({"d:1": 0.99})
    candidates = [
        Candidate("d:1", 0.03, None, ["dense:v0:r1"], {"chapter_no": 6}, {"dense": 0.41}),
        Candidate("d:2", 0.02, None, ["dense:v0:r2"], {"chapter_no": 5}, {"dense": 0.72}),
        Candidate("d:3", 0.01, None, ["sparse:v0:r1"], {"chapter_no": 9}, {"sparse": 9.0}),
        Candidate("d:4", 0.01, None, ["dense:v0:r3"], {"chapter_no": 3}, {"dense": -0.2}),
    ]
    node = build._traced("rerank", rk.make_rerank(s, lambda: reranker, None))
    state = initial_state("q", request_id="r1", debug=True)
    state["candidates"] = candidates
    out = node(state)
    kept = out["candidates"]
    # d:2 (0.72) first, d:1 (0.41) second; d:3 has no dense score and falls back to the fused
    # score; d:4's negative cosine clamps to 0. top_n cuts to two.
    assert [(c.chunk_id, c.rerank_score) for c in kept] == [("d:2", 0.72), ("d:1", 0.41)]
    assert reranker.calls == []
    a = spans(exp, "rerank")[0].attributes
    assert a["rag.rerank_enabled"] is False
    assert a["rag.n_in"] == 4 and a["rag.n_out"] == 2 and a["rag.score_top1"] == 0.72
    assert out["debug_info"]["rerank"]["top"] == [
        {"chunk_id": "d:2", "chapter_no": 5, "score": 0.72},
        {"chunk_id": "d:1", "chapter_no": 6, "score": 0.41},
    ]


def test_rerank_enabled_puts_the_kept_list_in_the_debug_info(
    store_path: Path,  # noqa: F811 - the test_rerank fixture
) -> None:
    s = switched(rerank={"top_n": 2})

    def run_node(reranker: FakeReranker, *, debug: bool) -> dict[str, Any]:
        node = build._traced("rerank", rk.make_rerank(s, lambda: reranker, store_path))
        state = initial_state("q", request_id="r1", debug=debug)
        state["candidates"] = [cand("c0"), cand("c1", 7)]
        return node(state)

    out = run_node(FakeReranker({"c1": 0.9, "c0": 0.4}), debug=True)
    assert out["debug_info"]["rerank"]["top"] == [
        {"chunk_id": "c1", "chapter_no": 7, "score": 0.9},
        {"chunk_id": "c0", "chapter_no": 6, "score": 0.4},
    ]
    assert "debug_info" not in run_node(FakeReranker({"c1": 0.9}), debug=False)


# --------------------------------------------------------------------------- debug field


def test_debug_field_carries_the_rerank_list_cost_and_tokens() -> None:
    s = load_settings()
    state = initial_state("q", request_id="r1", debug=True)
    state.update(
        status="no_information",
        abstain_reason="gate.low_top_score",
        cost_usd=0.00014,
        input_tokens=800,
        output_tokens=150,
        debug_info={"rerank": {"top": [{"chunk_id": "d:1", "chapter_no": 6, "score": 0.2}]}},
    )
    debug = respond_mod.build_response(state, s).debug
    assert debug["rerank"]["top"][0]["chapter_no"] == 6
    assert debug["cost_usd"] == pytest.approx(0.00014)
    assert debug["input_tokens"] == 800 and debug["output_tokens"] == 150


# --------------------------------------------------------------------------- run_tag


def test_run_tag_reaches_the_root_span_only_with_debug(
    tmp_path: Path,
    exporter: InMemorySpanExporter,  # noqa: F811 - the test_api fixture
) -> None:
    exp = exporter
    with _client(tmp_path, exp) as (client, _version):
        tagged = client.post(
            "/ask", json={"question": "hello", "debug": True, "run_tag": "m4-after"}
        )
        assert tagged.status_code == 200, tagged.text
        plain = client.post("/ask", json={"question": "hello", "run_tag": "m4-after"})
        assert plain.status_code == 200, plain.text
        too_long = client.post(
            "/ask", json={"question": "hello", "debug": True, "run_tag": "x" * 65}
        )
        assert too_long.status_code == 422
    roots = spans(exp, "rag.request")
    by_trace = {format(sp.context.trace_id, "032x"): sp.attributes for sp in roots}
    assert by_trace[tagged.json()["trace_id"]]["rag.run_tag"] == "m4-after"
    assert "rag.run_tag" not in by_trace[plain.json()["trace_id"]]


def test_run_tag_is_in_the_debug_field_of_the_response() -> None:
    s = load_settings()
    state = initial_state("q", request_id="r1", debug=True, run_tag="m4-before")
    state.update(status="no_information", abstain_reason="gate.low_top_score")
    assert respond_mod.build_response(state, s).debug["run_tag"] == "m4-before"
    untagged = initial_state("q", request_id="r1", debug=True)
    untagged.update(status="no_information", abstain_reason="gate.low_top_score")
    assert "run_tag" not in respond_mod.build_response(untagged, s).debug
