# "Done when" test for the respond node. Spec Section 6, Stage 18. Issue #23.
"""Stage 18: one builder for the `Response` record in all three statuses, labels to sources,
totals from the state, exactly one summary log line, the debug field, and the CLI view."""

import json
import time
from typing import Any

import pytest
import structlog
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError
from structlog.testing import capture_logs
from typer.testing import CliRunner

from flp_rag import cli, tracing
from flp_rag.contracts import Candidate, ContextBlock, Response, Source
from flp_rag.graph import build, markers
from flp_rag.graph.nodes import respond as rs
from flp_rag.graph.state import RagState, initial_state
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings
from tests.test_graph_build import stub_nodes


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


def block(label: str, ids: list[str], ch: int, title: str, start: int, end: int) -> ContextBlock:
    return ContextBlock(label, ids, f"[{label}] text", ch, title, start, end, 10)


BLOCKS = [
    block("S1", ["d:05:0038"], 5, "Property Promotion", 63, 66),
    block("S2", ["d:03:0020", "d:03:0021", "d:03:0022"], 3, "PHP's Type System", 49, 51),
    block("S3", ["d:06:0046", "d:06:0047"], 6, "Readonly Properties", 75, 77),
    block("S4", ["d:06:0050"], 6, "Readonly Properties", 80, 80),
]
LABEL_MAP = {
    "S1": {
        "chunk_ids": ["d:05:0038"],
        "parent_id": "p5",
        "kind": "chunk",
        "chapter_no": 5,
        "chapter_title": "Property Promotion",
        "section_title": "Constructor promotion",
        "page_printed_start": 63,
        "page_printed_end": 66,
    },
    "S2": {
        "chunk_ids": ["d:03:0020", "d:03:0021", "d:03:0022"],
        "parent_id": "p3",
        "kind": "parent",
        "chapter_no": 3,
        "chapter_title": "PHP's Type System",
        "section_title": "Other types",
        "page_printed_start": 49,
        "page_printed_end": 51,
    },
    "S3": {
        "chunk_ids": ["d:06:0046", "d:06:0047"],
        "parent_id": "p6",
        "kind": "chunks",
        "chapter_no": 6,
        "chapter_title": "Readonly Properties",
        "section_title": "Readonly",
        "page_printed_start": 75,
        "page_printed_end": 77,
    },
    "S4": {
        "chunk_ids": ["d:06:0050"],
        "parent_id": "p6b",
        "kind": "chunk",
        "chapter_no": 6,
        "chapter_title": "Readonly Properties",
        "section_title": "Readonly and clone",
        "page_printed_start": 80,
        "page_printed_end": 80,
    },
}


def cand(cid: str, score: float, titles: list[str] | None = None) -> Candidate:
    payload: dict[str, Any] = {"chapter_no": int(cid.split(":")[1])}
    if titles is not None:
        payload["section_titles"] = titles
    return Candidate(cid, score, score, ["dense"], payload)


CANDIDATES = [
    cand("d:05:0038", 0.91, ["Constructor promotion"]),
    cand("d:03:0020", 0.55, ["Other types"]),
    cand("d:03:0021", 0.72, ["Other types", "Enums"]),
    cand("d:03:0022", 0.61, ["Generics"]),
    cand("d:06:0046", 0.83, ["Readonly"]),
    cand("d:06:0050", 0.40, ["Readonly and clone"]),
]
ANSWER = "Promotion removes boilerplate [S1]. It does [S1]. Readonly is set once [S3]."


def answered_state(answer: str = ANSWER, **extra: Any) -> RagState:
    state = initial_state(
        "What is promotion?", request_id="r1", trace_id="a" * 32, index_version="v3-test"
    )
    return {
        **state,
        "candidates": CANDIDATES,
        "context": BLOCKS,
        "label_map": LABEL_MAP,
        "answer": answer,
        "status": "answered",
        "check_result": "ok",
        "prompt_version": "ab12cd34",
        "repair_calls": 0,
        "borderline": False,
        "top_score": 0.91,
        "input_tokens": 7400,
        "output_tokens": 300,
        "cost_usd": 0.0123,
        "timings_ms": {"retrieval.hybrid": 40, "llm.generate": 900},
        "visited": [
            "guard_input",
            "understand",
            "retrieve",
            "rerank",
            "gate",
            "assemble",
            "generate",
            "check_output",
        ],
        **extra,
    }


# --------------------------------------------------------------------------- sources


def test_cited_labels_give_one_source_each_in_label_order() -> None:
    sources = rs.sources_for(ANSWER, LABEL_MAP, CANDIDATES)
    assert [s.label for s in sources] == ["S1", "S3"]
    s1, s3 = sources
    assert s1.chunk_ids == ["d:05:0038"] and s3.chunk_ids == ["d:06:0046", "d:06:0047"]
    assert (s1.chapter_no, s1.chapter_title, s1.section_title) == (
        5,
        "Property Promotion",
        "Constructor promotion",
    )
    assert (s1.page_printed_start, s1.page_printed_end, s1.score) == (63, 66, 0.91)
    # S3: the highest rerank score among the block's chunks; one chunk was never retrieved.
    assert s3.score == 0.83 and s3.section_title == "Readonly" and s3.page_printed_end == 77


def test_labels_are_not_renumbered_and_combined_markers_become_canonical() -> None:
    answer = "The class keeps its state [S2, S4]."
    assert markers.canonical_markers(answer) == "The class keeps its state [S2][S4]."
    assert (
        markers.canonical_markers("[S1][S3] and [S2,S4] and [S9]")
        == "[S1][S3] and [S2][S4] and [S9]"
    )
    assert markers.canonical_markers("state [ S1 ] and [S2 ]") == "state [S1] and [S2]"
    assert markers.canonical_markers("DTOs【S1】 and 【S2, S3】.") == "DTOs[S1] and [S2][S3]."
    sources = rs.sources_for(answer, LABEL_MAP, CANDIDATES)
    assert [s.label for s in sources] == ["S2", "S4"]


def test_merged_sections_are_joined_in_order() -> None:
    (s2,) = rs.sources_for("See [S2].", LABEL_MAP, CANDIDATES)
    assert s2.section_title == "Other types + Enums + Generics"
    assert s2.score == 0.72 and s2.chunk_ids == ["d:03:0020", "d:03:0021", "d:03:0022"]
    # Without retrieved chunks the label map's section stands in.
    (s2,) = rs.sources_for("See [S2].", LABEL_MAP, [])
    assert s2.section_title == "Other types" and s2.score == 0.0


def test_a_cited_label_outside_the_label_map_is_a_bug() -> None:
    with pytest.raises(rs.UnknownLabel, match="S9"):
        rs.sources_for("Fine [S1] but [S9].", LABEL_MAP, CANDIDATES)


# --------------------------------------------------------------------------- the record


def test_answered_record_has_canonical_answer_sources_versions_and_models(settings) -> None:
    r = rs.build_response(answered_state("Promotion [S1, S3]."), settings)
    assert r.status == "answered" and r.answer == "Promotion [S1][S3]."
    assert [s.label for s in r.sources] == ["S1", "S3"] and r.trace_id == "a" * 32
    assert r.index_version == "v3-test" and r.prompt_version == "ab12cd34"
    assert r.timings_ms == {"retrieval.hybrid": 40, "llm.generate": 900}
    assert r.model == {
        "gen": settings.gen.model,
        "query": settings.query.model,
        "judge": settings.output.judge_model,
        "embed": settings.embed.model,
        "rerank": settings.rerank.model,
    }
    assert r.debug is None


def test_abstain_records_have_the_exact_text_no_sources_and_the_reason(settings) -> None:
    gate = answered_state(
        ABSTAIN_TEXT,
        status="no_information",
        abstain_reason="gate.no_passing_chunk",
        context=[],
        label_map={},
        prompt_version="",
        sources=[],
        debug=True,
    )
    r = rs.build_response(gate, settings)
    assert r.status == "no_information" and r.answer == ABSTAIN_TEXT and r.sources == []
    assert r.prompt_version == rs.default_prompt_version(settings) and len(r.prompt_version) == 8
    assert r.debug is not None and r.debug["abstain_reason"] == "gate.no_passing_chunk"

    unfaithful = answered_state(
        ABSTAIN_TEXT,
        status="no_information",
        abstain_reason="output.unfaithful",
        cost_usd=0.02,
        sources=[],
    )
    r = rs.build_response(unfaithful, settings)
    assert r.answer == ABSTAIN_TEXT and r.sources == []
    assert rs.summarize(unfaithful, r, settings).cost_usd == 0.02  # the abstain still cost


def test_error_record_uses_the_fixed_text_never_the_abstain_text(settings) -> None:
    state = initial_state("q", request_id="r1", trace_id="b" * 32, index_version="v3-test")
    r = rs.build_response(state, settings, error="RuntimeError: secret host refused")
    assert r.status == "error" and r.answer == settings.api.error_text
    assert r.answer != ABSTAIN_TEXT and "secret" not in r.answer and r.sources == []
    assert r.trace_id == "b" * 32 and r.index_version == "v3-test" and r.timings_ms == {}
    assert r.debug is None
    with pytest.raises(ValidationError):
        settings.api.model_copy(update={"error_text": ABSTAIN_TEXT}, deep=True).model_validate(
            {**settings.api.model_dump(), "error_text": ABSTAIN_TEXT}
        )


# --------------------------------------------------------------------------- totals and log line


def test_totals_and_the_summary_come_from_the_state(settings) -> None:
    state = answered_state(started_at=time.perf_counter() - 0.25, repair_calls=1, borderline=True)
    r = rs.build_response(state, settings)
    s = rs.summarize(state, r, settings)
    assert (s.input_tokens, s.output_tokens, s.cost_usd) == (7400, 300, 0.0123)
    assert s.request_id == "r1" and s.trace_id == "a" * 32 and s.status == "answered"
    assert s.abstain_reason is None and s.n_sources == 2 and s.repair_calls == 1 and s.borderline
    assert s.index_version == "v3-test" and s.prompt_version == "ab12cd34"
    assert s.config_hash == settings.config_hash and 240 <= s.latency_ms < 5000


def test_exactly_one_summary_line_without_question_or_answer_text(settings) -> None:
    node = rs.make_respond(settings)
    with capture_logs() as logs:
        out = node(answered_state())
    lines = [entry for entry in logs if entry["event"] == "request"]
    assert len(lines) == 1 and len(logs) == 1
    line = lines[0]
    for key in (
        "request_id",
        "trace_id",
        "status",
        "abstain_reason",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "index_version",
        "prompt_version",
        "config_hash",
        "n_sources",
        "repair_calls",
        "borderline",
    ):
        assert key in line, key
    text = json.dumps(line, default=str)
    assert "What is promotion" not in text and "boilerplate" not in text
    assert line["status"] == "answered" and line["n_sources"] == 2 and line["log_level"] == "info"
    assert out["response"].status == "answered" and out["sources"] == out["response"].sources


# --------------------------------------------------------------------------- debug


def test_debug_field_only_with_the_flag(settings) -> None:
    plain = rs.build_response(answered_state(), settings)
    assert plain.debug is None
    state = answered_state(
        debug=True,
        debug_info={
            "gate": {"decision": "pass", "reason": "gate.pass", "score_top1": 0.91, "n_pass": 6},
            "check_output": {"result": "ok", "failed_check": None, "sampled": False},
        },
        repair_calls=1,
    )
    r = rs.build_response(state, settings)
    d = r.debug
    assert d is not None and d["abstain_reason"] is None
    assert d["gate"]["score_top1"] == 0.91 and d["gate"]["tau_pass"] == settings.gate.tau_pass
    assert (
        d["gate"]["tau_low"] == settings.gate.tau_low
        and d["gate"]["tau_high"] == settings.gate.tau_high
    )
    assert [u["label"] for u in d["uncited_blocks"]] == ["S2", "S4"]
    assert (
        d["uncited_blocks"][0]["page_printed_start"] == 49
        and d["uncited_blocks"][0]["chapter_no"] == 3
    )
    assert d["repair_calls"] == 1 and d["check_output"]["failed_check"] is None
    assert d["timings_ms"] == {"retrieval.hybrid": 40, "llm.generate": 900}
    assert d["visited"][-1] == "respond"


# --------------------------------------------------------------------------- node, spans, timings


def test_node_span_attributes_and_stage_timings_in_the_graph(settings) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    nodes = build.default_nodes(build.Deps(settings=settings))
    stubs = stub_nodes().as_dict()

    def assemble(state: RagState) -> dict[str, Any]:
        return {"context": BLOCKS, "label_map": LABEL_MAP, "candidates": CANDIDATES}

    def generate(state: RagState) -> dict[str, Any]:
        return {
            "answer": "Promotion [S1, S3].",
            "prompt_version": "ab12cd34",
            "input_tokens": 1000,
            "output_tokens": 50,
            "cost_usd": 0.001,
        }

    mixed = build.GraphNodes(
        **{**stubs, "assemble": assemble, "generate": generate, "respond": nodes.respond}
    )
    with capture_logs() as logs:
        state = build.build_graph(mixed).invoke(
            initial_state("What is promotion?", request_id="r1", trace_id="c" * 32, debug=True)
        )
    response: Response = state["response"]
    assert response.answer == "Promotion [S1][S3]." and [s.label for s in response.sources] == [
        "S1",
        "S3",
    ]
    assert set(state["timings_ms"]) == {
        "guardrails.input",
        "query.understand",
        "retrieval.hybrid",
        "rerank",
        "relevance.gate",
        "context.assemble",
        "llm.generate",
        "guardrails.output",
        "response.build",
    }
    # The record is built inside the respond node, so its own timing lands in the state only.
    assert state["timings_ms"].keys() - response.timings_ms.keys() == {"response.build"}
    assert all(isinstance(v, int) and v >= 0 for v in response.timings_ms.values())
    span = next(s for s in exporter.get_finished_spans() if s.name == "response.build")
    a = span.attributes
    assert a["rag.status"] == "answered" and a["rag.n_sources"] == 2
    assert a["rag.answer_chars"] == len("Promotion [S1][S3].")
    assert len([e for e in logs if e["event"] == "request"]) == 1
    assert response.debug["gate"]["score_top1"] == 0.7  # from the stub gate


# --------------------------------------------------------------------------- CLI view


def test_format_response_prints_answer_sources_status_and_trace_id(settings) -> None:
    r = rs.build_response(answered_state(), settings)
    text = rs.format_response(r)
    lines = text.splitlines()
    assert lines[0] == ANSWER
    assert "S1 · Chapter 05 Property Promotion · Constructor promotion · p. 63-66" in lines
    assert "S3 · Chapter 06 Readonly Properties · Readonly · p. 75-77" in lines
    assert lines[-1] == f"status=answered trace_id={'a' * 32}"
    abstain = rs.build_response(
        answered_state(
            ABSTAIN_TEXT, status="no_information", abstain_reason="gate.low_top_score", sources=[]
        ),
        settings,
    )
    text = rs.format_response(abstain)
    assert text.splitlines()[0] == ABSTAIN_TEXT and "S1 ·" not in text
    assert text.splitlines()[-1].startswith("status=no_information")
    with_debug = rs.format_response(
        rs.build_response(answered_state(debug=True), settings), debug=True
    )
    assert "uncited_blocks" in with_debug and "timings_ms" in with_debug


def test_flp_ask_prints_the_response_record(monkeypatch: pytest.MonkeyPatch, settings) -> None:
    smalltalk = False

    def fake_graph(deps: build.Deps):
        nodes = build.default_nodes(deps)
        stubs = stub_nodes(smalltalk=smalltalk).as_dict()

        def assemble(state: RagState) -> dict[str, Any]:
            return {"context": BLOCKS, "label_map": LABEL_MAP, "candidates": CANDIDATES}

        def generate(state: RagState) -> dict[str, Any]:
            return {"answer": "Promotion removes boilerplate [S1].", "prompt_version": "ab12cd34"}

        return build.build_graph(
            build.GraphNodes(
                **{**stubs, "assemble": assemble, "generate": generate, "respond": nodes.respond}
            )
        )

    from flp_rag.api import app as api_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(api_mod, "build_deps", lambda s: build.Deps(settings=s))
    monkeypatch.setattr(build, "compile_default", fake_graph)
    monkeypatch.setattr(api_mod, "current_index_version", lambda s, d: ("v3-test", "c"))
    monkeypatch.setattr(cli.tracing, "configure_tracing", lambda s: None)
    monkeypatch.setattr(cli.tracing, "shutdown_tracing", lambda: None)

    result = CliRunner().invoke(cli.app, ["ask", "What is promotion?"])
    assert result.exit_code == 0, result.output
    out = [line for line in result.output.splitlines() if not line.startswith("{")]
    assert out[0] == "Promotion removes boilerplate [S1]."
    assert "S1 · Chapter 05 Property Promotion · Constructor promotion · p. 63-66" in out
    assert out[-1].startswith("status=answered trace_id=")

    smalltalk = True
    result = CliRunner().invoke(cli.app, ["ask", "hi there", "--debug"])
    assert result.exit_code == 0, result.output
    assert ABSTAIN_TEXT in result.output and "query.smalltalk" in result.output

    def broken(s: Settings) -> build.Deps:
        raise RuntimeError("secret://qdrant-host refused")

    monkeypatch.setattr(api_mod, "build_deps", broken)
    result = CliRunner().invoke(cli.app, ["ask", "What is promotion?"])
    assert result.exit_code == 1
    # The record and the stderr detail show no internals; the JSON ERROR log line holds them.
    shown = "\n".join(line for line in result.output.splitlines() if not line.startswith("{"))
    assert settings.api.error_text in shown and "status=error" in shown
    assert "qdrant-host" not in shown and "see trace" in shown
    assert any(line.startswith("{") and "qdrant-host" in line for line in result.output.splitlines())


def test_timings_of_a_node_that_runs_twice_add_up() -> None:
    from flp_rag.graph.state import sum_dicts

    assert sum_dicts({"llm.generate": 300}, {"llm.generate": 200, "guardrails.output": 5}) == {
        "llm.generate": 500,
        "guardrails.output": 5,
    }


def test_a_final_state_without_a_status_is_a_bug(settings) -> None:
    with pytest.raises(ValueError, match="status"):
        rs.build_response({"trace_id": "x"}, settings)


def test_source_contract_carries_every_chunk_of_the_block() -> None:
    s = Source("S1", ["a", "b"], 6, "R", "s", 75, 77, 0.9)
    assert json.loads(s.to_attrs()["rag.chunk_ids"]) == ["a", "b"]
    structlog.get_logger()  # the module is importable with the project's logging config
