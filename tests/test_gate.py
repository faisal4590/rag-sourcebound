# "Done when" test for the gate node. Spec Section 6, Stage 14. Issue #19.
"""Stage 14 relevance gate: the decision is a plain function of the rerank scores and the three
thresholds. Unit tests run with no graph, no model, and no index; the graph tests use stub nodes
around the real gate."""

from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Candidate
from flp_rag.graph import build
from flp_rag.graph.nodes import gate as g
from flp_rag.graph.state import RagState, initial_state
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings
from tests.test_graph_build import stub_nodes

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (ROOT / "config.yaml").read_text()


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


# --------------------------------------------------------------------------- decision function


@pytest.mark.parametrize(
    ("top", "decision", "reason", "borderline"),
    [
        (0.10, "abstain", "gate.low_top_score", False),
        (0.30, "abstain", "gate.no_passing_chunk", False),
        (0.50, "pass", "gate.borderline", True),
        (0.80, "pass", "gate.pass", False),
    ],
)
def test_the_four_zones(settings, top, decision, reason, borderline) -> None:
    d = g.decide([top, 0.05], settings.gate)
    assert (d.decision, d.reason, d.borderline, d.score_top1) == (decision, reason, borderline, top)


@pytest.mark.parametrize(
    ("top", "decision", "reason"),
    [
        (0.20, "abstain", "gate.no_passing_chunk"),
        (0.35, "pass", "gate.borderline"),
        (0.60, "pass", "gate.pass"),
    ],
)
def test_each_zone_includes_its_lower_edge(settings, top, decision, reason) -> None:
    d = g.decide([top], settings.gate)
    assert (d.decision, d.reason) == (decision, reason)


def test_no_candidates_abstains_with_its_own_reason(settings) -> None:
    d = g.decide([], settings.gate)
    assert (d.decision, d.reason, d.score_top1, d.n_pass, d.borderline) == (
        "abstain",
        "gate.no_candidates",
        None,
        0,
        False,
    )


def test_n_pass_counts_candidates_at_or_above_tau_pass(settings) -> None:
    d = g.decide([0.93, 0.5, 0.35, 0.349, 0.1], settings.gate)
    assert d.n_pass == 3 and d.decision == "pass"


@pytest.mark.parametrize("bad", [1.2, -0.1, float("nan"), float("inf")])
def test_a_score_outside_zero_to_one_is_an_error(settings, bad) -> None:
    with pytest.raises(g.ScoreOutOfRange, match="0 to 1"):
        g.decide([0.9, bad], settings.gate)


def test_the_range_bounds_are_inclusive(settings) -> None:
    assert g.decide([1.0], settings.gate).reason == "gate.pass"
    assert g.decide([0.0], settings.gate).reason == "gate.low_top_score"


def test_thresholds_must_be_strictly_ordered(tmp_path: Path) -> None:
    from tests.test_settings import _load

    with pytest.raises(ValueError) as exc:
        _load(tmp_path, SHIPPED.replace("tau_pass: 0.35", "tau_pass: 0.70"))
    assert "0.7" in str(exc.value) and "0.6" in str(exc.value) and "tau_pass" in str(exc.value)
    (tmp_path / "eq").mkdir()
    with pytest.raises(ValueError, match="tau_low < tau_pass < tau_high"):
        _load(tmp_path / "eq", SHIPPED.replace("tau_low: 0.20", "tau_low: 0.35"))
    (tmp_path / "eq2").mkdir()
    with pytest.raises(ValueError, match="tau_low < tau_pass < tau_high"):
        _load(tmp_path / "eq2", SHIPPED.replace("tau_high: 0.60", "tau_high: 0.35"))


# --------------------------------------------------------------------------- node in the graph


def candidates(*scores: float) -> list[Candidate]:
    return [
        Candidate(f"d:06:{i:04d}", s, s, ["dense"], {"chapter_no": 6}) for i, s in enumerate(scores)
    ]


def run_gate(
    settings: Settings, scores: list[float], *, debug: bool = False
) -> tuple[RagState, dict[str, Any]]:
    """Stub nodes around the real gate and the real respond node; returns state and span attrs."""
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    stubs = stub_nodes()
    nodes = build.default_nodes(build.Deps(settings=settings))

    def rerank(state: RagState) -> dict[str, Any]:
        return {"candidates": candidates(*scores)}

    mixed = build.GraphNodes(
        **{**stubs.as_dict(), "rerank": rerank, "gate": nodes.gate, "respond": nodes.respond}
    )
    state = build.build_graph(mixed).invoke(initial_state("q", request_id="r1", debug=debug))
    spans = {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}
    return state, spans


PART_4 = {
    "rag.gate_decision",
    "rag.score_top1",
    "rag.n_pass",
    "rag.tau_low",
    "rag.tau_pass",
    "rag.tau_high",
    "rag.borderline",
}


def test_pass_path_writes_every_attribute_and_hands_passing_candidates_on(settings) -> None:
    state, spans = run_gate(settings, [0.93, 0.5, 0.2])
    attrs = spans["relevance.gate"]
    assert PART_4 <= set(attrs) and "rag.abstain_reason" not in attrs
    assert attrs["rag.gate_decision"] == "pass" and attrs["rag.score_top1"] == pytest.approx(0.93)
    assert attrs["rag.n_pass"] == 2 and attrs["rag.borderline"] is False
    assert (attrs["rag.tau_low"], attrs["rag.tau_pass"], attrs["rag.tau_high"]) == (0.2, 0.35, 0.6)
    assert (
        state["gate_decision"] == "pass" and state["n_pass"] == 2 and state["borderline"] is False
    )
    assert [c.rerank_score for c in state["candidates"]] == [0.93, 0.5]
    assert "assemble" in state["visited"] and "generate" in state["visited"]


def test_borderline_continues_with_the_flag(settings) -> None:
    state, spans = run_gate(settings, [0.48], debug=True)
    attrs = spans["relevance.gate"]
    assert attrs["rag.borderline"] is True and attrs["rag.gate_decision"] == "pass"
    assert attrs["rag.gate_reason"] == "gate.borderline"
    assert state["gate_decision"] == "pass" and state["borderline"] is True
    assert "generate" in state["visited"]
    assert state["response"].debug["gate"]["reason"] == "gate.borderline"


def test_the_edge_value_passes_through_the_node_and_counts_in_n_pass(settings) -> None:
    state, spans = run_gate(settings, [0.35, 0.349])
    assert spans["relevance.gate"]["rag.gate_reason"] == "gate.borderline"
    assert state["n_pass"] == 1 and [c.rerank_score for c in state["candidates"]] == [0.35]


def test_abstain_path_writes_every_attribute_and_skips_generation(settings) -> None:
    state, spans = run_gate(settings, [0.27, 0.1])
    attrs = spans["relevance.gate"]
    assert PART_4 <= set(attrs) and attrs["rag.abstain_reason"] == "gate.no_passing_chunk"
    assert attrs["rag.gate_decision"] == "abstain" and attrs["rag.n_pass"] == 0
    assert state["answer"] == ABSTAIN_TEXT and state["status"] == "no_information"
    assert state["abstain_reason"] == "gate.no_passing_chunk" and state["sources"] == []
    assert state["visited"] == [
        "guard_input",
        "understand",
        "retrieve",
        "rerank",
        "gate",
        "respond",
    ]
    assert not {"context.assemble", "llm.generate", "guardrails.output"} & set(spans)


def test_no_candidates_in_the_graph(settings) -> None:
    state, spans = run_gate(settings, [])
    assert spans["relevance.gate"]["rag.abstain_reason"] == "gate.no_candidates"
    assert spans["relevance.gate"]["rag.score_top1"] == 0.0 and state["answer"] == ABSTAIN_TEXT


def test_debug_field_holds_the_reason_only_with_the_flag(settings) -> None:
    plain, _ = run_gate(settings, [0.04])
    debug, _ = run_gate(settings, [0.04], debug=True)
    assert plain["answer"] == debug["answer"] == ABSTAIN_TEXT
    assert plain["response"].debug is None
    assert debug["response"].debug["abstain_reason"] == "gate.low_top_score"
    assert debug["response"].debug["gate"]["reason"] == "gate.low_top_score"


def test_a_candidate_without_a_rerank_score_is_an_error(settings) -> None:
    nodes = build.default_nodes(build.Deps(settings=settings))
    state = initial_state("q", request_id="r1")
    state["candidates"] = [Candidate("d:06:0001", 0.7, None, ["dense"], {})]
    with pytest.raises(ValueError, match="rerank_score"):
        nodes.gate(state)
