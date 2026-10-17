# "Done when" test for Stage 13 cross-encoder reranking. Issue #26.
"""Scoring, the sigmoid applied once, the chapter-hint tie-breaker, the cut, the span, and the
failure rules, with a fake reranker of fixed scores. Two tests load the real model (`slow`);
one needs the live alias (`live`)."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import models as m
from flp_rag import tracing
from flp_rag.contracts import Candidate, Chunk, Parent, StandaloneQuery
from flp_rag.graph import build
from flp_rag.graph.nodes import rerank as rk
from flp_rag.graph.state import initial_state
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.parent_store import ParentStore


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


def payload(
    cid: str, chapter: int, title: str, section: str, start: int, end: int
) -> dict[str, Any]:
    return {
        "chunk_id": cid,
        "chapter_no": chapter,
        "chapter_title": title,
        "part": "Part 1",
        "section_title": section,
        "page_printed_start": start,
        "page_printed_end": end,
        "parent_id": f"p{chapter}",
        "has_code": False,
        "index_version": "v-test",
    }


def cand(cid: str, chapter: int = 6, fused: float = 0.02) -> Candidate:
    return Candidate(
        cid,
        fused,
        None,
        ["dense:v0:r1"],
        payload(cid, chapter, f"Title {chapter}", f"Sec {cid}", 70, 71),
        {"dense": 0.5},
    )


@pytest.fixture(scope="module")
def store_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A parent store whose chunks carry the display texts the reranker reads."""
    path = tmp_path_factory.mktemp("store") / "store.sqlite"
    chunks = [
        Chunk(
            f"c{i}",
            f"p{6 if i < 20 else 3}",
            f"Display text of chunk {i}.",
            "",
            5,
            payload(f"c{i}", 6 if i < 20 else 3, "T", f"Sec c{i}", 70, 71),
        )
        for i in range(32)
    ]
    parents = [Parent("p6", 6, "T", "text", 70, 71, 5), Parent("p3", 3, "T", "text", 40, 41, 5)]
    with ParentStore(path) as store:
        store.put_parents(parents)
        store.put_chunks(chunks)
    return path


class FakeReranker:
    """Fixed scores by chunk text; records every call. Scores are already 0..1."""

    model_name = "fake/reranker"
    device = "fake"

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        fail: Exception | None = None,
        truncated: int = 0,
    ) -> None:
        self.scores = scores or {}
        self.fail = fail
        self.truncated = truncated
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.fail is not None:
            raise self.fail
        return [next((s for key, s in self.scores.items() if key in t), 0.1) for t in texts]

    def truncated_pairs(self, query: str, texts: Sequence[str]) -> int:
        return self.truncated


def run(settings, store_path, candidates, reranker, *, hint=None, question="q", exporter=None):
    node = rk.make_rerank(settings, lambda: reranker, store_path)
    state = initial_state(question, request_id="r1")
    state["candidates"] = candidates
    if hint is not None:
        state["standalone"] = StandaloneQuery(
            question, question, "book_question", False, hint, [question]
        )
    traced = build._traced("rerank", node)
    return traced(state)


# --------------------------------------------------------------------------- pairs and scores


def test_pair_text_is_header_then_display_text() -> None:
    text = rk.pair_text(
        payload("d:06:0046", 6, "Readonly Properties", "Readonly and clone", 80, 81), "Body."
    )
    assert (
        text
        == "Front Line PHP > Part 1 > Chapter 06: Readonly Properties > Readonly and clone (p. 80-81)\nBody."
    )
    front = rk.pair_text(payload("d:00:0001", 0, "Foreword", "Foreword", 9, 9), "Hello.")
    assert front.startswith("Front Line PHP > Part 1 > Foreword > Foreword (p. 9)\n")


def test_sigmoid_values_of_example_2() -> None:
    assert [round(m.sigmoid(x), 3) for x in (4.2, 0.0, -3.1)] == [0.985, 0.5, 0.043]
    # Sigmoid of a value already in 0..1 lands in 0.50..0.73: the double-sigmoid band.
    assert 0.5 <= m.sigmoid(0.0) and m.sigmoid(1.0) < 0.74


def test_node_scores_sorts_cuts_and_records_every_score(settings, store_path) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    candidates = [cand(f"c{i}") for i in range(30)]
    scores = {f"chunk {i}.": (30 - i) / 31 for i in range(30)}
    scores["chunk 7."] = 0.99  # the best match sits at fused rank 8
    reranker = FakeReranker(scores, truncated=2)
    out = run(
        settings, store_path, candidates, reranker, question="Show the DTO", exporter=exporter
    )
    kept = out["candidates"]
    assert len(kept) == settings.rerank.top_n == 8 and kept[0].chunk_id == "c7"
    assert kept[0].rerank_score == pytest.approx(0.99) and kept[0].scores[
        "rerank"
    ] == pytest.approx(0.99)
    assert kept[0].scores["dense"] == 0.5 and kept[0].fused_score == 0.02  # nothing else changes
    assert [c.rerank_score for c in kept] == sorted((c.rerank_score for c in kept), reverse=True)
    ((query, texts),) = reranker.calls
    assert query == "Show the DTO" and len(texts) == 30
    assert texts[7] == rk.pair_text(candidates[7].payload, "Display text of chunk 7.")
    span = next(s for s in exporter.get_finished_spans() if s.name == "rerank")
    a = span.attributes
    assert a["openinference.span.kind"] == "RERANKER"
    assert (
        a["gen_ai.request.model"] == "fake/reranker" and a["reranker.model_name"] == "fake/reranker"
    )
    assert a["reranker.top_k"] == 8 and a["rag.n_in"] == 30 and a["rag.n_out"] == 8
    assert a["rag.score_top1"] == pytest.approx(0.99)
    assert a["rag.score_top3_mean"] == pytest.approx((0.99 + 30 / 31 + 29 / 31) / 3)
    all_scores = json.loads(a["rag.scores_all"])
    assert len(all_scores) == 30 and all_scores[0] == ["c7", pytest.approx(0.99)]
    assert [s for _, s in all_scores] == sorted((s for _, s in all_scores), reverse=True)
    assert a["rag.n_truncated"] == 2 and a["rag.hint_swaps"] == 0 and a["rag.device"] == "fake"


def test_fewer_than_top_n_are_all_kept(settings, store_path) -> None:
    out = run(settings, store_path, [cand(f"c{i}") for i in range(5)], FakeReranker())
    assert len(out["candidates"]) == 5


def test_zero_candidates_make_no_model_call(settings, store_path) -> None:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    reranker = FakeReranker()
    out = run(settings, store_path, [], reranker, exporter=exporter)
    assert out["candidates"] == [] and reranker.calls == []
    span = next(s for s in exporter.get_finished_spans() if s.name == "rerank")
    assert span.attributes["rag.n_in"] == 0 and span.attributes["rag.n_out"] == 0
    assert span.attributes["rag.score_top1"] == 0.0  # on every request, like the gate's attributes


def test_a_failing_model_raises_and_never_abstains(settings, store_path) -> None:
    with pytest.raises(RuntimeError, match="model crashed"):
        run(settings, store_path, [cand("c1")], FakeReranker(fail=RuntimeError("model crashed")))


def test_a_candidate_missing_from_the_store_is_an_error(settings, store_path) -> None:
    with pytest.raises(KeyError, match="nope"):
        run(settings, store_path, [cand("nope")], FakeReranker())


def test_no_store_path_is_a_configuration_error(settings) -> None:
    with pytest.raises(RuntimeError, match="store"):
        run(settings, None, [cand("c1")], FakeReranker())


# --------------------------------------------------------------------------- the chapter hint


def scored(*rows: tuple[str, int, float]) -> list[Candidate]:
    return [
        Candidate(cid, 0.0, s, [], {"chunk_id": cid, "chapter_no": ch}, {}) for cid, ch, s in rows
    ]


def test_hint_swaps_close_neighbors_and_leaves_scores_alone() -> None:
    ordered = scored(("A", 3, 0.812), ("B", 6, 0.806), ("C", 6, 0.5))
    out, swaps = rk.apply_hint(ordered, hint=6, margin=0.01)
    assert [c.chunk_id for c in out] == ["B", "A", "C"] and swaps == 1
    assert [c.rerank_score for c in out] == [0.806, 0.812, 0.5]  # scores unchanged
    assert max(c.rerank_score for c in out) == 0.812  # what the gate reads


def test_no_swap_when_the_gap_is_wide_or_without_a_hint() -> None:
    ordered = scored(("A", 3, 0.83), ("B", 6, 0.81))
    assert rk.apply_hint(ordered, hint=6, margin=0.01) == (ordered, 0)
    close = scored(("A", 3, 0.812), ("B", 6, 0.806))
    assert rk.apply_hint(close, hint=None, margin=0.01) == (close, 0)
    both = scored(("A", 6, 0.812), ("B", 6, 0.806))
    assert rk.apply_hint(both, hint=6, margin=0.01) == (both, 0)  # both match: nothing to prefer


def test_hint_is_a_tie_breaker_in_the_node_not_a_filter(settings, store_path) -> None:
    candidates = [cand("c0", chapter=6), cand("c25", chapter=3)]
    reranker = FakeReranker({"chunk 25.": 0.805, "chunk 0.": 0.80})
    out = run(settings, store_path, candidates, reranker, hint=6)
    assert [c.chunk_id for c in out["candidates"]] == ["c0", "c25"]
    assert out["candidates"][0].rerank_score == pytest.approx(0.80)
    without = run(settings, store_path, candidates, reranker, hint=3)
    assert [c.chunk_id for c in without["candidates"]] == ["c25", "c0"]


# --------------------------------------------------------------------------- the model wrapper


def test_bge_reranker_reads_its_settings_and_picks_a_device(settings) -> None:
    rr = m.get_reranker(settings)
    assert isinstance(rr, m.BgeReranker)
    assert rr.max_length == settings.rerank.max_length == 1024 and rr.batch_size == 16
    assert rr.requested_device == "auto" and rr.device == "auto"  # resolved at load, not here
    assert m.pick_device("cpu") == "cpu" and m.pick_device("auto") in ("cuda", "mps", "cpu")


def test_graph_wires_the_reranker_through_deps(settings, store_path) -> None:
    reranker = FakeReranker({"chunk 3.": 0.9})
    nodes = build.default_nodes(
        build.Deps(settings=settings, reranker=reranker, store_path=store_path)
    )
    state = initial_state("q", request_id="r1")
    state["candidates"] = [cand(f"c{i}") for i in range(10)]
    out = nodes.rerank(state)
    assert out["candidates"][0].chunk_id == "c3" and len(out["candidates"]) == 8
    assert reranker.calls and reranker.calls[0][0] == "q"


# --------------------------------------------------------------------------- slow: the real model


needs_reranker = pytest.mark.skipif(
    not (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-reranker-v2-m3"
    ).exists(),
    reason="bge-reranker-v2-m3 is not in the Hugging Face cache",
)


@pytest.mark.slow
@needs_reranker
def test_real_model_applies_the_sigmoid_exactly_once(settings) -> None:
    rr = m.get_reranker(settings)
    rr.warm_up()
    scores = rr.score(
        "How do readonly properties work in PHP?",
        [
            (
                "Front Line PHP > Part 1 > Chapter 06: Readonly Properties > Readonly (p. 75)\n"
                "A readonly property can only be initialized once, from inside the class scope, "
                "and cannot be changed afterwards."
            ),
            (
                "The boiling point of water at sea level is 100 degrees Celsius; mountain "
                "climbers need longer cooking times."
            ),
        ],
    )
    assert scores[0] > 0.9 and scores[1] < 0.1, (
        scores
    )  # a double sigmoid would land both in 0.50..0.73


@pytest.mark.slow
@needs_reranker
def test_real_model_counts_truncated_pairs(settings) -> None:
    rr = m.get_reranker(settings)
    short = "A readonly property can only be initialized once."
    long = " ".join(["readonly property code example"] * 400)
    assert rr.truncated_pairs("q", [short, long]) == 1


# --------------------------------------------------------------------------- live


@pytest.mark.live
@needs_reranker
def test_live_done_when_customer_dto_is_in_the_top_3() -> None:
    from tests.test_retrieve import _alias_up

    if not _alias_up():
        pytest.skip("needs Qdrant with the alias switched")
    from flp_rag.api.app import build_deps, current_index_version, default_index_dir
    from flp_rag.graph.nodes import retrieve as r
    from flp_rag.ingest.s06_s07_index import STORE_FILE

    settings = load_settings()
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    deps = build_deps(settings)
    version, _ = current_index_version(settings, deps)
    deps.store_path = default_index_dir(settings) / str(version) / STORE_FILE
    question = "Show an example of a DTO with promoted constructor properties"
    candidates = r.retrieve_hybrid(
        r.Variants(question),
        settings,
        client=deps.client,
        embedder=deps.embedder,
        sparse=deps.sparse,
    )
    nodes = build.default_nodes(deps)
    state = initial_state(question, request_id="r1")
    state["candidates"] = candidates
    kept = nodes.rerank(state)["candidates"]
    texts = r.display_texts(kept[:3], default_index_dir(settings))
    assert any("class CustomerDTO" in texts.get(c.chunk_id, "") for c in kept[:3]), [
        c.chunk_id for c in kept[:3]
    ]
