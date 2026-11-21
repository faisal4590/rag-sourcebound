# Stage 19 (issue #40): the cache lookup node, its route, and the store rule of Stage 18.
"""A hit skips Stages 11-17 and answers from the cache; only answered and faithful responses
without conversation history are stored; the abstain text is never cached."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flp_rag.contracts import Candidate, Source
from flp_rag.graph import build
from flp_rag.graph import cache as cache_mod
from flp_rag.graph.nodes import respond as respond_mod
from flp_rag.graph.state import RagState, initial_state
from flp_rag.settings import ABSTAIN_TEXT, load_settings
from flp_rag.stores.answer_cache import AnswerCache
from flp_rag.stores.session_store import SessionStore

VECTORS = {
    "How do readonly properties work?": [1.0, 0.0, 0.0],
    "how do readonly props work": [0.995, 0.0998, 0.0],  # cosine 0.995 with the first
    "What are enums?": [0.0, 1.0, 0.0],
}


class FakeEmbedder:
    model_name = "fake"
    dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [VECTORS[t] for t in texts]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_queries(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]


def settings_with_cache(enabled: bool = True):  # type: ignore[no-untyped-def]
    s = load_settings()
    return s.model_copy(update={"cache": s.cache.model_copy(update={"semantic_enabled": enabled})})


LABEL_MAP = {
    "S1": {
        "chunk_ids": ["d:06:0001"],
        "parent_id": "p",
        "kind": "chunk",
        "chapter_no": 6,
        "chapter_title": "Readonly Properties",
        "section_title": "Readonly",
        "page_printed_start": 75,
        "page_printed_end": 77,
    }
}


def stub_nodes(
    settings: Any,
    answers: AnswerCache | None,
    embedder: FakeEmbedder,
    sessions: SessionStore | None = None,
    *,
    gate: str = "pass",
    faithful: bool | None = True,
    ran: list[str] | None = None,
) -> build.GraphNodes:
    def guard_input(state: RagState) -> dict[str, Any]:
        return {"injection_suspected": False, "language": "en"}

    def understand(state: RagState) -> dict[str, Any]:
        if ran is not None:
            ran.append("understand")
        return {}

    def retrieve(state: RagState) -> dict[str, Any]:
        return {"candidates": [Candidate("d:06:0001", 0.7, 0.9, ["dense"], {"chapter_no": 6})]}

    def rerank(state: RagState) -> dict[str, Any]:
        return {}

    def gate_node(state: RagState) -> dict[str, Any]:
        if gate == "abstain":
            return {
                "gate_decision": "abstain",
                "status": "no_information",
                "answer": ABSTAIN_TEXT,
                "abstain_reason": "gate.low_top_score",
                "top_score": 0.1,
            }
        return {"gate_decision": "pass", "borderline": False, "top_score": 0.9}

    def assemble(state: RagState) -> dict[str, Any]:
        return {"context": [], "label_map": LABEL_MAP}

    def generate(state: RagState) -> dict[str, Any]:
        return {"answer": "Readonly properties are set once [S1].", "prompt_version": "abcd1234"}

    def check_output(state: RagState) -> dict[str, Any]:
        if faithful is False:
            return {
                "check_result": "abstain",
                "status": "no_information",
                "answer": ABSTAIN_TEXT,
                "abstain_reason": "output.unfaithful",
                "faithful": False,
            }
        return {"check_result": "ok", "status": "answered", "faithful": faithful}

    cache_lookup = (
        cache_mod.make_cache_lookup(settings, answers, lambda: embedder, None, sessions)
        if answers is not None
        else None
    )
    return build.GraphNodes(
        guard_input,
        understand,
        retrieve,
        rerank,
        gate_node,
        assemble,
        generate,
        check_output,
        respond_mod.make_respond(settings, sessions, answers),
        cache_lookup=cache_lookup,
    )


def run(nodes: build.GraphNodes, question: str, **kw: Any) -> RagState:
    return build.build_graph(nodes).invoke(
        initial_state(question, request_id="r1", index_version="v3", **kw)
    )


def test_second_ask_is_a_hit_that_skips_stages_11_to_17(tmp_path: Path) -> None:
    settings = settings_with_cache()
    embedder = FakeEmbedder()
    ran: list[str] = []
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers:
        nodes = stub_nodes(settings, answers, embedder, ran=ran)
        first = run(nodes, "How do readonly properties work?")
        assert first["status"] == "answered" and first["cache_hit"] is False
        assert first["visited"][:3] == ["guard_input", "cache_lookup", "understand"]
        assert answers.count() == 1 and ran == ["understand"]

        second = run(nodes, "how do readonly props work", debug=True)
        assert second["cache_hit"] is True and second["cache_similarity"] > 0.97
        assert second["visited"] == ["guard_input", "cache_lookup", "respond"]
        assert second["status"] == "answered" and second["answer"] == first["answer"]
        assert second["response"].sources == first["response"].sources
        # rows are keyed by the configured answer prompt, the version of the root span
        assert second["response"].prompt_version == respond_mod.default_prompt_version(settings)
        assert second["response"].debug["cache"] == {
            "hit": True,
            "similarity": second["cache_similarity"],
            "question": "How do readonly properties work?",
        }
        assert ran == ["understand"]  # Stage 11 did not run again
        assert answers.count() == 1  # a hit is not stored again

        assert run(nodes, "What are enums?")["cache_hit"] is False and answers.count() == 2


def test_abstains_and_unjudged_answers_are_never_stored(tmp_path: Path) -> None:
    settings = settings_with_cache()
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers:
        gate = run(stub_nodes(settings, answers, FakeEmbedder(), gate="abstain"), "What are enums?")
        assert gate["status"] == "no_information" and answers.count() == 0
        unfaithful = run(
            stub_nodes(settings, answers, FakeEmbedder(), faithful=False), "What are enums?"
        )
        assert unfaithful["abstain_reason"] == "output.unfaithful" and answers.count() == 0
        unjudged = run(
            stub_nodes(settings, answers, FakeEmbedder(), faithful=None), "What are enums?"
        )
        assert unjudged["status"] == "answered" and answers.count() == 0


def test_a_request_with_conversation_history_skips_the_cache_both_ways(tmp_path: Path) -> None:
    settings = settings_with_cache()
    embedder = FakeEmbedder()
    with (
        AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers,
        SessionStore(tmp_path / "sessions.sqlite", ttl_hours=24, excerpt_chars=100) as sessions,
    ):
        nodes = stub_nodes(settings, answers, embedder, sessions)
        run(nodes, "How do readonly properties work?")  # no session: stored
        assert answers.count() == 1
        sessions.add_turn("s1", question="earlier", status="answered", answer="x")
        state = run(nodes, "How do readonly properties work?", session_id="s1")
        assert state["cache_hit"] is False and state["cache_skipped"] == "history"
        assert state["visited"][:3] == ["guard_input", "cache_lookup", "understand"]
        assert embedder.calls == 1  # the history check comes before the embedding
        assert answers.count() == 1  # and nothing is stored for a follow-up question


def test_filters_judge_all_and_a_missing_index_version_skip_the_cache(tmp_path: Path) -> None:
    from flp_rag.graph.nodes.retrieve import RetrievalFilters

    settings = settings_with_cache()
    embedder = FakeEmbedder()
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers:
        nodes = stub_nodes(settings, answers, embedder)
        filtered = run(
            nodes, "What are enums?", filters=RetrievalFilters(chapter_no=(6,), has_code=None)
        )
        harness = run(nodes, "What are enums?", judge_all=True)
        no_index = build.build_graph(nodes).invoke(
            initial_state("What are enums?", request_id="r1")
        )
        assert filtered["cache_skipped"] == "filters" and harness["cache_skipped"] == "judge_all"
        assert no_index["cache_skipped"] == "no_index"
        assert all(s["status"] == "answered" for s in (filtered, harness, no_index))
        assert answers.count() == 0 and embedder.calls == 0  # nothing embedded, nothing stored


def test_a_broken_cache_file_is_a_miss_and_an_unstored_answer_not_an_error(tmp_path: Path) -> None:
    settings = settings_with_cache()
    answers = AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168)
    nodes = stub_nodes(settings, answers, FakeEmbedder())
    answers.close()  # every statement now raises sqlite3.ProgrammingError
    state = run(nodes, "What are enums?")
    assert state["status"] == "answered" and state["cache_hit"] is False
    assert state["response"].answer.endswith("[S1].")
    settings = settings_with_cache()
    broken = {
        "status": "answered",
        "faithful": True,
        "question_vector": [1.0, 0.0, 0.0],
        "question": "q",
    }
    response = respond_mod.build_response(
        {**broken, "label_map": LABEL_MAP, "trace_id": "t", "index_version": "v3"}, settings
    )
    assert cache_mod.store_answer(broken, response, answers, settings) == "ProgrammingError"


def test_route_and_graph_shape() -> None:
    assert cache_mod.route_after_cache({"cache_hit": True}) == "respond"
    assert cache_mod.route_after_cache({"cache_hit": False}) == "understand"
    assert cache_mod.route_after_cache({}) == "understand"
    settings = settings_with_cache(False)
    plain = stub_nodes(settings, None, FakeEmbedder())
    assert plain.cache_lookup is None and "cache_lookup" not in plain.as_dict()
    assert "cache_lookup" not in build.build_graph(plain).get_graph().nodes
    assert build.SPANS["cache_lookup"] == ("cache.lookup", "CHAIN")
    assert "cache_lookup" not in build.NODE_ORDER


def test_build_response_on_a_hit_uses_the_cached_sources() -> None:
    settings = load_settings()
    sources = [Source("S1", ["d:06:0001"], 6, "Readonly Properties", "Readonly", 75, 77, 0.9)]
    state = {
        "status": "answered",
        "answer": "Readonly properties are set once [S1].",
        "sources": sources,
        "cache_hit": True,
        "prompt_version": "abcd1234",
        "trace_id": "t",
        "index_version": "v3",
    }
    response = respond_mod.build_response(state, settings)
    assert response.sources == sources and response.prompt_version == "abcd1234"


def test_default_nodes_adds_the_node_only_when_enabled_and_a_store_exists(tmp_path: Path) -> None:
    on = settings_with_cache(True)
    with AnswerCache(tmp_path / "answers.sqlite", ttl_hours=168) as answers:
        nodes = build.default_nodes(build.Deps(settings=on, answers=answers))
        assert nodes.cache_lookup is not None
    assert build.default_nodes(build.Deps(settings=on)).cache_lookup is None
    assert (
        build.default_nodes(
            build.Deps(settings=settings_with_cache(False), answers=None)
        ).cache_lookup
        is None
    )
