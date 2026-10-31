# Stage 11 "Done when": the follow-up rewrite, the smalltalk abstain, the output checks, the fallback. Issue #27.
"""Unit tests for `graph/nodes/understand.py` with a scripted small model, plus two live cases."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Chunk, Parent, StandaloneQuery
from flp_rag.graph import build
from flp_rag.graph.nodes import respond as respond_mod
from flp_rag.graph.nodes import retrieve as r
from flp_rag.graph.nodes import understand as u
from flp_rag.graph.state import RagState, initial_state
from flp_rag.models import Completion, InvalidOutput, ModelConfigError, Price
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings
from flp_rag.stores.parent_store import ParentStore
from flp_rag.stores.session_store import SessionStore
from tests.test_graph_build import stub_nodes

STANDALONE = "What does the book say about enums in PHP?"
FULL: dict[str, Any] = {
    "standalone": STANDALONE,
    "intent": "book_question",
    "wants_code": False,
    "chapter_hint": 9,
    "paraphrases": ["How are enums used in PHP 8?", "What are PHP enumerations?", "Enums in PHP?"],
    "keywords": "enum enums backed cases",
}
HISTORY = [
    ("user", "Tell me about readonly properties"),
    ("assistant", "Readonly properties are set once [S1]."),
]
CHAPTERS = (
    (0, "Foreword"),
    (0, "Preface"),
    (1, "PHP Today"),
    (6, "Readonly Properties"),
    (9, "Enums"),
    (31, "In Closing"),
)
FACTS = u.IndexFacts(CHAPTERS, frozenset({"readonly", "enum", "customerdto", "match"}))


class FakeQueryModel:
    """Scripted outputs: a dict (returned as the JSON the schema asks for), a non-JSON string
    (`InvalidOutput`, as `LangChainChat` raises it), or an exception to raise."""

    def __init__(self, *outputs: Any, model_name: str = "fake:query") -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []
        self.model_name = model_name
        self.price_per_million_tokens = Price(input=0.1, output=0.4)

    def complete(
        self, system, messages, json_schema=None, temperature=0, max_tokens=None
    ) -> Completion:  # type: ignore[no-untyped-def]
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_schema": json_schema,
            }
        )
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, dict):
            return Completion(json.dumps(out), 800, 150, "stop", 5, self.model_name, json=out)
        raise InvalidOutput(f"model output is not JSON: {out!r}")

    def stream(self, system, messages, temperature=0):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def settings_for(**query: Any) -> Settings:
    base = load_settings()
    return base.model_copy(
        update={"query": base.query.model_copy(update={"retry_backoff_s": 0.0, **query})}
    )


def run(
    outputs: list[Any],
    question: str,
    *,
    history: list[tuple[str, str]] | None = None,
    settings: Settings | None = None,
    facts: u.IndexFacts = FACTS,
) -> tuple[u.Understanding, FakeQueryModel]:
    chat = FakeQueryModel(*outputs)
    s = settings or settings_for()
    prompt = u.load_query_prompt(s.config_path.parent / s.query.prompt_file)
    result = u.understand(
        question, list(history or []), chat=chat, prompt=prompt, facts=facts, settings=s
    )
    return result, chat


def user_message(chat: FakeQueryModel, call: int = 0) -> str:
    return chat.calls[call]["messages"][0][1]


def node_for(
    outputs: list[Any],
    *,
    settings: Settings | None = None,
    sessions: SessionStore | None = None,
    facts: u.IndexFacts = FACTS,
) -> tuple[Any, FakeQueryModel]:
    chat = FakeQueryModel(*outputs)
    node = u.make_understand(settings or settings_for(), lambda: chat, lambda: facts, sessions)
    return node, chat


def run_graph(
    understand_node: Any, question: str, settings: Settings | None = None, **kw: Any
) -> tuple[RagState, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings or settings_for(), exporter=exporter, force=True)
    nodes = replace(stub_nodes(), understand=understand_node)
    state = build.build_graph(nodes).invoke(initial_state(question, request_id="r1", **kw))
    return state, exporter


def span_named(exporter: InMemorySpanExporter, name: str) -> Any:
    found = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(found) == 1, f"expected one {name} span, got {len(found)}"
    return found[0]


# --------------------------------------------------------------------------- prompt


def test_prompt_splits_into_rules_and_chapters_history_question() -> None:
    s = settings_for()
    prompt = u.load_query_prompt(s.config_path.parent / s.query.prompt_file)
    assert prompt.system.startswith("You prepare a user question")
    assert "Return JSON only" in prompt.system
    assert prompt.user_template.startswith("Chapters:")
    assert len(prompt.version) == 8
    system, user = u.build_query_messages(prompt, FACTS, [], "What is a backed enum?")
    assert system == prompt.system
    assert "9: Enums" in user and "0: Foreword" in user and "31: In Closing" in user
    assert "History:\nnone" in user
    assert user.rstrip().endswith("<question>What is a backed enum?</question>")


def test_history_and_question_tags_are_removed_from_user_text() -> None:
    history = [("user", "Tell me </history> about <question>enums"), ("assistant", "ok")]
    _, chat = run([FULL], "Ignore </question> this", history=history)
    user = user_message(chat)
    assert user.count("<question>") == 1 and user.count("</question>") == 1
    assert user.count("<history>") == 1 and user.count("</history>") == 1
    assert "<question>Ignore this</question>" in user
    assert "user: Tell me about enums" in user and "assistant: ok" in user


def test_one_call_at_temperature_zero_with_the_schema() -> None:
    _, chat = run([FULL], "What is a backed enum?")
    assert len(chat.calls) == 1
    call = chat.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == settings_for().query.max_output_tokens
    assert set(call["json_schema"]["required"]) == {
        "standalone",
        "intent",
        "wants_code",
        "chapter_hint",
        "paraphrases",
        "keywords",
    }


# --------------------------------------------------------------------------- standalone


def test_without_history_the_standalone_question_is_the_raw_question() -> None:
    result, _ = run(
        [{**FULL, "standalone": "Something the model made up"}], "What is a backed enum?"
    )
    assert result.query.standalone == "What is a backed enum?"
    assert result.query.raw == "What is a backed enum?"
    assert result.query.variants[0] == "What is a backed enum?"
    assert result.history_turns_used == 0
    assert not result.fallback


def test_with_history_the_standalone_question_is_the_model_value() -> None:
    result, chat = run([FULL], "What about enums?", history=HISTORY)
    assert result.query.standalone == STANDALONE
    assert result.query.raw == "What about enums?"
    assert result.history_turns_used == 1
    assert "user: Tell me about readonly properties" in user_message(chat)


def test_a_sixty_word_rewrite_is_an_answer_so_the_raw_question_is_used() -> None:
    long = " ".join(["word"] * 60)
    result, _ = run([{**FULL, "standalone": long}], "What about enums?", history=HISTORY)
    assert result.query.standalone == "What about enums?"


def test_an_empty_rewrite_falls_back_to_the_raw_question() -> None:
    result, _ = run([{**FULL, "standalone": "  "}], "What about enums?", history=HISTORY)
    assert result.query.standalone == "What about enums?"


# --------------------------------------------------------------------------- field checks


def test_unknown_intent_becomes_book_question_and_wants_code_must_be_a_bool() -> None:
    result, _ = run([{**FULL, "intent": "banter", "wants_code": "yes"}], "q")
    assert result.query.intent == "book_question"
    assert result.query.wants_code is False
    result, _ = run([{**FULL, "intent": "meta", "wants_code": True}], "q")
    assert result.query.intent == "meta"
    assert result.query.wants_code is True


def test_chapter_hint_outside_the_index_becomes_none() -> None:
    assert run([{**FULL, "chapter_hint": 42}], "q")[0].query.chapter_hint is None
    assert run([{**FULL, "chapter_hint": "9"}], "q")[0].query.chapter_hint is None
    assert run([{**FULL, "chapter_hint": True}], "q")[0].query.chapter_hint is None
    assert run([{**FULL, "chapter_hint": None}], "q")[0].query.chapter_hint is None
    assert run([{**FULL, "chapter_hint": 9}], "q")[0].query.chapter_hint == 9
    assert run([{**FULL, "chapter_hint": 0}], "q")[0].query.chapter_hint == 0


def test_paraphrase_checks_drop_empty_duplicate_and_extra_entries() -> None:
    paraphrases = [
        "",
        "  What does the book say about enums in PHP?  ",  # equals the standalone question
        "How are enums used?",
        "how are ENUMS used?",  # duplicate, letter case ignored
        7,  # not text
        "Third paraphrase",
        "Fourth paraphrase",
        "Fifth paraphrase",  # over query.expansions
    ]
    result, _ = run([{**FULL, "paraphrases": paraphrases}], "What about enums?", history=HISTORY)
    assert result.query.variants == [
        STANDALONE,
        "How are enums used?",
        "Third paraphrase",
        "Fourth paraphrase",
    ]
    assert result.paraphrases_dropped == 5


def test_paraphrases_that_are_not_a_list_are_dropped() -> None:
    result, _ = run([{**FULL, "paraphrases": "one string"}], "q")
    assert result.query.variants == ["q"]
    assert result.paraphrases_dropped == 0


def test_a_paraphrase_that_loses_a_php_symbol_of_the_question_is_dropped() -> None:
    out = {
        **FULL,
        "paraphrases": ["How do immutable fields work?", "How does a readonly property behave?"],
    }
    result, _ = run([out], "How do readonly properties work?")
    assert result.query.variants == [
        "How do readonly properties work?",
        "How does a readonly property behave?",
    ]
    assert result.paraphrases_dropped == 1
    kept_all, _ = run(
        [out],
        "How do readonly properties work?",
        settings=settings_for(keep_question_symbols=False),
    )
    assert len(kept_all.query.variants) == 3


def test_keyword_variant_is_left_out_when_keywords_are_empty() -> None:
    full, _ = run([FULL], "What about enums?", history=HISTORY)
    assert full.query.keyword_variant == "enum enums backed cases"
    assert r.variants_from(full.query, full.query.raw).count == 5
    none, _ = run([{**FULL, "keywords": "   "}], "What about enums?", history=HISTORY)
    assert none.query.keyword_variant is None
    assert r.variants_from(none.query, none.query.raw).count == 4
    bad, _ = run([{**FULL, "keywords": ["enum"]}], "What about enums?", history=HISTORY)
    assert bad.query.keyword_variant is None


# --------------------------------------------------------------------------- smalltalk


SMALLTALK = {
    **FULL,
    "standalone": "hi there",
    "intent": "smalltalk",
    "chapter_hint": None,
    "paraphrases": [],
    "keywords": "",
}


def test_hi_there_abstains_with_reason_query_smalltalk_before_retrieval() -> None:
    node, _ = node_for([SMALLTALK])
    state, exporter = run_graph(node, "hi there")
    assert state["status"] == "no_information"
    assert state["answer"] == ABSTAIN_TEXT
    assert state["abstain_reason"] == "query.smalltalk"
    assert state["visited"] == ["guard_input", "understand", "respond"]
    names = {s.name for s in exporter.get_finished_spans()}
    assert "retrieval.hybrid" not in names and "rerank" not in names
    span = span_named(exporter, "query.understand")
    assert span.attributes["rag.intent"] == "smalltalk"
    assert span.attributes["rag.abstain_reason"] == "query.smalltalk"


def test_smalltalk_over_the_word_guard_continues_to_retrieval() -> None:
    question = "Thanks, that helps. And how do backed enums work in PHP?"  # 11 words
    node, _ = node_for([{**SMALLTALK, "standalone": question}])
    state, _ = run_graph(node, question)
    assert state["visited"][:3] == ["guard_input", "understand", "retrieve"]
    assert state["standalone"].intent == "smalltalk"


def test_other_and_meta_never_abstain_on_their_own() -> None:
    for intent in ("other", "meta"):
        node, _ = node_for([{**FULL, "intent": intent}])
        state, _ = run_graph(node, "What is the boiling point of water at sea level?")
        assert state["visited"][2] == "retrieve"


def test_policy_greet_does_not_abstain() -> None:
    node, _ = node_for([SMALLTALK], settings=settings_for(smalltalk_policy="greet"))
    state, _ = run_graph(node, "hi there", settings=settings_for(smalltalk_policy="greet"))
    assert state["visited"][2] == "retrieve"


# --------------------------------------------------------------------------- fallback


def test_two_invalid_outputs_give_the_fallback_and_the_request_continues() -> None:
    node, chat = node_for(["not json", "{still broken"])
    state, exporter = run_graph(node, "What is a backed enum?")
    assert len(chat.calls) == 2
    q = state["standalone"]
    assert q == StandaloneQuery(
        "What is a backed enum?",
        "What is a backed enum?",
        "book_question",
        False,
        None,
        ["What is a backed enum?"],
        None,
    )
    assert state["visited"][2] == "retrieve"
    span = span_named(exporter, "query.understand")
    assert span.attributes["rag.understand_fallback"] is True
    assert span.attributes["rag.variants"] == json.dumps(["What is a backed enum?"])
    classify = span_named(exporter, "query.classify")
    assert classify.attributes["rag.attempts"] == 2
    assert classify.parent.span_id == span.context.span_id


def test_two_timeouts_give_the_fallback() -> None:
    result, _ = run([TimeoutError("read timed out"), TimeoutError("again")], "q")
    assert result.fallback and result.attempts == 2
    assert result.query.variants == ["q"]
    assert result.error == "transport:TimeoutError"


def test_one_timeout_then_a_good_output_is_used() -> None:
    result, chat = run([TimeoutError("read timed out"), FULL], "What about enums?", history=HISTORY)
    assert not result.fallback and result.attempts == 2
    assert result.query.standalone == STANDALONE
    assert len(chat.calls) == 2


def test_a_missing_required_key_counts_as_invalid_output() -> None:
    partial = {"standalone": "q", "intent": "book_question"}
    result, _ = run([partial, FULL], "q")
    assert not result.fallback and result.attempts == 2


def test_a_rejected_json_from_the_provider_still_counts_its_tokens() -> None:
    """`LangChainChat.complete` raises `InvalidOutput` after the call went through; the tokens
    of that attempt are on the exception and count toward the totals."""
    spent = Completion("not json", 700, 1200, "length", 5, "fake:query")
    result, chat = run([InvalidOutput("cut", spent), FULL], "q")
    assert not result.fallback and result.attempts == 2
    assert result.input_tokens == 700 + 800 and result.output_tokens == 1200 + 150
    assert len(chat.calls) == 2
    twice, _ = run([InvalidOutput("cut", spent), InvalidOutput("cut", spent)], "q")
    assert twice.fallback and twice.error == "invalid_output"
    assert twice.input_tokens == 1400


def test_a_configuration_error_propagates() -> None:
    with pytest.raises(ModelConfigError):
        run([ModelConfigError("query.model is a placeholder")], "q")


def test_an_auth_error_is_not_retried_and_propagates() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    with pytest.raises(AuthenticationError):
        run([AuthenticationError("bad key")], "q")


# --------------------------------------------------------------------------- node and trace


def test_node_writes_the_part_6_attributes() -> None:
    node, _ = node_for([FULL])
    state, exporter = run_graph(node, "What about enums?", history=HISTORY)
    span = span_named(exporter, "query.understand")
    a = span.attributes
    assert a["rag.question_raw"] == "What about enums?"
    assert a["rag.question_standalone"] == STANDALONE
    assert a["rag.intent"] == "book_question"
    assert a["rag.wants_code"] is False
    assert a["rag.chapter_hint"] == 9
    assert json.loads(a["rag.variants"]) == [STANDALONE, *FULL["paraphrases"], FULL["keywords"]]
    assert a["rag.history_turns_used"] == 1
    assert a["rag.understand_fallback"] is False
    assert a["rag.paraphrases_dropped"] == 0
    classify = span_named(exporter, "query.classify")
    c = classify.attributes
    assert c["openinference.span.kind"] == "LLM"
    assert c["rag.rewrite_used"] is True
    assert c["gen_ai.request.model"] == "fake:query"
    assert c["gen_ai.usage.input_tokens"] == 800
    assert c["gen_ai.usage.output_tokens"] == 150
    assert "input.value" in c and "output.value" in c
    assert (
        c["rag.prompt_version"]
        == u.load_query_prompt(
            settings_for().config_path.parent / settings_for().query.prompt_file
        ).version
    )
    assert state["input_tokens"] == 800 and state["output_tokens"] == 150
    assert state["cost_usd"] == pytest.approx((800 * 0.1 + 150 * 0.4) / 1e6)


def test_question_text_is_not_recorded_when_capture_content_is_off() -> None:
    s = settings_for()
    quiet = s.model_copy(update={"trace": s.trace.model_copy(update={"capture_content": False})})
    node, _ = node_for([FULL], settings=quiet)
    _, exporter = run_graph(node, "What about enums?", settings=quiet, history=HISTORY)
    a = span_named(exporter, "query.understand").attributes
    assert "rag.question_raw" not in a and "rag.question_standalone" not in a
    assert "rag.variants" not in a and a["rag.n_variants"] == 5
    assert a["rag.intent"] == "book_question"
    c = span_named(exporter, "query.classify").attributes
    assert "input.value" not in c and "output.value" not in c


def test_node_reads_the_last_turns_of_the_session(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "s.sqlite", ttl_hours=24, excerpt_chars=300)
    for i in range(4):
        sessions.add_turn("s1", question=f"q{i}", status="answered", answer=f"a{i} [S1]")
    node, chat = node_for([FULL], sessions=sessions)
    state, exporter = run_graph(node, "What about enums?", session_id="s1")
    assert state["history"] == [
        ("user", "q1"),
        ("assistant", "a1 [S1]"),
        ("user", "q2"),
        ("assistant", "a2 [S1]"),
        ("user", "q3"),
        ("assistant", "a3 [S1]"),
    ]
    assert "user: q0" not in user_message(chat) and "user: q1" in user_message(chat)
    a = span_named(exporter, "query.understand").attributes
    assert a["rag.history_turns_used"] == 3
    assert span_named(exporter, "query.classify").attributes["rag.rewrite_used"] is True
    assert state["standalone"].standalone == STANDALONE
    sessions.close()


def test_node_without_a_session_sends_no_history(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "s.sqlite", ttl_hours=24, excerpt_chars=300)
    sessions.add_turn("s1", question="q0", status="answered", answer="a0")
    node, chat = node_for([FULL], sessions=sessions)
    state, exporter = run_graph(node, "What is a backed enum?")
    assert state["history"] == []
    assert "History:\nnone" in user_message(chat)
    assert span_named(exporter, "query.classify").attributes["rag.rewrite_used"] is False
    assert state["standalone"].standalone == "What is a backed enum?"
    sessions.close()


def test_respond_writes_one_turn_per_request_also_for_an_abstain(tmp_path: Path) -> None:
    s = settings_for()
    sessions = SessionStore(tmp_path / "s.sqlite", ttl_hours=24, excerpt_chars=300)
    understand, _ = node_for([FULL], sessions=sessions)
    nodes = replace(
        stub_nodes(), understand=understand, respond=respond_mod.make_respond(s, sessions)
    )
    tracing.configure_tracing(s, exporter=InMemorySpanExporter(), force=True)
    graph = build.build_graph(nodes)
    graph.invoke(
        initial_state("What about enums?", request_id="r1", session_id="s1", history=HISTORY)
    )
    turns = sessions.last_turns("s1", 3)
    assert [(t.turn_no, t.question, t.status) for t in turns] == [(1, STANDALONE, "answered")]
    assert turns[0].answer_excerpt == "Readonly properties are set once [S1]."
    abstain = replace(
        stub_nodes(gate="abstain"),
        understand=node_for([FULL], sessions=sessions)[0],
        respond=respond_mod.make_respond(s, sessions),
    )
    build.build_graph(abstain).invoke(
        initial_state("Capital of France?", request_id="r2", session_id="s1")
    )
    turns = sessions.last_turns("s1", 3)
    assert [(t.turn_no, t.status, t.answer_excerpt) for t in turns][1] == (
        2,
        "no_information",
        ABSTAIN_TEXT,
    )
    # no session id: nothing written
    fresh = replace(nodes, understand=node_for([FULL], sessions=sessions)[0])
    build.build_graph(fresh).invoke(initial_state("What about enums?", request_id="r3"))
    assert sessions.count() == 2
    sessions.close()


def test_a_session_store_error_does_not_fail_the_answer(tmp_path: Path) -> None:
    import sqlite3

    class BrokenStore:
        path = tmp_path / "broken.sqlite"

        def add_turn(self, *a: Any, **kw: Any) -> None:
            raise sqlite3.OperationalError("database is locked")

    s = settings_for()
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(s, exporter=exporter, force=True)
    nodes = replace(
        stub_nodes(),
        understand=node_for([FULL])[0],
        respond=respond_mod.make_respond(s, BrokenStore()),  # type: ignore[arg-type]
    )
    state = build.build_graph(nodes).invoke(
        initial_state("What about enums?", request_id="r1", session_id="s1", history=HISTORY)
    )
    assert state["status"] == "answered"
    assert (
        span_named(exporter, "response.build").attributes["rag.session_turn"] == "OperationalError"
    )
    ok = replace(nodes, understand=node_for([FULL])[0], respond=respond_mod.make_respond(s, None))
    build.build_graph(ok).invoke(initial_state("q", request_id="r2", session_id="s1"))
    written = [
        sp.attributes["rag.session_turn"]
        for sp in exporter.get_finished_spans()
        if sp.name == "response.build"
    ]
    assert written[-1] == "skipped"


def test_a_given_history_is_capped_at_history_turns() -> None:
    long_history = [(role, f"{role}{i}") for i in range(5) for role in ("user", "assistant")]
    two = settings_for(history_turns=2)
    node, chat = node_for([FULL], settings=two)
    state, exporter = run_graph(node, "What about enums?", settings=two, history=long_history)
    assert state["history"] == long_history[-4:]
    assert "user0" not in user_message(chat) and "user3" in user_message(chat)
    assert span_named(exporter, "query.understand").attributes["rag.history_turns_used"] == 2


# --------------------------------------------------------------------------- index facts


def test_index_facts_come_from_the_parent_store(tmp_path: Path) -> None:
    def chunk(i: int, chapter_no: int, title: str, symbols: list[str]) -> Chunk:
        payload = {"chapter_no": chapter_no, "chapter_title": title, "php_symbols": symbols}
        return Chunk(f"d:{i:02d}", "p", "text", "text", 10, payload)

    with ParentStore(tmp_path / "store.sqlite") as store:
        store.put_parents([Parent("p", 0, "Foreword", "text", 7, 8, 10)])
        store.put_chunks(
            [
                chunk(0, 0, "Preface", ["Needle"]),
                chunk(1, 0, "Foreword", []),
                chunk(2, 6, "Readonly Properties", ["readonly", "CustomerDTO"]),
                chunk(3, 6, "Readonly Properties", ["readonly"]),
                chunk(4, 9, "Enums", ["enum"]),
                Chunk("d:99", "p", "text", "text", 10, {"php_symbols": []}),  # no chapter
            ]
        )
    facts = u.IndexFacts.from_store(tmp_path / "store.sqlite")
    assert facts.chapters == (
        (0, "Preface"),
        (0, "Foreword"),
        (6, "Readonly Properties"),
        (9, "Enums"),
    )
    assert facts.chapter_numbers == frozenset({0, 6, 9})
    assert facts.symbols == frozenset({"needle", "readonly", "customerdto", "enum"})
    assert facts.chapter_list() == "0: Preface\n0: Foreword\n6: Readonly Properties\n9: Enums"


def test_index_facts_without_a_store_path_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="store"):
        u.IndexFacts.from_store(None)


def test_query_settings_have_the_stage_11_keys() -> None:
    s = load_settings()
    assert s.query.smalltalk_max_words == 8
    assert s.query.history_turns == 3 and s.query.expansions == 3
    assert s.query.max_standalone_words == 50
    assert s.query.prompt_file == Path("prompts/query_v1.md")
    assert s.query.temperature == 0
    assert s.session.ttl_hours == 24 and s.session.answer_excerpt_chars == 300
    assert str(s.session.store_path).endswith(".sqlite")


# --------------------------------------------------------------------------- live


@pytest.mark.live
def test_done_when_with_the_real_small_model() -> None:
    """The two "Done when" cases of spec Stage 11 with the configured `query.model` and the
    chapter list of the live index."""
    from flp_rag.api.app import current_index_version, default_index_dir
    from flp_rag.ingest.s06_s07_index import STORE_FILE
    from flp_rag.models import get_chat_model

    s = load_settings()
    version, _ = current_index_version(s, build.Deps(settings=s))
    assert version, "the Qdrant alias must point at an index"
    facts = u.IndexFacts.from_store(default_index_dir(s) / version / STORE_FILE)
    assert len(facts.chapters) == 33
    chat = get_chat_model(s, "query")
    prompt = u.load_query_prompt(s.config_path.parent / s.query.prompt_file)
    follow_up = u.understand(
        "What about enums?", list(HISTORY), chat=chat, prompt=prompt, facts=facts, settings=s
    )
    assert not follow_up.fallback
    assert "enum" in follow_up.query.standalone.lower()
    assert follow_up.query.intent == "book_question"
    greeting = u.understand("hi there", [], chat=chat, prompt=prompt, facts=facts, settings=s)
    assert not greeting.fallback
    assert greeting.query.intent == "smalltalk"
    assert u.should_abstain(greeting.query, s.query)
