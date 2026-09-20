"""Done-when tests for models.py, the provider factories. Spec Section 4.3. Issue #10.

Local models (bge-m3, bge-reranker-v2-m3, Qdrant/bm25) load from the Hugging Face cache and
skip when absent. Chat models are exercised through a fake LangChain model, never the network.
"""

import json
import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from flp_rag import models as m
from flp_rag.settings import load_settings


def _cached(repo: str) -> bool:
    home = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(home, "models--" + repo.replace("/", "--")))


needs_bge_m3 = pytest.mark.skipif(not _cached("BAAI/bge-m3"), reason="bge-m3 not in the HF cache")
needs_reranker = pytest.mark.skipif(
    not _cached("BAAI/bge-reranker-v2-m3"), reason="bge-reranker-v2-m3 not in the HF cache"
)


# --------------------------------------------------------------------------- model specs


def test_parse_model_spec_splits_provider_and_name() -> None:
    assert m.parse_model_spec("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    assert m.parse_model_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert m.parse_model_spec("openai:gpt-5-mini") == ("openai", "gpt-5-mini")


@pytest.mark.parametrize("spec", ["<generator-model-name>", "<small-model-name>", ""])
def test_placeholder_model_names_fail_with_a_clear_message(spec: str) -> None:
    with pytest.raises(m.ModelConfigError, match="config.yaml"):
        m.parse_model_spec(spec)


def test_unknown_provider_fails() -> None:
    with pytest.raises(m.ModelConfigError, match="provider"):
        m.parse_model_spec("mistral:large")
    with pytest.raises(m.ModelConfigError, match="provider"):
        m.parse_model_spec("claude-sonnet-5")  # no prefix


# --------------------------------------------------------------------------- chat adapter


def _fake_chat(text: str, usage: dict | None = None, stop: str = "end_turn") -> FakeMessagesListChatModel:
    msg = AIMessage(
        content=text,
        usage_metadata=usage or {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
        response_metadata={"stop_reason": stop},
    )
    return FakeMessagesListChatModel(responses=[msg])


def test_langchain_chat_complete_returns_a_completion() -> None:
    chat = m.LangChainChat(_fake_chat("No information found"), model_name="fake:one",
                           price=m.Price(input=1.0, output=2.0))
    out = chat.complete("You answer from the book.", [("user", "What is readonly?")])
    assert isinstance(out, m.Completion)
    assert out.text == "No information found"
    assert (out.input_tokens, out.output_tokens) == (12, 5)
    assert out.finish_reason == "end_turn"
    assert out.latency_ms >= 0 and out.model_name == "fake:one"
    assert chat.model_name == "fake:one" and chat.price_per_million_tokens.output == 2.0


def test_langchain_chat_json_schema_parses_and_validates() -> None:
    schema = {"type": "object", "required": ["intent", "wants_code"],
              "properties": {"intent": {"type": "string"}, "wants_code": {"type": "boolean"}}}
    chat = m.LangChainChat(_fake_chat('```json\n{"intent": "book_question", "wants_code": false}\n```'),
                           model_name="fake:json", price=m.Price(input=0, output=0))
    out = chat.complete("Classify.", [("user", "hi")], json_schema=schema)
    assert out.json == {"intent": "book_question", "wants_code": False}
    assert out.text.startswith("{")


def test_langchain_chat_json_schema_missing_key_raises() -> None:
    schema = {"type": "object", "required": ["intent"], "properties": {"intent": {"type": "string"}}}
    chat = m.LangChainChat(_fake_chat('{"other": 1}'), model_name="fake:json",
                           price=m.Price(input=0, output=0))
    with pytest.raises(m.InvalidOutput, match="intent"):
        chat.complete("Classify.", [("user", "hi")], json_schema=schema)


def test_langchain_chat_stream_yields_text() -> None:
    chat = m.LangChainChat(_fake_chat("Readonly properties cannot change."), model_name="fake:s",
                           price=m.Price(input=0, output=0))
    assert "".join(chat.stream("sys", [("user", "q")])) == "Readonly properties cannot change."


def test_ollama_temperature_goes_inside_options() -> None:
    from langchain_ollama import ChatOllama

    ollama = m.LangChainChat(ChatOllama(model="llama3.1:8b"), model_name="ollama:llama3.1:8b",
                             price=m.Price(input=0, output=0))
    assert ollama.invoke_kwargs(0.2) == {"options": {"temperature": 0.2}}
    # The installed client rejects a top-level temperature; the params builder must not leak one.
    params = ollama.llm._chat_params([], None, **ollama.invoke_kwargs(0.2))
    assert "temperature" not in params and params["options"]["temperature"] == 0.2
    other = m.LangChainChat(_fake_chat("x"), model_name="fake", price=m.Price(input=0, output=0))
    assert other.invoke_kwargs(0.0) == {"temperature": 0.0}


def test_finish_reason_reads_each_provider_key() -> None:
    for meta, expected in (({"stop_reason": "end_turn"}, "end_turn"),
                           ({"finish_reason": "stop"}, "stop"),
                           ({"done_reason": "length"}, "length"),
                           ({}, "unknown")):
        msg = AIMessage(content="ok", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        response_metadata=meta)
        chat = m.LangChainChat(FakeMessagesListChatModel(responses=[msg]), model_name="fake",
                               price=m.Price(input=0, output=0))
        assert chat.complete("s", [("user", "q")]).finish_reason == expected


def test_secrets_env_file_is_anchored_at_the_repo_root() -> None:
    from pathlib import Path

    assert Path(m.Secrets.model_config["env_file"]).parent == Path(m.__file__).resolve().parents[2]


def test_completion_cost_uses_the_price_table() -> None:
    c = m.Completion(text="x", input_tokens=2_000_000, output_tokens=500_000, finish_reason="stop",
                     latency_ms=1, model_name="fake")
    assert m.completion_cost_usd(c, m.Price(input=1.0, output=4.0)) == pytest.approx(2.0 + 2.0)
    assert m.completion_cost_usd(replace(c, output_tokens=0), m.Price(input=0, output=0)) == 0.0


def test_get_chat_model_builds_the_right_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    settings = load_settings()

    def with_models(gen: str, query: str, judge: str):
        prices = {**settings.prices_usd_per_million_tokens,
                  gen: m.Price(input=3, output=15), query: m.Price(input=1, output=5),
                  judge: m.Price(input=1, output=5)}
        return settings.model_copy(update={
            "gen": settings.gen.model_copy(update={"model": gen}),
            "query": settings.query.model_copy(update={"model": query}),
            "output": settings.output.model_copy(update={"judge_model": judge}),
            "prices_usd_per_million_tokens": prices,
        })

    s = with_models("anthropic:claude-sonnet-5", "openai:gpt-5-mini", "ollama:llama3.1:8b")
    gen, query, judge = (m.get_chat_model(s, role) for role in ("gen", "query", "judge"))
    assert type(gen.llm).__name__ == "ChatAnthropic" and gen.model_name == "anthropic:claude-sonnet-5"
    assert type(query.llm).__name__ == "ChatOpenAI"
    assert type(judge.llm).__name__ == "ChatOllama"
    assert gen.price_per_million_tokens.output == 15


def test_get_chat_model_rejects_placeholders_from_shipped_config() -> None:
    with pytest.raises(m.ModelConfigError, match="gen.model"):
        m.get_chat_model(load_settings(), "gen")


def test_secrets_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "co-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    secrets = m.Secrets(_env_file=None)
    assert secrets.cohere_api_key is not None and secrets.cohere_api_key.get_secret_value() == "co-test"
    assert secrets.anthropic_api_key is None
    assert "co-test" not in repr(secrets)


# --------------------------------------------------------------------------- rerankers


def test_sigmoid_maps_logits_to_unit_interval() -> None:
    assert m.sigmoid(0.0) == pytest.approx(0.5)
    assert 0.99 < m.sigmoid(10.0) < 1.0 and 0.0 < m.sigmoid(-10.0) < 0.01


def test_cohere_reranker_maps_scores_back_to_input_order() -> None:
    class FakeResult:
        def __init__(self, index: int, score: float) -> None:
            self.index, self.relevance_score = index, score

    class FakeResponse:
        def __init__(self) -> None:
            self.results = [FakeResult(2, 0.9), FakeResult(0, 0.4), FakeResult(1, 0.1)]

    class FakeClient:
        def rerank(self, **kwargs):
            assert kwargs["top_n"] == 3 and kwargs["model"] == "rerank-v3.5"
            return FakeResponse()

    rr = m.CohereReranker("rerank-v3.5", client=FakeClient())
    assert rr.score("q", ["a", "b", "c"]) == [0.4, 0.1, 0.9]
    assert rr.model_name == "rerank-v3.5"


@needs_reranker
def test_bge_reranker_scores_relevant_text_higher() -> None:
    rr = m.BgeReranker("BAAI/bge-reranker-v2-m3", batch_size=16)
    scores = rr.score("How do readonly properties work in PHP?",
                      ["A readonly property can only be initialized once, from inside the class.",
                       "The JIT compiler translates opcodes into machine code at runtime."])
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] > scores[1]


# --------------------------------------------------------------------------- embedders


@needs_bge_m3
def test_bge_m3_embedder_dim_norm_and_shapes() -> None:
    emb = m.get_embeddings(load_settings())
    assert isinstance(emb, m.BgeM3Embedder)
    assert emb.model_name == "BAAI/bge-m3" and emb.dim == 1024
    vectors = emb.embed_passages(["readonly properties", "the JIT compiler"])
    query = emb.embed_query("readonly properties")
    assert len(vectors) == 2 and len(vectors[0]) == 1024 and len(query) == 1024
    assert math.sqrt(sum(x * x for x in vectors[0])) == pytest.approx(1.0, abs=1e-3)
    # The same text as passage and as query lands in the same place for bge-m3.
    assert sum(a * b for a, b in zip(vectors[0], query, strict=True)) > 0.99


def test_openai_embedder_is_chosen_for_hosted_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    settings = load_settings()
    s = settings.model_copy(update={"embed": settings.embed.model_copy(
        update={"model": "text-embedding-3-small", "dim": 1536})})
    emb = m.get_embeddings(s)
    assert isinstance(emb, m.OpenAIEmbedder) and emb.dim == 1536


def test_openai_embedder_sends_dimensions_only_for_v3_models() -> None:
    calls: list[dict] = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            items = [SimpleNamespace(embedding=[0.0] * 3) for _ in kwargs["input"]]
            return SimpleNamespace(data=items)

    class FakeClient:
        embeddings = FakeEmbeddings()

    for name, expect_dims in (("text-embedding-3-small", True), ("text-embedding-ada-002", False)):
        emb = m.OpenAIEmbedder(name, 3, api_key="k")
        emb._client = FakeClient()
        assert len(emb.embed_passages(["a", "b"])) == 2
        assert ("dimensions" in calls[-1]) is expect_dims


def test_sparse_bm25_embedder_produces_sparse_vectors() -> None:
    sparse = m.get_sparse_embeddings(load_settings())
    assert sparse.model_name == "Qdrant/bm25"
    vectors = sparse.embed(["readonly properties readonly", "match expression"])
    assert len(vectors) == 2
    v = vectors[0]
    assert isinstance(v, m.SparseVector) and len(v.indices) == len(v.values) > 0
    assert len(set(v.indices)) == len(v.indices)
    assert all(x > 0 for x in v.values)
    assert json.dumps(v.to_dict())  # serializable for Qdrant
