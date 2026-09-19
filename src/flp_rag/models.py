# Factories: get_chat_model(), get_embeddings(), get_reranker(). The only file that names a provider. Spec Section 4.3.
"""Provider layer.

Three small interfaces, one class per provider behind each. Nothing outside this module names a
provider or reads an API key.

- `Embedder`: `embed_passages`, `embed_query`, `model_name`, `dim`. `BgeM3Embedder` (local),
  `OpenAIEmbedder` (hosted).
- `SparseEmbedder`: `embed` for the BM25 sparse vectors. `FastembedSparse`.
- `Reranker`: `score(query, texts) -> [0, 1]`, `model_name`. `BgeReranker` (local),
  `CohereReranker` (hosted).
- `ChatModel`: `complete`, `stream`, `model_name`, `price_per_million_tokens`. `LangChainChat`
  wraps one LangChain chat model: Anthropic, OpenAI, or Ollama.

Model names in `config.yaml`: chat models use `provider:model` (`anthropic:claude-sonnet-5`,
`openai:gpt-5-mini`, `ollama:llama3.1:8b`). Embedding names that start with `text-embedding`
are OpenAI; everything else is a local Hugging Face model. Reranker names that start with
`rerank` are Cohere; everything else is a local cross-encoder. API keys come from the
environment or `.env` through `Secrets`, never from `config.yaml`.
"""

import json
import math
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from flp_rag.settings import Price, Settings

ChatRole = Literal["gen", "query", "judge"]
PROVIDERS = ("anthropic", "openai", "ollama")

_PLACEHOLDER = re.compile(r"^\s*<.*>\s*$")
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ModelConfigError(ValueError):
    """A model name in config.yaml cannot be turned into a client."""


class InvalidOutput(ValueError):
    """A chat model returned text that does not satisfy the requested JSON schema."""


# --------------------------------------------------------------------------- secrets


class Secrets(BaseSettings):
    """API keys from the environment or `.env`. Never from config.yaml."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    def require(self, name: str) -> str:
        value: SecretStr | None = getattr(self, name)
        if value is None or not value.get_secret_value():
            raise ModelConfigError(f"{name.upper()} is not set; add it to .env or the environment")
        return value.get_secret_value()


# --------------------------------------------------------------------------- contracts


@dataclass(frozen=True)
class Completion:
    """One chat completion. The `llm.*` spans read these fields."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: int
    model_name: str
    json: dict[str, Any] | None = None

    def to_attrs(self) -> dict[str, str | int | float | bool]:
        return {
            "gen_ai.request.model": self.model_name,
            "gen_ai.usage.input_tokens": self.input_tokens,
            "gen_ai.usage.output_tokens": self.output_tokens,
            "gen_ai.response.finish_reasons": json.dumps([self.finish_reason]),
            "rag.latency_ms": self.latency_ms,
        }


def completion_cost_usd(completion: Completion, price: Price) -> float:
    return (completion.input_tokens * price.input + completion.output_tokens * price.output) / 1e6


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, list[int] | list[float]]:
        return {"indices": self.indices, "values": self.values}


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[SparseVector]: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


Message = tuple[Literal["user", "assistant"], str]


class ChatModel(Protocol):
    model_name: str
    price_per_million_tokens: Price

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> Completion: ...

    def stream(self, system: str, messages: Sequence[Message], temperature: float = 0) -> Iterator[str]: ...


# --------------------------------------------------------------------------- embedders


class BgeM3Embedder:
    """Local dense embeddings with sentence-transformers. Vectors are unit length."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        self.model_name, self.dim, self.batch_size = model_name, dim, batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            found = getter()
            if found != self.dim:
                raise ModelConfigError(f"{self.model_name} has {found} dimensions, config says {self.dim}")
        return self._model

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return [row.tolist() for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]


class OpenAIEmbedder:
    """Hosted embeddings through the OpenAI API."""

    def __init__(self, model_name: str, dim: int, api_key: str, batch_size: int = 32) -> None:
        self.model_name, self.dim, self.batch_size = model_name, dim, batch_size
        self._api_key = api_key
        self._client: Any = None

    def _load(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            params: dict[str, Any] = {"model": self.model_name, "input": batch}
            if self.model_name.startswith("text-embedding-3"):
                params["dimensions"] = self.dim  # ada-002 rejects the parameter
            response = self._load().embeddings.create(**params)
            out.extend(item.embedding for item in response.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_passages([text])[0]


class FastembedSparse:
    """BM25 sparse vectors with fastembed (`Qdrant/bm25`)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=[int(i) for i in emb.indices], values=[float(v) for v in emb.values])
            for emb in self._load().embed(list(texts))
        ]


# --------------------------------------------------------------------------- rerankers


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BgeReranker:
    """Local cross-encoder. Logits pass through a sigmoid so scores lie in [0, 1]."""

    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model_name, self.batch_size = model_name, batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._torch = torch
        return self._tokenizer, self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        tokenizer, model = self._load()
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                inputs = tokenizer(
                    [query] * len(batch), batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                logits = model(**inputs).logits.view(-1).tolist()
                scores.extend(sigmoid(float(x)) for x in logits)
        return scores


class CohereReranker:
    """Hosted reranking through Cohere. Scores are already in [0, 1]."""

    def __init__(self, model_name: str, api_key: str | None = None, client: Any = None) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client = client

    def _load(self) -> Any:
        if self._client is None:
            try:
                import cohere
            except ImportError as exc:  # pragma: no cover
                raise ModelConfigError("install the `cohere` package to use a Cohere reranker") from exc
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        response = self._load().rerank(
            model=self.model_name, query=query, documents=list(texts), top_n=len(texts)
        )
        scores = [0.0] * len(texts)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return scores


# --------------------------------------------------------------------------- chat


class LangChainChat:
    """One LangChain chat model behind the `ChatModel` interface."""

    def __init__(self, llm: BaseChatModel, *, model_name: str, price: Price) -> None:
        self.llm, self.model_name, self.price_per_million_tokens = llm, model_name, price

    def complete(
        self,
        system: str,
        messages: Sequence[Message],
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0,
    ) -> Completion:
        prompt = self._messages(system, messages, json_schema)
        started = time.perf_counter()
        response = self.llm.invoke(prompt, **self.invoke_kwargs(temperature))
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = _content_text(response)
        parsed = _parse_json(text, json_schema) if json_schema is not None else None
        usage = getattr(response, "usage_metadata", None) or {}
        meta = getattr(response, "response_metadata", None) or {}
        return Completion(
            text=json.dumps(parsed, ensure_ascii=False) if parsed is not None else text,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            # Anthropic: stop_reason. OpenAI: finish_reason. Ollama: done_reason.
            finish_reason=str(
                meta.get("stop_reason") or meta.get("finish_reason") or meta.get("done_reason") or "unknown"
            ),
            latency_ms=latency_ms,
            model_name=self.model_name,
            json=parsed,
        )

    def stream(self, system: str, messages: Sequence[Message], temperature: float = 0) -> Iterator[str]:
        for piece in self.llm.stream(self._messages(system, messages, None), **self.invoke_kwargs(temperature)):
            text = _content_text(piece)
            if text:
                yield text

    def invoke_kwargs(self, temperature: float) -> dict[str, Any]:
        """Per-call temperature in the form each client accepts. ChatOllama takes it inside
        `options`; a top-level `temperature` kwarg reaches the Ollama client and raises."""
        if type(self.llm).__name__ == "ChatOllama":
            return {"options": {"temperature": temperature}}
        return {"temperature": temperature}

    @staticmethod
    def _messages(
        system: str, messages: Sequence[Message], json_schema: dict[str, Any] | None
    ) -> list[BaseMessage]:
        if json_schema is not None:
            system = (
                f"{system}\n\nReturn JSON only, no prose, matching this JSON schema:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
        out: list[BaseMessage] = [SystemMessage(content=system)]
        for role, text in messages:
            out.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
        return out


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _parse_json(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    stripped = _JSON_FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise InvalidOutput(f"model output is not JSON: {exc}; got {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise InvalidOutput(f"model output is JSON but not an object: {text[:200]!r}")
    missing = [key for key in schema.get("required", []) if key not in data]
    if missing:
        raise InvalidOutput(f"model output lacks required keys {missing}: {text[:200]!r}")
    return data


# --------------------------------------------------------------------------- factories


def parse_model_spec(spec: str) -> tuple[str, str]:
    """`anthropic:claude-sonnet-5` -> (`anthropic`, `claude-sonnet-5`)."""
    if not spec or _PLACEHOLDER.match(spec):
        raise ModelConfigError(
            f"model name {spec!r} is a placeholder; set a real `provider:model` in config.yaml"
        )
    provider, sep, name = spec.partition(":")
    if not sep or provider not in PROVIDERS or not name:
        raise ModelConfigError(
            f"model name {spec!r} needs the form provider:model with provider in {PROVIDERS}"
        )
    return provider, name


def _chat_spec(settings: Settings, role: ChatRole) -> tuple[str, str]:
    return {
        "gen": ("gen.model", settings.gen.model),
        "query": ("query.model", settings.query.model),
        "judge": ("output.judge_model", settings.output.judge_model),
    }[role]


def get_chat_model(settings: Settings, role: ChatRole, secrets: Secrets | None = None) -> LangChainChat:
    """The chat model for one role: `gen` (answers), `query` (rewrite and classify), `judge`."""
    key, spec = _chat_spec(settings, role)
    try:
        provider, name = parse_model_spec(spec)
    except ModelConfigError as exc:
        raise ModelConfigError(f"{key}: {exc}") from exc
    secrets = secrets or Secrets()
    llm: BaseChatModel
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=name, api_key=secrets.require("anthropic_api_key"), temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=name, api_key=secrets.require("openai_api_key"), temperature=0)
    else:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=name, temperature=0)
    return LangChainChat(llm, model_name=spec, price=settings.price_for(spec))


def get_embeddings(settings: Settings, secrets: Secrets | None = None) -> Embedder:
    cfg = settings.embed
    if cfg.model.startswith("text-embedding"):
        secrets = secrets or Secrets()
        return OpenAIEmbedder(cfg.model, cfg.dim, secrets.require("openai_api_key"), cfg.batch_size)
    return BgeM3Embedder(cfg.model, cfg.dim, cfg.batch_size)


def get_sparse_embeddings(settings: Settings) -> SparseEmbedder:
    return FastembedSparse(settings.embed.sparse_model)


def get_reranker(settings: Settings, secrets: Secrets | None = None) -> Reranker:
    cfg = settings.rerank
    if cfg.model.startswith("rerank"):
        secrets = secrets or Secrets()
        return CohereReranker(cfg.model, api_key=secrets.require("cohere_api_key"))
    return BgeReranker(cfg.model, batch_size=cfg.batch_size)
