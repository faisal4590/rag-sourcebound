# Stage 16 - Generation. Prompt from prompts/, abstain and citation forms, repair. Spec Section 6, Stage 16.
"""Stage 16: the only stage that writes text for the user.

Plain functions of the question, the context blocks, the chat model, and `gen.*` in
`config.yaml` (spec Section 6, Stage 16; issue #21):

1. Prompt: `gen.prompt_file`, or `gen.strict_prompt_file` when the request is borderline and for
   every repair call. The rules before the `Context blocks:` marker go into the system message;
   the blocks and the question, wrapped in `<question>` tags with any such tags removed from the
   user text first, go into the user message. `prompt_version = sha256(file)[:8]`.
2. Call: `gen.temperature`, `gen.max_output_tokens` (the provider's limit; a cut answer counts as
   invalid output until open question 3 is answered).
3. Forms: exactly the abstain text (outer whitespace, quotes, emphasis, and a final period are
   tolerated, the exact text is returned) -> `gen.model_abstained`; at least one `[Sn]` marker
   -> answer; anything else -> invalid.
4. Failures: a transport error is retried once, then `GenerationUnavailable` is raised (status
   `error` at the API, never the abstain text). Invalid output gets one repair call with the
   strict prompt and the invalid output, within the request's `gen.max_repair_calls` budget
   shared with Stage 17; a second invalid output abstains with `gen.invalid_output`.
5. Trace: the stage span `llm.generate` (kind LLM) carries the `gen_ai.*` attributes, the
   prompt version, the cost from the price table, and the repair count. The LangChain
   instrumentor adds one span per model call when `trace.instrument_langchain` is on.

Streaming stays with Stage 9 for now (open question 4): the API streams the finished answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from flp_rag.contracts import ContextBlock
from flp_rag.graph.state import RagState
from flp_rag.models import (
    ChatModel,
    Completion,
    InvalidOutput,
    ModelConfigError,
    completion_cost_usd,
)
from flp_rag.settings import ABSTAIN_TEXT, Settings
from flp_rag.tracing import current_span

CONTEXT_MARKER = "Context blocks:"
MARKER = re.compile(r"\[S\d+\]")
QUESTION_TAG = re.compile(r"</?\s*question\s*>", re.IGNORECASE)
ABSTAIN_WRAP = re.compile(r"^[\s\"'`*_]+|[\s\"'`*_.]+$")
CUT_FINISH_REASONS = ("length", "max_tokens")
# Provider exception class names that mean "the call did not go through": retried once.
TRANSPORT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "OverloadedError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ResponseError",
    }
)
Form = Literal["abstain", "answer", "invalid"]


class GenerationUnavailable(RuntimeError):
    """The model could not be reached after the retry. Not an abstain: the book was not asked."""

    def __init__(self, message: str, completions: Sequence[Completion] = ()) -> None:
        super().__init__(message)
        self.completions = list(completions)  # calls that did go through before the failure


def is_transport_error(exc: BaseException) -> bool:
    """Timeouts, connection errors, rate limits, and server errors. Auth errors, bad requests,
    and programming errors are not retried."""
    if isinstance(exc, (InvalidOutput, ModelConfigError)):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return any(cls.__name__ in TRANSPORT_ERROR_NAMES for cls in type(exc).__mro__)


# --------------------------------------------------------------------------- prompts


@dataclass(frozen=True)
class Prompt:
    path: Path
    version: str
    system: str
    user_template: str  # holds {context} and {question}


@dataclass(frozen=True)
class Prompts:
    normal: Prompt
    strict: Prompt
    repair_note: str


def prompt_version(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def load_prompt(path: Path) -> Prompt:
    text = path.read_text()
    if CONTEXT_MARKER not in text:
        raise ValueError(f"{path} has no {CONTEXT_MARKER!r} line to split system and user parts")
    system, _, rest = text.partition(CONTEXT_MARKER)
    return Prompt(path, prompt_version(path), system.strip(), f"{CONTEXT_MARKER}{rest}".strip())


def load_prompts(settings: Settings) -> Prompts:
    root = settings.config_path.parent
    return Prompts(
        normal=load_prompt(root / settings.gen.prompt_file),
        strict=load_prompt(root / settings.gen.strict_prompt_file),
        repair_note=(root / settings.gen.repair_prompt_file).read_text().strip(),
    )


def escape_question(question: str) -> str:
    """Remove `<question>` and `</question>` from the user text so it cannot close the tag."""
    return " ".join(QUESTION_TAG.sub("", question).split())


def build_messages(
    prompt: Prompt, context: Sequence[ContextBlock], question: str
) -> tuple[str, str]:
    blocks = "\n\n".join(b.text for b in context)
    user = prompt.user_template.replace("{context}", blocks).replace(
        "{question}", escape_question(question)
    )
    return prompt.system, user


# --------------------------------------------------------------------------- forms


def classify(text: str, abstain_text: str = ABSTAIN_TEXT) -> Form:
    if ABSTAIN_WRAP.sub("", text) == abstain_text:
        return "abstain"
    if MARKER.search(text):
        return "answer"
    return "invalid"


# --------------------------------------------------------------------------- generation


@dataclass(frozen=True)
class GenResult:
    answer: str
    status: Literal["answered", "no_information"]
    abstain_reason: str | None
    prompt_version: str
    repair_calls: int
    attempts: int
    completions: list[Completion]
    cost_usd: float

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.completions)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.completions)


def _call_with_retry(
    chat: ChatModel,
    system: str,
    messages: list[tuple[str, str]],
    cfg: Any,
    earlier: Sequence[Completion] = (),
) -> tuple[Completion, int]:
    """One retry on a transport error, after `gen.retry_backoff_s`. Anything else is raised as
    it is: a bad key or a bug is not an outage."""
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            completion = chat.complete(
                system, messages, None, cfg.temperature, max_tokens=cfg.max_output_tokens
            )
            return completion, attempt
        except Exception as exc:
            if not is_transport_error(exc):
                raise
            last = exc
            if attempt == 1 and cfg.retry_backoff_s:
                time.sleep(cfg.retry_backoff_s)
    raise GenerationUnavailable(
        f"model {chat.model_name} failed after 2 attempts: {last}", earlier
    ) from last


def _form(completion: Completion) -> Form:
    if completion.finish_reason in CUT_FINISH_REASONS:
        return "invalid"
    return classify(completion.text)


def generate(
    question: str,
    context: Sequence[ContextBlock],
    *,
    borderline: bool,
    chat: ChatModel,
    settings: Settings,
    prompts: Prompts,
    repair_calls_used: int = 0,
) -> GenResult:
    cfg = settings.gen
    prompt = prompts.strict if borderline else prompts.normal
    system, user = build_messages(prompt, context, question)
    completion, attempts = _call_with_retry(chat, system, [("user", user)], cfg)
    completions = [completion]
    repairs = 0
    form = _form(completion)
    if form == "invalid" and repair_calls_used + repairs < cfg.max_repair_calls:
        strict_system, strict_user = build_messages(prompts.strict, context, question)
        repair_messages: list[tuple[str, str]] = [("user", strict_user)]
        if completion.text.strip():  # an empty assistant turn is rejected by some providers
            repair_messages.append(("assistant", completion.text))
        repair_messages.append(("user", prompts.repair_note))
        completion, more = _call_with_retry(chat, strict_system, repair_messages, cfg, completions)
        completions.append(completion)
        attempts += more
        repairs = 1
        prompt = prompts.strict  # the answer now comes from the strict prompt
        form = _form(completion)
    cost = sum(completion_cost_usd(c, chat.price_per_million_tokens) for c in completions)
    if form == "answer":
        return GenResult(
            completion.text.strip(),
            "answered",
            None,
            prompt.version,
            repairs,
            attempts,
            completions,
            cost,
        )
    reason = "gen.model_abstained" if form == "abstain" else "gen.invalid_output"
    return GenResult(
        ABSTAIN_TEXT, "no_information", reason, prompt.version, repairs, attempts, completions, cost
    )


# --------------------------------------------------------------------------- node


def make_generate(
    settings: Settings, chat_factory: Callable[[], ChatModel]
) -> Callable[[RagState], dict[str, Any]]:
    """The graph node. The chat model is built on first use, so the API starts without a key
    and a placeholder `gen.model` fails on the first request that reaches this stage."""
    prompts = load_prompts(settings)
    chat: ChatModel | None = None
    lock = threading.Lock()

    def model() -> ChatModel:
        nonlocal chat
        with lock:  # LangGraph runs sync nodes in a worker pool: build the client once
            if chat is None:
                chat = chat_factory()
            return chat

    def record(span: Any, llm: ChatModel, completions: Sequence[Completion], cost: float) -> None:
        for key, value in (
            ("gen_ai.system", llm.model_name.partition(":")[0]),
            ("gen_ai.request.model", llm.model_name),
            ("gen_ai.request.temperature", settings.gen.temperature),
            ("gen_ai.request.max_tokens", settings.gen.max_output_tokens),
            ("gen_ai.usage.input_tokens", sum(c.input_tokens for c in completions)),
            ("gen_ai.usage.output_tokens", sum(c.output_tokens for c in completions)),
            ("gen_ai.response.finish_reasons", json.dumps([c.finish_reason for c in completions])),
            ("rag.cost_usd", cost),
        ):
            span.set_attribute(key, value)

    def generate_node(state: RagState) -> dict[str, Any]:
        llm = model()
        standalone = state.get("standalone")
        question = standalone.standalone if standalone else state["question"]
        span = current_span()
        try:
            result = generate(
                question,
                state.get("context", []),
                borderline=bool(state.get("borderline")),
                chat=llm,
                settings=settings,
                prompts=prompts,
                repair_calls_used=int(state.get("repair_calls", 0)),
            )
        except GenerationUnavailable as exc:
            # The calls that went through before the outage still count and cost.
            partial = sum(
                completion_cost_usd(c, llm.price_per_million_tokens) for c in exc.completions
            )
            record(span, llm, exc.completions, partial)
            raise
        record(span, llm, result.completions, result.cost_usd)
        span.set_attribute("rag.prompt_version", result.prompt_version)
        span.set_attribute("rag.repair_calls", result.repair_calls)
        span.set_attribute("rag.attempts", result.attempts)
        out: dict[str, Any] = {
            "answer": result.answer,
            "status": result.status,
            "prompt_version": result.prompt_version,
            "repair_calls": int(state.get("repair_calls", 0)) + result.repair_calls,
            "cost_usd": float(state.get("cost_usd", 0.0)) + result.cost_usd,
            "abstain_reason": result.abstain_reason,
        }
        if result.abstain_reason:
            span.set_attribute("rag.abstain_reason", result.abstain_reason)
            out["sources"] = []
        return out

    return generate_node
