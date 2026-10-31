# Stage 11 - Query understanding. One small-model call: rewrite, classify, expand; smalltalk abstain; fallback. Spec Section 6, Stage 11.
"""Stage 11: the raw question becomes one standalone question, a classification, and the
search variants of Stage 12.

Plain functions of the question, the session history, the small chat model, the facts of the
index, and `query.*` in `config.yaml` (spec Section 6, Stage 11; issue #27):

1. History: the last `query.history_turns` turns of the session from the session store
   (Stage 18 writes them). Without a session id there is no history.
2. One call: `query.prompt_file` (Appendix C.4) does the rewrite, the classification, and the
   expansion in one JSON answer (`QUERY_SCHEMA`). The chapter list in the prompt comes from
   the parent store of the live index, never from a typed list. The history and the question
   go in tags, with any such tags removed from the user text first, as in Stage 16.
3. Checks (Part 3): without history the standalone question is the raw question, exactly;
   with history the model's rewrite is used when it has at most `query.max_standalone_words`
   words. An unknown intent is `book_question`; a chapter hint outside the index is none;
   paraphrases that are empty, equal to the standalone question, duplicates (letter case
   ignored), or that lose a `php_symbols` term named in the question are dropped, and at
   most `query.expansions` stay; empty keywords mean no keyword variant.
4. Smalltalk (Part 4): intent `smalltalk` under `query.smalltalk_policy: abstain` abstains with
   `query.smalltalk` only when the question has at most `query.smalltalk_max_words` words.
   Every other case continues; `other` never abstains on its own.
5. Fallback (Part 5): a transport error or an output that is not the JSON shape is retried
   once; a second failure gives the raw question as the only variant and
   `rag.understand_fallback = true`. The request continues: this stage improves retrieval,
   the pipeline works without it. A configuration or authentication error propagates.
6. Trace: `query.understand` is the parent (`rag.question_raw`, `rag.question_standalone`
   under `trace.capture_content`, `rag.intent`, `rag.wants_code`, `rag.chapter_hint`,
   `rag.variants` under `trace.capture_content`, `rag.history_turns_used`,
   `rag.understand_fallback`, `rag.paraphrases_dropped`); the model call is the child
   `query.classify` (kind LLM, with `gen_ai.*`, `rag.prompt_version`, `rag.rewrite_used`,
   `rag.attempts`, `rag.error` as a short code).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from flp_rag.contracts import StandaloneQuery
from flp_rag.graph.nodes.generate import is_transport_error, prompt_version
from flp_rag.graph.state import RagState
from flp_rag.models import ChatModel, Completion, InvalidOutput, completion_cost_usd
from flp_rag.settings import ABSTAIN_TEXT, QueryConfig, Settings
from flp_rag.stores.parent_store import ParentStore
from flp_rag.stores.session_store import SessionStore, history_from
from flp_rag.tracing import current_span, stage

log = structlog.get_logger()

INTENTS = ("book_question", "meta", "smalltalk", "other")
DEFAULT_INTENT = "book_question"
SMALLTALK_REASON = "query.smalltalk"
CHAPTERS_MARKER = "Chapters:"
NO_HISTORY = "none"
TAG = re.compile(r"</?\s*(question|history)\s*>", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone": {"type": "string"},
        "intent": {"type": "string", "enum": list(INTENTS)},
        "wants_code": {"type": "boolean"},
        "chapter_hint": {"type": ["integer", "null"]},
        "paraphrases": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "string"},
    },
    "required": ["standalone", "intent", "wants_code", "chapter_hint", "paraphrases", "keywords"],
}

History = Sequence[tuple[str, str]]


# --------------------------------------------------------------------------- prompt and facts


@dataclass(frozen=True)
class QueryPrompt:
    path: Path
    version: str
    system: str  # the rules above the `Chapters:` marker
    user_template: str  # holds {chapter_list}, {history}, {question}


def load_query_prompt(path: Path) -> QueryPrompt:
    text = path.read_text()
    if CHAPTERS_MARKER not in text:
        raise ValueError(f"{path} has no {CHAPTERS_MARKER!r} line to split system and user parts")
    system, _, rest = text.partition(CHAPTERS_MARKER)
    return QueryPrompt(
        path, prompt_version(path), system.strip(), f"{CHAPTERS_MARKER}{rest}".strip()
    )


@dataclass(frozen=True)
class IndexFacts:
    """What Stage 11 knows about the indexed book: its chapter list for the prompt and the
    hint check, and its `php_symbols` for the paraphrase check (open question 5)."""

    chapters: tuple[tuple[int, str], ...]
    symbols: frozenset[str]

    @classmethod
    def from_store(cls, path: Path | None) -> IndexFacts:
        if path is None:
            raise RuntimeError(
                "query understanding needs the parent store; Deps.store_path is not set"
            )
        with ParentStore.open_readonly(path) as store:
            return cls(tuple(store.chapters()), frozenset(store.php_symbols()))

    @property
    def chapter_numbers(self) -> frozenset[int]:
        return frozenset(no for no, _ in self.chapters)

    def chapter_list(self) -> str:
        return "\n".join(f"{no}: {title}" for no, title in self.chapters)


# --------------------------------------------------------------------------- messages


def strip_tags(text: str) -> str:
    """Remove `<question>`, `<history>` and their closing forms from user text, so the text
    cannot close the tag the prompt opens. Whitespace runs collapse to one space."""
    return " ".join(TAG.sub("", text).split())


def history_text(history: History) -> str:
    if not history:
        return NO_HISTORY
    lines = "\n".join(f"{role}: {strip_tags(text)}" for role, text in history)
    return f"<history>\n{lines}\n</history>"


def build_query_messages(
    prompt: QueryPrompt, facts: IndexFacts, history: History, question: str
) -> tuple[str, str]:
    user = (
        prompt.user_template.replace("{chapter_list}", facts.chapter_list())
        .replace("{history}", history_text(history))
        .replace("{question}", f"<question>{strip_tags(question)}</question>")
    )
    return prompt.system, user


# --------------------------------------------------------------------------- checks (Part 3)


def word_count(text: str) -> int:
    return len(text.split())


def question_symbols(question: str, symbols: frozenset[str]) -> frozenset[str]:
    """The `php_symbols` of the index that the question names, as lower-case words."""
    return frozenset(w.lower() for w in WORD.findall(question)) & symbols


def clean_paraphrases(
    raw: Any, *, standalone: str, limit: int, required: frozenset[str]
) -> tuple[list[str], int]:
    """Keep text entries that are not empty, not the standalone question, not duplicates, and
    that still name every required symbol; at most `limit`. Returns (kept, dropped)."""
    if not isinstance(raw, list):
        return [], 0
    kept: list[str] = []
    seen = {" ".join(standalone.split()).casefold()}
    for entry in raw:
        text = " ".join(entry.split()) if isinstance(entry, str) else ""
        key = text.casefold()
        if not text or key in seen or len(kept) >= limit:
            continue
        if required and not required <= question_symbols(text, required):
            continue
        seen.add(key)
        kept.append(text)
    return kept, len(raw) - len(kept)


def clean_output(
    data: Mapping[str, Any],
    *,
    raw_question: str,
    history_used: bool,
    cfg: QueryConfig,
    facts: IndexFacts,
) -> tuple[StandaloneQuery, int]:
    """Apply the checks of Part 3 to the model's JSON. Returns the query and the number of
    paraphrases dropped."""
    standalone = raw_question
    if history_used and isinstance(data.get("standalone"), str):
        rewrite = " ".join(data["standalone"].split())
        if rewrite and word_count(rewrite) <= cfg.max_standalone_words:
            standalone = rewrite
    intent = data.get("intent")
    if intent not in INTENTS:
        intent = DEFAULT_INTENT
    wants_code = data.get("wants_code") is True
    hint = data.get("chapter_hint")
    if isinstance(hint, bool) or not isinstance(hint, int) or hint not in facts.chapter_numbers:
        hint = None
    required = (
        question_symbols(standalone, facts.symbols) if cfg.keep_question_symbols else frozenset()
    )
    paraphrases, dropped = clean_paraphrases(
        data.get("paraphrases"), standalone=standalone, limit=cfg.expansions, required=required
    )
    keywords = data.get("keywords")
    keyword = " ".join(keywords.split()) if isinstance(keywords, str) else ""
    query = StandaloneQuery(
        raw=raw_question,
        standalone=standalone,
        intent=str(intent),
        wants_code=wants_code,
        chapter_hint=hint,
        variants=[standalone, *paraphrases],
        keyword_variant=keyword or None,
    )
    return query, dropped


def fallback_query(raw_question: str) -> StandaloneQuery:
    """Part 5: the raw question as the only variant, the default classification."""
    return StandaloneQuery(
        raw=raw_question,
        standalone=raw_question,
        intent=DEFAULT_INTENT,
        wants_code=False,
        chapter_hint=None,
        variants=[raw_question],
        keyword_variant=None,
    )


def should_abstain(query: StandaloneQuery, cfg: QueryConfig) -> bool:
    """Part 4: smalltalk under the abstain policy, and short enough to be a clear case."""
    return (
        query.intent == "smalltalk"
        and cfg.smalltalk_policy == "abstain"
        and word_count(query.raw) <= cfg.smalltalk_max_words
    )


# --------------------------------------------------------------------------- the model call


def parse_output(completion: Completion) -> dict[str, Any]:
    """The JSON object of the completion, or `InvalidOutput`."""
    data = completion.json
    if data is None:
        try:
            data = json.loads(completion.text)
        except json.JSONDecodeError as exc:
            raise InvalidOutput(
                f"query model output is not JSON: {completion.text[:200]!r}"
            ) from exc
    if not isinstance(data, dict):
        raise InvalidOutput(f"query model output is not a JSON object: {completion.text[:200]!r}")
    missing = [key for key in QUERY_SCHEMA["required"] if key not in data]
    if missing:
        raise InvalidOutput(f"query model output lacks {missing}: {completion.text[:200]!r}")
    return data


@dataclass(frozen=True)
class ClassifyOutcome:
    data: dict[str, Any] | None  # None when both attempts failed
    attempts: int
    completions: list[Completion]  # the calls that returned, for the token totals
    error: str | None


@stage("query.classify", kind="LLM")
def classify_query(
    system: str,
    user: str,
    *,
    chat: ChatModel,
    cfg: QueryConfig,
    version: str,
    history_used: bool,
) -> ClassifyOutcome:
    """Up to two attempts: a transport error or an output that is not the JSON shape is tried
    once more after `query.retry_backoff_s`. Anything else (a bad key, a placeholder model, a
    bug) is raised as it is. `error` is a short code (`invalid_output`, `transport:<class>`),
    never model text: the model's output can hold the user's question."""
    span = current_span()
    completions: list[Completion] = []
    error: str | None = None
    data: dict[str, Any] | None = None
    attempts = 0
    for attempt in (1, 2):
        attempts = attempt
        try:
            completion = chat.complete(
                system,
                [("user", user)],
                QUERY_SCHEMA,
                cfg.temperature,
                max_tokens=cfg.max_output_tokens,
            )
        except InvalidOutput as exc:
            # The provider parsed the JSON and rejected it; the call did go through, so its
            # tokens still count.
            if exc.completion is not None:
                completions.append(exc.completion)
            error = "invalid_output"
        except Exception as exc:
            if not is_transport_error(exc):
                raise
            error = f"transport:{type(exc).__name__}"
        else:
            completions.append(completion)
            try:
                data = parse_output(completion)
            except InvalidOutput:
                error = "invalid_output"
            else:
                error = None
                break
        if attempt == 1 and cfg.retry_backoff_s:
            time.sleep(cfg.retry_backoff_s)
    cost = sum(completion_cost_usd(c, chat.price_per_million_tokens) for c in completions)
    for key, value in (
        ("gen_ai.system", chat.model_name.partition(":")[0]),
        ("gen_ai.request.model", chat.model_name),
        ("gen_ai.request.temperature", cfg.temperature),
        ("gen_ai.request.max_tokens", cfg.max_output_tokens),
        ("gen_ai.usage.input_tokens", sum(c.input_tokens for c in completions)),
        ("gen_ai.usage.output_tokens", sum(c.output_tokens for c in completions)),
        ("gen_ai.response.finish_reasons", json.dumps([c.finish_reason for c in completions])),
        ("rag.cost_usd", cost),
        ("rag.prompt_version", version),
        ("rag.rewrite_used", history_used),
        ("rag.attempts", attempts),
        ("rag.error", error or ""),
    ):
        span.set_attribute(key, value)
    return ClassifyOutcome(data, attempts, completions, error)


# --------------------------------------------------------------------------- the stage


@dataclass(frozen=True)
class Understanding:
    query: StandaloneQuery
    history_turns_used: int
    fallback: bool
    paraphrases_dropped: int
    attempts: int
    completions: list[Completion]
    cost_usd: float
    error: str | None = None

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.completions)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.completions)


def understand(
    question: str,
    history: History,
    *,
    chat: ChatModel,
    prompt: QueryPrompt,
    facts: IndexFacts,
    settings: Settings,
) -> Understanding:
    """One request: the model call with the history, then the checks, or the fallback."""
    cfg = settings.query
    history_used = bool(history)
    system, user = build_query_messages(prompt, facts, history, question)
    outcome = classify_query(
        system,
        user,
        chat=chat,
        cfg=cfg,
        version=prompt.version,
        history_used=history_used,
    )
    cost = sum(completion_cost_usd(c, chat.price_per_million_tokens) for c in outcome.completions)
    turns_used = sum(1 for role, _ in history if role == "user")
    if outcome.data is None:
        log.warning("query understanding fell back to the raw question", error=outcome.error)
        return Understanding(
            fallback_query(question),
            turns_used,
            True,
            0,
            outcome.attempts,
            outcome.completions,
            cost,
            outcome.error,
        )
    query, dropped = clean_output(
        outcome.data, raw_question=question, history_used=history_used, cfg=cfg, facts=facts
    )
    return Understanding(
        query, turns_used, False, dropped, outcome.attempts, outcome.completions, cost
    )


# --------------------------------------------------------------------------- node


def make_understand(
    settings: Settings,
    chat_factory: Callable[[], ChatModel],
    facts_factory: Callable[[], IndexFacts],
    sessions: SessionStore | None,
) -> Callable[[RagState], dict[str, Any]]:
    """The graph node. The chat model and the index facts are built on first use under a lock,
    so the API starts without a key and a placeholder `query.model` fails on the first request.
    The session store comes from `Deps`; without one there is never any history."""
    prompt = load_query_prompt(settings.config_path.parent / settings.query.prompt_file)
    cfg = settings.query
    chat: ChatModel | None = None
    facts: IndexFacts | None = None
    lock = threading.Lock()

    def model() -> ChatModel:
        nonlocal chat
        with lock:
            if chat is None:
                chat = chat_factory()
            return chat

    def index_facts() -> IndexFacts:
        nonlocal facts
        with lock:
            if facts is None:
                facts = facts_factory()
            return facts

    def load_history(state: RagState) -> list[tuple[str, str]]:
        given = list(state.get("history") or [])
        session_id = state.get("session_id")
        if given or not session_id or sessions is None:
            return given[-2 * cfg.history_turns :] if cfg.history_turns else []
        return history_from(sessions.last_turns(session_id, cfg.history_turns))

    def understand_node(state: RagState) -> dict[str, Any]:
        question = state["question"]
        span = current_span()
        span.set_attribute("rag.understand_enabled", cfg.enabled)
        if not cfg.enabled:  # the Milestone 3 passthrough: no history, no model call (issue #28)
            query = fallback_query(question)
            span.set_attribute("rag.intent", query.intent)
            span.set_attribute("rag.n_variants", 1)
            span.set_attribute("rag.understand_fallback", False)
            return {"standalone": query, "history": []}
        history = load_history(state)
        result = understand(
            question, history, chat=model(), prompt=prompt, facts=index_facts(), settings=settings
        )
        query = result.query
        variants = [*query.variants, *([query.keyword_variant] if query.keyword_variant else [])]
        if settings.trace.capture_content:  # the variants are question text too
            limit = settings.trace.attr_string_max_chars
            span.set_attribute("rag.question_raw", question[:limit])
            span.set_attribute("rag.question_standalone", query.standalone[:limit])
            span.set_attribute("rag.variants", json.dumps(variants, ensure_ascii=False)[:limit])
        for key, value in (
            ("rag.intent", query.intent),
            ("rag.wants_code", query.wants_code),
            ("rag.n_variants", len(variants)),
            ("rag.history_turns_used", result.history_turns_used),
            ("rag.understand_fallback", result.fallback),
            ("rag.paraphrases_dropped", result.paraphrases_dropped),
            ("rag.cost_usd", result.cost_usd),
        ):
            span.set_attribute(key, value)
        if query.chapter_hint is not None:
            span.set_attribute("rag.chapter_hint", query.chapter_hint)
        out: dict[str, Any] = {
            "standalone": query,
            "history": history,
            "input_tokens": int(state.get("input_tokens", 0)) + result.input_tokens,
            "output_tokens": int(state.get("output_tokens", 0)) + result.output_tokens,
            "cost_usd": float(state.get("cost_usd", 0.0)) + result.cost_usd,
        }
        if should_abstain(query, cfg):
            span.set_attribute("rag.abstain_reason", SMALLTALK_REASON)
            out.update(
                status="no_information", answer=ABSTAIN_TEXT, abstain_reason=SMALLTALK_REASON
            )
        return out

    return understand_node
