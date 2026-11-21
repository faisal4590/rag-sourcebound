# Stage 5, Milestone 7 rule 5: one small-model summary per chunk, prepended to embedding_text. #37.
"""Contextual summaries (spec Stage 5, rules 5 and 7; issue #37).

One small-model call per chunk writes one or two sentences on what the chunk covers inside its
chapter. The summary goes in front of `embedding_text`, so the dense vector carries the context
the chunk text alone lacks; `display_text` never changes. The prompt is `enrich.summary_prompt_file`
(`prompts/enrich_v1.md`, versioned like the answer prompt: `prompt_version = sha256(file)[:8]`),
split at the `Chunk:` line into the system rules and the user template.

Rules:
- Cache by `sha256(prompt_version + chunk_id + display_text)` in `enrich.summary_cache_path`;
  a rerun with the same prompt and chunks makes no call.
- A transport error (timeout, connection, rate limit, server error) or an invalid output (empty,
  or over `enrich.summary_max_words`) is retried up to `enrich.summary_retry_attempts` times; the
  wait doubles from `enrich.summary_retry_backoff_s`, and a `Retry-After` header wins when it is
  longer; a header over `enrich.summary_retry_after_max_s` (a daily quota) and an output cut by
  `enrich.summary_max_output_tokens` are not retried. A chunk that still has no summary is kept
  without one and counted as a failure; `ModelConfigError` and auth errors propagate.
- `enrich.summary_pause_s` sleeps between model calls (not cache hits): the Groq free tier
  allows 30 requests per minute.
- Every call counts toward `rag.llm_calls`, `gen_ai.usage.*`, and `rag.cost_usd` of `ingest.enrich`.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import structlog

from flp_rag.contracts import Chunk
from flp_rag.graph.nodes.generate import is_transport_error
from flp_rag.models import ChatModel, InvalidOutput, completion_cost_usd
from flp_rag.settings import EnrichConfig
from flp_rag.stores.summary_cache import SummaryCache

log = structlog.get_logger()

CHUNK_MARKER = "Chunk:"
SUMMARY_FIELD, PROMPT_FIELD = "context_summary", "enrich_prompt_version"
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)
_LABEL = re.compile(r"^summary\s*:\s*", re.IGNORECASE)
CUT_FINISH_REASONS = ("length", "max_tokens")
_QUOTES = "\"'`“”‘’ \t\r\n"


# --------------------------------------------------------------------------- prompt


@dataclass(frozen=True)
class SummaryPrompt:
    path: Path
    version: str
    system: str
    user_template: str  # from the `Chunk:` line, with {chapter_no} ... {text} placeholders


def load_summary_prompt(path: Path) -> SummaryPrompt:
    text = path.read_text()
    if CHUNK_MARKER not in text:
        raise ValueError(f"{path} has no {CHUNK_MARKER!r} line to split system and user parts")
    system, _, rest = text.partition(CHUNK_MARKER)
    version = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return SummaryPrompt(path, version, system.strip(), f"{CHUNK_MARKER}{rest}".strip())


def user_message(prompt: SummaryPrompt, chunk: Chunk) -> str:
    p = chunk.payload
    return prompt.user_template.format(
        chapter_no=p.get("chapter_no", "?"),
        chapter_title=p.get("chapter_title", ""),
        section_title=p.get("section_title", ""),
        page_start=p.get("page_printed_start", "?"),
        page_end=p.get("page_printed_end", "?"),
        text=chunk.display_text,
    )


def summary_key(prompt_version: str, chunk_id: str, display_text: str) -> str:
    return hashlib.sha256((prompt_version + chunk_id + display_text).encode("utf-8")).hexdigest()


def check_summary(text: str, max_words: int) -> str:
    """The summary as one line of plain text, or `InvalidOutput`."""
    cleaned = _FENCE.sub("", text).strip(_QUOTES)
    cleaned = _LABEL.sub("", cleaned).strip(_QUOTES)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise InvalidOutput("empty summary")
    words = len(cleaned.split())
    if words > max_words:
        raise InvalidOutput(f"summary has {words} words, over {max_words}")
    return cleaned


def retry_after_s(exc: BaseException) -> float | None:
    """The `Retry-After` seconds of a rate-limit response, when the client exposes them."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def with_summary(chunk: Chunk, summary: str | None, prompt_version: str) -> Chunk:
    """The chunk with the summary in front of `embedding_text` and recorded in the payload.
    `summary` None records the failure (the field is present, null) and leaves the text."""
    payload = {**chunk.payload, SUMMARY_FIELD: summary, PROMPT_FIELD: prompt_version}
    if summary is None:
        return replace(chunk, payload=payload)
    return replace(chunk, embedding_text=f"{summary}\n{chunk.embedding_text}", payload=payload)


# --------------------------------------------------------------------------- the summarizer


@dataclass(frozen=True)
class Tally:
    """The model usage of one or more summaries; `+` adds two."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    failures: int = 0
    cache_hits: int = 0

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            self.calls + other.calls,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            round(self.cost_usd + other.cost_usd, 10),
            self.failures + other.failures,
            self.cache_hits + other.cache_hits,
        )


class Summarizer:
    """Writes the summary of one chunk through the cache, the retries, and the pause."""

    def __init__(
        self,
        chat: ChatModel,
        prompt: SummaryPrompt,
        cfg: EnrichConfig,
        cache: SummaryCache,
        *,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self.chat, self.prompt, self.cfg, self.cache, self.sleep = chat, prompt, cfg, cache, sleep
        self.called = False  # a pause is due before a model call that follows another

    def apply(self, chunk: Chunk) -> tuple[Chunk, Tally]:
        summary, tally = self.summarize(chunk)
        return with_summary(chunk, summary, self.prompt.version), tally

    def summarize(self, chunk: Chunk) -> tuple[str | None, Tally]:
        key = summary_key(self.prompt.version, chunk.chunk_id, chunk.display_text)
        cached = self.cache.get(key, model=self.chat.model_name)
        if cached is not None:
            return cached, Tally(cache_hits=1)
        summary, tally = self._call_with_retries(chunk)
        if summary is not None:
            self.cache.put(
                key,
                chunk_id=chunk.chunk_id,
                prompt_version=self.prompt.version,
                model=self.chat.model_name,
                summary=summary,
                input_tokens=tally.input_tokens,
                output_tokens=tally.output_tokens,
            )
        return summary, tally

    def _call_with_retries(self, chunk: Chunk) -> tuple[str | None, Tally]:
        cfg = self.cfg
        tally = Tally()
        wait = cfg.summary_retry_backoff_s
        user = user_message(self.prompt, chunk)
        for attempt in range(1, cfg.summary_retry_attempts + 1):
            if self.called and cfg.summary_pause_s > 0:
                self.sleep(cfg.summary_pause_s)
            self.called = True
            try:
                completion = self.chat.complete(
                    self.prompt.system,
                    [("user", user)],
                    None,
                    cfg.summary_temperature,
                    max_tokens=cfg.summary_max_output_tokens,
                )
            except Exception as exc:
                if not is_transport_error(exc):
                    raise
                tally += Tally(calls=1)
                error: BaseException = exc
            else:
                tally += Tally(
                    calls=1,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    cost_usd=completion_cost_usd(completion, self.chat.price_per_million_tokens),
                )
                try:
                    return check_summary(completion.text, cfg.summary_max_words), tally
                except InvalidOutput as exc:
                    error = exc
                if completion.finish_reason in CUT_FINISH_REASONS:
                    # The model spent the budget on its reasoning: the same call gives the
                    # same cut at temperature 0, so the retries would only cost calls.
                    log.warning("summary cut by max_output_tokens", chunk_id=chunk.chunk_id)
                    break
            header = retry_after_s(error)
            if header is not None and header > cfg.summary_retry_after_max_s:
                # A daily quota, not a burst: waiting would hang the run for hours.
                log.warning("summary rate limit beyond the cap", retry_after_s=header)
                break
            log.warning(
                "summary retry",
                chunk_id=chunk.chunk_id,
                attempt=attempt,
                error=type(error).__name__,
            )
            if attempt < cfg.summary_retry_attempts:
                self.sleep(max(wait, header) if header is not None else wait)
                wait *= 2
        return None, tally + Tally(failures=1)
