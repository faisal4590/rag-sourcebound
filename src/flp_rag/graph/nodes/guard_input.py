# Stage 10 - Input guardrails. Spec Section 6, Stage 10.
"""Stage 10: reject bad input, mark suspicious input, record the language.

Rules (spec Section 6, Stage 10):
1. Fewer than 1 or more than `guard.max_chars` characters (surrounding whitespace ignored):
   raise `InputRejected` with the message `question must be 1-2000 characters`. Stage 9
   answers HTTP 400 with that text.
2. Scan for the injection patterns in `guard.injection_patterns` and for a base64 run of at
   least `guard.base64_min_run` characters. A match sets `injection_suspected`. Never block.
3. Detect the language with py3langid (pure Python, no download, under a millisecond). A
   question shorter than `guard.language_min_chars`, or one the detector is less than
   `guard.language_min_confidence` sure about (code-only questions), is assumed English.
4. Rate limiting is Milestone 5 (`guard.rate_limit_per_min` is read there).

This stage never answers and never abstains. The span is `guardrails.input`, opened by
`build_graph`; the node sets `guard.chars`, `guard.injection_suspected`, `guard.language`
on every call, the rejected ones included, then raises.
"""

import functools
import re
import threading
from collections.abc import Callable
from typing import Any

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from flp_rag.graph.state import RagState
from flp_rag.settings import GuardConfig, Settings
from flp_rag.tracing import current_span

REJECT_MESSAGE = "question must be 1-{max_chars} characters"
ENGLISH = "en"
Compiled = tuple[tuple[str, re.Pattern[str]], ...]

_identifier_lock = threading.Lock()


class InputRejected(ValueError):
    """The question failed rule 1. The message is the exact text the client sees."""


def reject_message(guard: GuardConfig) -> str:
    return REJECT_MESSAGE.format(max_chars=guard.max_chars)


# --------------------------------------------------------------------------- rule 1


def check_length(question: str, guard: GuardConfig) -> None:
    if not 1 <= len(question.strip()) <= guard.max_chars:
        raise InputRejected(reject_message(guard))


# --------------------------------------------------------------------------- rule 2


@functools.lru_cache(maxsize=4)
def compile_patterns(guard: GuardConfig) -> Compiled:
    """The configured regexes plus the base64-run rule, compiled once per config."""
    patterns = [(p, re.compile(p, re.IGNORECASE)) for p in guard.injection_patterns]
    base64_run = rf"[A-Za-z0-9+/]{{{guard.base64_min_run},}}={{0,2}}"
    patterns.append((f"base64 run >= {guard.base64_min_run}", re.compile(base64_run)))
    return tuple(patterns)


def _scan(question: str, compiled: Compiled) -> tuple[str, ...]:
    return tuple(name for name, rx in compiled if rx.search(question))


def scan_injection(question: str, guard: GuardConfig) -> tuple[str, ...]:
    """The names of the patterns that matched. Empty means nothing suspicious."""
    return _scan(question, compile_patterns(guard))


# --------------------------------------------------------------------------- rule 3


@functools.cache
def _identifier() -> LanguageIdentifier:
    """One model per process, with normalised probabilities so a confidence floor applies."""
    with _identifier_lock:
        return LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


def detect_language(question: str, guard: GuardConfig) -> str:
    """ISO 639-1 code. Short or low-confidence text is assumed English (rule 3)."""
    text = question.strip()
    if len(text) < guard.language_min_chars:
        return ENGLISH
    code, confidence = _identifier().classify(text)
    if float(confidence) < guard.language_min_confidence:
        return ENGLISH
    return str(code)


# --------------------------------------------------------------------------- node


def make_guard_input(settings: Settings) -> Callable[[RagState], dict[str, Any]]:
    guard = settings.guard
    compiled = compile_patterns(guard)
    _identifier()  # load the model at build time, not on the first request

    def guard_input(state: RagState) -> dict[str, Any]:
        question = state["question"]
        hits = _scan(question, compiled)
        language = detect_language(question, guard)
        span = current_span()
        span.set_attribute("guard.chars", len(question))
        span.set_attribute("guard.injection_suspected", bool(hits))
        span.set_attribute("guard.injection_patterns", list(hits))
        span.set_attribute("guard.language", language)
        check_length(question, guard)
        return {"injection_suspected": bool(hits), "language": language}

    return guard_input
