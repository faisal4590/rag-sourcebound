# The correctness judge of spec 8.3 (issue #34): a second model call that scores an answer 0/1/2 against the expected answer.
"""`prompts/judge_correctness_v1.md`, split at `Question:` into the system rules and the user
template. One retry after `eval.judge_pause_s`; `ModelConfigError` propagates."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flp_rag.contracts import EvalCase
from flp_rag.models import ChatModel, ModelConfigError, get_chat_model
from flp_rag.settings import Settings

QUESTION_MARKER = "Question:"
SCORES = (0, 1, 2)
CORRECTNESS_SCHEMA = {
    "type": "object",
    "required": ["score", "reason"],
    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
}
Judge = Callable[[EvalCase, str], tuple[int, str]]


@dataclass(frozen=True)
class CorrectnessPrompt:
    system: str
    user_template: str
    version: str  # sha256(file)[:8], recorded in the run file


def load_correctness_prompt(path: Path) -> CorrectnessPrompt:
    raw = path.read_bytes()
    text = raw.decode()
    if QUESTION_MARKER not in text:
        raise ValueError(f"{path} has no {QUESTION_MARKER!r} marker")
    system, _, user = text.partition(QUESTION_MARKER)
    version = hashlib.sha256(raw).hexdigest()[:8]
    return CorrectnessPrompt(system.strip(), (QUESTION_MARKER + user).strip(), version)


def build_correctness_messages(
    prompt: CorrectnessPrompt, question: str, expected: str, answer: str
) -> tuple[str, str]:
    user = (
        prompt.user_template.replace("{question}", question)
        .replace("{expected}", expected)
        .replace("{answer}", answer)
    )
    return prompt.system, user


def parse_correctness(data: Mapping[str, Any] | None) -> tuple[int, str]:
    if data is None or data.get("score") not in SCORES:
        raise ValueError(f"correctness verdict has the wrong shape: {data!r}")
    return int(data["score"]), str(data.get("reason") or "")


def make_judge(settings: Settings, chat: ChatModel | None = None) -> Judge:
    """One call per answered answerable case, `eval.judge_pause_s` apart (the free tier's
    tokens per minute), retried once; a verdict of the wrong shape counts as a failed call."""
    cfg = settings.eval
    prompt = load_correctness_prompt(settings.config_path.parent / cfg.correctness_prompt_file)
    model = chat or get_chat_model(settings, "judge")

    def once(case: EvalCase, answer: str) -> tuple[int, str]:
        system, user = build_correctness_messages(
            prompt, case.question, case.expected_answer or "", answer
        )
        completion = model.complete(
            system,
            [("user", user)],
            json_schema=CORRECTNESS_SCHEMA,
            temperature=cfg.correctness_temperature,
            max_tokens=cfg.correctness_max_output_tokens,
        )
        return parse_correctness(completion.json)

    def judge(case: EvalCase, answer: str) -> tuple[int, str]:
        try:
            return once(case, answer)
        except ModelConfigError:
            raise
        except Exception:  # noqa: BLE001 - one retry for a transport error or a bad verdict
            time.sleep(cfg.judge_pause_s)
            return once(case, answer)

    return judge
