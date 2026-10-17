# Stage 17 - Output checks. Citation check, faithfulness judge, code check. Spec Section 6, Stage 17.
"""Stage 17: the last abstain point. Nothing after this stage changes the answer.

Plain functions of the answer, the label map and the context from Stage 15, and `output.*` in
`config.yaml` (spec Section 6, Stage 17; issue #22):

1. Early stop: the exact abstain text (the Stage 16 tolerance applies) keeps Stage 16's status
   and reason. No check runs.
2. Fixed checks, cheapest first, the first failure stops the rest: mixed abstain (the abstain
   text in any letter case next to other text), citations (every `[Sn]` marker must be in the
   label map; one unknown or malformed marker fails), code (every fenced block in the answer
   (backtick or tilde fences, an unclosed fence runs to the end) must be part of
   one fenced block of the context after normalization: trimmed lines, no empty lines, one
   space per run, `...` for the book's `…`; when the context has no fences, an index built
   before the blocks table, whole block texts stand in).
3. Judge: `output.judge_model` with `output.judge_prompt_file` on every borderline request, on
   every request when the harness sets `judge_all`, and otherwise on a sample of
   `output.faithfulness_sample_rate` decided by a hash of the request id. `supported: false`
   abstains with `output.unfaithful`; no repair can add evidence. A judge that fails (transport
   error after the retry, or output that is not the verdict JSON) abstains with
   `output.judge_unavailable` when the request had to be judged and otherwise delivers the
   answer with the error on the span (open question 1). A placeholder judge model is a
   configuration error and propagates on every judged request.
4. Repair or abstain: a failed fixed check goes back to Stage 16 for one repair call when the
   shared `gen.max_repair_calls` budget has one left (`repair`), else `output.invalid_after_repair`.
   The repair message is the line that names the failed check plus the rewrite note, both from
   `output.repair_prompt_file`. A repaired answer passes through every check again.
5. Trace: `guardrails.output` is the parent; `check.citations` and `check.code` (TOOL) and
   `check.faithfulness` (LLM) are children. A request that is not judged has no judge span.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flp_rag.contracts import ContextBlock
from flp_rag.graph.markers import find_malformed, find_markers
from flp_rag.graph.nodes.generate import (
    Prompt,
    call_with_retry,
    classify,
    load_prompt,
    prompt_version,
)
from flp_rag.graph.state import RagState
from flp_rag.models import ChatModel, Completion, InvalidOutput, completion_cost_usd
from flp_rag.settings import ABSTAIN_TEXT, OutputConfig, Settings
from flp_rag.tracing import current_span, stage

CheckResult = Literal["ok", "repair", "abstain"]
Failure = Literal["mixed_abstain", "citations", "code"]

FENCE_LINE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
ELLIPSIS = "…"
REPAIR_SECTIONS: tuple[str, ...] = ("mixed_abstain", "citations", "code", "rewrite")
SECTION_HEADING = re.compile(r"^## (\w+)[ \t]*$", re.MULTILINE)
PLACEHOLDER = re.compile(r"\{(context|answer)\}")
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "unsupported_claims"],
}


# --------------------------------------------------------------------------- prompts


@dataclass(frozen=True)
class RepairMessages:
    """`## <section>` blocks of `output.repair_prompt_file`: one line per failed check and the
    rewrite note that closes every repair message."""

    path: Path
    version: str
    sections: dict[str, str]

    def message(self, failure: Failure, **values: str) -> str:
        line = self.sections[failure]
        for key, value in values.items():
            line = line.replace(f"{{{key}}}", value)
        return f"{line}\n\n{self.sections['rewrite']}"


@dataclass(frozen=True)
class CheckPrompts:
    judge: Prompt
    repair: RepairMessages


def parse_repair_sections(text: str) -> dict[str, str]:
    parts = SECTION_HEADING.split(text)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    missing = [name for name in REPAIR_SECTIONS if name not in sections]
    if missing:
        raise ValueError(f"repair prompt file lacks the sections {missing}")
    return sections


def load_repair_messages(path: Path) -> RepairMessages:
    return RepairMessages(path, prompt_version(path), parse_repair_sections(path.read_text()))


def load_check_prompts(settings: Settings) -> CheckPrompts:
    root = settings.config_path.parent
    return CheckPrompts(
        judge=load_prompt(root / settings.output.judge_prompt_file),
        repair=load_repair_messages(root / settings.output.repair_prompt_file),
    )


def build_judge_messages(
    prompt: Prompt, context: Sequence[ContextBlock], answer: str
) -> tuple[str, str]:
    """One pass over the template, so an answer that contains `{context}` is not expanded."""
    values = {"context": "\n\n".join(b.text for b in context), "answer": answer}
    return prompt.system, PLACEHOLDER.sub(lambda m: values[m.group(1)], prompt.user_template)


# --------------------------------------------------------------------------- fixed checks


def is_mixed_abstain(answer: str, abstain_text: str = ABSTAIN_TEXT) -> bool:
    """The abstain text, in any letter case, together with other text."""
    return classify(answer, abstain_text) != "abstain" and abstain_text.lower() in answer.lower()


@dataclass(frozen=True)
class CitationCheck:
    markers_found: list[str]
    markers_unknown: list[str]

    @property
    def valid(self) -> bool:
        return not self.markers_unknown

    def to_attrs(self) -> dict[str, Any]:
        return {
            "rag.markers_found": json.dumps(self.markers_found),
            "rag.markers_unknown": json.dumps(self.markers_unknown),
        }


@stage("check.citations", kind="TOOL")
def check_citations(answer: str, label_map: Mapping[str, Any]) -> CitationCheck:
    """Unknown labels, then malformed groups (`[S1-S9]`, `[s9]`), in reading order."""
    found = find_markers(answer)
    unknown = sorted({m for m in found if m not in label_map}, key=lambda m: int(m[1:]))
    return CitationCheck(found, unknown + find_malformed(answer))


def normalize_code(text: str) -> str:
    """Trim every line, drop empty lines, one space per whitespace run, `...` for `…`."""
    lines = (" ".join(line.split()) for line in text.replace(ELLIPSIS, "...").splitlines())
    return "\n".join(line for line in lines if line)


def fenced_code(text: str) -> list[str]:
    """The bodies of the fenced blocks, as a Markdown reader sees them: a fence of three or
    more backticks or tildes opens a block, a line of the same character closes it, an unclosed
    block runs to the end, and a one-line ```` ```code``` ```` holds the text between its fences.
    Empty blocks are left out: they hold no code to check."""
    blocks: list[str] = []
    open_fence: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        fence = FENCE_LINE.match(line)
        if open_fence is None:
            if fence is None:
                continue
            marker, rest = fence.group(1), fence.group(2)
            if marker in rest:  # one-line block
                blocks.append(rest.split(marker, 1)[0])
                continue
            open_fence, body = marker, []
        elif (
            fence
            and fence.group(1)[0] == open_fence[0]
            and len(fence.group(1)) >= len(open_fence)
            and not fence.group(2).strip()
        ):
            blocks.append("\n".join(body))
            open_fence = None
        else:
            body.append(line)
    if open_fence is not None:
        blocks.append("\n".join(body))
    return [block for block in blocks if block.strip()]


@dataclass(frozen=True)
class CodeCheck:
    code_blocks: int
    code_blocks_unmatched: int
    plain_text: bool  # the context had no fences: whole block texts were searched instead

    @property
    def ok(self) -> bool:
        return self.code_blocks_unmatched == 0

    def to_attrs(self) -> dict[str, Any]:
        return {
            "rag.code_blocks": self.code_blocks,
            "rag.code_blocks_unmatched": self.code_blocks_unmatched,
            "rag.code_check_plain_text": self.plain_text,
        }


def context_code(context: Sequence[ContextBlock]) -> tuple[list[str], bool]:
    fenced = [normalize_code(code) for b in context for code in fenced_code(b.text)]
    if fenced:
        return fenced, False
    return [normalize_code(b.text) for b in context], True


@stage("check.code", kind="TOOL")
def check_code(answer: str, context: Sequence[ContextBlock]) -> CodeCheck:
    """Every fenced block of the answer must be part of one code block of the context."""
    haystacks, plain = context_code(context)
    blocks = [normalize_code(code) for code in fenced_code(answer)]
    unmatched = sum(1 for code in blocks if not any(code in hay for hay in haystacks))
    return CodeCheck(len(blocks), unmatched, plain)


# --------------------------------------------------------------------------- judge


def is_sampled(request_id: str, rate: float) -> bool:
    """A hash of the request id, so a replay makes the same decision."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.sha256(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def should_judge(*, borderline: bool, judge_all: bool, request_id: str, rate: float) -> bool:
    return borderline or judge_all or is_sampled(request_id, rate)


@dataclass(frozen=True)
class Verdict:
    supported: bool
    unsupported_claims: list[str]
    completion: Completion
    cost_usd: float
    prompt_version: str

    def to_attrs(self) -> dict[str, Any]:
        c = self.completion
        return {
            "gen_ai.system": c.model_name.partition(":")[0],
            "gen_ai.request.model": c.model_name,
            "gen_ai.usage.input_tokens": c.input_tokens,
            "gen_ai.usage.output_tokens": c.output_tokens,
            "gen_ai.response.finish_reasons": json.dumps([c.finish_reason]),
            "rag.cost_usd": self.cost_usd,
            "rag.prompt_version": self.prompt_version,
            "rag.faithful": self.supported,
            "rag.unsupported_claims": json.dumps(self.unsupported_claims, ensure_ascii=False),
            "rag.faithfulness_sampled": True,
        }


def parse_verdict(completion: Completion) -> tuple[bool, list[str]]:
    """`supported` and the claims. A verdict that says supported but lists claims is read as
    not supported: the listed claim is the evidence, the flag is the summary."""
    data = completion.json
    if data is None:
        raise InvalidOutput(f"judge returned no JSON verdict: {completion.text!r}")
    supported, claims = data.get("supported"), data.get("unsupported_claims")
    if not isinstance(supported, bool) or not isinstance(claims, list):
        raise InvalidOutput(f"judge verdict has the wrong shape: {completion.text!r}")
    return supported and not claims, [str(claim) for claim in claims]


@stage("check.faithfulness", kind="LLM")
def judge_faithfulness(
    answer: str,
    context: Sequence[ContextBlock],
    *,
    chat: ChatModel,
    prompt: Prompt,
    cfg: OutputConfig,
    backoff_s: float,
) -> Verdict:
    """One judge call: the context blocks and the answer in, the verdict JSON out. Raises
    `GenerationUnavailable` after the retry, `InvalidOutput` on a verdict that is not JSON of
    the right shape, and whatever else the provider raises; the caller applies the judge
    failure policy to all of them."""
    system, user = build_judge_messages(prompt, context, answer)
    completion, _ = call_with_retry(
        chat,
        system,
        [("user", user)],
        temperature=cfg.judge_temperature,
        max_tokens=cfg.judge_max_output_tokens,
        backoff_s=backoff_s,
        json_schema=JUDGE_SCHEMA,
    )
    supported, claims = parse_verdict(completion)
    cost = completion_cost_usd(completion, chat.price_per_million_tokens)
    return Verdict(supported, claims, completion, cost, prompt.version)


# --------------------------------------------------------------------------- decision


@dataclass(frozen=True)
class Outcome:
    result: CheckResult
    answer: str  # the checked answer, or the abstain text
    abstain_reason: str | None = None  # None on `ok`, and on the early stop (Stage 16's stays)
    early_stop: bool = False
    failure: Failure | None = None
    repair_message: str | None = None
    mixed_abstain: bool = False
    citations: CitationCheck | None = None
    code: CodeCheck | None = None
    sampled: bool = False
    verdict: Verdict | None = None
    judge_error: str | None = None
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def check_output(
    answer: str,
    *,
    label_map: Mapping[str, Any],
    context: Sequence[ContextBlock],
    borderline: bool,
    request_id: str,
    repair_calls: int,
    settings: Settings,
    judge: Callable[[], ChatModel],
    prompts: CheckPrompts,
    judge_all: bool = False,
) -> Outcome:
    if classify(answer) == "abstain":
        return Outcome("abstain", ABSTAIN_TEXT, early_stop=True)
    budget_left = repair_calls < settings.gen.max_repair_calls

    def rejected(failure: Failure, checks: dict[str, Any], **values: str) -> Outcome:
        if budget_left:
            message = prompts.repair.message(failure, **values)
            return Outcome("repair", answer, failure=failure, repair_message=message, **checks)
        return Outcome(
            "abstain", ABSTAIN_TEXT, "output.invalid_after_repair", failure=failure, **checks
        )

    if is_mixed_abstain(answer):
        return rejected("mixed_abstain", {"mixed_abstain": True})
    citations = check_citations(answer, label_map)
    checks: dict[str, Any] = {"citations": citations}
    if not citations.valid:
        markers = ", ".join(f"[{m}]" for m in citations.markers_unknown)
        return rejected("citations", checks, markers=markers)
    if settings.output.code_verbatim:
        code = check_code(answer, context)
        checks["code"] = code
        if not code.ok:
            return rejected("code", checks)
    must_judge = borderline or judge_all
    sampled = should_judge(
        borderline=borderline,
        judge_all=judge_all,
        request_id=request_id,
        rate=settings.output.faithfulness_sample_rate,
    )
    if not sampled:
        return Outcome("ok", answer, **checks)
    chat = judge()  # a placeholder model raises ModelConfigError here, on every judged request
    try:
        verdict = judge_faithfulness(
            answer,
            context,
            chat=chat,
            prompt=prompts.judge,
            cfg=settings.output,
            backoff_s=settings.gen.retry_backoff_s,
        )
    except Exception as exc:  # noqa: BLE001 - every judge error follows the same policy
        error = f"{type(exc).__name__}: {exc}"
        if must_judge:
            return Outcome(
                "abstain",
                ABSTAIN_TEXT,
                "output.judge_unavailable",
                sampled=True,
                judge_error=error,
                **checks,
            )
        return Outcome("ok", answer, sampled=True, judge_error=error, **checks)
    judged = {"sampled": True, "verdict": verdict, "cost_usd": verdict.cost_usd, **checks}
    if not verdict.supported:
        return Outcome("abstain", ABSTAIN_TEXT, "output.unfaithful", **judged)
    return Outcome("ok", answer, **judged)


# --------------------------------------------------------------------------- node


def make_check_output(
    settings: Settings, judge_factory: Callable[[], ChatModel]
) -> Callable[[RagState], dict[str, Any]]:
    """The graph node. The judge model is built on the first judged request, so the API starts
    without a key and a placeholder `output.judge_model` fails on the first request that is
    sampled or borderline (a configuration error, not a judge failure)."""
    prompts = load_check_prompts(settings)
    judge: ChatModel | None = None
    lock = threading.Lock()

    def model() -> ChatModel:
        nonlocal judge
        with lock:
            if judge is None:
                judge = judge_factory()
            return judge

    def record(span: Any, outcome: Outcome, reason: str | None) -> None:
        attrs: dict[str, Any] = {
            "rag.check_result": outcome.result,
            "rag.early_stop": outcome.early_stop,
            "rag.mixed_abstain": outcome.mixed_abstain,
            "rag.faithfulness_sampled": outcome.sampled,
            "rag.cost_usd": outcome.cost_usd,
            "rag.repair_prompt_version": prompts.repair.version,
            "rag.abstain_reason": reason,
            "rag.failed_check": outcome.failure,
            "rag.judge_error": outcome.judge_error,
        }
        if outcome.citations is not None:
            attrs["rag.citations_valid"] = outcome.citations.valid
        if outcome.code is not None:
            attrs["rag.code_verbatim_ok"] = outcome.code.ok
        if outcome.verdict is not None:
            attrs["rag.faithful"] = outcome.verdict.supported
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)

    def check_node(state: RagState) -> dict[str, Any]:
        answer = state.get("answer", "")
        outcome = check_output(
            answer,
            label_map=state.get("label_map", {}),
            context=state.get("context", []),
            borderline=bool(state.get("borderline")),
            request_id=state["request_id"],
            repair_calls=int(state.get("repair_calls", 0)),
            settings=settings,
            judge=model,
            prompts=prompts,
            judge_all=bool(state.get("judge_all")),
        )
        reason = outcome.abstain_reason or (
            state.get("abstain_reason") if outcome.early_stop else None
        )
        span = current_span()
        record(span, outcome, reason)
        replaced = outcome.result != "ok" and not outcome.early_stop
        if replaced and settings.trace.capture_content:
            span.set_attribute("rag.rejected_answer", answer)
        out: dict[str, Any] = {
            "check_result": outcome.result,
            "cost_usd": float(state.get("cost_usd", 0.0)) + outcome.cost_usd,
        }
        if outcome.result == "ok":
            out |= {"status": "answered", "answer": outcome.answer, "repair_failure": None}
        elif outcome.result == "repair":
            out |= {"repair_failure": outcome.repair_message, "rejected_answer": answer}
        else:
            out |= {"status": "no_information", "answer": ABSTAIN_TEXT, "sources": []}
            if outcome.abstain_reason:
                out["abstain_reason"] = outcome.abstain_reason
            if replaced:
                out["rejected_answer"] = answer
        if state.get("debug"):
            out["debug_info"] = {
                **state.get("debug_info", {}),
                "check_output": {
                    "result": outcome.result,
                    "failed_check": outcome.failure,
                    "sampled": outcome.sampled,
                    "faithful": outcome.verdict.supported if outcome.verdict else None,
                    "judge_error": outcome.judge_error,
                },
            }
        return out

    return check_node
