# Issue #36, spec 8.7 rule 1: eval/online_judge.py samples answered traces, judges them, annotates them.
"""Against a fake Phoenix (REST spans by name, batched root lookup, span annotations read and
write) and a fake judge."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from flp_rag.models import Completion, ModelConfigError, Price
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import online_judge

BLOCK = {
    "label": "S1",
    "source_ids": ["d:6:1"],
    "text": "A readonly property can be set once.",
    "chapter_no": 6,
    "chapter_title": "Readonly",
    "page_printed_start": 75,
    "page_printed_end": 76,
    "token_count": 9,
}


def span(
    name: str, trace: str, output: Any, *, start: str = "2026-11-08T01:00:00+00:00"
) -> dict[str, Any]:
    return {
        "name": name,
        "context": {"trace_id": trace, "span_id": f"{name}-{trace}"},
        "parent_id": f"root-{trace}",
        "start_time": start,
        "attributes": {
            "output.value": output if not isinstance(output, dict) else json.dumps(output)
        },
    }


def trace_spans(
    trace: str,
    *,
    status: str = "answered",
    answer: str = "It is set once [S1]",
    check: str | None = "ok",
) -> list[dict[str, Any]]:
    out = [
        span("llm.generate", trace, {"answer": answer, "status": status}),
        span("context.assemble", trace, {"context": [BLOCK], "label_map": {}}),
    ]
    if check is not None:
        final = "answered" if check == "ok" else "no_information"
        out.append(span("guardrails.output", trace, {"status": final}))
    return out


def settings_fast() -> Any:
    s = load_settings()
    return s.model_copy(update={"eval": s.eval.model_copy(update={"judge_pause_s": 0})})


class FakePhoenix:
    """The three endpoints the script uses, with the live shapes (data / next_cursor,
    context.trace_id / span_id, flat dotted attributes)."""

    def __init__(
        self, spans: list[dict[str, Any]], scored: set[str] | None = None, *, page: int = 1000
    ) -> None:
        self.spans, self.scored, self.page = spans, set(scored or ()), page
        self.posted: list[dict[str, Any]] = []
        self.requests: list[str] = []

    def transport(self) -> httpx.MockTransport:
        def paged(items: list[dict[str, Any]], params: dict[str, Any]) -> httpx.Response:
            start = int(params.get("cursor") or 0)
            chunk = items[start : start + self.page]
            nxt = str(start + self.page) if start + self.page < len(items) else None
            return httpx.Response(200, json={"data": chunk, "next_cursor": nxt})

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(
                f"{request.method} {request.url.path}?{request.url.query.decode()}"
            )
            params = dict(request.url.params)
            multi = request.url.params.get_list
            if request.url.path.endswith("/span_annotations") and request.method == "GET":
                hits = [
                    {"name": "faithfulness_online", "span_id": s}
                    for s in multi("span_ids")
                    if s in self.scored
                ]
                if not hits:
                    return httpx.Response(
                        404, json={"detail": "None of the supplied span_ids exist"}
                    )
                return paged(hits, params)
            if request.url.path.endswith("/spans"):
                if params.get("parent_id") == "null":
                    roots = [
                        {"context": {"trace_id": t, "span_id": f"root-{t}"}}
                        for t in multi("trace_id")
                        if any(s["context"]["trace_id"] == t for s in self.spans)
                    ]
                    return paged(roots, params)
                assert "start_time" in params and "end_time" in params
                return paged([s for s in self.spans if s["name"] == params["name"]], params)
            if request.url.path == "/v1/span_annotations":
                self.posted.extend(json.loads(request.content)["data"])
                return httpx.Response(200, json={"data": []})
            return httpx.Response(404)

        return httpx.MockTransport(handler)


class FakeJudge:
    model_name, price_per_million_tokens = "fake:judge", Price(input=0, output=0)

    def __init__(self, unfaithful: set[str] | None = None, fail: set[str] | None = None) -> None:
        self.unfaithful, self.fail = set(unfaithful or ()), set(fail or ())
        self.calls: list[str] = []

    def complete(self, system: str, messages: Any, **kw: Any) -> Completion:
        user = messages[0][1]
        self.calls.append(user)
        if any(f"answer-{t}" in user for t in self.fail):
            raise RuntimeError("429 from the provider")
        bad = any(f"answer-{t}" in user for t in self.unfaithful)
        data = {"supported": not bad, "unsupported_claims": ["claim x"] if bad else []}
        return Completion(
            json=data,
            text=json.dumps(data),
            input_tokens=1,
            output_tokens=1,
            finish_reason="stop",
            latency_ms=1,
            model_name="fake:judge",
        )


def test_candidates_are_one_per_trace_and_skip_abstains_withdrawals_and_cut_values() -> None:
    spans = (
        trace_spans("t1")
        + trace_spans("t2", status="no_information", answer="No information found")
        + trace_spans("t3", check="abstain")  # the output checks withdrew the answer
        + trace_spans("t4")
    )
    # a repair call: the first generate span of t4 is the rejected answer, the later one counts
    spans.insert(
        0,
        span(
            "llm.generate",
            "t4",
            {"answer": "old [S9]", "status": "answered"},
            start="2026-11-08T00:59:00+00:00",
        ),
    )
    spans[1]["attributes"]["output.value"] = spans[1]["attributes"]["output.value"][
        :20
    ]  # t1 cut at content_max_chars
    generate = [s for s in spans if s["name"] == "llm.generate"]
    assemble = [s for s in spans if s["name"] == "context.assemble"]
    checks = [s for s in spans if s["name"] == "guardrails.output"]
    cands, counts = online_judge.candidates_from(generate, assemble, checks)
    assert [c.trace_id for c in cands] == ["t4"] and cands[0].answer == "It is set once [S1]"
    assert cands[0].context[0].label == "S1" and cands[0].context[0].page_printed_start == 75
    assert counts == {"traces": 4, "not_answered": 1, "withdrawn": 1, "unparseable": 1}


def test_a_context_block_with_an_extra_key_still_loads() -> None:
    block = {**BLOCK, "future_field": 1}
    assert online_judge._block(block) is not None
    assert online_judge._block({"label": "S1"}) is None  # too few fields -> skipped, no crash


def test_sample_target_and_seeded_sample() -> None:
    assert online_judge.sample_target(answered=10, already_scored=0, share=0.2) == 2
    assert (
        online_judge.sample_target(answered=10, already_scored=2, share=0.2) == 0
    )  # a rerun adds nothing
    assert online_judge.sample_target(answered=3, already_scored=0, share=0.2) == 1  # at least one
    assert online_judge.sample_target(answered=0, already_scored=0, share=0.2) == 0
    cands = [online_judge.Candidate(f"t{i}", [], "a") for i in range(10)]
    a = online_judge.sample(cands, target=3, seed=42)
    b = online_judge.sample(
        list(reversed(cands)), target=3, seed=42
    )  # order of arrival is irrelevant
    assert [c.trace_id for c in a] == [c.trace_id for c in b] and len(a) == 3


def test_run_samples_judges_and_annotates_the_root_span_and_skips_scored_traces() -> None:
    spans: list[dict[str, Any]] = []
    for i in range(10):
        spans += trace_spans(f"t{i}", answer=f"answer-t{i} [S1]")
    phoenix = FakePhoenix(spans, scored={"root-t0"})  # t0 was scored by an earlier run today
    judge = FakeJudge(unfaithful={"t3", "t5"})
    now = datetime(2026, 11, 8, 6, 0, tzinfo=UTC)
    summary = online_judge.run(
        settings_fast(),
        hours=24,
        share=0.2,
        seed=42,
        chat=judge,
        transport=phoenix.transport(),
        now=now,
    )
    assert summary["traces"] == 10 and summary["answered"] == 10 and summary["already_scored"] == 1
    assert (
        summary["sampled"] == 1 == summary["judged"] == len(judge.calls) == len(phoenix.posted)
    )  # 20% of 10 minus 1
    a = phoenix.posted[0]
    assert (
        a["name"] == "faithfulness_online"
        and a["span_id"].startswith("root-")
        and a["span_id"] != "root-t0"
    )
    assert (
        a["identifier"] == f"faithfulness_online:{a['span_id'][5:]}"
        and a["metadata"]["sampled_on"] == "2026-11-08"
    )
    assert "A readonly property can be set once." in judge.calls[0]
    assert any("start_time=2026-11-07T06" in r for r in phoenix.requests)
    assert sum(1 for r in phoenix.requests if "parent_id=null" in r) == 1  # one batched root lookup


def test_pagination_follows_next_cursor() -> None:
    spans: list[dict[str, Any]] = []
    for i in range(5):
        spans += trace_spans(f"t{i}")
    phoenix = FakePhoenix(spans, page=2)
    summary = online_judge.run(
        settings_fast(), hours=24, share=0.2, seed=42, dry_run=True, transport=phoenix.transport()
    )
    assert summary["answered"] == 5 and summary["sampled"] == 1
    assert sum(1 for r in phoenix.requests if "name=llm.generate" in r) == 3  # 5 spans, pages of 2


def test_a_failed_judge_call_is_counted_and_the_run_goes_on() -> None:
    spans: list[dict[str, Any]] = []
    for i in range(4):
        spans += trace_spans(f"t{i}", answer=f"answer-t{i} [S1]")
    phoenix = FakePhoenix(spans)
    chosen = [
        c.trace_id
        for c in online_judge.sample(
            [online_judge.Candidate(f"t{i}", [], "") for i in range(4)], target=4, seed=42
        )
    ]
    judge = FakeJudge(fail={chosen[0]})
    summary = online_judge.run(
        settings_fast(), hours=24, share=1.0, seed=42, chat=judge, transport=phoenix.transport()
    )
    assert summary["sampled"] == 4 and summary["judged"] == 3 and summary["judge_failed"] == 1
    assert len(phoenix.posted) == 3  # each verdict posted right away

    class Broken(FakeJudge):
        def complete(self, system: str, messages: Any, **kw: Any) -> Completion:
            raise ModelConfigError("placeholder judge model")

    with pytest.raises(ModelConfigError):
        online_judge.run(
            settings_fast(),
            hours=24,
            share=1.0,
            seed=42,
            chat=Broken(),
            transport=FakePhoenix(spans).transport(),
        )


def test_dry_run_counts_without_a_judge_call_or_a_write() -> None:
    phoenix = FakePhoenix(trace_spans("t1") + trace_spans("t2"))
    judge = FakeJudge()
    summary = online_judge.run(
        settings_fast(),
        hours=24,
        share=0.2,
        seed=42,
        dry_run=True,
        chat=judge,
        transport=phoenix.transport(),
    )
    assert summary["sampled"] == 1 and "judged" not in summary
    assert judge.calls == [] and phoenix.posted == []


def test_annotation_carries_the_unsupported_claims_as_explanation() -> None:
    v = online_judge.Verdict("t1", False, ["claim a", "claim b"])
    a = online_judge.annotation_for(v, "root-t1", day="2026-11-08")
    assert a["result"] == {"score": 0.0, "label": "unfaithful", "explanation": "claim a; claim b"}
    ok = online_judge.annotation_for(
        online_judge.Verdict("t2", True, []), "root-t2", day="2026-11-08"
    )
    assert ok["result"] == {"score": 1.0, "label": "faithful"} and ok["annotator_kind"] == "LLM"
