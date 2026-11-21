# Issue #29, Parts 1 and 2: the span registry loads and the attribute rules hold.
"""The registry file is the one list of spans; every span has a kind and a parent that exists;
the rule checker flags a null value, a long text, an oversized JSON value, and a content
attribute recorded with the switch off."""

from __future__ import annotations

import json

import pytest

from flp_rag import span_registry as reg

REG = reg.load_registry()


def test_registry_lists_every_request_and_ingestion_span_with_kind_and_parent() -> None:
    assert set(REG.request) >= {
        "rag.request",
        "guardrails.input",
        "cache.lookup",
        "query.understand",
        "query.classify",
        "retrieval.hybrid",
        "retrieval.dense",
        "retrieval.sparse",
        "retrieval.fuse",
        "rerank",
        "relevance.gate",
        "context.assemble",
        "llm.generate",
        "guardrails.output",
        "check.citations",
        "check.code",
        "check.faithfulness",
        "response.build",
    }
    assert set(REG.ingest) == {
        "ingest.run",
        "ingest.parse",
        "ingest.structure",
        "ingest.clean",
        "ingest.chunk",
        "ingest.enrich",
        "ingest.embed",
        "embed.batch",
        "ingest.index",
        "ingest.verify",
    }
    for table in (REG.request, REG.ingest):
        for name, spec in table.items():
            assert spec.kind in reg.KINDS, name
            assert spec.parent is None or spec.parent in table, (name, spec.parent)
            for key in (*spec.required, *spec.conditional, *spec.content):
                assert key in REG.attributes, f"{name}: {key} is not in the attribute catalog"
    assert REG.request["rag.request"].parent is None and REG.ingest["ingest.run"].parent is None
    assert REG.request["retrieval.dense"].parent == "retrieval.hybrid"


def test_the_issue_additions_are_in_the_registry() -> None:
    assert {"rag.n_chunk_fallbacks", "rag.n_skipped"} <= set(
        REG.request["context.assemble"].required
    )
    assert "check.code" in REG.request
    assert {"rag.n_truncated", "rag.hint_swaps"} <= set(REG.request["rerank"].required)
    assert "rag.understand_fallback" in REG.request["query.understand"].required
    assert "rag.run_tag" in REG.request["rag.request"].conditional
    assert REG.attributes["rag.chapter_hint"].sentinel == -1
    assert REG.attributes["rag.abstain_reason"].sentinel == "none"


def test_expected_spans_follow_the_paths_of_part_7() -> None:
    answered = REG.expected_spans("answered")
    assert {
        "rag.request",
        "guardrails.input",
        "query.understand",
        "query.classify",
        "retrieval.hybrid",
        "retrieval.dense",
        "retrieval.fuse",
        "rerank",
        "relevance.gate",
        "context.assemble",
        "llm.generate",
        "guardrails.output",
        "check.citations",
        "response.build",
    } <= answered
    assert "cache.lookup" not in answered  # optional: only when the cache is on
    assert "cache.lookup" in REG.expected_spans("answered", cache_enabled=True)
    gate = REG.expected_spans("gate_abstain")
    assert {"context.assemble", "llm.generate", "guardrails.output"}.isdisjoint(gate)
    assert "relevance.gate" in gate and "response.build" in gate
    small = REG.expected_spans("smalltalk")
    assert not any(n.startswith("retrieval") or n == "rerank" for n in small)
    assert REG.expected_spans("cache_hit") == {
        "rag.request",
        "guardrails.input",
        "cache.lookup",
        "response.build",
    }
    assert "response.build" not in REG.expected_spans("generate_error")
    with pytest.raises(KeyError):
        REG.expected_spans("no_such_path")


def view(
    name: str,
    attrs: dict,
    *,
    parent: str | None = "rag.request",
    kind: str | None = None,
    status: str = "OK",
) -> reg.SpanView:
    spec = REG.request[name]
    return reg.SpanView(name, parent, kind or spec.kind, status, attrs)


def gate_attrs(**over: object) -> dict:
    base = {
        "rag.gate_decision": "pass",
        "rag.gate_reason": "gate.pass",
        "rag.abstain_reason": "none",
        "rag.score_top1": 0.9,
        "rag.n_pass": 3,
        "rag.tau_low": 0.3,
        "rag.tau_pass": 0.35,
        "rag.tau_high": 0.65,
        "rag.borderline": False,
        "rag.latency_ms": 2,
    }
    return {**base, **over}


def test_rule_checks_flag_missing_null_long_and_oversized_values() -> None:
    ok = reg.check_span(REG, view("relevance.gate", gate_attrs()), capture_content=True)
    assert ok == []
    missing = reg.check_span(
        REG,
        view("relevance.gate", {k: v for k, v in gate_attrs().items() if k != "rag.n_pass"}),
        capture_content=True,
    )
    assert any("rag.n_pass" in m and "missing" in m for m in missing)
    null = reg.check_span(
        REG,
        view("relevance.gate", gate_attrs(**{"rag.abstain_reason": None})),
        capture_content=True,
    )
    assert any("null" in m for m in null)
    long_text = reg.check_span(
        REG,
        view("relevance.gate", gate_attrs(**{"rag.gate_reason": "x" * 1200})),
        capture_content=True,
    )
    assert any("1000" in m for m in long_text)
    big = json.dumps(list(range(2000)))
    assert len(big) > 4000
    oversized = reg.check_span(
        REG, view("rerank", {"rag.scores_all": big}, kind="RERANKER"), capture_content=True
    )
    assert any("4000" in m and "rag.scores_all" in m for m in oversized)
    wrong_type = reg.check_span(
        REG, view("relevance.gate", gate_attrs(**{"rag.n_pass": "3"})), capture_content=True
    )
    assert any("type" in m for m in wrong_type)
    wrong_kind = reg.check_span(
        REG, view("relevance.gate", gate_attrs(), kind="LLM"), capture_content=True
    )
    assert any("kind" in m for m in wrong_kind)
    wrong_parent = reg.check_span(
        REG, view("relevance.gate", gate_attrs(), parent="rerank"), capture_content=True
    )
    assert any("parent" in m for m in wrong_parent)


def test_content_attributes_are_a_violation_when_the_switch_is_off() -> None:
    attrs = gate_attrs(
        **{"input.value": "the question", "retrieval.documents.0.document.content": "book text"}
    )
    assert reg.check_span(REG, view("relevance.gate", attrs), capture_content=True) == []
    off = reg.check_span(REG, view("relevance.gate", attrs), capture_content=False)
    assert len([m for m in off if "content" in m]) == 2


def test_an_error_span_is_checked_for_kind_and_parent_only() -> None:
    errored = view("llm.generate", {"rag.latency_ms": 5}, kind="LLM", status="ERROR")
    assert reg.check_span(REG, errored, capture_content=True) == []
    assert reg.check_span(
        REG,
        view("llm.generate", {"rag.latency_ms": 5}, kind="CHAIN", status="ERROR"),
        capture_content=True,
    )


def test_check_trace_compares_the_span_set_and_sums_the_cost() -> None:
    spans = [
        reg.SpanView("rag.request", None, "CHAIN", "OK", {"rag.cost_usd": 0.5}),
        reg.SpanView("guardrails.input", "rag.request", "CHAIN", "OK", {}),
        reg.SpanView(
            "cache.lookup",
            "rag.request",
            "CHAIN",
            "OK",
            {"cache.hit": True, "cache.similarity": 0.99},
        ),
        reg.SpanView("response.build", "rag.request", "CHAIN", "OK", {}),
    ]
    problems = reg.check_trace(REG, spans, "cache_hit", capture_content=True, cache_enabled=True)
    # the set is right; the attribute gaps are listed per span
    assert not any("unexpected span" in p or "missing span" in p for p in problems)
    assert any("guardrails.input" in p and "guard.chars" in p for p in problems)
    fewer = spans[:2]
    problems = reg.check_trace(REG, fewer, "cache_hit", capture_content=True, cache_enabled=True)
    assert any("missing span" in p and "response.build" in p for p in problems)
    extra = [*spans, reg.SpanView("rerank", "rag.request", "RERANKER", "OK", {})]
    problems = reg.check_trace(REG, extra, "cache_hit", capture_content=True, cache_enabled=True)
    assert any("unexpected span" in p and "rerank" in p for p in problems)
    # cost: the root carries the sum of the LLM spans
    llm = [
        reg.SpanView("rag.request", None, "CHAIN", "OK", {"rag.cost_usd": 0.3}),
        reg.SpanView("llm.generate", "rag.request", "LLM", "OK", {"rag.cost_usd": 0.2}),
        reg.SpanView("check.faithfulness", "guardrails.output", "LLM", "OK", {"rag.cost_usd": 0.1}),
    ]
    assert reg.cost_mismatch(llm) is None
    llm[0] = reg.SpanView("rag.request", None, "CHAIN", "OK", {"rag.cost_usd": 0.25})
    assert reg.cost_mismatch(llm) == pytest.approx(0.05)


def test_classify_path_reads_the_root_and_the_span_names() -> None:
    root = {"rag.status": "answered", "rag.abstain_reason": "none", "cache.hit": False}
    assert (
        reg.classify_path(root, ["rag.request", "llm.generate", "guardrails.output"]) == "answered"
    )
    assert reg.classify_path(root, ["rag.request", "llm.generate", "llm.generate"]) == "repair"
    assert (
        reg.classify_path({**root, "cache.hit": True}, ["rag.request", "cache.lookup"])
        == "cache_hit"
    )
    assert (
        reg.classify_path(
            {"rag.status": "no_information", "rag.abstain_reason": "gate.low_top_score"}, []
        )
        == "gate_abstain"
    )
    assert (
        reg.classify_path(
            {"rag.status": "no_information", "rag.abstain_reason": "query.smalltalk"}, []
        )
        == "smalltalk"
    )
    assert (
        reg.classify_path(
            {"rag.status": "no_information", "rag.abstain_reason": "output.unfaithful"}, []
        )
        == "unfaithful"
    )
    assert (
        reg.classify_path({"rag.status": "error", "rag.abstain_reason": "guard.rejected"}, [])
        == "guard_reject"
    )
    assert (
        reg.classify_path({"rag.status": "error", "rag.abstain_reason": "none"}, ["llm.generate"])
        == "generate_error"
    )
    assert (
        reg.classify_path(
            {"rag.status": "no_information", "rag.abstain_reason": "gen.model_abstained"}, []
        )
        == "generate_abstain"
    )
    assert reg.classify_path({"rag.status": "error", "rag.abstain_reason": "timeout"}, []) is None
    assert "generate_abstain" in REG.paths and "check.citations" not in REG.expected_spans(
        "generate_abstain"
    )
