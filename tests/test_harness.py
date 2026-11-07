# Issue #34: the evaluation harness of spec Sections 8.2-8.6 against a fake API; no model, no network.
"""Metrics math, the run protocol, the run file, the baseline comparison, the Phoenix upload."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from flp_rag.contracts import EvalCase
from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import correctness
import harness
import metrics

PAGES = {
    "d:5:1": (64, 65),
    "d:5:2": (66, 66),
    "d:6:1": (75, 79),
    "d:9:1": (119, 120),
    "d:1:1": (11, 12),
}


def pages_of(chunk_id: str) -> tuple[int, int] | None:
    return PAGES.get(chunk_id)


def case(cid: str, group: str = "answerable", **kw: Any) -> EvalCase:
    base: dict[str, Any] = {
        "case_id": cid,
        "group": group,
        "question": f"q {cid}",
        "expected_answer": "the answer" if group == "answerable" else None,
        "expected_pages_printed": [64, 65] if group == "answerable" else [],
        "expected_symbol": None,
        "expected_abstain": group != "answerable",
        "golden_version": "v1",
        "evidence_quotes": ["x"] if group == "answerable" else [],
        "kind": "definition" if group == "answerable" else "offtopic",
        "chapter_no": 5 if group == "answerable" else None,
    }
    base.update(kw)
    return EvalCase(**base)


def body(
    *,
    status: str = "answered",
    top: list[str] = ("d:5:1", "d:6:1"),  # type: ignore[assignment]
    fused: list[str] | None = None,
    sources: list[tuple[str, int, int]] = (("d:5:1", 64, 65),),  # type: ignore[assignment]
    answer: str = "A CustomerDTO class [S1]",
    reason: str | None = None,
    faithful: bool | None = True,
    cost: float = 0.001,
    timings: dict[str, int] | None = None,
    trace_id: str = "a" * 32,
) -> dict[str, Any]:
    return {
        "answer": answer if status == "answered" else "No information found",
        "status": status,
        "sources": [
            {
                "label": f"S{i + 1}",
                "chunk_ids": [cid],
                "chapter_no": 5,
                "chapter_title": "T",
                "section_title": "S",
                "page_printed_start": a,
                "page_printed_end": b,
                "score": 0.9,
            }
            for i, (cid, a, b) in enumerate(sources if status == "answered" else [])
        ],
        "trace_id": trace_id,
        "timings_ms": timings or {"rerank": 100},
        "model": {"gen": "fake"},
        "index_version": "v3-test",
        "prompt_version": "abcd1234",
        "debug": {
            "abstain_reason": reason,
            "timings_ms": timings or {"rerank": 100, "llm.generate": 50},
            "cost_usd": cost,
            "rerank": {"top": [{"chunk_id": c, "chapter_no": 5, "score": 0.5} for c in top]},
            "retrieval": {"fused": list(fused if fused is not None else top)},
            "check_output": {
                "faithful": faithful if status == "answered" else None,
                "judge_error": None,
            },
        },
    }


# --------------------------------------------------------------------------- hits and ranks


def test_pages_hit_uses_the_one_page_tolerance() -> None:
    assert metrics.pages_hit(64, 65, [66], tolerance=1)
    assert not metrics.pages_hit(64, 65, [67], tolerance=1)
    assert metrics.pages_hit(70, 80, [60, 75], tolerance=1)
    assert not metrics.pages_hit(70, 80, [], tolerance=1)


def test_first_hit_rank_and_ndcg() -> None:
    ranked = ["d:6:1", "d:5:1", "d:9:1", "d:5:2"]
    assert metrics.first_hit_rank(ranked, pages_of, [64, 66], tolerance=1) == 2
    assert metrics.first_hit_rank(["d:6:1"], pages_of, [64], tolerance=1) is None
    assert metrics.first_hit_rank(["unknown"], pages_of, [64], tolerance=1) is None
    # hits at ranks 2 and 4 of 4: DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3)
    hits = [False, True, False, True]
    assert metrics.ndcg_at(hits, 10) == pytest.approx(
        (1 / 1.58496 + 1 / 2.32193) / (1 + 1 / 1.58496), abs=1e-4
    )
    assert metrics.ndcg_at([True], 10) == 1.0 and metrics.ndcg_at([False, False], 10) == 0.0


# --------------------------------------------------------------------------- one case


def test_evaluate_case_reads_ranks_citations_code_and_faithfulness() -> None:
    c = case("g005", kind="code", expected_symbol="CustomerDTO")
    r = harness.evaluate_case(
        c, body(top=["d:6:1", "d:5:1"], fused=["d:9:1", "d:5:2"]), 1234, pages_of, tolerance=1
    )
    assert r.case_id == "g005" and r.status == "answered" and r.elapsed_ms == 1234
    assert r.rank_rerank == 2 and r.rank_fused == 2
    assert r.cited_total == 1 and r.cited_hits == 1
    assert (
        r.code_present is True
        and r.faithful is True
        and r.answer_chars == len("A CustomerDTO class [S1]")
    )
    assert r.cost_usd == 0.001 and r.timings_ms["rerank"] == 100 and r.trace_id == "a" * 32


def test_evaluate_case_abstain_and_error_rows() -> None:
    off = case("o001", "out_of_domain")
    r = harness.evaluate_case(
        off,
        body(status="no_information", reason="gate.low_top_score", top=[]),
        300,
        pages_of,
        tolerance=1,
    )
    assert r.status == "no_information" and r.abstained and r.abstain_reason == "gate.low_top_score"
    assert (
        r.rank_rerank is None
        and r.cited_total == 0
        and r.code_present is None
        and r.answer_chars is None
    )
    err = harness.evaluate_case(
        off, {"status": "error", "detail": "transport: boom"}, 5, pages_of, tolerance=1
    )
    assert err.status == "error" and err.error == "transport: boom" and not err.abstained


def test_code_presence_is_only_scored_on_code_cases_and_is_case_insensitive() -> None:
    plain = harness.evaluate_case(case("g001"), body(), 1, pages_of, tolerance=1)
    assert plain.code_present is None
    code = harness.evaluate_case(
        case("g005", kind="code", expected_symbol="customerdto"),
        body(answer="class CustomerDTO [S1]"),
        1,
        pages_of,
        tolerance=1,
    )
    assert code.code_present is True


# --------------------------------------------------------------------------- metrics


def results() -> list[harness.CaseResult]:
    rows = [
        harness.evaluate_case(
            case("g001"),
            body(top=["d:5:1"], cost=0.01, timings={"rerank": 100, "llm.generate": 1000}),
            2000,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("g002", kind="code", expected_symbol="Foo"),
            body(
                top=["d:6:1", "d:9:1", "d:5:1"],
                fused=["d:6:1"],
                answer="no symbol [S1]",
                faithful=False,
                cost=0.03,
            ),
            3000,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("g003"),
            body(top=["d:6:1"], fused=["d:6:1", "d:5:1"], sources=[("d:6:1", 75, 79)]),
            4000,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("g004"),
            body(status="no_information", reason="gate.low_top_score", top=[]),
            500,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("o001", "out_of_domain"),
            body(status="no_information", reason="gate.low_top_score", top=[]),
            400,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("o002", "out_of_domain"),
            body(status="no_information", reason="gen.model_abstained", top=[]),
            6000,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("u001", "in_domain_unanswerable", kind="unanswerable"),
            body(top=["d:6:1"], sources=[("d:6:1", 75, 79)], cost=0.02),
            7000,
            pages_of,
            tolerance=1,
        ),
    ]
    rows[0] = harness.with_correctness(rows[0], 2, "matches")
    rows[1] = harness.with_correctness(rows[1], 1, "partial")
    rows[2] = harness.with_correctness(rows[2], 0, "wrong")
    rows[3] = harness.with_correctness(rows[3], 0, harness.ABSTAINED_REASON)
    return rows


def test_metrics_follow_the_spec_definitions() -> None:
    m = harness.compute_metrics(results(), load_settings().eval)
    # answerable: g001 hit@1, g002 hit@3, g003 miss, g004 abstained (no retrieval list -> miss)
    assert m["recall_at_5"]["value"] == pytest.approx(2 / 4)
    assert m["mrr_at_10"]["value"] == pytest.approx((1 + 1 / 3 + 0 + 0) / 4)
    assert m["recall_at_30_prerank"]["value"] == pytest.approx(
        2 / 4
    )  # g001 fused=top hit, g003 fused hit
    # generation over the answered answerable cases g001, g002, g003
    assert m["faithfulness"]["value"] == pytest.approx(2 / 3)
    assert m["correctness"]["value"] == pytest.approx(
        (2 + 1 + 0 + 0) / 4 / 2
    )  # g004 abstained -> 0
    assert m["citation_accuracy"]["value"] == pytest.approx(
        2 / 3
    )  # g001, g002 cite d:5:1; g003 cites d:6:1
    assert m["code_presence"]["value"] == 0.0 and m["code_presence"]["n"] == 1
    # abstention: abstains g004, o001, o002 (3); expected abstain o001, o002, u001 (3)
    assert m["abstain_precision"]["value"] == pytest.approx(2 / 3)
    assert m["abstain_recall"]["value"] == pytest.approx(2 / 3)
    assert m["abstain_f1"]["value"] == pytest.approx(2 / 3)
    assert m["gate_share"]["value"] == pytest.approx(2 / 3)  # two gate.* of three abstains
    assert m["p95_latency_s"]["value"] == pytest.approx(6.7, abs=0.05)
    assert m["cost_per_answered_usd"]["value"] == pytest.approx((0.01 + 0.03 + 0.001 + 0.02) / 4)
    assert m["answer_length_median"]["target"] is None
    assert m["recall_at_5"]["pass"] is False and m["recall_at_5"]["target"] == 0.85
    assert m["p95_latency_s"]["pass"] is False and m["cost_per_answered_usd"]["pass"] is True


def test_latency_and_cost_breakdown() -> None:
    extra = harness.breakdown(results())
    assert extra["span_latency_ms"]["rerank"]["p50"] == 100
    assert extra["cost_usd_mean"]["answered"] == pytest.approx((0.01 + 0.03 + 0.001 + 0.02) / 4)
    assert extra["cost_usd_mean"]["abstained"] == pytest.approx(0.001)
    assert extra["n"] == {"answered": 4, "abstained": 3, "error": 0}


# --------------------------------------------------------------------------- baseline


def test_regressions_are_drops_over_the_tolerance_on_target_metrics() -> None:
    now = {
        "recall_at_5": {"value": 0.80},
        "abstain_f1": {"value": 0.95},
        "p95_latency_s": {"value": 9.0},
        "answer_length_median": {"value": 10},
    }
    base = {
        "metrics": {
            "recall_at_5": {"value": 0.85},
            "abstain_f1": {"value": 0.97},
            "p95_latency_s": {"value": 3.0},
            "answer_length_median": {"value": 900},
        }
    }
    names = tuple(load_settings().eval.targets_min)
    regressions = harness.compare_with_baseline(now, base, tolerance=0.03, names=names)
    assert [r["metric"] for r in regressions] == ["recall_at_5"]
    assert regressions[0]["baseline"] == 0.85 and regressions[0]["current"] == 0.80
    assert harness.compare_with_baseline(now, {}, tolerance=0.03, names=names) == []
    # a metric the baseline has and this run could not compute is a regression too
    lost = harness.compare_with_baseline(
        {"recall_at_5": {"value": None}},
        {"metrics": {"recall_at_5": {"value": 0.9}}},
        tolerance=0.03,
        names=names,
    )
    assert [r["metric"] for r in lost] == ["recall_at_5"] and lost[0]["current"] is None


# --------------------------------------------------------------------------- upload


def test_upload_builds_one_annotation_per_score_on_the_root_span() -> None:
    posted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/graphql":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "projects": {
                            "edges": [
                                {
                                    "node": {
                                        "name": "flp-rag",
                                        "spans": {
                                            "edges": [
                                                {
                                                    "node": {
                                                        "context": {
                                                            "spanId": "root1",
                                                            "traceId": "a" * 32,
                                                        }
                                                    }
                                                }
                                            ]
                                        },
                                    }
                                }
                            ]
                        }
                    }
                },
            )
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"data": []})

    rows = results()[:1]
    n, missing = harness.upload_scores(
        rows,
        "http://phoenix:6006",
        "flp-rag",
        tag="eval-t",
        timeout_s=5,
        transport=httpx.MockTransport(handler),
    )
    names = sorted(a["name"] for a in posted[0]["data"])
    assert (
        n == len(names)
        and missing == 0
        and "correctness" in names
        and "retrieval_hit_at_5" in names
        and "faithfulness" in names
    )
    assert all(
        a["span_id"] == "root1" and a["annotator_kind"] in ("LLM", "CODE")
        for a in posted[0]["data"]
    )
    assert all(a["identifier"] for a in posted[0]["data"])  # upserts on a rerun


# --------------------------------------------------------------------------- the run


class FakeApi:
    def __init__(self, bodies: dict[str, dict[str, Any]]) -> None:
        self.bodies, self.requests = bodies, []  # type: ignore[var-annotated]

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={"status": "ok", "index_version": "v3-test", "models": {"gen": "fake"}},
                )
            payload = json.loads(request.content)
            self.requests.append(payload)
            out = self.bodies[payload["question"]]
            return httpx.Response(200 if out["status"] != "error" else 504, json=out)

        return httpx.MockTransport(handler)


def test_run_writes_the_run_file_and_exits_1_on_regression(tmp_path: Path) -> None:
    cases = [case("g001"), case("o001", "out_of_domain")]
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases))
    api = FakeApi(
        {
            "q g001": body(),
            "q o001": body(status="no_information", reason="gate.low_top_score", top=[]),
        }
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"metrics": {"recall_at_5": {"value": 1.0}, "abstain_f1": {"value": 1.0}}})
    )
    out = tmp_path / "run.json"
    settings = load_settings().model_copy(
        update={"eval": load_settings().eval.model_copy(update={"runs_dir": tmp_path / "runs"})}
    )
    code = harness.run(
        settings,
        api_url="http://api",
        golden=golden,
        out=out,
        baseline=baseline,
        pages_of=pages_of,
        judge=lambda c, answer: (2, "ok"),
        upload=False,
        pause_s=0,
        transport=api.transport(),
    )
    assert code == 0
    assert all(r["debug"] is True and r["judge_all"] is True and r["run_tag"] for r in api.requests)
    run = json.loads(out.read_text())
    assert (
        run["golden_version"] == "v1" and run["n_cases"] == 2 and run["index_version"] == "v3-test"
    )
    assert (
        run["metrics"]["recall_at_5"]["value"] == 1.0
        and run["metrics"]["correctness"]["value"] == 1.0
    )
    assert len(run["cases"]) == 2 and run["cases"][0]["correctness"] == 2
    assert run["models"]["gen"] == "fake" and run["models"]["judge"] and run["error_share"] == 0
    assert not list((tmp_path / "runs").glob("*.partial.jsonl"))
    assert run["judge_error_share"] == 0
    # a drop in the baseline comparison fails the run
    baseline.write_text(
        json.dumps(
            {
                "metrics": {
                    "recall_at_5": {"value": 1.0},
                    "faithfulness": {"value": 1.0},
                    "abstain_f1": {"value": 1.0},
                }
            }
        )
    )
    api2 = FakeApi(
        {
            "q g001": body(faithful=False),
            "q o001": body(status="no_information", reason="gate.low_top_score", top=[]),
        }
    )
    code = harness.run(
        load_settings(),
        api_url="http://api",
        golden=golden,
        out=out,
        baseline=baseline,
        pages_of=pages_of,
        judge=lambda c, a: (2, "ok"),
        upload=False,
        pause_s=0,
        transport=api2.transport(),
    )
    assert code == 1


def test_default_out_path_has_timestamp_index_and_prompt_version() -> None:
    p = harness.default_out(
        Path("eval/runs"), "v3-bgem3-d4ebab9e", "287346c0", stamp="20261107T2030"
    )
    assert p == Path("eval/runs/20261107T2030_v3-bgem3-d4ebab9e_287346c0.json")


def test_correctness_prompt_and_parse() -> None:
    prompt = correctness.load_correctness_prompt(ROOT / "prompts" / "judge_correctness_v1.md")
    system, user = correctness.build_correctness_messages(prompt, "q?", "expected", "given")
    assert "q?" in user and "expected" in user and "given" in user and "Question:" not in system
    assert correctness.parse_correctness({"score": 2, "reason": "same"}) == (2, "same")
    with pytest.raises(ValueError):
        correctness.parse_correctness({"score": 3})


# --------------------------------------------------------------------------- the API side


def test_judge_all_and_the_fused_list_reach_the_graph_only_with_debug() -> None:
    """`judge_all` is a request field read with `debug: true` only (the API side mirrors
    `run_tag`); the retrieve node puts the fused chunk ids in the debug field, which is what
    Recall@30 before the reranker reads."""
    import flp_rag.graph.nodes.retrieve as retrieve_mod
    from flp_rag.api.schemas import AskRequest
    from flp_rag.contracts import Candidate
    from flp_rag.graph import build
    from flp_rag.graph.state import initial_state

    assert AskRequest(question="q", debug=True, judge_all=True).judge_all is True
    assert AskRequest(question="q").judge_all is False
    assert initial_state("q", request_id="r1", debug=True)["judge_all"] is False
    assert initial_state("q", request_id="r1", judge_all=True)["judge_all"] is True
    fake = [Candidate("d:1", 0.5, None), Candidate("d:2", 0.4, None)]
    original = retrieve_mod.retrieve_hybrid
    retrieve_mod.retrieve_hybrid = lambda *a, **k: fake  # type: ignore[assignment]
    try:
        s = load_settings()
        nodes = build.default_nodes(build.Deps(settings=s))
        out = nodes.retrieve(initial_state("q", request_id="r1", debug=True))
        assert out["debug_info"]["retrieval"]["fused"] == ["d:1", "d:2"]
        assert out["candidates"] == fake
        assert "debug_info" not in nodes.retrieve(initial_state("q", request_id="r1"))
    finally:
        retrieve_mod.retrieve_hybrid = original  # type: ignore[assignment]


def test_faithfulness_counts_an_unfaithful_abstain_and_correctness_scores_an_abstain_zero() -> None:
    """Stage 17 turns `supported: false` into an `output.unfaithful` abstain, so the verdict
    must be read on abstained rows too (review of #34)."""
    b = body(status="no_information", reason="output.unfaithful", top=["d:5:1"])
    b["debug"]["check_output"]["faithful"] = False
    r = harness.evaluate_case(case("g001"), b, 1, pages_of, tolerance=1)
    assert r.faithful is False and r.abstained and r.rank_rerank == 1
    rows = [harness.with_correctness(r, 0, harness.ABSTAINED_REASON)]
    m = harness.compute_metrics(rows, load_settings().eval)
    assert m["faithfulness"]["value"] == 0.0 and m["correctness"]["value"] == 0.0
    assert m["recall_at_5"]["value"] == 1.0  # a gate-path abstain still carries its list


def test_error_rows_count_in_latency_and_abstention_and_can_fail_the_run(tmp_path: Path) -> None:
    rows = [
        harness.evaluate_case(
            case("o001", "out_of_domain"),
            {"status": "error", "detail": "504"},
            20_000,
            pages_of,
            tolerance=1,
        ),
        harness.evaluate_case(
            case("o002", "out_of_domain"),
            body(status="no_information", reason="gate.low_top_score", top=[]),
            400,
            pages_of,
            tolerance=1,
        ),
    ]
    m = harness.compute_metrics(rows, load_settings().eval)
    assert m["abstain_recall"]["value"] == 0.5 and m["p95_latency_s"]["value"] > 6
    cases = [case("g001"), case("o001", "out_of_domain")]
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases))
    api = FakeApi(
        {
            "q g001": {"status": "error", "detail": "boom"},
            "q o001": {"status": "error", "detail": "boom"},
        }
    )
    code = harness.run(
        load_settings(),
        api_url="http://api",
        golden=golden,
        out=tmp_path / "run.json",
        baseline=tmp_path / "none.json",
        pages_of=pages_of,
        judge=False,
        upload=False,
        pause_s=0,
        transport=api.transport(),
    )
    assert code == 1  # every request errored: over eval.max_error_share


def test_upload_failure_is_a_warning_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = [case("o001", "out_of_domain")]
    golden = tmp_path / "golden.jsonl"
    golden.write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200, json={"status": "ok", "index_version": "v3-test", "models": {}}
            )
        if request.url.path == "/ask":
            return httpx.Response(
                200, json=body(status="no_information", reason="gate.low_top_score", top=[])
            )
        return httpx.Response(503, text="phoenix down")

    code = harness.run(
        load_settings(),
        api_url="http://api",
        golden=golden,
        out=tmp_path / "run.json",
        baseline=tmp_path / "none.json",
        pages_of=pages_of,
        judge=False,
        upload=True,
        pause_s=0,
        transport=httpx.MockTransport(handler),
    )
    assert code == 0 and "WARNING: Phoenix upload failed" in capsys.readouterr().out


def test_judge_retries_once_and_lets_a_config_error_through() -> None:
    from flp_rag.models import Completion, ModelConfigError, Price

    class Chat:
        model_name, price_per_million_tokens = "fake:judge", Price(input=0, output=0)

        def __init__(self, outputs: list[Any]) -> None:
            self.outputs, self.calls = outputs, 0

        def complete(self, system: str, messages: Any, **kw: Any) -> Completion:
            self.calls += 1
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return Completion(
                json=out,
                text=json.dumps(out),
                input_tokens=1,
                output_tokens=1,
                finish_reason="stop",
                latency_ms=1,
                model_name="fake:judge",
            )

    s = load_settings().model_copy(
        update={"eval": load_settings().eval.model_copy(update={"judge_pause_s": 0})}
    )
    flaky = Chat([{"score": 3}, {"score": 1, "reason": "partial"}])
    assert (
        harness.make_judge(s, chat=flaky)(case("g001"), "an answer") == (1, "partial")
        and flaky.calls == 2
    )
    with pytest.raises(ModelConfigError):
        harness.make_judge(s, chat=Chat([ModelConfigError("placeholder")]))(case("g001"), "x")
    assert metrics.METRIC_NAMES[0] == "recall_at_5"


def test_score_cases_counts_judge_failures_and_lets_a_config_error_through() -> None:
    from flp_rag.models import ModelConfigError

    bodies = [(case("g001"), body(), 10), (case("g002"), body(), 10)]

    def flaky(c: EvalCase, answer: str) -> tuple[int, str]:
        if c.case_id == "g001":
            raise RuntimeError("429")
        return 2, "ok"

    rows, failed = harness.score_cases(bodies, pages_of, flaky, tolerance=1, judge_pause_s=0)
    assert (
        failed == 1 and rows[0].correctness is None and "judge error" in rows[0].correctness_reason
    )
    assert rows[1].correctness == 2

    def broken(c: EvalCase, answer: str) -> tuple[int, str]:
        raise ModelConfigError("placeholder judge model")

    with pytest.raises(ModelConfigError):
        harness.score_cases(bodies, pages_of, broken, tolerance=1, judge_pause_s=0)
