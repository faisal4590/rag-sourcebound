# Metric definitions of spec Sections 8.2-8.5 for the evaluation harness (issue #34). Pure functions, no I/O.
"""Hits, ranks, per-case results, the metric table, the latency and cost breakdown, and the
baseline comparison. `eval/harness.py` drives it."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from flp_rag.contracts import EvalCase
from flp_rag.settings import EvalConfig

GATE_PREFIX = "gate."  # abstain reasons decided at Stage 14
RECALL_K, MRR_K, FUSED_K = 5, 10, 30  # the k of recall_at_5, mrr_at_10 / ndcg_at_10, recall_at_30
METRIC_NAMES = (
    "recall_at_5",
    "mrr_at_10",
    "ndcg_at_10",
    "recall_at_30_prerank",
    "faithfulness",
    "correctness",
    "citation_accuracy",
    "code_presence",
    "abstain_precision",
    "abstain_recall",
    "abstain_f1",
    "gate_share",
    "p95_latency_s",
    "cost_per_answered_usd",
    "answer_length_median",
)
PagesOf = Callable[[str], tuple[int, int] | None]


# --------------------------------------------------------------------------- hits and ranks


def pages_hit(start: int, end: int, expected: Sequence[int], *, tolerance: int) -> bool:
    """The chunk's printed page range, widened by the tolerance, holds an expected page."""
    return any(start - tolerance <= page <= end + tolerance for page in expected)


def hit_flags(
    ranked: Sequence[str], pages_of: PagesOf, expected: Sequence[int], *, tolerance: int
) -> list[bool]:
    flags: list[bool] = []
    for chunk_id in ranked:
        pages = pages_of(chunk_id)
        hit = pages is not None and pages_hit(pages[0], pages[1], expected, tolerance=tolerance)
        flags.append(hit)
    return flags


def first_hit_rank(
    ranked: Sequence[str], pages_of: PagesOf, expected: Sequence[int], *, tolerance: int
) -> int | None:
    for rank, hit in enumerate(hit_flags(ranked, pages_of, expected, tolerance=tolerance), 1):
        if hit:
            return rank
    return None


def ndcg_at(hits: Sequence[bool], k: int) -> float:
    """Binary gains over the top k; the ideal ranks the hits found in the list first (the
    golden set names evidence, not every relevant chunk, so that is the ideal we know)."""
    top = list(hits[:k])
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(top) if hit)
    ideal = sum(1 / math.log2(i + 2) for i in range(sum(top)))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Inclusive linear interpolation, like numpy's default."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = p * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


# --------------------------------------------------------------------------- one case


@dataclass(frozen=True)
class CaseResult:
    """Everything the harness reads from one response, plus the correctness score added later.
    The reranked list is at most `rerank.top_n` long (8), so MRR@10 and nDCG@10 see 8 ranks."""

    case_id: str
    group: str
    kind: str
    expected_abstain: bool
    status: str
    abstain_reason: str | None
    trace_id: str
    elapsed_ms: int
    timings_ms: dict[str, int]
    cost_usd: float
    rank_rerank: int | None
    rank_fused: int | None
    ndcg_10: float | None
    cited_total: int
    cited_hits: int
    code_present: bool | None
    faithful: bool | None  # the Stage 17 verdict, also on an `output.unfaithful` abstain
    judge_error: str | None
    answer_chars: int | None
    correctness: int | None = None
    correctness_reason: str | None = None
    error: str | None = None

    @property
    def abstained(self) -> bool:
        return self.status == "no_information"

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    @property
    def failed(self) -> bool:
        return self.status == "error"


def evaluate_case(
    case: EvalCase, body: Mapping[str, Any], elapsed_ms: int, pages_of: PagesOf, *, tolerance: int
) -> CaseResult:
    """No model call here. An answerable case without a reranked list (a smalltalk or guard
    abstain, an error) is a retrieval miss; a gate abstain still carries its list."""
    status = str(body.get("status") or "error")
    debug = body.get("debug") or {}
    answered = status == "answered"
    answer = str(body.get("answer") or "")
    top_ids = [e["chunk_id"] for e in (debug.get("rerank") or {}).get("top") or []]
    fused_ids = list((debug.get("retrieval") or {}).get("fused") or [])[:FUSED_K]
    expected = list(case.expected_pages_printed)
    rank_rerank = rank_fused = None
    ndcg = None
    if case.group == "answerable":
        flags = hit_flags(top_ids, pages_of, expected, tolerance=tolerance)
        rank_rerank = next((i for i, hit in enumerate(flags, 1) if hit), None)
        ndcg = ndcg_at(flags, MRR_K)
        rank_fused = first_hit_rank(fused_ids, pages_of, expected, tolerance=tolerance)
    sources = (body.get("sources") or []) if answered else []
    cited_hits = sum(
        1
        for s in sources
        if pages_hit(
            int(s["page_printed_start"]), int(s["page_printed_end"]), expected, tolerance=tolerance
        )
    )
    code_present = None
    if answered and case.kind == "code" and case.expected_symbol:
        code_present = case.expected_symbol.lower() in answer.lower()
    check = debug.get("check_output") or {}
    timings = debug.get("timings_ms") or body.get("timings_ms") or {}
    return CaseResult(
        case_id=case.case_id,
        group=case.group,
        kind=case.kind,
        expected_abstain=case.expected_abstain,
        status=status,
        abstain_reason=debug.get("abstain_reason"),
        trace_id=str(body.get("trace_id") or ""),
        elapsed_ms=elapsed_ms,
        timings_ms={k: int(v) for k, v in timings.items()},
        cost_usd=float(debug.get("cost_usd") or 0.0),
        rank_rerank=rank_rerank,
        rank_fused=rank_fused,
        ndcg_10=ndcg,
        cited_total=len(sources),
        cited_hits=cited_hits,
        code_present=code_present,
        faithful=check.get("faithful"),
        judge_error=check.get("judge_error"),
        answer_chars=len(answer) if answered else None,
        error=str(body.get("detail")) if status == "error" else None,
    )


def with_correctness(result: CaseResult, score: int, reason: str) -> CaseResult:
    return replace(result, correctness=score, correctness_reason=reason)


# --------------------------------------------------------------------------- the metric table


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _metric(value: float | None, n: int, target: float | None, *, higher: bool) -> dict[str, Any]:
    ok = None
    if target is not None and value is not None:
        ok = value >= target if higher else value <= target
    return {"value": value, "n": n, "target": target, "pass": ok}


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_metrics(results: Sequence[CaseResult], cfg: EvalConfig) -> dict[str, dict[str, Any]]:
    """Spec 8.2-8.5. Retrieval over the answerable cases; faithfulness over the judged
    answerable cases (an `output.unfaithful` abstain counts as unfaithful); correctness over the
    answerable cases that did not error (an abstain scores 0); abstention and latency over
    every case, an error row being neither an abstain nor a correct answer."""
    unknown = (set(cfg.targets_min) | set(cfg.targets_max)) - set(METRIC_NAMES)
    if unknown:
        raise ValueError(f"eval targets name unknown metrics: {sorted(unknown)}")
    answerable = [r for r in results if r.group == "answerable"]
    answered = [r for r in answerable if r.answered]
    abstains = [r for r in results if r.abstained]
    expected_abstain = [r for r in results if r.expected_abstain]
    true_abstains = [r for r in abstains if r.expected_abstain]
    faithful = [r.faithful for r in answerable if r.faithful is not None]
    correct = [r.correctness for r in answerable if r.correctness is not None and not r.failed]
    code = [r.code_present for r in answered if r.code_present is not None]
    cited_total = sum(r.cited_total for r in answered)
    cited_hits = sum(r.cited_hits for r in answered)
    all_answered = [r for r in results if r.answered]
    lengths = [r.answer_chars for r in all_answered if r.answer_chars is not None]
    precision = len(true_abstains) / len(abstains) if abstains else None
    recall = len(true_abstains) / len(expected_abstain) if expected_abstain else None
    in_top = [r.rank_rerank is not None and r.rank_rerank <= RECALL_K for r in answerable]
    reciprocal = [
        1 / r.rank_rerank if r.rank_rerank and r.rank_rerank <= MRR_K else 0.0 for r in answerable
    ]
    fused_hit = [r.rank_fused is not None and r.rank_fused <= FUSED_K for r in answerable]
    gate = [bool(r.abstain_reason and r.abstain_reason.startswith(GATE_PREFIX)) for r in abstains]
    values: dict[str, tuple[float | None, int]] = {
        "recall_at_5": (_share(in_top), len(answerable)),
        "mrr_at_10": (statistics.fmean(reciprocal) if reciprocal else None, len(answerable)),
        "ndcg_at_10": (
            statistics.fmean([r.ndcg_10 or 0.0 for r in answerable]) if answerable else None,
            len(answerable),
        ),
        "recall_at_30_prerank": (_share(fused_hit), len(answerable)),
        "faithfulness": (_share(faithful), len(faithful)),
        "correctness": (statistics.fmean(correct) / 2 if correct else None, len(correct)),
        "citation_accuracy": (cited_hits / cited_total if cited_total else None, cited_total),
        "code_presence": (_share(code), len(code)),
        "abstain_precision": (precision, len(abstains)),
        "abstain_recall": (recall, len(expected_abstain)),
        "abstain_f1": (_f1(precision, recall), len(results)),
        "gate_share": (_share(gate), len(abstains)),
        "p95_latency_s": (
            percentile([r.elapsed_ms for r in results], 0.95) / 1000 if results else None,
            len(results),
        ),
        "cost_per_answered_usd": (
            statistics.fmean([r.cost_usd for r in all_answered]) if all_answered else None,
            len(all_answered),
        ),
        "answer_length_median": (statistics.median(lengths) if lengths else None, len(lengths)),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (value, n) in values.items():
        if name in cfg.targets_min:
            out[name] = _metric(value, n, cfg.targets_min[name], higher=True)
        elif name in cfg.targets_max:
            out[name] = _metric(value, n, cfg.targets_max[name], higher=False)
        else:
            out[name] = _metric(value, n, None, higher=True)
    return out


def breakdown(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Spec 8.5: p50 and p95 per span and the mean cost for answered and abstained cases."""
    spans: dict[str, list[float]] = {}
    for r in results:
        for name, ms in r.timings_ms.items():
            spans.setdefault(name, []).append(ms)
    answered = [r.cost_usd for r in results if r.answered]
    abstained = [r.cost_usd for r in results if r.abstained]
    wall = [r.elapsed_ms for r in results]
    return {
        "span_latency_ms": {
            name: {"p50": percentile(v, 0.5), "p95": percentile(v, 0.95), "n": len(v)}
            for name, v in sorted(spans.items())
        },
        "request_latency_ms": {"p50": percentile(wall, 0.5), "p95": percentile(wall, 0.95)},
        "cost_usd_mean": {
            "answered": statistics.fmean(answered) if answered else None,
            "abstained": statistics.fmean(abstained) if abstained else None,
        },
        "n": {
            "answered": len(answered),
            "abstained": len(abstained),
            "error": sum(1 for r in results if r.failed),
        },
    }


def compare_with_baseline(
    metrics: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    *,
    tolerance: float,
    names: Sequence[str],
) -> list[dict[str, Any]]:
    """Spec 8.6 rule 4 over the higher-is-better metrics (`names`, the `targets_min` keys): a
    value more than `tolerance` under the baseline, or a metric the baseline has and this run
    could not compute, is a regression."""
    base_metrics = baseline.get("metrics") or {}
    out: list[dict[str, Any]] = []
    for name in names:
        before = (base_metrics.get(name) or {}).get("value")
        if before is None:
            continue
        now = (metrics.get(name) or {}).get("value")
        if now is None:
            out.append({"metric": name, "baseline": before, "current": None, "drop": None})
        elif before - now > tolerance:
            out.append({"metric": name, "baseline": before, "current": now, "drop": before - now})
    return out


def golden_version(cases: Sequence[EvalCase]) -> str:
    versions = {c.golden_version for c in cases}
    return versions.pop() if len(versions) == 1 else "mixed"
