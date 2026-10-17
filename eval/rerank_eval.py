# Part 8 of issue #26: reranker gain over fusion, score separation for the gate, and latency.
"""Measure Stage 13 on the live index.

Two question sets: the 30 answerable golden questions and 10 off-topic questions
(`eval/offtopic_v1.jsonl`). For every question: hybrid retrieval, then the reranker.

Reports MRR@10 after fusion against MRR@10 after the reranker, Recall@8 (only 8 candidates go
on), the best reranker score per question against the gate thresholds (answerable should sit
above `gate.tau_pass`, off-topic below `gate.tau_low`), and the p50 / p95 latency of the rerank
stage on this machine.

    uv run python eval/rerank_eval.py --json eval/runs/rerank_<date>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flp_rag import tracing
from flp_rag.api.app import build_deps, current_index_version, default_index_dir
from flp_rag.contracts import Candidate, Chunk, EvalCase, read_jsonl
from flp_rag.graph.nodes import rerank as rk
from flp_rag.graph.nodes import retrieve as r
from flp_rag.ingest.s06_s07_index import STORE_FILE
from flp_rag.ingest.s08_verify import normalize_text
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.parent_store import ParentStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_retrieval import mrr_at, recall_at

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_v1.jsonl"
OFFTOPIC = ROOT / "eval" / "offtopic_v1.jsonl"


def hit_rank(
    candidates: Sequence[Candidate], texts: dict[str, str], quotes: Sequence[str]
) -> int | None:
    norm_quotes = [normalize_text(q) for q in quotes]
    for rank, c in enumerate(candidates, start=1):
        text = normalize_text(texts.get(c.chunk_id, ""))
        if any(q in text for q in norm_quotes):
            return rank
    return None


def percentile(values: Sequence[float], p: float) -> float:
    """Inclusive linear interpolation, like numpy's default; stated with n in the table."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return float(cuts[min(98, max(0, round(p * 100) - 1))])


def run_case(case: EvalCase, settings: Settings, deps: Any, store: ParentStore) -> dict[str, Any]:
    """One question: hybrid retrieval, then the reranker without a chapter hint (Stage 11 is
    not built yet, so production runs the same way today)."""
    fused = r.retrieve_hybrid(
        r.Variants(case.question),
        settings,
        client=deps.client,
        embedder=deps.embedder,
        sparse=deps.sparse,
        cache=deps.cache,
    )
    started = time.perf_counter()
    result = rk.rerank(case.question, fused, reranker=deps.reranker, store=store, settings=settings)
    latency_ms = (time.perf_counter() - started) * 1000
    texts = {
        c.chunk_id: (store.get_chunk(c.chunk_id) or Chunk("", "", "", "", 0, {})).display_text
        for c in result.scored
    }
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "abstain": case.expected_abstain,
        "rank_fused": hit_rank(fused, texts, case.evidence_quotes)
        if case.evidence_quotes
        else None,
        "rank_reranked": hit_rank(result.scored, texts, case.evidence_quotes)
        if case.evidence_quotes
        else None,
        "score_top1": result.scores_all[0][1] if result.scores_all else 0.0,
        "n_truncated": result.n_truncated,
        "latency_ms": latency_ms,
    }


def report(rows: Sequence[dict[str, Any]], settings: Settings, device: str) -> dict[str, Any]:
    answerable = [row for row in rows if not row["abstain"]]
    offtopic = [row for row in rows if row["abstain"]]
    gate = settings.gate
    fused_ranks = [row["rank_fused"] for row in answerable]
    reranked = [row["rank_reranked"] for row in answerable]
    best_on = [row["score_top1"] for row in answerable]
    best_off = [row["score_top1"] for row in offtopic]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "n_answerable": len(answerable),
        "n_offtopic": len(offtopic),
        "mrr10_fused": mrr_at(fused_ranks, 10),
        "mrr10_reranked": mrr_at(reranked, 10),
        "recall8_fused": recall_at(fused_ranks, 8),
        "recall8_reranked": recall_at(reranked, 8),
        "recall3_reranked": recall_at(reranked, 3),
        "n_truncated_total": sum(row["n_truncated"] for row in rows),
        "score_top1_answerable": {
            "min": min(best_on, default=0.0),
            "p05": percentile(best_on, 0.05),
            "median": statistics.median(best_on) if best_on else 0.0,
            "max": max(best_on, default=0.0),
            "above_tau_pass": sum(1 for s in best_on if s >= gate.tau_pass),
        },
        "score_top1_offtopic": {
            "min": min(best_off, default=0.0),
            "median": statistics.median(best_off) if best_off else 0.0,
            "max": max(best_off, default=0.0),
            "below_tau_low": sum(1 for s in best_off if s < gate.tau_low),
        },
        "thresholds": {
            "tau_low": gate.tau_low,
            "tau_pass": gate.tau_pass,
            "tau_high": gate.tau_high,
        },
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "device": device,
        "rows": list(rows),
    }


def print_table(rep: dict[str, Any]) -> None:
    n = rep["n_answerable"] + rep["n_offtopic"]
    print(
        f"answerable n={rep['n_answerable']}, off-topic n={rep['n_offtopic']}, device {rep['device']}"
    )
    print(f"MRR@10  fused {rep['mrr10_fused']:.3f} -> reranked {rep['mrr10_reranked']:.3f}")
    print(
        f"Recall@8 fused {rep['recall8_fused']:.3f} -> reranked {rep['recall8_reranked']:.3f}; Recall@3 reranked {rep['recall3_reranked']:.3f}"
    )
    on, off, t = rep["score_top1_answerable"], rep["score_top1_offtopic"], rep["thresholds"]
    print(
        f"best score, answerable: min {on['min']:.3f} p05 {on['p05']:.3f} median {on['median']:.3f} max {on['max']:.3f}; {on['above_tau_pass']}/{rep['n_answerable']} at or above tau_pass {t['tau_pass']}"
    )
    print(
        f"best score, off-topic:  min {off['min']:.3f} median {off['median']:.3f} max {off['max']:.3f}; {off['below_tau_low']}/{rep['n_offtopic']} below tau_low {t['tau_low']}"
    )
    lat = rep["latency_ms"]
    print(
        f"rerank latency ms over all {n} questions: p50 {lat['p50']:.0f} p95 {lat['p95']:.0f} max {lat['max']:.0f}; truncated pairs {rep['n_truncated_total']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    tracing.configure_tracing(settings)
    cases = [*read_jsonl(GOLDEN, EvalCase), *read_jsonl(OFFTOPIC, EvalCase)]
    deps = build_deps(settings)
    version, _ = current_index_version(settings, deps)
    store_path = default_index_dir(settings) / str(version) / STORE_FILE
    deps.reranker.warm_up()  # type: ignore[union-attr]
    device = getattr(deps.reranker, "device", "unknown")
    try:
        with ParentStore.open_readonly(store_path) as store:
            rows = [run_case(case, settings, deps, store) for case in cases]
    finally:
        if deps.cache is not None:
            deps.cache.close()
        tracing.shutdown_tracing()
    rep = report(rows, settings, device)
    print_table(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
