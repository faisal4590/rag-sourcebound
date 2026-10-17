# Part 8 of issue #25: dense, sparse, and fused retrieval measured on the live Qdrant index.
"""Measure Stage 12 on the live index with the 30 golden questions.

Three runs over `eval/golden_v1.jsonl`: dense only, sparse only (Qdrant BM25 with the IDF
modifier), and the fused list. A hit is a chunk whose display text contains one of the case's
evidence quotes, the same rule as `eval/chunk_retrieval.py`, so the numbers sit next to the
chunk quality report. Reports Recall@5, Recall@30, MRR@10 overall and per kind, plus the paired
sign test of fused against each single retriever.

    uv run python eval/hybrid_retrieval.py            # prints the table
    uv run python eval/hybrid_retrieval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flp_rag import tracing
from flp_rag.contracts import Candidate, EvalCase, read_jsonl
from flp_rag.graph.nodes import retrieve as r
from flp_rag.ingest.s08_verify import normalize_text
from flp_rag.models import get_embeddings, get_sparse_embeddings
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.embedding_cache import EmbeddingCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_retrieval import mrr_at, recall_at, sign_test_p

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_v1.jsonl"
RUNS = ("dense", "sparse", "fused")


def hit_rank(
    candidates: Sequence[Candidate], texts: dict[str, str], quotes: Sequence[str]
) -> int | None:
    norm_quotes = [normalize_text(q) for q in quotes]
    for rank, c in enumerate(candidates, start=1):
        text = normalize_text(texts.get(c.chunk_id, ""))
        if any(q in text for q in norm_quotes):
            return rank
    return None


def lists_for(
    case: EvalCase, settings: Settings, deps: dict[str, Any]
) -> dict[str, list[Candidate]]:
    """The three rankings for one question. Dense and sparse come from the same two lists that
    the fused run uses, so the comparison is paired. The single-retriever runs go through
    `fuse` too, which for one list is the list itself. The sibling rule is left out: it never
    fires on this book and would not change a rank."""
    variants = r.Variants(case.question)
    client, embedder, sparse, cache = (
        deps["client"],
        deps["embedder"],
        deps["sparse"],
        deps["cache"],
    )
    dense = r.retrieve_dense_lists(
        variants.dense_queries,
        settings,
        filters=None,
        client=client,
        embedder=embedder,
        cache=cache,
        collection=settings.index.alias,
    )
    sparse_lists = r.retrieve_sparse_lists(
        variants.sparse_queries,
        settings,
        filters=None,
        client=client,
        sparse=sparse,
        collection=settings.index.alias,
    )
    k, top = settings.retrieval.rrf_k, settings.retrieval.top_k_fused
    return {
        "dense": r.fuse(dense, rrf_k=k, top_k=top),
        "sparse": r.fuse(sparse_lists, rrf_k=k, top_k=top),
        "fused": r.fuse([*dense, *sparse_lists], rrf_k=k, top_k=top),
    }


def evaluate(cases: Sequence[EvalCase], settings: Settings, deps: dict[str, Any]) -> dict[str, Any]:
    ranks: dict[str, list[int | None]] = {run: [] for run in RUNS}
    kinds: list[str] = []
    overlap: list[float] = []
    index_dir = settings.config_path.parent / "data" / "index"
    for case in cases:
        rankings = lists_for(case, settings, deps)
        texts = r.display_texts([c for lst in rankings.values() for c in lst], index_dir)
        for run in RUNS:
            ranks[run].append(hit_rank(rankings[run], texts, case.evidence_quotes))
        kinds.append(case.kind)
        overlap.append(r.overlap_ratio(rankings["fused"]))
    return {"ranks": ranks, "kinds": kinds, "overlap": overlap, "n": len(cases)}


def metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    return {
        "recall@5": recall_at(ranks, 5),
        "recall@30": recall_at(ranks, 30),
        "mrr@10": mrr_at(ranks, 10),
    }


def paired_sign(a: Sequence[int | None], b: Sequence[int | None]) -> dict[str, Any]:
    """Per question: is the rank in `a` better (lower) than in `b`? A miss counts as the worst."""

    def key(rank: int | None) -> float:
        return float("inf") if rank is None else rank

    better = sum(1 for x, y in zip(a, b, strict=True) if key(x) < key(y))
    worse = sum(1 for x, y in zip(a, b, strict=True) if key(x) > key(y))
    return {
        "better": better,
        "worse": worse,
        "ties": len(a) - better - worse,
        "p": sign_test_p(better, worse),
    }


def report(result: dict[str, Any]) -> dict[str, Any]:
    ranks = result["ranks"]
    out: dict[str, Any] = {
        "n": result["n"],
        "overall": {run: metrics(ranks[run]) for run in RUNS},
        "by_kind": {},
    }
    for kind in sorted(set(result["kinds"])):
        idx = [i for i, k in enumerate(result["kinds"]) if k == kind]
        out["by_kind"][kind] = {run: metrics([ranks[run][i] for i in idx]) for run in RUNS}
        out["by_kind"][kind]["n"] = len(idx)
    out["sign_test"] = {
        "fused_vs_dense": paired_sign(ranks["fused"], ranks["dense"]),
        "fused_vs_sparse": paired_sign(ranks["fused"], ranks["sparse"]),
        "sparse_vs_dense": paired_sign(ranks["sparse"], ranks["dense"]),
    }
    out["overlap_ratio_mean"] = (
        sum(result["overlap"]) / len(result["overlap"]) if result["overlap"] else 0.0
    )
    out["misses"] = {run: [i for i, rk in enumerate(ranks[run]) if rk is None] for run in RUNS}
    return out


def print_table(rep: dict[str, Any]) -> None:
    print(
        f"n = {rep['n']} golden questions, live alias; mean overlap ratio {rep['overlap_ratio_mean']:.2f}"
    )
    print(f"{'run':<8}{'Recall@5':>10}{'Recall@30':>11}{'MRR@10':>9}")
    for run in RUNS:
        m = rep["overall"][run]
        print(f"{run:<8}{m['recall@5']:>10.3f}{m['recall@30']:>11.3f}{m['mrr@10']:>9.3f}")
    for kind, runs in rep["by_kind"].items():
        print(
            f"  kind={kind} (n={runs['n']}): "
            + "  ".join(
                f"{run} R@5 {runs[run]['recall@5']:.2f} MRR {runs[run]['mrr@10']:.2f}"
                for run in RUNS
            )
        )
    for name, st in rep["sign_test"].items():
        print(
            f"sign test {name}: better {st['better']}, worse {st['worse']}, ties {st['ties']}, p = {st['p']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = parser.parse_args(argv)
    settings = load_settings()
    tracing.configure_tracing(settings)
    cases = list(read_jsonl(args.golden, EvalCase))
    with EmbeddingCache(settings.config_path.parent / settings.embed.cache_path) as cache:
        deps = {
            "client": r.qdrant_client(settings),
            "embedder": get_embeddings(settings),
            "sparse": get_sparse_embeddings(settings),
            "cache": cache,
        }
        try:
            rep = report(evaluate(cases, settings, deps))
        finally:
            deps["client"].close()
            tracing.shutdown_tracing()
    print_table(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
