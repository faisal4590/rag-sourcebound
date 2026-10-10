# Level 3 retrieval test: dense and BM25 over the structural chunks and a flat-splitter baseline. Issue #62.
"""Retrieval test for the chunk set, issue #62 Level 3.

Run: `uv run python eval/chunk_retrieval.py [--golden eval/golden_v1.jsonl] [--no-baseline]`

For every golden question, rank every chunk two ways and record the rank of the first hit:
- dense: embed the question with the configured embedder (bge-m3, unit length, no prefix) and
  take the dot product with the dense vectors of `data/vectors` (the parquet file, no Qdrant);
- BM25: `rank_bm25.BM25Okapi` over lowercase word tokens of `display_text`.
A chunk is a hit when its normalized `display_text` (lowercase, whitespace collapsed) contains a
normalized evidence quote of the question. Metrics, overall and per question kind:
Recall@5, Recall@20, MRR@10 (0 when no hit in the top 10), and evidence share (tokens of the
matched quote / tokens of the first hit chunk, mean over questions with a hit).

Baseline: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(cl100k_base, 400, 50)` over the
clean text of the whole book, blocks joined with blank lines, embedded with the same model.
Same questions, same metrics. Paired per-question rank comparisons carry a two-sided sign test
p-value (ties dropped). Answer completeness counts the clean code blocks that sit whole inside one
chunk, and the code questions whose top-5 holds the whole block of the evidence line.
Results go to `eval/runs/chunks_<date>_<chunk_config_hash[:8]>.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from rank_bm25 import BM25Okapi

from flp_rag.contracts import Block, Chunk, EvalCase, read_jsonl
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.ingest.s08_verify import normalize_text
from flp_rag.models import Embedder, get_embeddings
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.tracing import git_sha

ROOT = Path(__file__).resolve().parents[1]
RECALL_KS = (5, 20)
MRR_K = 10
TOP_N_RECORDED = 3
BASELINE_CHUNK_TOKENS = 400  # issue #62 Level 3 baseline definition
BASELINE_OVERLAP_TOKENS = 50
WORD = re.compile(r"\w+")


# --------------------------------------------------------------------------- corpus


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    pages: tuple[int, int] | None  # printed pages, None for baseline chunks


@dataclass(frozen=True)
class Corpus:
    name: str
    docs: list[Doc]
    dense: np.ndarray  # (n_docs, dim), unit length rows

    @property
    def normalized(self) -> list[str]:
        return [normalize_text(d.text) for d in self.docs]


def bm25_tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def structural_corpus(chunks: Sequence[Chunk], vectors_path: Path) -> Corpus:
    table = pq.read_table(vectors_path).to_pydict()
    by_id = {cid: np.asarray(v, dtype=np.float32) for cid, v in zip(table["chunk_id"], table["dense"], strict=True)}
    docs = [Doc(c.chunk_id, c.display_text, (int(c.payload["page_printed_start"]), int(c.payload["page_printed_end"]))) for c in chunks]
    return Corpus("structural", docs, np.stack([by_id[c.chunk_id] for c in chunks]))


def baseline_texts(blocks: Sequence[Block], encoding_name: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name, chunk_size=BASELINE_CHUNK_TOKENS, chunk_overlap=BASELINE_OVERLAP_TOKENS
    )
    return splitter.split_text("\n\n".join(b.text for b in blocks))


def embed_cached(texts: Sequence[str], embedder: Embedder, cache: EmbeddingCache) -> np.ndarray:
    cached = cache.get_many(embedder.model_name, texts)
    missing = [t for t in texts if t not in cached]
    if missing:
        fresh = embedder.embed_passages(missing)
        cache.put_many(embedder.model_name, list(zip(missing, fresh, strict=True)))
        cached.update(dict(zip(missing, fresh, strict=True)))
    return np.asarray([cached[t] for t in texts], dtype=np.float32)


def baseline_corpus(blocks: Sequence[Block], embedder: Embedder, cache: EmbeddingCache, encoding_name: str) -> Corpus:
    texts = baseline_texts(blocks, encoding_name)
    docs = [Doc(f"base:{i:04d}", t, None) for i, t in enumerate(texts)]
    return Corpus("baseline", docs, embed_cached(texts, embedder, cache))


# --------------------------------------------------------------------------- ranking and metrics


def dense_ranking(query: np.ndarray, dense: np.ndarray) -> list[int]:
    return [int(i) for i in np.argsort(-(dense @ query), kind="stable")]


def bm25_ranking(bm25: BM25Okapi, question: str) -> list[int]:
    scores = np.asarray(bm25.get_scores(bm25_tokens(question)))
    return [int(i) for i in np.argsort(-scores, kind="stable")]


def first_hit(ranking: Sequence[int], normalized_docs: Sequence[str], quotes: Sequence[str]) -> tuple[int | None, str | None]:
    """1-based rank of the first document that contains a quote, and that quote."""
    norm_quotes = [normalize_text(q) for q in quotes]
    for rank, idx in enumerate(ranking, start=1):
        text = normalized_docs[idx]
        for raw, norm in zip(quotes, norm_quotes, strict=True):
            if norm in text:
                return rank, raw
    return None, None


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr_at(ranks: Sequence[int | None], k: int = MRR_K) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def evidence_share(quote: str, hit_text: str, enc: tiktoken.Encoding) -> float:
    return len(enc.encode(quote)) / max(1, len(enc.encode(hit_text)))


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    kind: str
    chapter_no: int | None
    rank: int | None
    hit_doc: str | None
    evidence_share: float | None
    top: list[dict[str, Any]]


def evaluate(cases: Sequence[EvalCase], corpus: Corpus, rankings: Sequence[Sequence[int]], enc: tiktoken.Encoding) -> list[CaseResult]:
    normalized = corpus.normalized
    out: list[CaseResult] = []
    for case, ranking in zip(cases, rankings, strict=True):
        rank, quote = first_hit(ranking, normalized, case.evidence_quotes)
        hit = corpus.docs[ranking[rank - 1]] if rank is not None else None
        share = evidence_share(quote, hit.text, enc) if hit is not None and quote is not None else None
        top = [{"doc_id": corpus.docs[i].doc_id, "pages": corpus.docs[i].pages} for i in ranking[:TOP_N_RECORDED]]
        out.append(CaseResult(case.case_id, case.kind, case.chapter_no, rank, hit.doc_id if hit else None, share, top))
    return out


def aggregate(results: Sequence[CaseResult]) -> dict[str, Any]:
    ranks = [r.rank for r in results]
    shares = [r.evidence_share for r in results if r.evidence_share is not None]
    return {
        "n": len(results),
        **{f"recall@{k}": round(recall_at(ranks, k), 4) for k in RECALL_KS},
        f"mrr@{MRR_K}": round(mrr_at(ranks), 4),
        "evidence_share": round(sum(shares) / len(shares), 4) if shares else None,
        "no_hit": sum(1 for r in ranks if r is None),
    }


def by_kind(results: Sequence[CaseResult]) -> dict[str, dict[str, Any]]:
    kinds = sorted({r.kind for r in results})
    return {k: aggregate([r for r in results if r.kind == k]) for k in kinds}


def sign_test_p(better: int, worse: int) -> float | None:
    """Two-sided exact sign test with ties dropped: the probability, under a fair coin, of a
    split at least as uneven as the observed one. None when every pair is a tie."""
    n = better + worse
    if n == 0:
        return None
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-question comparison of the rank of the first hit, system a against system b: how many
    questions a ranks better (lower rank), equal, and worse, plus the sign test p-value. A miss
    counts as the worst rank."""
    counts = {"better": 0, "equal": 0, "worse": 0}
    for ra, rb in zip(a, b, strict=True):
        x = ra["rank"] if ra["rank"] is not None else float("inf")
        y = rb["rank"] if rb["rank"] is not None else float("inf")
        counts["better" if x < y else "equal" if x == y else "worse"] += 1
    return {**counts, "sign_test_p": sign_test_p(counts["better"], counts["worse"])}


# --------------------------------------------------------------------------- code-block completeness


def code_blocks_contained(blocks: Sequence[Block], corpus: Corpus, enc: tiktoken.Encoding, code_max_tokens: int) -> dict[str, int]:
    """How many clean code blocks sit whole inside at least one chunk of the corpus. A code block
    over `code_max_tokens` is split into a code group by design and is counted apart."""
    normalized = corpus.normalized
    total = whole = by_design = 0
    for b in blocks:
        if b.type != "code":
            continue
        total += 1
        if len(enc.encode(b.text)) > code_max_tokens:
            by_design += 1
            continue
        if any(normalize_text(b.text) in t for t in normalized):
            whole += 1
    return {"code_blocks": total, "whole_in_one_chunk": whole, "split_by_design": by_design,
            "split": total - whole - by_design}


def evidence_block(case: EvalCase, blocks: Sequence[Block]) -> Block | None:
    """The clean code block that holds one of the question's evidence quotes."""
    for b in blocks:
        if b.type == "code" and any(normalize_text(q) in normalize_text(b.text) for q in case.evidence_quotes):
            return b
    return None


def code_answers_whole(cases: Sequence[EvalCase], blocks: Sequence[Block], corpus: Corpus, rankings: Sequence[Sequence[int]], k: int) -> dict[str, Any]:
    """For the questions whose evidence is a code line: how many have a top-k chunk that holds
    the whole code block the line belongs to."""
    normalized = corpus.normalized
    whole: list[str] = []
    considered: list[str] = []
    for case, ranking in zip(cases, rankings, strict=True):
        block = evidence_block(case, blocks)
        if block is None:
            continue
        considered.append(case.case_id)
        target = normalize_text(block.text)
        if any(target in normalized[i] for i in ranking[:k]):
            whole.append(case.case_id)
    return {"k": k, "questions": len(considered), "whole_block_in_top_k": len(whole), "cases": whole}


# --------------------------------------------------------------------------- run


def run_system(name: str, cases: Sequence[EvalCase], corpus: Corpus, rankings: Sequence[Sequence[int]], enc: tiktoken.Encoding) -> dict[str, Any]:
    results = evaluate(cases, corpus, rankings, enc)
    return {"name": name, "overall": aggregate(results), "by_kind": by_kind(results), "cases": [asdict(r) for r in results]}


def query_vectors(cases: Sequence[EvalCase], embedder: Embedder, cache: EmbeddingCache) -> np.ndarray:
    vectors = []
    for case in cases:
        v = cache.get_query(embedder.model_name, case.question)
        if v is None:
            v = embedder.embed_query(case.question)
            cache.put_query(embedder.model_name, case.question, v)
        vectors.append(v)
    return np.asarray(vectors, dtype=np.float32)


def run_all(
    settings: Settings,
    cases: Sequence[EvalCase],
    chunks: Sequence[Chunk],
    blocks: Sequence[Block],
    vectors_path: Path,
    *,
    baseline: bool,
    no_header_ablation: bool = False,
) -> dict[str, Any]:
    enc = tokenizer(settings.chunk)
    embedder = get_embeddings(settings)
    structural = structural_corpus(chunks, vectors_path)
    systems: list[dict[str, Any]] = []
    with EmbeddingCache(settings.config_path.parent / settings.embed.cache_path) as cache:
        queries = query_vectors(cases, embedder, cache)
        corpora = [structural]
        if baseline:
            corpora.append(baseline_corpus(blocks, embedder, cache, settings.chunk.tokenizer))
        if no_header_ablation:
            # Same chunks, embedded from display_text alone: what the chunk header contributes.
            ablation = Corpus("structural_no_header", structural.docs, embed_cached([d.text for d in structural.docs], embedder, cache))
    completeness: dict[str, Any] = {"corpus": {}, "questions": {}}
    code_max = settings.chunk.code_max_tokens
    for corpus in corpora:
        bm25 = BM25Okapi([bm25_tokens(d.text) for d in corpus.docs])
        dense_r = [dense_ranking(q, corpus.dense) for q in queries]
        bm25_r = [bm25_ranking(bm25, c.question) for c in cases]
        systems.append(run_system(f"{corpus.name}_dense", cases, corpus, dense_r, enc))
        systems.append(run_system(f"{corpus.name}_bm25", cases, corpus, bm25_r, enc))
        completeness["corpus"][corpus.name] = code_blocks_contained(blocks, corpus, enc, code_max)
        completeness["questions"][f"{corpus.name}_dense"] = code_answers_whole(cases, blocks, corpus, dense_r, RECALL_KS[0])
        completeness["questions"][f"{corpus.name}_bm25"] = code_answers_whole(cases, blocks, corpus, bm25_r, RECALL_KS[0])
    if no_header_ablation:
        abl_r = [dense_ranking(q, ablation.dense) for q in queries]
        systems.append(run_system("structural_no_header_dense", cases, ablation, abl_r, enc))
        completeness["questions"]["structural_no_header_dense"] = code_answers_whole(cases, blocks, ablation, abl_r, RECALL_KS[0])
    named = {s["name"]: s for s in systems}
    pairs: dict[str, dict[str, int]] = {}
    if baseline:
        for mode in ("dense", "bm25"):
            pairs[f"structural_vs_baseline_{mode}"] = paired(named[f"structural_{mode}"]["cases"], named[f"baseline_{mode}"]["cases"])
    if no_header_ablation:
        pairs["structural_vs_no_header_dense"] = paired(named["structural_dense"]["cases"], named["structural_no_header_dense"]["cases"])
    return {
        "paired": pairs,
        "answer_completeness": completeness,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "chunk_config_hash": settings.chunk_config_hash,
        "git_sha": git_sha(),
        "embed_model": settings.embed.model,
        "n_cases": len(cases),
        "corpora": {c.name: {"n_docs": len(c.docs)} for c in corpora},
        "baseline": {"splitter": "RecursiveCharacterTextSplitter.from_tiktoken_encoder", "encoding": settings.chunk.tokenizer,
                     "chunk_size": BASELINE_CHUNK_TOKENS, "chunk_overlap": BASELINE_OVERLAP_TOKENS},
        "systems": systems,
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"chunk_retrieval: {report['n_cases']} questions, hash {report['chunk_config_hash']}, "
          + ", ".join(f"{n} {c['n_docs']} docs" for n, c in report["corpora"].items()))
    print(f"  {'system':18} {'kind':11} {'n':>3} {'R@5':>6} {'R@20':>6} {'MRR@10':>7} {'evid':>6} {'miss':>4}")
    for s in report["systems"]:
        rows = [("all", s["overall"])] + sorted(s["by_kind"].items())
        for kind, m in rows:
            print(f"  {s['name']:18} {kind:11} {m['n']:3d} {m['recall@5']:6.2f} {m['recall@20']:6.2f} {m['mrr@10']:7.2f} "
                  f"{(m['evidence_share'] if m['evidence_share'] is not None else 0):6.2f} {m['no_hit']:4d}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden_v1.jsonl")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-header-ablation", action="store_true", help="also embed display_text without the chunk header")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    settings = load_settings()
    doc_id = args.doc_id or min((args.data_dir / "chunks").glob("*.jsonl")).stem
    cases = [c for c in read_jsonl(args.golden, EvalCase) if c.evidence_quotes]
    chunks = list(read_jsonl(args.data_dir / "chunks" / f"{doc_id}.jsonl", Chunk))
    blocks = list(read_jsonl(args.data_dir / "clean" / f"{doc_id}.jsonl", Block))
    report = run_all(settings, cases, chunks, blocks, args.data_dir / "vectors" / f"{doc_id}.parquet",
                     baseline=not args.no_baseline, no_header_ablation=args.no_header_ablation)
    print_summary(report)
    for name, counts in report["paired"].items():
        print(f"  paired {name}: better {counts['better']}, equal {counts['equal']}, worse {counts['worse']}, sign test p={counts['sign_test_p']}")
    for name, c in report["answer_completeness"]["corpus"].items():
        print(f"  code blocks {name}: {c['whole_in_one_chunk']}/{c['code_blocks']} whole in one chunk, {c['split_by_design']} split by design, {c['split']} split")
    for name, c in report["answer_completeness"]["questions"].items():
        print(f"  code questions {name}: whole block in top {c['k']}: {c['whole_block_in_top_k']}/{c['questions']}")
    out = args.out or (ROOT / "eval" / "runs" / f"chunks_{report['date'].replace('-', '')}_{settings.chunk_config_hash[:8]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"  report -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
