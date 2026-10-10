"""Unit tests for the Level 3 retrieval metrics in eval/chunk_retrieval.py. Issue #62. No model runs."""

import sys
from pathlib import Path

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from flp_rag.contracts import Block, EvalCase
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.settings import load_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import chunk_retrieval as cr


@pytest.fixture(scope="module")
def enc():
    return tokenizer(load_settings().chunk)


def case(cid: str, question: str, quotes: list[str], kind: str = "definition") -> EvalCase:
    return EvalCase(cid, "answerable", question, None, [1], None, False, "v1", quotes, kind, 1)


DOCS = [
    cr.Doc("d:01:0000", "Readonly properties are set once. Later writes throw an Error.", (75, 76)),
    cr.Doc("d:01:0001", "Enums represent a collection of constant values; backed enums carry a value.", (115, 116)),
    cr.Doc("d:01:0002", "Match does strict type checks, like === instead of ==.", (145, 145)),
]


def test_recall_and_mrr() -> None:
    ranks = [1, 3, None, 12, 6]
    assert cr.recall_at(ranks, 5) == pytest.approx(2 / 5)
    assert cr.recall_at(ranks, 20) == pytest.approx(4 / 5)
    assert cr.mrr_at(ranks, 10) == pytest.approx((1 + 1 / 3 + 1 / 6) / 5)
    assert cr.recall_at([], 5) == 0.0 and cr.mrr_at([]) == 0.0


def test_first_hit_ignores_case_and_whitespace() -> None:
    normalized = [cr.normalize_text(d.text) for d in DOCS]
    rank, quote = cr.first_hit([2, 1, 0], normalized, ["BACKED   enums carry a value"])
    assert (rank, quote) == (2, "BACKED   enums carry a value")
    assert cr.first_hit([0, 1, 2], normalized, ["not in the book"]) == (None, None)


def test_evidence_share_is_quote_tokens_over_chunk_tokens(enc) -> None:
    share = cr.evidence_share("set once", "Readonly properties are set once. Later writes throw an Error.", enc)
    assert 0 < share < 0.5


def test_bm25_ranks_the_matching_document_first() -> None:
    bm25 = BM25Okapi([cr.bm25_tokens(d.text) for d in DOCS])
    assert cr.bm25_ranking(bm25, "What is a backed enum?")[0] == 1
    assert cr.bm25_ranking(bm25, "strict type checks in match")[0] == 2


def test_dense_ranking_orders_by_dot_product() -> None:
    dense = np.eye(3, dtype=np.float32)
    assert cr.dense_ranking(np.array([0.1, 0.9, 0.3], dtype=np.float32), dense) == [1, 2, 0]


def test_evaluate_and_aggregate_per_kind(enc) -> None:
    corpus = cr.Corpus("t", DOCS, np.eye(3, dtype=np.float32))
    cases = [case("a", "readonly?", ["set once"], "definition"), case("b", "enum?", ["backed enums"], "definition"),
             case("c", "match?", ["nothing here"], "comparison")]
    rankings = [[0, 1, 2], [2, 0, 1], [0, 1, 2]]
    results = cr.evaluate(cases, corpus, rankings, enc)
    assert [r.rank for r in results] == [1, 3, None]
    assert results[0].hit_doc == "d:01:0000" and results[2].hit_doc is None
    assert results[0].top[0] == {"doc_id": "d:01:0000", "pages": (75, 76)}
    agg = cr.aggregate(results)
    assert agg["recall@5"] == pytest.approx(2 / 3, abs=1e-4) and agg["no_hit"] == 1
    kinds = cr.by_kind(results)
    assert kinds["definition"]["recall@5"] == 1.0 and kinds["comparison"]["recall@5"] == 0.0
    assert kinds["comparison"]["evidence_share"] is None


def test_baseline_splitter_respects_the_token_budget(enc) -> None:
    blocks = [Block(f"d:001:{i:03d}", "paragraph", " ".join(f"word{j}" for j in range(150)), "P", 1, "C", "S", 1, -1, i, False)  # type: ignore[arg-type]
              for i in range(6)]
    texts = cr.baseline_texts(blocks, "cl100k_base")
    assert len(texts) >= 3 and all(len(enc.encode(t)) <= cr.BASELINE_CHUNK_TOKENS for t in texts)


def test_paired_counts_better_equal_worse_and_treats_a_miss_as_worst() -> None:
    a = [{"rank": 1}, {"rank": 3}, {"rank": None}, {"rank": 2}]
    b = [{"rank": 2}, {"rank": 3}, {"rank": 7}, {"rank": 1}]
    assert cr.paired(a, b) == {"better": 1, "equal": 1, "worse": 2, "sign_test_p": 1.0}
    assert cr.paired(b, a)["better"] == 2


def test_sign_test_matches_the_binomial_tails() -> None:
    assert cr.sign_test_p(7, 5) == pytest.approx(0.774, abs=0.001)
    assert cr.sign_test_p(5, 2) == pytest.approx(0.453, abs=0.001)
    assert cr.sign_test_p(3, 2) == 1.0
    assert cr.sign_test_p(10, 0) == pytest.approx(2 / 1024)
    assert cr.sign_test_p(0, 0) is None


def test_code_block_completeness_counts_whole_split_and_by_design(enc) -> None:
    code = "class A\n{\n    public int $x;\n}"
    huge = "\n".join(f"$v{i} = {i};" for i in range(400))
    blocks = [Block("d:001:000", "code", code, "P", 1, "C", "S", 1, -1, 0, True),  # type: ignore[arg-type]
              Block("d:001:001", "code", "echo 1;", "P", 1, "C", "S", 1, -1, 1, True),  # type: ignore[arg-type]
              Block("d:001:002", "code", huge, "P", 1, "C", "S", 1, -1, 2, True),  # type: ignore[arg-type]
              Block("d:001:003", "paragraph", "prose", "P", 1, "C", "S", 1, -1, 3, False)]  # type: ignore[arg-type]
    docs = [cr.Doc("c0", "Intro\n\n" + code, (1, 1)), cr.Doc("c1", "echo", (1, 1))]
    corpus = cr.Corpus("t", docs, np.eye(2, dtype=np.float32))
    assert cr.code_blocks_contained(blocks, corpus, enc, 900) == {"code_blocks": 3, "whole_in_one_chunk": 1, "split_by_design": 1, "split": 1}
    q = case("a", "show class A", ["public int $x;"], "code")
    assert cr.evidence_block(q, blocks).block_id == "d:001:000"
    assert cr.code_answers_whole([q], blocks, corpus, [[1, 0]], 5) == {"k": 5, "questions": 1, "whole_block_in_top_k": 1, "cases": ["a"]}
    assert cr.code_answers_whole([q], blocks, corpus, [[1, 0]], 1)["whole_block_in_top_k"] == 0
    assert cr.code_answers_whole([case("b", "prose?", ["prose"])], blocks, corpus, [[0, 1]], 5)["questions"] == 0
