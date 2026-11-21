# Issue #82: the page-based hit flags of eval/rerank_eval.py, without the live index.
"""Precision@5 after fusion and after the reranker uses the harness rule: a chunk hits when its
printed page range, widened by the tolerance, meets an expected page."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flp_rag.contracts import Candidate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import rerank_eval

PAGES = {"d:5:1": (64, 65), "d:5:2": (66, 66), "d:6:1": (75, 79), "d:9:1": (119, 120)}


def cand(chunk_id: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, fused_score=0.5, rerank_score=None)


def test_page_flags_follow_the_harness_hit_rule() -> None:
    ranked = [cand("d:6:1"), cand("d:5:1"), cand("d:9:1"), cand("d:5:2"), cand("unknown")]
    flags = rerank_eval.page_flags(ranked, PAGES.get, [64], tolerance=1)
    assert flags == [False, True, False, False, False]
    assert rerank_eval.page_flags(ranked, PAGES.get, [66], tolerance=1) == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert rerank_eval.page_flags([], PAGES.get, [64], tolerance=1) == []


def test_precision5_rows_average_over_the_answerable_cases_only() -> None:
    rows = [
        {
            "abstain": False,
            "group": "answerable",
            "precision5_fused": 0.2,
            "precision5_reranked": 0.4,
        },
        {
            "abstain": False,
            "group": "answerable",
            "precision5_fused": 0.0,
            "precision5_reranked": 0.2,
        },
        {
            "abstain": True,
            "group": "out_of_domain",
            "precision5_fused": None,
            "precision5_reranked": None,
        },
    ]
    fused, reranked = rerank_eval.precision5_means(rows)
    assert fused == pytest.approx(0.1) and reranked == pytest.approx(0.3)
    assert rerank_eval.precision5_means([]) == (0.0, 0.0)
