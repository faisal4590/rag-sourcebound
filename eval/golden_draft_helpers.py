# Helper fields for eval/golden_v1.draft.jsonl: overlap and term_elsewhere. Issue #62, Level 3 prep.
"""Recompute `overlap` and `term_elsewhere` for every case in the golden draft and print a table.

Run: `uv run python eval/golden_draft_helpers.py [--show-sentences]`

- overlap: share of the question's content stems (the Qdrant/bm25 pipeline of Stage 6: lowercase,
  English stop words removed, Snowball stemmed) that also appear in its evidence quotes.
- term_elsewhere: chapters other than the question's own whose `data/clean` text contains
  `main_term` as a whole word, case-insensitive (`(?<!\\w)term(?!\\w)`, so `FFI` does not match
  `efficient`). `main_term` is read from the file and may be edited by hand.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from fastembed import SparseTextEmbedding
from fastembed.sparse.bm25 import remove_non_alphanumeric

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "eval" / "golden_v1.draft.jsonl"
CLEAN = ROOT / "data" / "clean" / "bca146b9df2d0a2b.jsonl"
SENTENCE_END = re.compile(r"(?<=[.!?:])\s+")


def term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def first_sentence_with(term: str, blocks: list[dict]) -> str:
    pat = term_pattern(term)
    for b in blocks:
        if pat.search(b["text"]):
            for sentence in SENTENCE_END.split(" ".join(b["text"].split())):
                if pat.search(sentence):
                    return f"p{b['page_printed']}: {sentence}"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-sentences", action="store_true", help="print the first sentence per term_elsewhere chapter")
    args = ap.parse_args()

    model = SparseTextEmbedding(load_settings().embed.sparse_model).model

    def stems(text: str) -> set[str]:
        return set(model._stem(model.tokenizer.tokenize(remove_non_alphanumeric(text))))

    blocks = [json.loads(line) for line in CLEAN.read_text().splitlines()]
    by_chapter: dict[int, list[dict]] = collections.defaultdict(list)
    for b in blocks:
        by_chapter[b["chapter_no"]].append(b)
    rows = [json.loads(line) for line in DRAFT.read_text().splitlines()]
    for r in rows:
        q = stems(r["question"])
        e = set().union(*(stems(x) for x in r["evidence_quotes"]))
        r["overlap"] = round(len(q & e) / len(q), 2) if q else 0.0
        pat = term_pattern(r["main_term"])
        r["term_elsewhere"] = sorted(
            ch for ch, bs in by_chapter.items() if ch != r["chapter_no"] and any(pat.search(b["text"]) for b in bs)
        )
    DRAFT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    print(f"{'id':5} {'ch':>2} {'kind':10} {'overlap':>7}  {'main_term':22} {'term_elsewhere':24} question")
    for r in sorted(rows, key=lambda r: -r["overlap"]):
        te = ",".join(f"{c:02d}" for c in r["term_elsewhere"]) or "-"
        print(f"{r['case_id']:5} {r['chapter_no']:02d} {r['kind']:10} {r['overlap']:7.2f}  {r['main_term'][:22]:22} {te[:24]:24} {r['question']}")
        if args.show_sentences:
            for ch in r["term_elsewhere"]:
                print(f"        ch{ch:02d} {first_sentence_with(r['main_term'], by_chapter[ch])[:200]}")
    kinds = sorted({r["kind"] for r in rows})
    print("\nmean overlap:", round(sum(r["overlap"] for r in rows) / len(rows), 3), "| by kind:",
          {k: round(sum(r["overlap"] for r in rows if r["kind"] == k) / sum(1 for r in rows if r["kind"] == k), 2) for k in kinds})


if __name__ == "__main__":
    main()
