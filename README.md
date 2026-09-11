# flp-rag

Question answering over the book *Front Line PHP* (PHP 8.2 edition). Answers come from the book only.
If the book does not contain the answer, the system returns the exact text `No information found`.

The full specification is `docs/front-line-php-rag-spec.md`. The architecture figure is `docs/rag-architecture-flowchart.png`.

## Start here

1. Copy `.env.example` to `.env`. Fill in one chat model key.
2. Run `uv sync` (or `pip install -e ".[dev]"`). Then run `make up`.
3. Do Milestone 0 from the spec, Section 11: one traced `hello` span visible at http://localhost:6006.

## Layout

| Path | Holds |
|---|---|
| `src/flp_rag/ingest/` | Stages 1-8, your own code |
| `src/flp_rag/graph/` | Stages 9-18 as a LangGraph, one node file per stage |
| `src/flp_rag/models.py` | The only file that names a model provider |
| `prompts/` | One file per prompt version |
| `eval/` | Golden dataset, smoke queries, harness, calibration |
| `data/raw/` | The PDF |
| `tests/` | One test file per "Done when" line |

Every Python file holds a one-line comment that names its stage and the spec section. No code exists yet.

## Milestones

M0 environment (1 evening) -> M1 parse and chunk (1 weekend) -> M2 embed and index (2 evenings) -> M3 answer and abstain (1 weekend) -> M4 hybrid, rerank, query understanding (1 weekend) -> M5 observability (1 weekend) -> M6 evaluation (1 weekend) -> M7 advanced (1-2 weekends).
