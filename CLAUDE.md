# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

`flp-rag` answers questions about PHP 8 from one book only: *Front Line PHP* (Brent Roose, PHP 8.2
edition, `data/raw/front-line-php-revised-for-php-82.pdf`). Every answer cites printed page numbers.
If the book does not contain the answer, the system returns the exact text `No information found`.

The full specification is `docs/front-line-php-rag-spec.md`. It is the source of truth. Read the
relevant stage section before writing or changing a stage. Each stage there follows one template:
Purpose, Input and output, Rules, Configuration, Trace span, Done when.

**Current state:** scaffold only. Every Python file under `src/`, `eval/`, and `tests/` holds one
comment line that names its stage and spec section. No implementation exists yet. Milestone 0 is next.

## Hard rules (from the spec)

1. **Abstain text is exact.** `ABSTAIN_TEXT = "No information found"`. Case-sensitive, no period, no
   extra words. The CLI and API return the same string. Never return a partial answer with a
   disclaimer. Three stages may produce it: relevance gate (14), generator (16), output checks (17).
   Each records a reason code on the span (`query.smalltalk`, `gate.low_top_score`,
   `gate.no_passing_chunk`, `gen.model_abstained`, `gen.invalid_output`, `output.unfaithful`,
   `output.invalid_after_repair`).
2. **No constants in code.** Every tunable value comes from `config.yaml` via `settings.py`.
   API keys come from environment variables (`.env`), never from `config.yaml`.
3. **One span per stage.** Every stage opens one OpenTelemetry span with the name in its
   "Trace span" line, through the `@stage` decorator in `tracing.py`. Business code never imports
   the tracer directly. Errors are recorded on the span and re-raised. Never swallow.
4. **Every stage has a "Done when" test** in `tests/`. Write it before the implementation (TDD).
5. **Never edit a prompt file in place.** Copy `prompts/answer_v1.md` to `answer_v2.md`, then change
   `config.yaml`. `prompt_version = sha256(file)[:8]` goes on every trace.
6. **Never build a partial index.** If any chunk has no vector after retries, fail the run. The
   Qdrant alias `flp_chunks_current` switches only after Stage 8 verification passes.
7. **Cite printed pages**, not PDF pages. `printed = pdf - 2`.
8. **Chapter numbers come from PDF bookmarks**, never from the printed `CHAPTER NN` label (the
   Type Variance chapter is mislabeled 21 in print; it is 25).
9. **Never split a code block** across chunks. Never re-wrap, spell-correct, or re-quote code.
10. **Pin model names** in `config.yaml`. The manifest and every trace carry them.

## Commands

```bash
uv sync --extra dev --extra local   # or: pip install -e ".[dev,local]"
cp .env.example .env         # fill in one chat model key
make up                      # docker compose: Qdrant :6333, Phoenix :6006
make up-local                # same services without Docker: .local/qdrant binary + `phoenix serve`
make status                  # both services answer?
make down / make down-local
make ingest                  # flp ingest -> Stages 1-8, prints verification report
make serve                   # uvicorn flp_rag.api.app:app --reload --port 8000
make test                    # pytest -q  (the "Done when" tests)
make eval                    # flp eval -> harness, compares eval/baseline.json, exit 1 on regression
make calibrate               # flp calibrate -> threshold sweep, spec 8.4
ruff check src tests eval    # lint, line length 100, py312
pytest tests/test_s04_chunk.py -q   # one stage's test
flp ask "What is a readonly property?"
```

Python 3.12 required. `uv` is installed at `~/.local/bin/uv`. Docker is **not** installed on the dev
machine, so `make up-local` is the working path. `.local/` (Qdrant binary, storage, Phoenix data,
pid files) is gitignored. Run Python through `uv run ...`.

## Architecture

Three planes. Stage numbers are the spec's numbers and appear in file names and comments.

| Plane | Stages | Where |
|---|---|---|
| Offline ingestion, run once per PDF version | 1-8 | `src/flp_rag/ingest/` |
| Online query, run once per request | 9-19 | `src/flp_rag/graph/` + `src/flp_rag/api/` |
| Observability and evaluation, always on | spec 7, 8 | `tracing.py`, `eval/` |

**Ingestion** (`ingest/run.py` runs them in order): `s01_parse` (pdfplumber words with font, size,
position; delete 8pt words and `top < 50` to strip running headers) -> `s02_structure` (group into
typed blocks by font: `chapter_title`, `heading`, `callout`, `code`, `paragraph`) -> `s03_clean`
(NFC, nbsp, zero-width, whitespace, page-number lines) -> `s04_chunk` (structure-aware, 120-600
tokens, code blocks whole, parent per section) -> `s05_enrich` (payload fields + `php_symbols`) ->
`s06_s07_index` (bge-m3 dense + Qdrant/bm25 sparse into a versioned Qdrant collection, SQLite
parent store, `manifest.json`) -> `s08_verify` (coverage, duplicates, 10 smoke queries, alias switch).

Each stage writes `data/{stage}/{doc_id}.jsonl`. A stage reads only the file of the stage before it,
so one stage can be rerun without the PDF.

**Query** (a LangGraph `StateGraph` in `graph/build.py`, one node per stage in `graph/nodes/`):
`guard_input` (10) -> `understand` (11) -> `retrieve` (12, hybrid dense + sparse, RRF k=60) ->
`rerank` (13, bge-reranker-v2-m3, top 8) -> `gate` (14, conditional edge: `pass` or `abstain`) ->
`assemble` (15, parents, `[S1]` labels, 6000-token budget, book order) -> `generate` (16) ->
`check_output` (17, citations, faithfulness judge, code verbatim) -> `respond` (18).
`graph/cache.py` is Stage 19 (Milestone 7). `graph/state.py` holds `RagState`.

**Shared:** `contracts.py` (dataclasses from spec Section 10, each with `to_attrs()`),
`settings.py` (loads `config.yaml`, computes `config_hash`), `models.py` (the only file that names a
model provider: `get_chat_model()`, `get_embeddings()`, `get_reranker()`), `stores/` (SQLite
embedding cache and parent store), `cli.py` (Typer: `flp ask|ingest|eval|calibrate`).

### Spec vs scaffold divergences

The spec (Section 4.1) says "no orchestration framework in the core". The scaffold chose LangGraph
plus LangChain components for the online pipeline. Keep that choice. Keep each node a thin,
plain-function stage so traces stay readable, and keep the spec's span names and attributes
regardless of what LangChain auto-instrumentation emits. Stages 6 and 7 live in one file
(`s06_s07_index.py`). Node files are named by function, not stage number.

## Key defaults (see `config.yaml` for all)

| Concern | Value |
|---|---|
| Embedding | `BAAI/bge-m3`, 1024 dims, local CPU, batch 32, SQLite cache |
| Sparse | `Qdrant/bm25` via fastembed |
| Reranker | `BAAI/bge-reranker-v2-m3`, sigmoid to 0-1, top 8 |
| Gate thresholds | `tau_low 0.20`, `tau_pass 0.35`, `tau_high 0.60` (recalibrate per spec 8.4) |
| Chunks | target 350 tokens, min 120, max 600, code max 900, `cl100k_base` |
| Generator | temperature 0, `max_output_tokens 800`, `prompts/answer_v1.md` |
| Tracing | OTLP/HTTP to Phoenix at `http://localhost:6006/v1/traces`, `service.name = flp-rag` |
| Qdrant | `flp_chunks_{index_version}`, alias `flp_chunks_current`, named vectors `dense` + `sparse_bm25` |

`gen.model`, `query.model`, and `output.judge_model` are placeholders (`<generator-model-name>`).
Set them before Milestone 3.

## Tracing conventions

- Root span per request: `rag.request` (kind `CHAIN`). One child per stage. Span tree in spec 7.4.
- Ingestion root: `ingest.run`. Children `ingest.parse` ... `ingest.verify`. Spec 7.5.
- Attribute namespaces: `rag.*`, `guard.*`, `cache.*` (project), `gen_ai.*` (OTel GenAI
  conventions on every model call), `openinference.span.kind` and `retrieval.documents.{i}.*`
  (OpenInference, Phoenix renders them). Full catalog in spec 7.6.
- Every span gets `rag.latency_ms` from the decorator. Every `LLM` span gets `rag.cost_usd` from
  the price table in `config.yaml`. The root sums them.
- Logs: structlog JSON to stdout, `trace_id` and `span_id` injected. One `INFO` line per request.
- A request that abstains at the gate has no `context.assemble`, `llm.generate`, or
  `guardrails.output` span. The absence is the signal.

## Evaluation

- `eval/golden_v1.jsonl`: target 100 cases (60 answerable, 20 in-domain unanswerable, 20
  out-of-domain). Currently 7 example cases. `eval/smoke_queries.jsonl`: 10 queries used by Stage 8.
- `eval/baseline.json` is empty (`{}`). Create it after the first full run (Milestone 6).
- Targets: Recall@5 >= 0.85, Abstain F1 >= 0.90, Faithfulness >= 0.95, Citation accuracy >= 0.90,
  p95 latency <= 6 s, cost <= USD 0.02 per answered question, trace coverage 100%.
- A hit is a chunk whose printed page range intersects the expected pages, tolerance 1 page.
- Before adding an "in-domain unanswerable" case, grep `data/clean/` for the key term. It must be absent.

## Canonical test fixtures

- **PDF page 66** (printed 64): the `class CustomerDTO` example. It must be one `code` block with
  correct indentation, land in exactly one chunk, and list `customerdto` in `php_symbols`.
  `grep -c "class CustomerDTO" data/chunks/*.jsonl` must return 1.
- **Chapter 6** (Readonly Properties, printed 75-84): `How do readonly properties work?` must hit
  Chapter 6 in the top 10 of both dense and sparse lists.
- `hi there` -> `No information found`, reason `query.smalltalk`.
- `What is the capital of France?` -> `No information found`.
- After Stage 1, the string `Front Line PHP` must appear zero times in `data/parsed/`.
- `rag.chapters_found` after Stage 2 must equal 33 (Foreword, Preface, 30 chapters, In Closing).

## Milestones

M0 environment (hello span in Phoenix) -> M1 parse and chunk (250-350 chunks, CustomerDTO test) ->
M2 embed and index (`VERIFY OK: 10/10 smoke queries hit`) -> M3 answer and abstain (FastAPI, CLI,
gate, output checks; reranker stubbed as `rerank_score = fused_score`) -> M4 hybrid, rerank, query
understanding -> M5 full observability -> M6 evaluation harness and calibration -> M7 advanced
(contextual summaries, hypothetical questions, decomposition, semantic cache, sandwich order).
Do not start a milestone before the previous "Done when" line passes. Details in spec Section 11.

## Working conventions

- Python 3.12, `ruff` with line length 100. Type hints on every function. Dataclasses for contracts,
  pydantic for settings and API schemas.
- One stage is one function with a typed input and typed output. Keep files under 400 lines.
- Prefer immutable updates: return new records, do not mutate inputs.
- `data/` and `eval/runs/` are gitignored except `.gitkeep`. The PDF in `data/raw/` is local only.
- `.env` is never committed. `.env.example` lists the expected keys.
- Debugging a wrong answer: follow the runbook in spec 7.11, starting from the `trace_id` in the
  response. Then add the case to the golden dataset and rerun `make eval`.
- Commit messages: `<type>: <description>` with types feat, fix, refactor, docs, test, chore, perf, ci.
