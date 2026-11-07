# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

`flp-rag` answers questions about PHP 8 from one book only: *Front Line PHP* (Brent Roose, PHP 8.2
edition, `data/raw/front-line-php-revised-for-php-82.pdf`). Every answer cites printed page numbers.
If the book does not contain the answer, the system returns the exact text `No information found`.

The full specification is `docs/front-line-php-rag-spec.md`. It is the source of truth. Read the
relevant stage section before writing or changing a stage. Each stage there follows one template:
Purpose, Input and output, Rules, Configuration, Trace span, Done when.

**Current state:** Milestone 2 complete, Milestone 3 in progress. Ingestion Stages 1-8 under
`ingest/`; `make ingest` verifies and switches the alias. Query side: `graph/state.py` (`RagState`,
`initial_state`), `graph/build.py` (`build_graph`, `GraphNodes`, `Deps`, `default_nodes`,
`compile_default`; every node runs under its spec span, routing after `understand` and `gate`),
Stage 12 hybrid retrieval in `graph/nodes/retrieve.py`, `flp retrieve` (dense listing only), and the
Stage 9 FastAPI app in `api/app.py` (`create_app`, `POST /ask` with SSE, `POST /feedback` stub,
`GET /health`; `make serve` runs it on :8000). Every response carries `trace_id` and the
`X-Trace-Id` header; a middleware opens `rag.request` per request, unhandled errors answer 500
(501 for a `NotImplementedError` placeholder) inside that span, timeouts answer 504. Stage 10 input guardrails in `graph/nodes/guard_input.py` (length check raises `InputRejected`,
which the API answers as HTTP 400 `question must be 1-2000 characters`; injection regexes and the
base64 run length come from `guard.*` in `config.yaml`; language via `py3langid`, questions under
`guard.language_min_chars` assumed English). Stage 14 relevance gate in `graph/nodes/gate.py`:
`decide(scores, settings.gate)` is a plain function of the rerank scores and `gate.tau_*`
(strictly ordered, checked at load), zones per the spec table with each zone including its
lower edge, reasons `gate.no_candidates`, `gate.low_top_score`, `gate.no_passing_chunk`,
`gate.borderline`, `gate.pass`; a score outside 0..1 or a candidate without `rerank_score` is
an error. The node writes the Part 4 span attributes on every request, hands only the passing
candidates and `n_pass` on, and puts the decision in the response `debug` field when the request
has `debug: true`. Until #26 the rerank stub feeds the dense cosine clamped to 0..1, so the
thresholds are on the wrong scale and must be recalibrated when the reranker lands. Stage 15
context assembly in `graph/nodes/assemble.py`: `assemble(candidates, store, settings, enc)`
selects by `gate.tau_pass` and `context.max_blocks`, groups by `parent_id`, fills
`context.max_tokens` in score order (whole parent, else the passing chunks, else the best
chunk, else skipped), sorts by chapter and printed page, labels `[S1]`..., keeps code in fenced
`php` blocks, and writes `context` plus `label_map` to the state and the seven
`context.assemble` attributes. The parent store gained a `blocks` table (clean blocks per chunk,
in order) that Stage 7 fills from `data/clean`; without it the context falls back to plain
text. `Deps.store_path` names the store; the API sets it from the index version behind the
alias. `context.order` accepts only `position` until Milestone 7. Stage 16 generation in
`graph/nodes/generate.py`: `generate(question, context, borderline, chat, settings, prompts)`
builds the system message from the rules above the `Context blocks:` marker of the prompt file
and the user message from the blocks and the `<question>`-wrapped question (tags in the user
text removed first); `gen.strict_prompt_file` for borderline requests and every repair call;
`prompt_version = sha256(file)[:8]` and `prompts/registry.json` pins every prompt file (a test
fails on an in-place edit). Output forms: the exact abstain text (whitespace, quotes, emphasis,
and a final period tolerated) -> `gen.model_abstained`; an `[Sn]` marker -> answered; anything
else or a cut answer -> one repair call (`gen.repair_prompt_file`, budget `gen.max_repair_calls`
shared with Stage 17 through `state.repair_calls`) then `gen.invalid_output`. A transport error
is retried once, then `GenerationUnavailable` (HTTP 500 with the trace id, never the abstain
text). The `llm.generate` span carries `gen_ai.*`, `rag.prompt_version`, `rag.cost_usd`,
`rag.repair_calls`. The model comes from `Deps.chat` or `get_chat_model(settings, "gen")` on
first use; a placeholder model name fails with `ModelConfigError` on the first request
that reaches the stage. Stage 17 output checks in `graph/nodes/check_output.py`:
`check_output(answer, label_map=, context=, borderline=, request_id=, repair_calls=, settings=,
judge=, prompts=, judge_all=)` returns an `Outcome` with `result` `ok` / `repair` / `abstain`.
Order: early stop on the exact abstain text (Stage 16's reason stays) -> mixed abstain (any
letter case) -> citations (`check.citations` span; one unknown `[Sn]` fails) -> code
(`check.code` span; every fenced block in the answer must be part of one fenced block of the
context after normalization: trimmed lines, no empty lines, one space per run, `...` for `…`;
a context without fences falls back to whole block texts and sets
`rag.code_check_plain_text`) -> judge (`check.faithfulness` span, LLM; `output.judge_model`,
`output.judge_prompt_file`; always for borderline or `judge_all`, else a sha256 sample of the
request id at `output.faithfulness_sample_rate`). `supported: false` -> `output.unfaithful`,
no repair. Judge failure (transport after retry, or bad verdict JSON) -> `output.judge_unavailable`
when the request had to be judged, else the answer is delivered with `rag.judge_error`;
`ModelConfigError` propagates. A failed fixed check goes back to Stage 16 (`repair_failure`
in the state: the named line plus the rewrite note from `output.repair_prompt_file`, sections
`## mixed_abstain|citations|code|rewrite`) when `repair_calls < gen.max_repair_calls`, else
`output.invalid_after_repair`. `route_after_check` sends `repair` to `generate`; `generate()`
takes `repair_of` and `repair_note` and skips the first call. One marker parser for Stages
16-18 in `graph/markers.py` (`[S1]`, `[S1][S3]`, `[S1, S3]`; `(S1)` is none). Stage 18 response
in `graph/nodes/respond.py`: `build_response(state, settings, error=None)` is the one builder
for all three statuses (the API calls it with `error=` for 400, 504, 500, 501, so every error
has the same shape: status `error`, `api.error_text`, no sources, plus `detail`);
`sources_for` maps each distinct cited label to one `Source` in citation order with every
`chunk_ids` of the block (contract changed from `chunk_id`), the section titles of the
block's chunks joined with " + ", the printed pages, and the highest rerank score; a cited
label outside the label map raises `UnknownLabel`; `canonical_markers` turns `[S1, S3]` into
`[S1][S3]`. `summarize` + `log_request` write exactly one `request` INFO line per request
(request id, trace id, status, reason, latency, tokens, cost, versions, config hash,
n_sources, repair_calls, borderline; never the question or the answer). Timings come from the
graph wrapper (`timings_ms` per span name, merge reducer), token totals from the
model-calling nodes (`input_tokens`, `output_tokens`), `index_version` and `started_at` from
Stage 9 through `initial_state`. The debug field (only with `debug: true`) holds the reason,
the gate score and thresholds, the uncited blocks, repair count, failed check, timings, and the
path. `flp ask` is a thin client of the API (issue #24): it posts to `POST /ask` at `--api-url`,
then `FLP_API_URL`, then `cli.api_url`, with `cli.timeout_s` (must exceed `api.timeout_s`),
renders the JSON record (answer, sources table with label, chapter, section, printed pages,
score, the debug field with `--debug`, `trace_id=` last), and exits 0 answered, 2 no
information, 1 error or no API; `--stream` reads the SSE events and prints a withdrawn note
when `done` is not answered; `--json` prints the raw record; `--retrieve-only` runs
`flp retrieve`, the Milestone 2 chunk listing. Every default node is real; no stub is left.
`trace.instrument_langchain` (default false) adds OpenInference LangChain spans to the same
provider. `EmbeddingCache` is thread-safe (LangGraph runs sync nodes in a worker pool).
**Chunk set evaluated and frozen (issue #62):** 202 chunks, `chunk_config_hash d4ebab9ed10ca7e7`.
The hash covers the parse, structure, clean, and chunk config sections plus the source of
Stages 1-5, so any chunker change forces a new index version. Gates: `eval/chunk_checks.py`
(Level 1, exit 1 on failure), `eval/chunk_retrieval.py` (Level 3: dense and BM25 over the
parquet vectors against a flat 400/50 splitter baseline, metrics per question kind),
`eval/chunk_review.md` (Level 2 human read), `eval/golden_v1.jsonl` (100 cases since #33, 60 with
`evidence_quotes`, `kind`, `chapter_no`), report in `eval/reports/`. Rules that now differ from
the spec are listed in `docs/spec-changes-pending.md`; the spec itself is unchanged. Test
markers `live` (needs the Qdrant alias on the current index) and `slow` (PyMuPDF glyph oracle)
are excluded from the default run. The live index holds this set:
`make ingest` runs Stages 7 and 8; the alias points at `flp_chunks_v3-bgem3-d4ebab9e`.
Milestone 3 is complete (Groq models set, live loop verified). Milestone 4 in progress:
Stage 12 hybrid retrieval (#25) done: `retrieve_hybrid(variants, settings, filters=, client=,
embedder=, sparse=, cache=, collection=)` runs one dense batch (`embed_queries` through the
query cache, one `query_batch_points`) and one sparse batch (`FastembedSparse.embed_query`,
the BM25 query mode with a weight of 1 per term; Qdrant adds the IDF) over `Variants`
(`variants_from(standalone, question)`: standalone + paraphrases to dense, plus the
`keyword_variant` to sparse only), fuses with RRF (`fuse(lists, rrf_k=, top_k=)`: 1/(k+rank)
summed, ties by best rank then chunk id), adds `code_group_id` siblings after the cut
(`add_code_group_siblings`), and writes `retrieval.hybrid` (`rag.n_variants`, `rag.n_dense_hits`,
`rag.n_sparse_hits`, `rag.n_fused`, `rag.n_siblings_added`, `rag.overlap_ratio`,
`rag.top_fused_score`, documents) with children `retrieval.dense`, `retrieval.sparse`,
`retrieval.fuse`. `Candidate.provenance` entries read `dense:v0:r3`; `Candidate.scores` holds
the best raw score per retriever and the rerank stub reads `scores["dense"]` until #26. An
empty sparse list is fine; a Qdrant or model failure raises. `Deps.sparse` carries the BM25
encoder. Part 8 on the live index (`eval/hybrid_retrieval.py`, 30 golden questions): Recall@5
and Recall@30 are 1.0 for dense, sparse, and fused; MRR@10 dense 0.859, sparse 0.925, fused
0.900; mean overlap 0.47. Stage 13 rerank (#26) done: `graph/nodes/rerank.py`,
`rerank(question, candidates, reranker=, store=, settings=, chapter_hint=)` builds one pair per
candidate (`pair_text`: the chunk header rebuilt from the payload, without the part title the
payload lacks, plus the display text from the parent store), scores with `Deps.reranker`
(`BgeReranker`: sigmoid on the logit exactly once, `rerank.max_length` 1024 with the model's
own tokenizer, `rerank.device` auto = cuda > mps > cpu, loaded under a lock and warmed at API
start), sorts by score then chunk id, runs one `apply_hint` pass (neighbors closer than
`rerank.hint_margin` swap when only the lower one is in the hinted chapter; scores never
change), keeps `rerank.top_n`, and writes `rag.scores_all` (chunk id, score pairs, best first),
`rag.score_top1`, `rag.score_top3_mean`, `rag.n_truncated`, `rag.hint_swaps`, `rag.device`,
`reranker.*`, `gen_ai.request.model` on the `rerank` span. Zero candidates: no model call. The
rerank stub is gone; the gate reads real cross-encoder scores. `eval/rerank_eval.py` (Part 8)
measures the gain over fusion, the score separation against the gate thresholds, and the
stage latency on every golden case (since #33 all 100; before, 30 answerable plus 10 off-topic).
Measured 2026-10-17 on the M1 Pro (MPS): answerable best scores 0.72-1.00 (median 0.98),
off-topic 0.001-0.018, so the default gate thresholds hold without recalibration; Recall@3 after
the reranker 1.0; MRR@10 0.900 (fused) -> 0.872 (rank 1 and 2 swaps); rerank latency about 5-10 s
per request for 30 pairs of 300-700 tokens, far over the 1.5 s budget and the cause of 20 s API
timeouts. Decided 2026-11-07 (#76): the reranker is `BAAI/bge-reranker-base` with
`rerank.max_length` 512 (the model's limit; `BgeReranker` refuses a larger value at load with
`ModelConfigError`). Same 40 questions, same machine: rerank p50 1.3 s (v2-m3 at 1024: 6.1 s,
at 512: 4.1 s), MRR@10 0.911 (26 rank-1 against 23), Recall@3 0.933 (g012 and g022 drop to
rank 4, both still in the top 8 and in the context), Recall@8 1.0, answerable best scores
0.40-1.00 (two in the borderline zone, g022 and g024, so the judge runs for them), off-topic
at most 0.001. `eval/rerank_eval.py --rerank-model/--max-length/--top-k-fused` trial a setting
without editing `config.yaml`; reports in `eval/runs/rerank_20261107_*.json`. Stage 11 query
understanding (#27) done: `graph/nodes/understand.py`, `understand(question, history, chat=,
prompt=, facts=, settings=)` makes one small-model call (`query.model`, `query.prompt_file` =
`prompts/query_v1.md` split at `Chapters:` into system and user parts, `QUERY_SCHEMA`,
temperature `query.temperature`, `query.max_output_tokens`, `query.reasoning_effort`) that
rewrites, classifies, and expands at once. `IndexFacts.from_store(store_path)` reads the
chapter list (`ParentStore.chapters()`, in book order, 33 entries) and the `php_symbols` set
of the live index; the chapter list goes into the prompt and bounds `chapter_hint`. Checks
(Part 3): no history -> the standalone question is the raw question exactly; with history ->
the rewrite when it has at most `query.max_standalone_words` (50) words; unknown intent ->
`book_question`; `wants_code` must be a bool; paraphrases dropped when empty, equal to the
standalone question, duplicates (case-insensitive), not text, over `query.expansions`, or
(with `query.keep_question_symbols`) when they lose a `php_symbols` word the question names;
empty keywords -> no `keyword_variant`. Smalltalk abstains (`query.smalltalk`, before
retrieval) only under `smalltalk_policy: abstain` and at most `query.smalltalk_max_words` (8)
words. A transport error or a non-JSON output is retried once after `query.retry_backoff_s`,
then the fallback (raw question, `book_question`, one variant) with
`rag.understand_fallback = true`; `ModelConfigError` and auth errors propagate. The session
store is new (`stores/session_store.py`, `session.store_path` = `data/cache/sessions.sqlite`,
`session.ttl_hours` 24, `session.answer_excerpt_chars` 300): Stage 18 (`record_turn`) writes
one row per request with a session id (standalone question, status, answer excerpt, time),
also for an abstain; Stage 11 reads the last `query.history_turns` rows into `state.history`
(`history_from`) and tags the history and the question (`strip_tags` first). `Deps.sessions`
and `Deps.query_chat` carry the store and the small model; the API builds the store and closes
it; a store error at Stage 18 is recorded (`rag.session_turn` = `written`, `skipped`, or the
error class) and logged, and the answer is still delivered. Spans: `query.understand`
(`rag.question_raw`, `rag.question_standalone`, `rag.variants` under `capture_content`,
`rag.intent`, `rag.wants_code`, `rag.chapter_hint`, `rag.n_variants`,
`rag.history_turns_used`, `rag.understand_fallback`, `rag.paraphrases_dropped`,
`rag.cost_usd`) with the child `query.classify` (LLM: `gen_ai.*`, `rag.prompt_version`,
`rag.rewrite_used`, `rag.attempts`, `rag.error` as a short code, never model text). An
`InvalidOutput` from `LangChainChat.complete` carries the `Completion`, so the tokens of a
rejected JSON answer still count. Live: "What about
enums?" after the readonly question rewrites to a standalone enum question; "hi there" abstains
with `query.smalltalk`. Issue #28 (the Milestone 4 close): three switches reproduce the
Milestone 3 pipeline on the same code and index, `query.enabled`, `retrieval.sparse_enabled`,
`rerank.enabled` (false = the raw question only / dense lists only / the dense cosine as the
score, the old stub); each side has its own `config_hash`. The request field `run_tag` (with
`debug: true` only) lands on the root span as `rag.run_tag` and in the debug field, which
also holds `rerank.top` (kept candidates with chapter and score), `cost_usd`, and the token
totals. `eval/smoke_run.py --tag <tag> --repeats 3 --pause 55` runs the smoke queries with a
warm-up and writes `eval/runs/smoke_<tag>_<date>.json`; the pause is needed on the Groq free
tier (8,000 tokens per minute for the answer model; one answered request uses about 6,000,
so back-to-back requests get 429s, and the generation retry does not wait for the
`Retry-After`). The comparison is `docs/notes/m4-before-after.md`; git tags `m3` and `m4`
mark the two sides. Milestone 6 started out of order (the user's call; Milestone 5 is open):
the golden dataset is complete (#33, 100 cases) and the evaluation harness (#34) is
`eval/harness.py` (the driver) with `eval/metrics.py` (the metric definitions, pure),
`eval/correctness.py` (the correctness judge), `eval/phoenix_upload.py` (span annotations), run by `make eval` / `flp eval [--api-url] [--out] [--no-upload] [--no-judge]
[--pause] [--limit]` or `uv run python eval/harness.py`. It posts every golden case to the
running API with `debug: true`, `judge_all: true` (a new request field, read with `debug` only,
that forces the Stage 17 faithfulness judge), and one `run_tag` per run, `eval.pause_s` apart
(55 s on the Groq free tier, so a full run is about 95 minutes); reads ranks from the debug
field (`rerank.top` after the reranker, the new `retrieval.fused` list before it) and maps
chunk ids to printed pages through the parent store of the index the API reports; a hit is a
page intersection within `eval.page_tolerance`. Metrics (spec 8.2-8.5, targets in
`eval.targets_min` / `eval.targets_max`): Recall@5, MRR@10, nDCG@10 (binary gains),
Recall@30 before the reranker (an abstained answerable case counts as a miss), faithfulness
(the API's verdict, read on abstained rows too because `supported: false` abstains), correctness
(a second judge, `prompts/judge_correctness_v1.md`, score 0/1/2 against `expected_answer`,
mean / 2; an abstained answerable case scores 0 without a call; calls `eval.judge_pause_s`
apart, one retry, `ModelConfigError` propagates), citation accuracy (pooled over cited sources),
code presence, abstain precision / recall / F1, gate share (`gate.*` reasons over all
abstains), p95 latency (client wall time), cost per answered question, answer length. The run
file `eval/runs/{timestamp}_{index_version}_{prompt_version}.json` holds the metrics with
pass marks, per-span p50/p95, the cost split answered/abstained, the regressions against
`eval/baseline.json` (a higher-is-better metric more than `eval.regression_tolerance` under
the baseline, or one the baseline has and the run could not compute; exit 1), the error
share (over `eval.max_error_share` also exits 1), the model names, the git sha, and one row
per case; while a run goes, every response lands in `eval/runs/<run_tag>.partial.jsonl`. Per-case scores go to Phoenix as span annotations
on the root span of each request (`POST /v1/span_annotations`, root span id by GraphQL from
`trace_id`; identifier `<run_tag>:<case_id>:<name>` so a rerun upserts). The calibration
sweep (#35) is `eval/calibrate.py`, run by `make calibrate` (= `flp calibrate --local`, a dry
run) or `flp calibrate --run <harness run file> [--write]`: scores come from the `scores` column
of a harness run (the reranker scores of `rerank.top`, which the debug field carries also when
the gate abstained, so the spec's run with the gate switched off is not needed) or from a
local pass of hybrid retrieval plus the reranker over the golden cases with the raw question
(no model call, about three minutes); then spec 8.4: `tau_pass` from 0.05 to 0.95 in steps of
0.05 under the rule "no score at or above tau_pass -> abstain", the highest abstain F1 wins and
the middle of the plateau on a tie; `tau_low` = p05 and `tau_high` = p50 of the answerable
best scores. An unordered result (p05 above the chosen `tau_pass` happens when the groups
separate cleanly) is reported with the table and never written. `--write` replaces only the
three values inside the `gate:` block of `config.yaml` (comments kept, the reranker named) and
adds a `gate` block to the index manifest. Reports in `eval/runs/calibrate_<stamp>.json`.
Calibrated
2026-11-07 (UTC) on the 100 golden cases with `bge-reranker-base`: F1 0.963 on the plateau 0.30-0.40
(two answerable cases under it, g048 0.063 and g046 0.205; one in-domain case above it, u004
0.626), so `tau_pass` stays 0.35; p05 of the answerable best scores is 0.47, above `tau_pass`,
so `tau_low` is clamped to 0.30; the spec's p50 for `tau_high` is 0.986, which would make 28 of
the 58 passing answers borderline (strict prompt plus the judge on every one), so
`eval.calibration.high_percentile` is 0.10 (the user's call) and `tau_high` is 0.654 (0.6542), 4 of 58
borderline, close to the old 0.60. Issue #36 closes
Milestone 6: `eval/baseline.json` is the full 100-case run `eval/runs/20261108_baseline_v1.json`
(`flp eval --update-baseline` makes the run that just passed the baseline, `--baseline-from
<run file>` adopts an earlier one; both refuse an incomplete file and record `baseline_from` and
`baseline_set_at`); the regression rule and the weekly review are written down in
`docs/notes/evaluation-loop.md`. Baseline numbers (2026-11-08, 100 cases, 0 errors): Recall@5
1.000, MRR@10 0.983, Recall@30 1.000, faithfulness 1.000, correctness 0.883, citation accuracy
0.966, abstain precision 0.851 / recall 1.000 / F1 0.920 (the Milestone 6 bar is 0.90), gate
share 0.851, p95 7.3 s; the marks under target are seven answerable cases that abstained (g046
and g048 at the gate, g014/g021/g037 `output.invalid_after_repair`, g022 `gen.model_abstained`,
g058 `output.judge_unavailable`) and the free-tier latency. The spec 8.7 online loop is `eval/online_judge.py`
(`make online-judge`): the `llm.generate` spans of the last `eval.online.hours` from the
Phoenix REST API (`GET /v1/projects/<project>/spans`, simple filters, paged), the answered
ones not yet carrying a `faithfulness_online` annotation, a seeded 20% sample
(`eval.online.sample_share`, `eval.online.seed`), the Stage 17 judge on the context blocks
and the answer read from `input.value` / `output.value`, and the verdict as the annotation
`faithfulness_online` on the root span (identifier `faithfulness_online:<trace_id>`, so a rerun
upserts). Rules 2-4 of 8.7 (weekly thumbs-down review, add failures to the golden set, bump
`golden_version`) are manual steps in the same note. Next: Milestone 5 (full observability).
**Trap fixed, not in the spec:** JetBrains Mono ligatures read as `=` in the PDF text layer.
Stage 1 repairs them by glyph id through `parse.glyph_tables` (generated by
`tools/build_glyph_table.py` from JetBrains Mono v2.221). 1,607 glyphs recovered on the shipped
PDF. Every other Python file under `src/`, `eval/`, and `tests/` still
holds one comment line. `CONTEXT.md` is the glossary; read it before naming anything.

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
flp ask "What is a readonly property?"      # through the API (make serve first); exit 0/2/1
flp retrieve "readonly"                      # local chunk listing, no model
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
`rerank` (13, bge-reranker-base, top 8) -> `gate` (14, conditional edge: `pass` or `abstain`) ->
`assemble` (15, parents, `[S1]` labels, 6000-token budget, book order) -> `generate` (16) ->
`check_output` (17, citations, faithfulness judge, code verbatim) -> `respond` (18).
`graph/cache.py` is Stage 19 (Milestone 7). `graph/state.py` holds `RagState`.

**Shared:** `contracts.py` (dataclasses from spec Section 10, each with `to_attrs()`),
`settings.py` (loads `config.yaml`, computes `config_hash`), `models.py` (the only file that names a
model provider: `get_chat_model()`, `get_embeddings()`, `get_reranker()`), `stores/` (SQLite
embedding cache and parent store), `cli.py` (Typer: `flp ask|retrieve|ingest|eval|calibrate`).

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
| Reranker | `BAAI/bge-reranker-base`, `max_length` 512, sigmoid to 0-1, top 8 |
| Gate thresholds | `tau_low 0.20`, `tau_pass 0.35`, `tau_high 0.60` (recalibrate per spec 8.4) |
| Chunks | target 350 tokens, min 120, max 600, code max 900, `cl100k_base` |
| Generator | `groq:openai/gpt-oss-120b`, temperature 0, `max_output_tokens 2000`, `reasoning_effort low`, `prompts/answer_v1.md` |
| Tracing | OTLP/HTTP to Phoenix at `http://localhost:6006/v1/traces`, `service.name = flp-rag` |
| Qdrant | `flp_chunks_{index_version}`, alias `flp_chunks_current`, named vectors `dense` + `sparse_bm25` |

`gen.model` is `groq:openai/gpt-oss-120b`; `query.model` and `output.judge_model` are
`groq:openai/gpt-oss-20b` (Groq free tier, OpenAI protocol at `GROQ_BASE_URL`, key
`GROQ_API_KEY` in `.env`; price rows 0). Both are reasoning models: their thinking counts toward
`gen.max_output_tokens` (2000) and `output.judge_max_output_tokens` (1200), and
`gen.reasoning_effort` / `output.judge_reasoning_effort` (`low`) go to the OpenAI protocol only.
Measured on the free tier: an answered question takes 15-25 s, over the 6 s p95 target; the
latency sits in the model calls, not in the pipeline. Providers: `anthropic`, `openai`, `groq`, `ollama`, as
`provider:model`, each with a row in `prices_usd_per_million_tokens`. Embedding names that
start with `text-embedding` and reranker names that start with `rerank` select the hosted
providers; anything else is a local Hugging Face model. `models.py` is the only file that names a
provider or reads an API key (`Secrets`, from `.env`). Only the Groq key is set on this machine.

## Local models and where they live

No model weights are in the repository. The libraries download each model on first use into a
user-level cache and reuse it afterwards. A fresh clone fetches them on the first `flp ingest`.

| Model | Used by | Cache path | Size |
|---|---|---|---|
| `BAAI/bge-m3` | `BgeM3Embedder`, dense vectors | `~/.cache/huggingface/hub/models--BAAI--bge-m3` | about 4 GB |
| `BAAI/bge-reranker-base` | `BgeReranker`, Stage 13 | `~/.cache/huggingface/hub/models--BAAI--bge-reranker-base` | about 1.1 GB |
| `Qdrant/bm25` | `FastembedSparse`, sparse vectors | `$TMPDIR/fastembed_cache/models--Qdrant--bm25` | under 1 MB |

Set `HF_HOME` to move the Hugging Face cache. Set `FASTEMBED_CACHE_PATH` to move the fastembed
cache, which otherwise sits in the system temp directory and can vanish on reboot. Delete a cache
directory to force a re-download. Tests that load these models skip when the cache lacks them.
The only model-derived files in git are the two glyph tables under `src/flp_rag/ingest/glyph_tables/`.

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
  out-of-domain). Complete since issue #33: 60 answerable (two per chapter; kinds definition,
  code, comparison, how-to; every case with exact `evidence_quotes` and the printed page of the
  quote), 20 in-domain unanswerable (`kind` `unanswerable`, each key term verified absent from
  `data/clean/`), 20 out-of-domain (`kind` `offtopic`; `eval/offtopic_v1.jsonl` is folded in). `eval/smoke_queries.jsonl`: 10 queries used by Stage 8.
- `eval/baseline.json` is the full 100-case run of 2026-11-08 (issue #36). After a deliberate
  improvement, `flp eval --update-baseline` (or `--baseline-from <run file>`) in the same commit.
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
- A test that talks to the live Qdrant must use a throwaway collection name (rewrite
  `index_version` to `v999-test-...` in a temp copy of the enriched data) and delete only that.
  A test once reused the real `index_version` and its cleanup deleted the production collection.
- Debugging a wrong answer: follow the runbook in spec 7.11, starting from the `trace_id` in the
  response. Then add the case to the golden dataset and rerun `make eval`.
- Commit messages: `<type>: <description>` with types feat, fix, refactor, docs, test, chore, perf, ci.
