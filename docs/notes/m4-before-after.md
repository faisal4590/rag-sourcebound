# Milestone 4, before and after

The 10 smoke queries through the full API twice: once with the Milestone 3 pipeline (dense
retrieval, the raw question as the only variant, the dense cosine as the "rerank" score) and
once with the Milestone 4 pipeline (Stage 11 query understanding, Stage 12 hybrid retrieval
with BM25 and RRF, Stage 13 cross-encoder reranking). Same code, same index, same models;
only three switches in `config.yaml` differ (issue #28, Part 1). The smoke queries are a
regression check, not an evaluation: the retrieval measurements are Part 8 of #25 and #26,
the full evaluation is Milestone 6.

**Short version.** Retrieval did not get worse on any smoke query: every expected chapter
ranks first on both sides, except the JIT question, which ranks second on both sides. The
Milestone 4 pipeline sends a context one third the size to the answer model (median 1,768
tokens against 4,397) and abstains on small talk before any search. It costs about 9 seconds
more per request, 7.5 of them in the reranker on this laptop; the median request is over the
6-second target. The "after" run could not prove the answered status: the free-tier daily
token limit of the answer model ran out during the run, so 27 of 30 requests failed at
Stage 16 with 429 (issue #74). The stages before generation ran normally in every request and
their numbers come from the trace spans.

## Setup

| Item | Value |
|---|---|
| Date | 2026-10-31 |
| Code | branch `feat/issue-28-before-after` on top of `7b8a922` (master after #27); tag `m3` marks the last Milestone 3 commit `aaef076`, tag `m4` the commit that closes this issue |
| Index | `v3-bgem3-d4ebab9e` (202 chunks, `chunk_config_hash d4ebab9ed10ca7e7`), alias `flp_chunks_current` |
| Before | `query.enabled: false`, `retrieval.sparse_enabled: false`, `rerank.enabled: false`; `config_hash 62af1ced` |
| After | all three `true` (the shipped configuration); `config_hash f1814ded` |
| Models | generator `groq:openai/gpt-oss-120b`, query and judge `groq:openai/gpt-oss-20b` (Groq free tier), dense `BAAI/bge-m3`, sparse `Qdrant/bm25`, reranker `BAAI/bge-reranker-v2-m3` on MPS (M1 Pro, 16 GB) |
| Gate | `tau_low 0.20`, `tau_pass 0.35`, `tau_high 0.60` on both sides (on the dense cosine before, on the reranker score after) |
| Caveat | During the before run the API still built and warmed the reranker and the BM25 encoder although both were switched off (fixed in the same PR: `build_deps` now leaves disabled components unbuilt). This changes memory and startup, not the measured stage times: the disabled stages ran in 0 to 1 ms |
| Protocol | `eval/smoke_run.py --tag <tag> --repeats 3 --pause 55`: one warm-up request dropped, then the 10 queries three times, `debug: true`, tagged with `run_tag` (never `session_id`). The pause keeps the answer model under the free-tier limit of 8,000 tokens per minute (Findings 1) |
| Runs | `eval/runs/smoke_m4-before_20261031.json`, `eval/runs/smoke_m4-after_20261031.json` |

## Results per query

Rank: the position of the first candidate from the expected chapter in the reranked list
(`debug.rerank.top`, 8 entries). For the 27 "after" rows whose request failed at Stage 16 the
rank comes from `rag.scores_all` on the `rerank` span of the same trace (`rank_source` in the
run file); the two sources agree on the 3 rows that have both.

| # | Query | Expected chapter | Rank before (3 runs) | Rank after (3 runs) | Status before | Status after | Cited before | Cited after |
|---|---|---|---|---|---|---|---|---|
| 1 | What is constructor property promotion? | 5 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 2 | How do readonly properties work? | 6 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 3 | How do I use named arguments? | 7 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 4 | How do I define an attribute class? | 8 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 5 | What are backed enums? | 9 | 1, 1, 1 | 1, 1, 1 | answered 3 | answered 2, error 1 | 3/3 | 2/2 answered |
| 6 | What is the difference between match and switch? | 13 | 1, 1, 1 | 1, 1, 1 | answered 3 | answered 1, error 2 | 3/3 | 1/1 answered |
| 7 | What does the never return type mean? | 3 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 8 | How does the JIT compiler work in PHP 8? | 21 | 2, 2, 2 | 2, 2, 2 | answered 3 | error 3 | 3/3 | 0/3 |
| 9 | What are fibers? | 27 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |
| 10 | What is new in PHP 8.2? | 30 | 1, 1, 1 | 1, 1, 1 | answered 3 | error 3 | 3/3 | 0/3 |

Every "error" in the after column is a Stage 16 failure after retrieval and reranking had
finished: 25 are `GenerationUnavailable` on a 429 "tokens per day" answer from Groq, 2 are the
20-second API timeout while the client library retried the same 429. Every answered request
on both sides cited the expected chapter.

Other per-request values of the after run (from the spans, all 30 requests):

| Value | After |
|---|---|
| Variants sent to Stage 12 | 5 for 17 requests, 3 for 8, 1 for 5 (the model returned no usable paraphrases; see Findings 4) |
| `rag.overlap_ratio` (dense and sparse agree on a fused chunk) | median 0.63, range 0.37 to 0.83 |
| `rag.score_top1` after the reranker | median 0.987, minimum 0.600 (the attribute question) |
| `rag.n_pass` (candidates at or above `tau_pass`) | median 4 (before: 8, the whole top_n, since every dense cosine in the top 8 was above 0.35) |
| Context tokens for the answer model | median 1,768 (before: 4,397) |
| `chapter_hint` | the expected chapter for 9 of 10 questions; none for the `never` question |

## One trace, read: "How do readonly properties work?"

Before: trace `3c1067abf4418e96d8f01c81a60ea4e2` (run 1). After: trace
`757b990d98bf8acb5ae736287786af48` (run 1). The per-list ranks of the after side come from an
offline run of `retrieve_dense_lists` and `retrieve_sparse_lists` with the same variants,
because the retriever spans record counts and top scores but not the hit ids (#25 decided
against recording chunk text or ids per list). The best Chapter 6 chunk is
`bca146b9df2d0a2b:06:0045` on both sides.

| Step | Span | Before | After |
|---|---|---|---|
| 1 | `rag.request` | status answered, 18,979 ms (this run paid a repair call, see below; the other two runs took 1,379 and 1,251 ms), cost 0, `config_hash 62af1ced`, `run_tag m4-before` | status error (`timeout`, Stage 16 hit the daily limit), 20,026 ms, `config_hash f1814ded`, `run_tag m4-after` |
| 2 | `query.understand` | disabled: raw question, 1 variant, 0 ms | standalone = the raw question (no history), intent `book_question`, `chapter_hint 6`, `wants_code true`, 5 variants: the question, "What is the behavior of readonly properties in PHP?", "Explain how readonly properties function in PHP.", "Describe the mechanics of readonly properties in PHP.", keyword variant "readonly properties PHP"; 2,222 ms (two attempts: the first answer was not the JSON shape) |
| 3 | `retrieval.dense` | 1 list of 30, top cosine 0.697; Chapter 6 chunks at ranks 1 to 5, `:06:0045` first | 4 lists of 30, top cosines 0.697, 0.742, 0.739, 0.722; `:06:0045` first in every list, Chapter 6 at ranks 1 to 5 in three lists and 1 to 4 plus 6 in one |
| 4 | `retrieval.sparse` | none | 5 lists of 30, top BM25 11.7 to 15.3; `:06:0045` first in four lists and second in one (behind `:06:0043`); the Chapter 30 chunk `:30:0193` takes rank 2 in the list of the second paraphrase |
| 5 | `retrieval.hybrid` | 30 fused from 1 list, `top_fused_score 0.0164` (= 1/61), no overlap value | 270 hits fused to 30, `rag.overlap_ratio 0.63`, `top_fused_score 0.147`; `:06:0045` first with provenance in all 9 lists; fused top 5 all Chapter 6 |
| 6 | `rerank` | stub: dense cosine as score, 0 ms; `score_top1 0.697`, `score_top3_mean 0.681`; top 5 all Chapter 6, then `:30:0193` (0.567), `:08:0056` (0.565) | `score_top1 0.982`, `score_top3_mean 0.975`, `hint_swaps 0`, `n_truncated 0`, device mps, 7,013 ms; top 5 all Chapter 6 (0.98, 0.97, 0.97, 0.76, 0.71), then `:30:0193` at 0.237 and `:09:0066` at 0.132 |
| 7 | `relevance.gate` | pass, zone 3 (0.697 >= `tau_high`), `n_pass 8`: every candidate in the top 8 passes on the cosine scale | pass, zone 3 (0.982), `n_pass 5`: the three off-chapter candidates fall under `tau_pass` |
| 8 | `context.assemble` | `n_blocks 7`, `n_parents_used 7`, 4,397 tokens, sources from Chapters 5, 6, 8, 30 | `n_blocks 4`, `n_parents_used 4`, 1,417 tokens, sources from Chapter 6 only |
| 9 | `llm.generate` | 4,593 input tokens; the first answer failed the code check (plain-text fallback, the index has no `blocks` table), one repair call of 17.4 s; answer cites S2 and S3 | 1,653 input tokens requested; both attempts answered 429 (daily limit); `GenerationUnavailable` after 54.5 s in the worker thread, 34 s after the API had answered 504 |

Reading: the dense retriever already found the right chunk first; the Milestone 4 stages did
not change the top rank here. They changed what surrounds it. The reranker pushes the three
off-chapter candidates from 0.57 (a cosine the gate accepts) to 0.03 to 0.24 (scores the gate
rejects), so the context holds Chapter 6 only and is one third the size. The price is 7 s of
reranking and 0.8 to 2.2 s of query understanding.

## Time and cost

Medians over the 30 requests of each side. Before: every request answered. After: the stage
times come from the spans for the 27 rows that failed at Stage 16; `llm.generate` and the wall
time are over the 3 answered rows only.

| Stage (median ms) | Before | After |
|---|---|---|
| `guardrails.input` | 1 | 12 |
| `query.understand` | 0 (disabled) | 804 (range 343 to 4,769; 7 of 30 requests needed the second attempt) |
| `retrieval.hybrid` | 18 (1 dense list) | 76 (up to 4 dense and 5 sparse lists; 1,336 ms in the trace above, where the query embeddings were not yet cached) |
| `rerank` | 0 (stub) | 7,469 (30 pairs on MPS) |
| `relevance.gate` | 0 | 1 |
| `context.assemble` | 11 | 9 |
| `llm.generate` | 1,125 | 1,242 (3 answered requests) |
| `guardrails.output` | 2 | 3 |
| Whole request, client wall time | 1,296 | 10,504 (all 30; 11,079 over the 3 answered) |
| Answer-model input tokens per request | 4,593 (trace above; context 4,397) | 1,653 (trace above; context median 1,768) |
| Cost per request, USD | 0 (free tier) | 0 (free tier) |
| Requests answered | 30/30 | 3/30 (Findings 1) |

Where the 9 extra seconds go: 7.5 s reranker, 0.8 s query understanding, 0.06 s hybrid
retrieval. The answer model's input shrinks by about 2,900 tokens per request, which on the
free tier is the difference between 1 and 3 requests per minute.

Small talk, one request per side (`run_tag` on the root span):

| | Before (trace `97b2e795120641dc962830b1ce3eab10`) | After (trace `007739fcf2e0f7dacc6ec1eec53a5c59`) |
|---|---|---|
| Path | guard, understand, retrieve, rerank, gate (pass, borderline, cosine 0.464), assemble, generate, check, respond | guard, understand, respond |
| Result | `No information found`, reason `gen.model_abstained`: the answer model read 8 blocks and declined | `No information found`, reason `query.smalltalk`: no retrieval span |
| Time | 920 ms, one answer-model call | 423 ms, one small-model call |

## Findings

1. **The free tier, not the pipeline, decided the after run.** Groq limits `gpt-oss-120b` to
   8,000 tokens per minute and 200,000 per day. The first before run without a pause lost 17
   of 30 requests to the per-minute limit (the client library retried the 429 for about 98 s
   while the API answered 504 at 20 s); the paced rerun answered 30 of 30. The after run then
   met the daily limit: 27 of 30 requests failed at Stage 16 with `Rate limit reached ... tokens
   per day (TPD): Limit 200000, Used 199856`. Issue #74 tracks the two code changes this asks
   for (honor `Retry-After`, stop the worker when the request is gone). The rerun of the after
   side needs a day with a fresh budget: 30 requests at about 1,900 tokens each is 57,000
   tokens, well inside it, which the smaller context makes possible.
2. **No smoke query got worse.** Ranks are identical on both sides (1 everywhere, 2 for the JIT
   question on both). The smoke questions are easy topic questions whose chapter the dense
   retriever already finds first; the gain of hybrid search and the reranker shows on the
   golden set (#25: MRR@10 0.859 dense to 0.900 fused; #26: off-topic best scores 0.001 to
   0.018 against 0.72 to 1.00 for answerable questions), not here.
3. **The reranker changes the context, not the top rank.** `n_pass` drops from 8 to a median
   of 4 and the context from 4,397 to 1,768 tokens, because the reranker separates
   off-chapter candidates (0.03 to 0.24) from on-chapter ones (0.7 to 1.0) where the cosine
   put them all between 0.55 and 0.70. The gate thresholds of the spec therefore work as
   intended only with the reranker; before, zone 3 held every candidate.
4. **Stage 11 is noisier than its unit tests suggest.** 7 of 30 requests needed the second
   attempt (the first answer of `gpt-oss-20b` was not the JSON shape), and 5 requests ended
   with one variant: the model wrote paraphrases that failed the Part 3 checks or none at all
   (`rag.paraphrases_dropped` is 0 on those, so the model returned an empty list). The
   `never` question got no chapter hint. Retrieval still ranked every expected chapter first,
   so this did not cost a smoke query; the Milestone 6 golden run should report the variant
   count per request. Also open: `query.understand` is 2.2 s when both attempts run, 0.4 to
   0.9 s otherwise.
5. **Latency is over the target on this machine.** The median request is 10.5 s against the
   6-second p95 target of the spec, before any of the free-tier waits. The reranker alone is
   7.5 s for 30 pairs of 300 to 700 tokens on MPS (#26 measured 5 to 10 s). The decision from
   #26 is still open: `retrieval.top_k_fused: 20`, `rerank.max_length: 512`, a smaller reranker
   (`BAAI/bge-reranker-base`), or a longer `api.timeout_s`. Nothing in the pipeline itself
   needs a fix for this.
6. **Two trace details worth a Milestone 5 fix.** The `retrieval.hybrid` span appears twice per
   request (the node wrapper and the `@stage` decorator on `retrieve_hybrid` share the name).
   The retriever spans record counts and top scores but not the hit ids, so a per-list rank
   needs an offline rerun (step 3 and 4 above).

## Milestone 4 close (Part 7)

| Check | Result |
|---|---|
| All 10 smoke queries have the expected chapter in the reranked top 5 (after) | Pass: rank 1 for 9, rank 2 for the JIT question, in all three runs |
| No smoke query has a lower rank after than before | Pass: identical ranks on both sides |
| The `CustomerDTO` question ranks its chunk in the top 3 | Pass: `bca146b9df2d0a2b:05:0036` is second on both sides (dense cosine 0.528 behind `:05:0038` at 0.552 before; reranker 0.897 behind `:07:0046` at 0.903 after). Offline check with the same code paths, since the daily limit blocked the API |
| `hi there` abstains with `query.smalltalk` and no retrieval spans | Pass (after); before it reached the answer model and abstained with `gen.model_abstained` |
| All 10 answered in both runs | Before pass (30/30). After **not shown**: 3 of 30 answered, the rest failed at Stage 16 on the daily token limit (Findings 1). Rerun `eval/smoke_run.py --tag m4-after --repeats 3 --pause 55` on a fresh day |
| Median request under 6 s | **Fail**: 10.5 s after (1.3 s before); 7.5 s of it is the reranker (Findings 5) |
| Records | Note committed; tag `m3` set; tag `m4` set on the closing commit; Milestone 4 closes when the two open checks are settled |

## Links

- Run files: `eval/runs/smoke_m4-before_20261031.json`, `eval/runs/smoke_m4-after_20261031.json` (force-added; `eval/runs/` is gitignored). The after file carries `rank_source` and `timings_source` per row and a `note` on the recovery from the spans
- Trace ids (Phoenix keeps traces 30 days): before readonly `3c1067abf4418e96d8f01c81a60ea4e2`, `d170e8510b3028d056ef7674e68c875c`, `4373010dc057f7d90ae0ed6ebc74909a`; after readonly `757b990d98bf8acb5ae736287786af48`, `da787a1cd02ad649de9a938a24cd5861`, `7643a96bba2dca7d655f3e69d567c14b`; small talk before `97b2e795120641dc962830b1ce3eab10`, after `007739fcf2e0f7dacc6ec1eec53a5c59`; a daily-limit failure `8c23168ae9ee266fd719af6c05ea1f3f`
- Part 8 of #25 (hybrid retrieval on the 30 golden questions): Recall@5 and Recall@30 1.0 for dense, sparse, and fused; MRR@10 dense 0.859, sparse 0.925, fused 0.900; mean overlap 0.47 (`eval/runs/hybrid_20261017.json`)
- Part 8 of #26 (reranker): answerable best scores 0.72 to 1.00 (median 0.98), off-topic 0.001 to 0.018; Recall@3 1.0; MRR@10 0.900 fused to 0.872 reranked; rerank latency p50 about 7 s on MPS (`eval/runs/rerank_20261017.json`)
- Issues opened from this note: #74 (Stage 16 retry ignores `Retry-After`; free-tier token limits)
