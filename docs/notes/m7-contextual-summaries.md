# Milestone 7, item 1: contextual summaries (issue #37)

Date: 2026-11-21 and 2026-11-22 (UTC). Branch `feat/issue-37-contextual-summaries`.

## What was built

One small-model call per chunk (`enrich.summary_model`, `groq:openai/gpt-oss-20b`) writes one
or two sentences on what the chunk covers inside its chapter. The summary is the first line of
`embedding_text`; `display_text` never changes. The prompt is `prompts/enrich_v1.md`
(`prompt_version` = sha256 of the file, first 8 hex). The summaries are cached in
`data/cache/summaries.sqlite` by `sha256(prompt_version + chunk_id + display_text)`.

## The build

| Measure | Value |
|---|---|
| Chunks summarized | 202 of 202 |
| Model calls that went through | 202 (plus 45 retries on `429 Too Many Requests`) |
| Input tokens, mean per chunk | 549 |
| Output tokens, mean per chunk | 106 (reasoning included) |
| Summary length, words | min 13, median 43, max 62 |
| Cost | USD 0 (free tier) |
| Wall time of the first run | about 30 minutes, paced by the free-tier limits |

The first run stopped with 9 chunks over the failure bound (rate-limit retries exhausted);
a test process summarizing the same chunks ran at the same time and took part of the quota.
Every summary landed in the cache, so the second run made no model call and built the index.

Four summaries picked with `random.seed(42)`:

- `:26:0164` "Explains why PHP's stateless request/response cycle makes it simple to start,
  but also introduces the growing async PHP community ... Features: MVC pattern, async
  community, PHP request/response cycle."
- `:04:0029` "Explores why PHP's type system matters ... covers strong vs weak, static vs
  dynamic typing ... Features: type system, static analysis, strong typing, weak typing ..."
- `:02:0006` "The chunk explains why PHP 6 was never released ... It mentions PHP 6 and PHP 7."
- `:29:0190` "The Baseline section explains how to generate a baseline file with Psalm or
  PHPStan ... Features: baseline, Psalm, PHPStan."

## Retrieval before the change (index `v3-bgem3-d4ebab9e`, 60 answerable golden cases)

`eval/hybrid_retrieval.py` (`eval/runs/hybrid_20261122_before.json`):

| Run | Recall@5 | Recall@30 | MRR@10 |
|---|---|---|---|
| dense | 0.983 | 1.000 | 0.871 |
| sparse | 1.000 | 1.000 | 0.901 |
| fused | 1.000 | 1.000 | 0.900 |

`eval/rerank_eval.py` (`eval/runs/rerank_20261122_before.json`): MRR@10 fused 0.900, reranked
0.936; Recall@3 reranked 0.967; answerable best score min 0.063, p05 0.470, median 0.986;
off-topic max 0.626; rerank latency p50 1.34 s.

## Retrieval after the change (index `v4-bgem3-716fbd64`, same 60 cases)

`eval/hybrid_retrieval.py` (`eval/runs/hybrid_20261122_summaries.json`):

| Run | Recall@5 | Recall@30 | MRR@10 | before -> after |
|---|---|---|---|---|
| dense | 0.983 | 1.000 | 0.899 | MRR 0.871 -> 0.899 |
| sparse | 1.000 | 1.000 | 0.901 | unchanged (BM25 reads `display_text`) |
| fused | 1.000 | 1.000 | 0.938 | MRR 0.900 -> 0.938 |

Paired per case (`eval/runs/rerank_20261122_*.json`, `rank_fused`): 6 cases better, 2 worse,
52 ties; two-sided sign test p = 0.29. The moves: g013, g015, g026, g045, g046, g049 from rank
2 to 1; g022 from 1 to 2; g038 from 2 to 4. Rank-1 share of the fused list 0.80 -> 0.88.

`eval/rerank_eval.py` (`eval/runs/rerank_20261122_summaries.json`): MRR@10 reranked 0.936 ->
0.936, Recall@3 0.967 -> 0.967, Recall@8 1.000 -> 1.000. Paired per case, `rank_reranked`:
60 ties of 60. The gate scores are identical to the last digit (answerable median 0.986,
off-topic max 0.626), so the gate decisions are identical too.

## Reading

The summary changes the dense vector only. It moves the fused list a little (more rank-1
hits, not significant at n = 60), and the reranker then ranks the same candidates the same
way, because it reads the chunk text (the header and `display_text`), which the summary does
not touch. Every candidate that the summary lifted was already in the fused top 30 before, so
the reranker saw the same set. The retrieval of this book is at the ceiling for this golden
set: Recall@5 1.000 and Recall@30 1.000 before and after, MRR@10 after the reranker 0.936
before and after.

## The `make eval` run (Milestone 7 Done-when): invalid, quota exhausted

`flp eval --api-url http://localhost:8001` against `v4-bgem3-716fbd64`, 100 cases, run file
`eval/runs/20261121T185442_v4-bgem3-716fbd64_287346c0.json`. The run is not a measurement of
the index: the free tier's daily token limit of `openai/gpt-oss-20b` (200,000 tokens per day)
was reached during the run. That model is the query model (Stage 11), the faithfulness judge
(Stage 17), and the correctness judge of the harness, and the summaries build had used about
132,000 of those tokens the same day (plus a test process that summarized the same chunks).
The result: 9 answered, 70 abstained, 21 errors (the limit is 10%), 8 correctness judges
failed with `429 ... tokens per day (TPD): Limit 200000`, and every retrieval metric fell
because an errored or wrongly abstained case counts as a miss (Recall@5 0.867, MRR@10 0.850,
abstain F1 0.491 against the baseline 1.000, 0.983, 0.920). The harness exited 1 on the error
share and the regressions, as designed. The run is recorded here as the Done-when asks, and
marked invalid; a clean rerun needs a day with the daily quota unused.

## Decision

Revert to the index without summaries, keep the code switched off. The retrieval evaluation
(no model calls, so untouched by the quota) is the evidence: the reranked list, which the
gate, the context, and the answer read, is identical on all 60 answerable cases, so the
summary cannot change an answer of this golden set. The fused-list gain (MRR@10 0.900 ->
0.938) would matter only if the reranker were switched off (`rerank.enabled: false`).

- `config.yaml`: `enrich.contextual_summary: false` (the spec default).
- Alias `flp_chunks_current` -> `flp_chunks_v3-bgem3-d4ebab9e` again; `flp_chunks_v4-bgem3-716fbd64`
  and its manifest stay for a rerun of the harness. Stage 8 pruned `flp_chunks_v2-bgem3-a8ea0fc5`
  during the build (retention).
- `data/cache/summaries.sqlite` holds the 202 summaries: a rebuild with the switch on makes
  no model call.
- The chunk config hash with the switch off is `f32bf8f588b0d7b7`, so the next `make ingest`
  builds a new version of the same 202 chunks from the embedding cache (no model, a few
  minutes) and switches the alias to it.

To try again: wait for the daily quota, set the switch on, `make ingest` (no summary calls:
all cached), `make serve`, `make eval`. A better test of the idea would switch the reranker
off as well, or use a golden set with cases the fused list misses.
