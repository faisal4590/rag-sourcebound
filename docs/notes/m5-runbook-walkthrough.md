# Milestone 5 runbook walkthrough (spec 7.11, issue #32)

One wrong answer, caused on purpose, debugged from its `trace_id` with the tools of
Milestone 5. Date: 2026-11-21. Index `v3-bgem3-d4ebab9e`, prompt `287346c0`.

## The fault

`config.yaml` was changed to `gate.tau_pass: 0.99` and `gate.tau_high: 0.995` (the defaults
are 0.35 and 0.6542), the API restarted, and one question asked that the book answers:

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is a readonly property?"}'
```

The response: status `no_information`, answer `No information found`, trace id
`6ff1dbca87370031bcf017f9234a3782`. A wrong answer: Chapter 6 (Readonly Properties, printed
75-84) answers the question.

## The walk

```bash
make trace TRACE=6ff1dbca87370031bcf017f9234a3782
# = uv run python eval/trace_show.py 6ff1dbca87370031bcf017f9234a3782
```

The output, step by step (the runbook of spec 7.11):

    trace 6ff1dbca87370031bcf017f9234a3782: 11 spans
    
    1. root span rag.request
       status no_information  abstain_reason gate.no_passing_chunk  latency 7168 ms  cost USD 0.0  cache.hit False
       index v3-bgem3-d4ebab9e  prompt 287346c0  config 4eba68b6
    
    2. query.understand
       raw        What is a readonly property?
       standalone What is a readonly property?
       intent book_question  chapter_hint 6  variants 5  fallback False
    
    3. retrieval.hybrid
       dense hits 120  sparse hits 150  fused 30  overlap 0.6333333333333333
        1. bca146b9df2d0a2b:06:0045  ch 6  Readonly Classes                          p. 83-84  score 0.147
        2. bca146b9df2d0a2b:06:0042  ch 6  Readonly Properties                       p. 77-79  score 0.145
        3. bca146b9df2d0a2b:06:0043  ch 6  No default value                          p. 79-81  score 0.143
        4. bca146b9df2d0a2b:06:0041  ch 6  Readonly Properties                       p. 75-77  score 0.140
        5. bca146b9df2d0a2b:06:0044  ch 6  Cloning                                   p. 82-82  score 0.139
        6. bca146b9df2d0a2b:30:0193  ch 30  Dynamic Properties                        p. 310-310  score 0.136
        7. bca146b9df2d0a2b:05:0038  ch 5  How It Works                              p. 68-70  score 0.123
        8. bca146b9df2d0a2b:30:0192  ch 30  Dynamic Properties                        p. 308-309  score 0.122
        9. bca146b9df2d0a2b:05:0037  ch 5  How It Works                              p. 65-67  score 0.120
       10. bca146b9df2d0a2b:05:0040  ch 5  Inheritance                               p. 74-74  score 0.113
       -> is the expected chapter in this list? If not, the fault is before the reranker.
    
    4. rerank
       model BAAI/bge-reranker-base  in 30  out 8  top1 0.8927624114818737  top3 mean 0.8393408747866626  truncated 4  hint swaps 0
        1. bca146b9df2d0a2b:06:0045  ch 6  Readonly Classes                          p. 83-84  score 0.893
        2. bca146b9df2d0a2b:06:0042  ch 6  Readonly Properties                       p. 77-79  score 0.836
        3. bca146b9df2d0a2b:06:0043  ch 6  No default value                          p. 79-81  score 0.789
        4. bca146b9df2d0a2b:06:0044  ch 6  Cloning                                   p. 82-82  score 0.582
        5. bca146b9df2d0a2b:06:0041  ch 6  Readonly Properties                       p. 75-77  score 0.576
        6. bca146b9df2d0a2b:30:0193  ch 30  Dynamic Properties                        p. 310-310  score 0.035
        7. bca146b9df2d0a2b:09:0066  ch 9  Backed enums with interfaces              p. 120-121  score 0.022
        8. bca146b9df2d0a2b:24:0157  ch 24  Internals                                 p. 260-260  score 0.000
       -> is the expected chunk in the top 8? If it is below, the fault is the reranker.
    
    5. relevance.gate
       decision abstain  reason gate.no_passing_chunk  top1 0.8927624114818737  n_pass 0  borderline False
       tau_low 0.3  tau_pass 0.99  tau_high 0.995
       -> the gate stopped the request: top score 0.8927624114818737 against tau_pass 0.99. Lower tau_pass, or the chunk is not good enough.
    
    6. context.assemble
       absent: no context was built (the gate abstained, or an earlier stage stopped)
    
    7. generation and output checks
       llm.generate absent: no answer model ran
    
    8. next: add the question to eval/golden_v1.jsonl with the expected pages and run make eval

## The reading

| Step | What the span says | Conclusion |
|---|---|---|
| 1 root | `rag.status no_information`, `rag.abstain_reason gate.no_passing_chunk`, cost 0, no cache hit | the gate stopped the request, not the model |
| 2 understand | standalone question equals the raw question, intent `book_question`, chapter hint 6, 5 variants | Stage 11 understood the question |
| 3 retrieval | the five Chapter 6 chunks are ranks 1-5 of the fused list | the fault is not before the reranker |
| 4 rerank | top 1 at 0.893, the five Chapter 6 chunks at 0.58-0.89, the off-topic ones under 0.04 | the reranker separates the chapter cleanly |
| 5 gate | `tau_pass 0.99`, top score 0.893, `n_pass 0` | the threshold is above every score; the configuration is the fault |
| 6 assemble | absent | the absence is the signal, as spec 7.4 says |
| 7 generate | absent | no model ran, so the cost is 0 |

The tool prints the conclusion of step 5 as a hint: "the gate stopped the request: top score
0.8927624114818737 against tau_pass 0.99. Lower tau_pass, or the chunk is not good enough."

## The fix

`config.yaml` restored (`tau_pass 0.35`, `tau_high 0.6542`). The dashboard shows the fault
as well: `make dashboard HOURS=1` lists the request under
`no_information:gate.no_passing_chunk` in panel 1, and with more than
`alerts.abstain_min_requests` such requests in an hour the abstain-rate rule fires.

## What the walk needed

- the `trace_id` of the response (the `X-Trace-Id` header, the `trace_id` field, the CLI line);
- Phoenix running with the trace (the request spans land within seconds);
- `eval/trace_show.py`, which reads the spans of the trace over the Phoenix REST API and prints
  the eight steps; without it, the same attributes are in the Phoenix trace view.

The runbook ended with step 8: add the question to `eval/golden_v1.jsonl` and run `make eval`.
Chapter 6 has two golden cases already (g006 and g036), and every answerable case abstains
under this gate, so the harness reports the fault as abstain recall 0 for the answerable
cases and a regression against `eval/baseline.json` (exit 1). The question itself was not
added: the fault was the configuration, not a gap in the set.
