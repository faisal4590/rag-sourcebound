# Dashboards and alerts (spec 7.9 and 7.10, issue #32)

Phoenix (20.9) stores the traces and the span annotations. It has no dashboard builder and no
alerting. The project therefore has two versions of the five panels and the five rules.

## 1. Without a metrics backend: `make dashboard`, `make alerts`

`eval/dashboard.py` reads the spans of a time window from the Phoenix REST API and computes
the five panels of spec 7.9 as tables, and evaluates the five rules of 7.10 on the same spans.

```bash
make dashboard HOURS=24              # the five panels, text; --json writes the report
make alerts                          # the five rules; exit 1 when one fires
make trace TRACE=<trace id>          # the debug runbook of 7.11 on one trace
```

The thresholds come from the `alerts:` block of `config.yaml`. The panels:

| Panel | Source | What it shows |
|---|---|---|
| 1 Requests and outcomes | `rag.request` roots of `/ask` | requests per hour by `rag.status` and `rag.abstain_reason` |
| 2 Latency by stage | every span | p50 and p95 per span name, ms |
| 3 Quality | `rerank`, `check.faithfulness`, `relevance.gate`, `retrieval.hybrid` | the top-1 score histogram in 0.05 buckets, the fail rate of the judged answers (all, borderline, not borderline), the overlap histogram |
| 4 Cost | `LLM` spans and `embed.batch`, the roots | USD per hour by model, tokens per request (mean, p50, p95), cache hit rate |
| 5 Feedback and evaluation | the `user_feedback` span annotations, `eval/runs/*.json` | ratings per day; the golden-set scores of every evaluation run by index and prompt version |

## 2. With Grafana: `grafana-flp-rag.json` and `alerts.yaml`

When `trace.metrics_endpoint` points at an OpenTelemetry Collector that feeds Prometheus, the
metrics of spec 7.7 (`src/flp_rag/metrics.py`) carry the same panels and rules in Grafana.

1. Collector: an `otlp` receiver (HTTP, the endpoint of `trace.metrics_endpoint`) and a
   `prometheus` exporter with `add_metric_suffixes: false`, so the names stay the OTel ones:
   `rag_requests`, `rag_stage_latency_ms_bucket`, `rag_tokens`, `rag_cost_usd`,
   `rag_rerank_score_top1_bucket`, `rag_retrieval_overlap_ratio_bucket`, `rag_errors`,
   `rag_user_feedback`. Attribute names become labels with `.` as `_` (`span_name`,
   `gen_ai_request_model`, `abstain_reason`, `cache_hit`).
2. Grafana: import `grafana-flp-rag.json` or provision it from a file; the dashboard variable
   `DS_PROMETHEUS` (type datasource) selects the Prometheus datasource either way. Provision
   `alerts.yaml` with `DS_PROMETHEUS` set to the datasource uid in the provisioning environment.
   Every alert query is an instant query over its own window, reduced to one number before the
   threshold or math node; a rule without data is `OK`, a failing query is an error.

`rag_requests` carries the label `http_route`, and every request query filters on
`http_route="/ask"`, so the health, page, and feedback roots do not inflate the rates. The
latency histogram has no route label: the p95 rule counts every `rag.request` root, which
lowers the p95 when health checks are frequent.

Three numbers are approximations in PromQL and exact in the span version:

- the faithfulness fail rate is `unfaithful / (answered + unfaithful)` over the request
  counter; the unjudged answers (the sample rate) stay in the denominator, so it understates
  the judged rate that `make dashboard` reads from the `check.faithfulness` spans;
- tokens per request divides by every `/ask` request; the span version averages over the
  requests that called a model (a cache hit or a gate abstain has no tokens);
- the golden-set table reads the gauge `flp_eval_metric`, meant for a small exporter job over
  `eval/runs/*.json` that is not shipped; the span version reads the run files directly.

## The gap, stated

Phoenix cannot host the panels or fire the rules. Until a Grafana stack is deployed, the rules
are saved queries: `make alerts` runs them on demand (a cron line makes it periodic), and
`eval/runs/dashboard_<date>.json` keeps a report when `--json` is given. Exit codes for the
cron line: 0 quiet, 1 a rule fired, 2 Phoenix unreachable. The fetch always covers the longest
rule window (24 h for the cost rule), whatever `--hours` says; the panels use `--hours`.
The feedback panel counts the ratings of the requests in the window, by the day of the
rating, so a rating given today on an older request is out of a short window.
