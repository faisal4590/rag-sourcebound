# The evaluation loop: baseline, regression check, online sample

Milestone 6 closes with this loop (spec 8.6 and 8.7, issues #34 to #36). Three pieces: the
harness, the baseline, and the online sample. Everything is run from the repository root with
the services up (`make up-local`) and the API running (`make serve`).

## The baseline rule

`eval/baseline.json` is a copy of one accepted evaluation run. `make eval` compares every new run
with it and exits 1 when a higher-is-better metric drops by more than `eval.regression_tolerance`
(0.03), or when the baseline has a metric the run could not compute.

The rule: **after a deliberate improvement, copy the new run to `eval/baseline.json` in the same
commit as the change.** A reviewer then sees the code change and the new numbers together, and
the next run measures against the new level. Never update the baseline in a commit of its own,
and never to make a red run green.

```bash
uv run flp eval --update-baseline          # the run that just finished with exit 0 becomes the baseline
uv run flp eval --baseline-from eval/runs/20261108_baseline_v1.json   # adopt a run made earlier
```

Both refuse an incomplete run file (no metrics or no cases). The baseline keeps the whole run,
plus `baseline_from` (which file) and `baseline_set_at` (when).

When a run regresses, the report names the metric, the baseline value and the current value:

```text
REGRESSION recall_at_5: baseline 0.917 -> 0.867
```

## The first baseline

`eval/runs/20261108_baseline_v1.json`: all 100 golden cases against the API with the base
reranker, the calibrated gate (`tau_low` 0.30, `tau_pass` 0.35, `tau_high` 0.654), the Groq
free-tier models, and `judge_all`. The numbers are in the run file and in the pull request of
issue #36.

## The online sample (spec 8.7)

Rule 1 is a script. The other three rules are a weekly habit.

1. **Sample and judge.** `make online-judge` reads the `llm.generate` spans of the last 24 hours
   from Phoenix, keeps the answered ones that have no `faithfulness_online` annotation yet,
   takes a 20% sample (seeded shuffle, `eval.online.seed` 42), runs the Stage 17 faithfulness
   judge on each (the same prompt and model as the API), and writes the verdict to the root span
   of the trace as the annotation `faithfulness_online` (score 1 or 0, label, the unsupported
   claims as the explanation). Run it once a day; a rerun on the same day skips what is scored.
   `--dry-run` counts and samples without a model call or a write.
2. **Review thumbs-down once a week.** In Phoenix, filter the `flp-rag` project on the
   feedback annotation that issue #31 adds (the CLI keypress and `POST /feedback`; the name is
   set there) and on `faithfulness_online` = 0. Open each trace: the question, the context blocks, the answer, and
   the gate decision are on the spans.
3. **Add each real failure to the golden set** with the correct printed pages and an exact
   evidence quote (`eval/golden_v1.jsonl`; the schema test checks the file).
4. **Bump the golden version** when cases are added (`golden_version` on every case, the harness
   records it in every run), and rerun `make eval` so the baseline carries the new version.
