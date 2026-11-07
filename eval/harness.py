# Evaluation harness. Metrics of Sections 8.2 to 8.5. Uploads scores to Phoenix. Regression check against baseline.json.
"""Run every golden case against the running API and score it (issue #34).

    make eval                                     # = uv run flp eval
    uv run python eval/harness.py [--api-url URL] [--golden FILE] [--out FILE] [--baseline FILE]
                                  [--no-upload] [--no-judge] [--pause S] [--limit N] [--tag TAG]
                                  [--update-baseline]
    uv run python eval/harness.py --baseline-from eval/runs/<run file>.json   # adopt a past run

Every request carries `debug: true`, `judge_all: true` (the Stage 17 faithfulness judge runs
on every answer), and one `run_tag` for the whole run, `eval.pause_s` apart. Ranks come from the
debug field (`rerank.top` after the reranker, `retrieval.fused` before it); chunk ids map to
printed pages through the parent store of the index the API reports; a hit is a page
intersection within `eval.page_tolerance`. The metric definitions are in `eval/metrics.py`, the
correctness judge in `eval/correctness.py`, the Phoenix upload in `eval/phoenix_upload.py`.

The run file is `eval/runs/{timestamp}_{index_version}_{prompt_version}.json`; while the run is
going, every response is appended to `eval/runs/<run_tag>.partial.jsonl`, which is removed once
the run file is written. Exit 1 when a higher-is-better metric dropped by more than
`eval.regression_tolerance` against `eval/baseline.json` (or the baseline has it and this run
could not compute it), or when more than `eval.max_error_share` of the requests errored.

The baseline rule (spec 8.6 rule 5, issue #36): after a deliberate improvement, copy the new run
to `eval/baseline.json` in the same commit. `--update-baseline` does the copy at the end of a
run that finished with exit code 0; `--baseline-from <run file>` copies a run made earlier.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from flp_rag.api.app import default_index_dir
from flp_rag.contracts import EvalCase, read_jsonl
from flp_rag.ingest.s06_s07_index import STORE_FILE
from flp_rag.models import ModelConfigError
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.parent_store import ParentStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correctness import Judge, load_correctness_prompt, make_judge
from metrics import (
    CaseResult,
    PagesOf,
    breakdown,
    compare_with_baseline,
    compute_metrics,
    evaluate_case,
    golden_version,
    with_correctness,
)
from phoenix_upload import upload_scores

ROOT = Path(__file__).resolve().parents[1]
ABSTAINED_REASON = "abstained: an answerable case with no answer scores 0"
STAMP = "%Y%m%dT%H%M%S"


# --------------------------------------------------------------------------- helpers


def phoenix_base_url(settings: Settings) -> str:
    """The Phoenix host from the OTLP endpoint (`.../v1/traces`)."""
    endpoint = settings.trace.exporter_endpoint
    return endpoint.split("/v1/")[0] if "/v1/" in endpoint else endpoint


def store_pages(settings: Settings, index_version: str) -> PagesOf:
    """Chunk id -> printed page range from the parent store of the index the API answers with.
    Fails at once when the store is missing, before any request is sent."""
    path = default_index_dir(settings) / index_version / STORE_FILE
    store = ParentStore.open_readonly(path)
    cache: dict[str, tuple[int, int] | None] = {}

    def pages_of(chunk_id: str) -> tuple[int, int] | None:
        if chunk_id not in cache:
            chunk = store.get_chunk(chunk_id)
            cache[chunk_id] = (
                (int(chunk.payload["page_printed_start"]), int(chunk.payload["page_printed_end"]))
                if chunk is not None
                else None
            )
        return cache[chunk_id]

    return pages_of


def default_out(
    runs_dir: Path, index_version: str, prompt_version: str, *, stamp: str | None = None
) -> Path:
    stamp = stamp or datetime.now(UTC).strftime(STAMP)
    return runs_dir / f"{stamp}_{index_version or 'noindex'}_{prompt_version or 'noprompt'}.json"


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def health(client: httpx.Client, api_url: str) -> dict[str, Any]:
    """`GET /health`: the index version and the model names, before the first question."""
    r = client.get(f"{api_url.rstrip('/')}/health")
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict) or not body.get("index_version"):
        raise RuntimeError(f"{api_url} reports no index version: {str(body)[:200]}")
    return body


def ask(
    client: httpx.Client, api_url: str, question: str, *, tag: str
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    try:
        response = client.post(
            f"{api_url.rstrip('/')}/ask",
            json={"question": question, "debug": True, "judge_all": True, "run_tag": tag},
        )
        body = response.json()
        if not isinstance(body, dict):
            body = {"status": "error", "detail": f"non-object body: {str(body)[:100]}"}
    except (httpx.HTTPError, ValueError) as exc:
        body = {"status": "error", "detail": f"transport: {type(exc).__name__}: {exc}"}
    return body, int((time.perf_counter() - started) * 1000)


def print_report(
    metrics: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Any],
    regressions: Sequence[Mapping[str, Any]],
    *,
    baseline_found: bool,
) -> None:
    print(f"{'metric':24} {'value':>8} {'target':>8}  mark    n")
    for name, m in metrics.items():
        value = "-" if m["value"] is None else f"{m['value']:.3f}"
        target = "-" if m["target"] is None else f"{m['target']:g}"
        mark = {True: "pass", False: "FAIL", None: "report"}[m["pass"]]
        print(f"{name:24} {value:>8} {target:>8}  {mark:6}  {m['n']}")
    n = extra["n"]
    print(
        f"answered {n['answered']}, abstained {n['abstained']}, error {n['error']}, "
        f"judge errors {n.get('judge_errors', 0)}"
    )
    cost = extra["cost_usd_mean"]
    print(f"mean cost USD: answered {cost['answered']}, abstained {cost['abstained']}")
    spans = ", ".join(
        f"{k} {v['p50']:.0f}/{v['p95']:.0f}" for k, v in extra["span_latency_ms"].items()
    )
    print(f"span p50 / p95 ms: {spans}")
    if not baseline_found:
        print("no baseline: eval/baseline.json is missing or empty, nothing compared (issue #36)")
    for r in regressions:
        now = "not computed" if r["current"] is None else f"{r['current']:.3f}"
        print(f"REGRESSION {r['metric']}: baseline {r['baseline']:.3f} -> {now}")


def update_baseline(
    run_file: Path, baseline: Path, *, expected_cases: int | None = None
) -> dict[str, Any]:
    """Copy a run file over the baseline (spec 8.6 rule 5). Only a complete run qualifies: it
    has metrics and one row per case, and, when `expected_cases` is given (the golden size), it
    covers all of them, so a `--limit` run cannot become the baseline. Metrics under their
    target and error rows are printed, not refused: the rule speaks of a deliberate choice.
    The copy is written to a temporary file and moved into place, and carries `baseline_from`
    (relative to the repository when possible) and `baseline_set_at`."""
    data = json.loads(run_file.read_text())
    cases = data.get("cases") or []
    if not data.get("metrics") or not cases:
        raise ValueError(f"{run_file} is not a complete run file (no metrics or no cases)")
    if data.get("n_cases") != len(cases):
        raise ValueError(f"{run_file}: n_cases {data.get('n_cases')} but {len(cases)} case rows")
    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(f"{run_file} holds {len(cases)} cases, the golden set {expected_cases}")
    failing = sorted(name for name, m in data["metrics"].items() if m.get("pass") is False)
    if failing:
        print(f"note: the new baseline has metrics under target: {', '.join(failing)}")
    for key in ("error_share", "judge_error_share"):
        if data.get(key):
            print(f"note: the new baseline has {key} {data[key]:.0%}")
    try:
        origin = str(run_file.resolve().relative_to(ROOT))
    except ValueError:
        origin = str(run_file)
    record = {
        **data,
        "baseline_from": origin,
        "baseline_set_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    tmp = baseline.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(baseline)
    return record


# --------------------------------------------------------------------------- the run


def score_cases(
    bodies: Sequence[tuple[EvalCase, dict[str, Any], int]],
    pages_of: PagesOf,
    judge: Judge | None,
    *,
    tolerance: int,
    judge_pause_s: float,
) -> tuple[list[CaseResult], int]:
    """Read every response, then the correctness judge on the answered answerable cases
    (`judge_pause_s` apart); an abstained answerable case scores 0 without a model call.
    Returns the rows and the number of failed judge calls (`ModelConfigError` propagates)."""
    results: list[CaseResult] = []
    judged = failed = 0
    for case, body, elapsed_ms in bodies:
        r = evaluate_case(case, body, elapsed_ms, pages_of, tolerance=tolerance)
        if case.group == "answerable" and r.abstained:
            r = with_correctness(r, 0, ABSTAINED_REASON)
        elif judge is not None and r.answered and case.group == "answerable":
            if judged and judge_pause_s:
                time.sleep(judge_pause_s)
            judged += 1
            try:
                r = with_correctness(r, *judge(case, str(body.get("answer") or "")))
            except ModelConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded on the row; the metric's n shrinks
                failed += 1
                r = replace(r, correctness_reason=f"judge error: {type(exc).__name__}: {exc}")
                print(f"{case.case_id} correctness judge failed: {type(exc).__name__}: {exc}")
        results.append(r)
    return results, failed


def run(
    settings: Settings,
    *,
    api_url: str,
    golden: Path,
    out: Path | None = None,
    baseline: Path | None = None,
    pages_of: PagesOf | None = None,
    judge: Judge | None | bool = None,
    upload: bool | None = None,
    pause_s: float | None = None,
    limit: int | None = None,
    tag: str | None = None,
    transport: httpx.BaseTransport | None = None,
    update_baseline_after: bool = False,
) -> int:
    """Run the harness. `judge=False` skips the correctness judge; `pages_of` and `transport`
    are the injection points of the tests. With `update_baseline_after`, a run that ends with
    exit code 0 becomes the new baseline. Returns the process exit code."""
    cfg = settings.eval
    root = settings.config_path.parent
    cases = list(read_jsonl(golden, EvalCase))[: limit or None]
    if not cases:
        raise ValueError(f"{golden} holds no cases")
    tag = tag or f"{cfg.run_tag_prefix}-{datetime.now(UTC).strftime(STAMP)}"
    pause = cfg.pause_s if pause_s is None else pause_s
    do_upload = cfg.upload_scores if upload is None else upload
    runs_dir = root / cfg.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    partial = runs_dir / f"{tag}.partial.jsonl"
    bodies: list[tuple[EvalCase, dict[str, Any], int]] = []
    with httpx.Client(transport=transport, timeout=cfg.timeout_s) as client:
        info = health(client, api_url)
        index_version = str(info["index_version"])
        # everything that can fail before the 95-minute loop fails here
        pages = pages_of or store_pages(settings, index_version)
        judge_fn: Judge | None
        judge_fn = None if judge is False else (judge if callable(judge) else make_judge(settings))
        baseline_path = baseline or root / cfg.baseline_file
        baseline_data = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        compute_metrics([], cfg)  # unknown target names fail here, not after the loop
        partial.write_text("")
        for i, case in enumerate(cases):
            if i and pause:
                time.sleep(pause)
            body, elapsed_ms = ask(client, api_url, case.question, tag=tag)
            bodies.append((case, body, elapsed_ms))
            with partial.open("a") as fh:
                fh.write(
                    json.dumps({"case_id": case.case_id, "elapsed_ms": elapsed_ms, "body": body})
                    + "\n"
                )
            status = body.get("status") or "error"
            print(f"{case.case_id} {status:>14} {elapsed_ms:>6} ms  {case.question[:60]}")
    prompt_version = str(
        next((b.get("prompt_version") for _, b, _ in bodies if b.get("prompt_version")), "")
    )
    results, judge_failures = score_cases(
        bodies, pages, judge_fn, tolerance=cfg.page_tolerance, judge_pause_s=cfg.judge_pause_s
    )
    metrics = compute_metrics(results, cfg)
    extra = breakdown(results)
    extra["n"]["judge_errors"] = judge_failures
    baseline_found = bool(baseline_data.get("metrics"))
    regressions = compare_with_baseline(
        metrics, baseline_data, tolerance=cfg.regression_tolerance, names=tuple(cfg.targets_min)
    )
    error_share = extra["n"]["error"] / len(results)
    out_path = out or default_out(runs_dir, index_version, prompt_version)
    record = {
        "run_tag": tag,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "golden_file": str(golden),
        "golden_version": golden_version(cases),
        "n_cases": len(cases),
        "index_version": index_version,
        "prompt_version": prompt_version,
        "correctness_prompt_version": (
            load_correctness_prompt(root / cfg.correctness_prompt_file).version
            if judge_fn is not None
            else None
        ),
        "config_hash": settings.config_hash,
        "models": {
            **{k: str(v) for k, v in (info.get("models") or {}).items()},
            "query": settings.query.model,
            "judge": settings.output.judge_model,
        },
        "api_url": api_url,
        "metrics": metrics,
        "breakdown": extra,
        "regressions": regressions,
        "error_share": error_share,
        "judge_error_share": judge_failures / len(results),
        "cases": [dataclasses.asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    partial.unlink(missing_ok=True)
    print_report(metrics, extra, regressions, baseline_found=baseline_found)
    if do_upload:
        try:
            sent, missing = upload_scores(
                results,
                phoenix_base_url(settings),
                cfg.phoenix_project,
                tag=tag,
                timeout_s=cfg.timeout_s,
                transport=transport,
            )
            print(f"uploaded {sent} span annotations to Phoenix; {missing} traces not found")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"WARNING: Phoenix upload failed, scores stay in the run file: {exc!r}")
    print(f"run file -> {out_path}")
    if error_share > cfg.max_error_share:
        print(f"FAIL: {error_share:.0%} of the requests errored (limit {cfg.max_error_share:.0%})")
        return 1
    if judge_failures / len(results) > cfg.max_error_share:
        print(
            f"FAIL: {judge_failures} correctness judge calls failed "
            f"(limit {cfg.max_error_share:.0%} of the cases)"
        )
        return 1
    if regressions:
        return 1
    if update_baseline_after:
        update_baseline(out_path, baseline_path, expected_cases=len(cases))
        print(f"baseline updated from this run -> {baseline_path}; commit it with the change")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-url", default=None, help="default cli.api_url")
    parser.add_argument("--golden", type=Path, default=None, help="default eval.golden_file")
    parser.add_argument("--out", type=Path, default=None, help="default eval/runs/<stamp>_...")
    parser.add_argument("--baseline", type=Path, default=None, help="default eval.baseline_file")
    parser.add_argument("--no-upload", action="store_true", help="no span annotations to Phoenix")
    parser.add_argument("--no-judge", action="store_true", help="skip the correctness judge")
    parser.add_argument("--pause", type=float, default=None, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=None, help="first N cases only")
    parser.add_argument("--tag", default=None, help="the run_tag; default <prefix>-<timestamp>")
    parser.add_argument(
        "--update-baseline", action="store_true", help="a run with exit 0 becomes the baseline"
    )
    parser.add_argument(
        "--baseline-from",
        type=Path,
        default=None,
        help="copy this run file to the baseline (--baseline names the destination) and stop",
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    root = settings.config_path.parent
    if args.baseline_from is not None:
        if args.update_baseline:
            parser.error("--baseline-from and --update-baseline exclude each other")
        target = args.baseline or root / settings.eval.baseline_file
        golden_n = sum(
            1 for _ in read_jsonl(args.golden or root / settings.eval.golden_file, EvalCase)
        )
        update_baseline(args.baseline_from, target, expected_cases=golden_n)
        print(f"baseline <- {args.baseline_from} ({target})")
        return 0
    return run(
        settings,
        api_url=args.api_url or settings.cli.api_url,
        golden=args.golden or root / settings.eval.golden_file,
        out=args.out,
        baseline=args.baseline,
        judge=False if args.no_judge else None,
        upload=False if args.no_upload else None,
        pause_s=args.pause,
        limit=args.limit,
        tag=args.tag,
        update_baseline_after=args.update_baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
