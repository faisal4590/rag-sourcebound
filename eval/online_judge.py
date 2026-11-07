# Spec 8.7 online loop (issue #36): sample yesterday's answered traces, judge faithfulness, annotate the trace.
"""Score a sample of production traces with the faithfulness judge (spec 8.7, rule 1).

    uv run python eval/online_judge.py [--hours 24] [--sample 0.2] [--seed 42] [--dry-run]
    make online-judge                                     # the defaults above

1. Read three span kinds of the last `--hours` from Phoenix (`GET /v1/projects/<project>/spans`,
   simple filters, paged): `llm.generate` (the answer and its status in `output.value`),
   `context.assemble` (the context blocks in `output.value`), `guardrails.output` (whether the
   output checks kept the answer). One candidate per trace: the last generate span, kept only
   when the checks left the answer standing.
2. Find the root span of each trace (batched), drop the traces that already carry a
   `faithfulness_online` annotation (`GET /v1/projects/<project>/span_annotations`, batched),
   and take the sample: `--sample` of all answered traces of the window, minus the ones scored
   before, so a rerun on the same day adds nothing. The sample is a seeded shuffle over the
   candidates sorted by trace id.
3. Run the Stage 17 faithfulness judge (`output.judge_model`, `output.judge_prompt_file`) on the
   context blocks and the answer, as the API does, `eval.judge_pause_s` apart; one failed call
   is counted and skipped, `ModelConfigError` stops the run.
4. Write each verdict at once to the root span as the annotation `faithfulness_online` (score
   1.0 or 0.0, label `faithful` / `unfaithful`, the unsupported claims as the explanation).

A span whose recorded value was cut at `trace.content_max_chars` cannot be parsed; such traces
are counted as `unparseable` and skipped. Rules 2-4 of spec 8.7 are manual and written down in
`docs/notes/evaluation-loop.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from flp_rag.contracts import ContextBlock
from flp_rag.graph.nodes.check_output import JUDGE_SCHEMA, build_judge_messages, parse_verdict
from flp_rag.graph.nodes.generate import load_prompt
from flp_rag.models import ChatModel, ModelConfigError, get_chat_model
from flp_rag.settings import Settings, load_settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import phoenix_base_url

ANNOTATION = "faithfulness_online"
GENERATE_SPAN, ASSEMBLE_SPAN, CHECK_SPAN = "llm.generate", "context.assemble", "guardrails.output"
PAGE = 1000  # the endpoint's maximum page size
BATCH = 100  # trace ids or span ids per request
BLOCK_FIELDS = {f.name for f in dataclasses.fields(ContextBlock)}


@dataclass(frozen=True)
class Candidate:
    trace_id: str
    context: list[ContextBlock]
    answer: str


@dataclass(frozen=True)
class Verdict:
    trace_id: str
    faithful: bool
    unsupported_claims: list[str]


# --------------------------------------------------------------------------- phoenix reads


def _get(client: httpx.Client, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Every page of a Phoenix list endpoint."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        query = {**params, "limit": PAGE, **({"cursor": cursor} if cursor else {})}
        r = client.get(url, params=query)
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("data") or [])
        cursor = body.get("next_cursor")
        if not cursor:
            return out


def spans_named(
    client: httpx.Client,
    phoenix_url: str,
    project: str,
    name: str,
    *,
    since: datetime,
    until: datetime,
) -> list[dict[str, Any]]:
    params = {"name": name, "start_time": since.isoformat(), "end_time": until.isoformat()}
    return _get(client, f"{phoenix_url.rstrip('/')}/v1/projects/{project}/spans", params)


def root_span_ids(
    client: httpx.Client, phoenix_url: str, project: str, trace_ids: Iterable[str]
) -> dict[str, str]:
    """trace id -> root span id, `BATCH` traces per request."""
    ids = sorted(set(trace_ids))
    out: dict[str, str] = {}
    for start in range(0, len(ids), BATCH):
        params: dict[str, Any] = {"trace_id": ids[start : start + BATCH], "parent_id": "null"}
        for span in _get(client, f"{phoenix_url.rstrip('/')}/v1/projects/{project}/spans", params):
            out[str(span["context"]["trace_id"])] = str(span["context"]["span_id"])
    return out


def scored_span_ids(
    client: httpx.Client, phoenix_url: str, project: str, span_ids: Iterable[str]
) -> set[str]:
    """The root spans that already carry the annotation. Phoenix answers 404 when none of the
    ids exist; that is 'none scored', every other error raises."""
    ids = sorted(set(span_ids))
    out: set[str] = set()
    url = f"{phoenix_url.rstrip('/')}/v1/projects/{project}/span_annotations"
    for start in range(0, len(ids), BATCH):
        params: dict[str, Any] = {
            "span_ids": ids[start : start + BATCH],
            "include_annotation_names": [ANNOTATION],
            "limit": PAGE,
        }
        r = client.get(url, params=params)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        out |= {
            str(a["span_id"]) for a in r.json().get("data") or [] if a.get("name") == ANNOTATION
        }
    return out


# --------------------------------------------------------------------------- candidates


def _json(value: Any) -> Any:
    """A recorded `input.value` / `output.value`: JSON text, or None when it was cut."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def _block(raw: dict[str, Any]) -> ContextBlock | None:
    data = {k: v for k, v in raw.items() if k in BLOCK_FIELDS}
    try:
        return ContextBlock(**data)
    except TypeError:
        return None


def candidates_from(
    generate: Sequence[dict[str, Any]],
    assemble: Sequence[dict[str, Any]],
    checks: Sequence[dict[str, Any]],
) -> tuple[list[Candidate], dict[str, int]]:
    """One candidate per trace: the last `llm.generate` span (a repair makes two), kept when it
    answered and the output checks kept the answer, with the blocks of the trace's
    `context.assemble` span. Returns the candidates and the counts of what was dropped."""
    counts = {"traces": 0, "not_answered": 0, "withdrawn": 0, "unparseable": 0}
    last_generate: dict[str, dict[str, Any]] = {}
    for span in sorted(generate, key=lambda s: str(s.get("start_time") or "")):
        last_generate[str(span["context"]["trace_id"])] = span
    contexts = {str(s["context"]["trace_id"]): s for s in assemble}
    outcomes = {str(s["context"]["trace_id"]): s for s in checks}
    counts["traces"] = len(last_generate)
    out: list[Candidate] = []
    for trace_id, span in last_generate.items():
        result = _json((span.get("attributes") or {}).get("output.value"))
        if not isinstance(result, dict):
            counts["unparseable"] += 1
            continue
        if result.get("status") != "answered" or not result.get("answer"):
            counts["not_answered"] += 1
            continue
        check = outcomes.get(trace_id)
        if check is not None:
            outcome = _json((check.get("attributes") or {}).get("output.value"))
            if isinstance(outcome, dict) and outcome.get("status") == "no_information":
                counts["withdrawn"] += 1
                continue
        context_span = contexts.get(trace_id)
        assembled = _json((context_span or {}).get("attributes", {}).get("output.value"))
        blocks = (
            [
                b
                for b in (
                    _block(raw)
                    for raw in (assembled or {}).get("context") or []
                    if isinstance(raw, dict)
                )
                if b is not None
            ]
            if isinstance(assembled, dict)
            else []
        )
        if not blocks:
            counts["unparseable"] += 1
            continue
        out.append(Candidate(trace_id, blocks, str(result["answer"])))
    return out, counts


def sample(candidates: Sequence[Candidate], *, target: int, seed: int) -> list[Candidate]:
    """A seeded shuffle over the candidates sorted by trace id, then the first `target`."""
    order = sorted(candidates, key=lambda c: c.trace_id)
    random.Random(seed).shuffle(order)
    return order[: max(0, target)]


def sample_target(answered: int, already_scored: int, share: float) -> int:
    """`share` of all answered traces of the window, minus the ones scored before: a rerun on
    the same day adds nothing; at least one when nothing is scored yet."""
    if answered == 0:
        return 0
    wanted = max(1, math.ceil(answered * share))
    return max(0, wanted - already_scored)


# --------------------------------------------------------------------------- judge and write


def judge(candidate: Candidate, *, chat: ChatModel, settings: Settings) -> Verdict:
    prompt = load_prompt(settings.config_path.parent / settings.output.judge_prompt_file)
    system, user = build_judge_messages(prompt, candidate.context, candidate.answer)
    completion = chat.complete(
        system,
        [("user", user)],
        json_schema=JUDGE_SCHEMA,
        temperature=settings.output.judge_temperature,
        max_tokens=settings.output.judge_max_output_tokens,
    )
    supported, claims = parse_verdict(completion)
    return Verdict(candidate.trace_id, supported, claims)


def annotation_for(verdict: Verdict, span_id: str, *, day: str) -> dict[str, Any]:
    explanation = "; ".join(verdict.unsupported_claims)[:1000]
    return {
        "name": ANNOTATION,
        "span_id": span_id,
        "annotator_kind": "LLM",
        "result": {
            "score": 1.0 if verdict.faithful else 0.0,
            "label": "faithful" if verdict.faithful else "unfaithful",
            **({"explanation": explanation} if explanation else {}),
        },
        "identifier": f"{ANNOTATION}:{verdict.trace_id}",
        "metadata": {"sampled_on": day, "source": "eval/online_judge.py"},
    }


def post_annotation(client: httpx.Client, phoenix_url: str, annotation: dict[str, Any]) -> None:
    r = client.post(
        f"{phoenix_url.rstrip('/')}/v1/span_annotations?sync=false", json={"data": [annotation]}
    )
    r.raise_for_status()


# --------------------------------------------------------------------------- the run


def run(
    settings: Settings,
    *,
    hours: float,
    share: float,
    seed: int,
    dry_run: bool = False,
    chat: ChatModel | None = None,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Returns the summary that is also printed as one JSON line."""
    phoenix_url = phoenix_base_url(settings)
    project = settings.eval.phoenix_project
    until = now or datetime.now(UTC)
    since = until - timedelta(hours=hours)
    summary: dict[str, Any] = {"since": since.isoformat(), "until": until.isoformat()}
    with httpx.Client(transport=transport, timeout=settings.eval.timeout_s) as client:
        window = {"since": since, "until": until}
        generate = spans_named(client, phoenix_url, project, GENERATE_SPAN, **window)
        assemble = spans_named(client, phoenix_url, project, ASSEMBLE_SPAN, **window)
        checks = spans_named(client, phoenix_url, project, CHECK_SPAN, **window)
        candidates, counts = candidates_from(generate, assemble, checks)
        roots = root_span_ids(client, phoenix_url, project, (c.trace_id for c in candidates))
        with_root = [c for c in candidates if c.trace_id in roots]
        scored = scored_span_ids(client, phoenix_url, project, roots.values())
        fresh = [c for c in with_root if roots[c.trace_id] not in scored]
        target = sample_target(len(with_root), len(with_root) - len(fresh), share)
        chosen = sample(fresh, target=target, seed=seed)
        summary |= counts | {
            "answered": len(with_root),
            "no_root_span": len(candidates) - len(with_root),
            "already_scored": len(with_root) - len(fresh),
            "sampled": len(chosen),
        }
        if dry_run:
            print(json.dumps(summary))
            return summary
        model = chat or get_chat_model(settings, "judge")
        day = until.date().isoformat()
        judged = unfaithful = failed = 0
        for i, c in enumerate(chosen):
            if i and settings.eval.judge_pause_s:
                time.sleep(settings.eval.judge_pause_s)
            try:
                verdict = judge(c, chat=model, settings=settings)
            except ModelConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failed call is counted, the loop goes on
                failed += 1
                print(f"{c.trace_id} judge failed: {type(exc).__name__}")
                continue
            post_annotation(
                client, phoenix_url, annotation_for(verdict, roots[c.trace_id], day=day)
            )
            judged += 1
            unfaithful += 0 if verdict.faithful else 1
            print(f"{c.trace_id} {'faithful' if verdict.faithful else 'UNFAITHFUL'}")
    summary |= {"judged": judged, "unfaithful": unfaithful, "judge_failed": failed}
    print(json.dumps(summary))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hours", type=float, default=None, help="default eval.online.hours")
    parser.add_argument(
        "--sample", type=float, default=None, help="default eval.online.sample_share"
    )
    parser.add_argument("--seed", type=int, default=None, help="default eval.online.seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="count and sample; no judge, no write"
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    online = settings.eval.online
    run(
        settings,
        hours=args.hours if args.hours is not None else online.hours,
        share=args.sample if args.sample is not None else online.sample_share,
        seed=args.seed if args.seed is not None else online.seed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
