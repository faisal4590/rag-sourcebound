# The debug runbook of spec 7.11 on one trace, read from Phoenix. Issue #32.
"""Walk one trace the way the runbook does.

    uv run python eval/trace_show.py <trace_id>

Step by step, the spans the runbook names and the attributes it reads: the root (status and
abstain reason), `query.understand` (what the system thought the user meant), the retrieval
documents (is the expected chapter there at all), the rerank scores, the gate decision against
its thresholds, the context budget, and the generation and output checks. A span that is
absent is printed as absent: after a gate abstain, the absence of `context.assemble` and
`llm.generate` is the signal. The script reads; the conclusion is the reader's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from flp_rag.settings import Settings, load_settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoenix_api import phoenix_base_url, trace_spans

Row = dict[str, Any]
EXCERPT = 300


def attrs(row: Row) -> dict[str, Any]:
    return row.get("attributes") or {}


def by_name(rows: list[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for r in sorted(rows, key=lambda r: str(r.get("start_time", ""))):
        out.setdefault(r["name"], []).append(r)
    return out


def documents(a: dict[str, Any], prefix: str, limit: int) -> list[str]:
    """`<prefix>.<i>.document.*` as one line per document: chunk id, chapter, section, pages."""
    lines = []
    for i in range(limit):
        key = f"{prefix}.{i}.document"
        if f"{key}.id" not in a:
            break
        meta = metadata_of(a, key)
        pages = f"{meta.get('page_printed_start', '?')}-{meta.get('page_printed_end', '?')}"
        score = a.get(f"{key}.score")
        shown = f"{score:.3f}" if isinstance(score, int | float) else "?"
        lines.append(
            f"{i + 1:2}. {a[f'{key}.id']}  ch {meta.get('chapter_no', '?')}  "
            f"{str(meta.get('section_title', ''))[:40]:40}  p. {pages}  score {shown}"
        )
    return lines


def metadata_of(a: dict[str, Any], key: str) -> dict[str, Any]:
    """The document metadata: Phoenix parses the JSON string the app writes into
    `<key>.metadata.<field>` keys; an unparsed string is read here; anything else is empty."""
    flat = {k.rsplit(".", 1)[1]: v for k, v in a.items() if k.startswith(f"{key}.metadata.")}
    raw = a.get(f"{key}.metadata")
    if isinstance(raw, dict):
        return {**raw, **flat}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return {**parsed, **flat}
    return flat


def excerpt(value: Any) -> str:
    text = str(value) if value is not None else ""
    text = text.replace("\n", " ")
    return text[:EXCERPT] + ("…" if len(text) > EXCERPT else "")


def render(rows: list[Row], trace_id: str) -> list[str]:
    """The runbook steps as lines of text."""
    spans = by_name(rows)
    out = [f"trace {trace_id}: {len(rows)} spans"]

    def first(name: str) -> dict[str, Any] | None:
        return attrs(spans[name][0]) if name in spans else None

    root = first("rag.request") or {}
    out += [
        "",
        "1. root span rag.request",
        (
            f"   status {root.get('rag.status')}  abstain_reason {root.get('rag.abstain_reason')}  "
            f"latency {root.get('rag.latency_ms')} ms  cost USD {root.get('rag.cost_usd')}  "
            f"cache.hit {root.get('cache.hit')}"
        ),
        (
            f"   index {root.get('rag.index_version')}  prompt {root.get('rag.prompt_version')}  "
            f"config {root.get('rag.config_hash')}"
        ),
    ]
    u = first("query.understand")
    out += ["", "2. query.understand"]
    if u is None:
        out.append(
            "   absent: the request never reached Stage 11 (a cache hit or an input rejection)"
        )
    else:
        out += [
            f"   raw        {excerpt(u.get('rag.question_raw', '(content off)'))}",
            f"   standalone {excerpt(u.get('rag.question_standalone', '(content off)'))}",
            (
                f"   intent {u.get('rag.intent')}  chapter_hint {u.get('rag.chapter_hint')}  "
                f"variants {u.get('rag.n_variants')}  fallback {u.get('rag.understand_fallback')}"
            ),
        ]
        if u.get("rag.abstain_reason") not in (None, "none"):
            out.append(f"   -> Stage 11 abstained: {u.get('rag.abstain_reason')}")
    h = first("retrieval.hybrid")
    out += ["", "3. retrieval.hybrid"]
    if h is None:
        out.append("   absent: no retrieval ran")
    else:
        out.append(
            f"   dense hits {h.get('rag.n_dense_hits')}  sparse hits {h.get('rag.n_sparse_hits')}  "
            f"fused {h.get('rag.n_fused')}  overlap {h.get('rag.overlap_ratio')}"
        )
        out += ["   " + line for line in documents(h, "retrieval.documents", 10)]
        out.append(
            "   -> is the expected chapter in this list? If not, the fault is before the reranker."
        )
    rr = first("rerank")
    out += ["", "4. rerank"]
    if rr is None:
        out.append("   absent")
    else:
        out.append(
            f"   model {rr.get('gen_ai.request.model')}  in {rr.get('rag.n_in')}  "
            f"out {rr.get('rag.n_out')}  "
            f"top1 {rr.get('rag.score_top1')}  top3 mean {rr.get('rag.score_top3_mean')}  "
            f"truncated {rr.get('rag.n_truncated')}  hint swaps {rr.get('rag.hint_swaps')}"
        )
        out += ["   " + line for line in documents(rr, "reranker.output_documents", 8)]
        out.append(
            "   -> is the expected chunk in the top 8? If it is below, the fault is the reranker."
        )
    g = first("relevance.gate")
    out += ["", "5. relevance.gate"]
    if g is None:
        out.append("   absent")
    else:
        out.append(
            f"   decision {g.get('rag.gate_decision')}  reason {g.get('rag.gate_reason')}  "
            f"top1 {g.get('rag.score_top1')}  n_pass {g.get('rag.n_pass')}  "
            f"borderline {g.get('rag.borderline')}"
        )
        out.append(
            f"   tau_low {g.get('rag.tau_low')}  tau_pass {g.get('rag.tau_pass')}  "
            f"tau_high {g.get('rag.tau_high')}"
        )
        if g.get("rag.gate_decision") == "abstain":
            out.append(
                f"   -> the gate stopped the request: top score {g.get('rag.score_top1')} against "
                f"tau_pass {g.get('rag.tau_pass')}. "
                "Lower tau_pass, or the chunk is not good enough."
            )
    c = first("context.assemble")
    out += ["", "6. context.assemble"]
    if c is None:
        out.append(
            "   absent: no context was built (the gate abstained, or an earlier stage stopped)"
        )
    else:
        out.append(
            f"   blocks {c.get('rag.n_blocks')}  parents {c.get('rag.n_parents_used')}  "
            f"tokens {c.get('rag.tokens_used')}  truncated {c.get('rag.truncated')}  "
            f"sources {c.get('rag.source_ids')}"
        )
        if c.get("rag.truncated") is True:
            out.append(
                "   -> the budget cut a block: raise context.max_tokens or lower context.max_blocks"
            )
    out += ["", "7. generation and output checks"]
    gens = spans.get("llm.generate", [])
    if not gens:
        out.append("   llm.generate absent: no answer model ran")
    for i, gen in enumerate(gens, 1):
        a = attrs(gen)
        out.append(
            f"   llm.generate #{i}: model {a.get('gen_ai.request.model')}  "
            f"prompt {a.get('rag.prompt_version')}  "
            f"repair_calls {a.get('rag.repair_calls')}  attempts {a.get('rag.attempts')}  "
            f"tokens {a.get('gen_ai.usage.input_tokens')}/{a.get('gen_ai.usage.output_tokens')}"
        )
        if "output.value" in a:
            out.append(f"      output: {excerpt(a['output.value'])}")
    for chk in spans.get("guardrails.output", []):
        a = attrs(chk)
        out.append(
            f"   guardrails.output: result {a.get('rag.check_result')}  "
            f"citations_valid {a.get('rag.citations_valid')}  "
            f"code_verbatim_ok {a.get('rag.code_verbatim_ok')}  faithful {a.get('rag.faithful')}  "
            f"failed_check {a.get('rag.failed_check')}  "
            f"abstain_reason {a.get('rag.abstain_reason')}"
        )
    for j in spans.get("check.faithfulness", []):
        a = attrs(j)
        out.append(
            f"   check.faithfulness: faithful {a.get('rag.faithful')}  "
            f"claims {excerpt(a.get('rag.unsupported_claims'))}"
        )
    errors = [r for r in rows if r.get("status_code") == "ERROR"]
    if errors:
        out += ["", "spans with status ERROR: " + ", ".join(r["name"] for r in errors)]
    out += [
        "",
        (
            "8. next: add the question to eval/golden_v1.jsonl with the expected pages "
            "and run make eval"
        ),
    ]
    return out


def run(settings: Settings, trace_id: str, *, transport: httpx.BaseTransport | None = None) -> int:
    url, project = phoenix_base_url(settings), settings.eval.phoenix_project
    with httpx.Client(timeout=settings.eval.timeout_s, transport=transport) as client:
        rows = trace_spans(client, url, project, trace_id)
    if not rows:
        print(f"no spans for trace {trace_id} in project {project!r} at {url}")
        return 1
    print("\n".join(render(rows, trace_id)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_id")
    args = parser.parse_args(argv)
    return run(load_settings(), args.trace_id)


if __name__ == "__main__":
    sys.exit(main())
