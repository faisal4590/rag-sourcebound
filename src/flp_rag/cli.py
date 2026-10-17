# Typer CLI: flp ask, flp retrieve, flp ingest, flp eval, flp calibrate. Spec Appendix D.
"""Command line entry point.

`ask` is a thin client of the API (issue #24): it sends the question to `POST /ask`, prints the
`Response` record by status, and exits with 0 (answered), 2 (no information), or 1 (error or
no API). It never builds text of its own, so the CLI and the API cannot drift apart.
`retrieve` (also `ask --retrieve-only`) lists chunks from the local retriever without a model
call. `eval` and `calibrate` land in Milestone 6.
"""

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from flp_rag import tracing
from flp_rag.ingest import run as run_mod
from flp_rag.settings import ABSTAIN_TEXT, Settings, load_settings

app = typer.Typer(no_args_is_help=True, help="Front Line PHP RAG.")

EXIT_ANSWERED, EXIT_ERROR, EXIT_NO_INFORMATION = 0, 1, 2
API_URL_ENV = "FLP_API_URL"
SSE_CONTENT_TYPE = "text/event-stream"
WITHDRAWN = "--- Answer withdrawn after the checks ---"

# Tests inject an `httpx.MockTransport`; None means the real network.
http_transport: httpx.BaseTransport | None = None


def _not_yet(issue: int) -> None:
    typer.echo(f"Available from Milestone 6 (issue #{issue}).", err=True)
    raise typer.Exit(code=EXIT_ERROR)


PREVIEW_CHARS = 200


def index_dir(settings: Settings) -> Path:
    """data/index next to config.yaml, where Stage 7 writes the parent stores."""
    return settings.config_path.parent / "data" / "index"


def query_cache_path(settings: Settings) -> Path:
    return settings.config_path.parent / settings.embed.cache_path


# --------------------------------------------------------------------------- flp ask


def api_url(flag: str | None, settings: Settings) -> str:
    """The flag, then `FLP_API_URL`, then `cli.api_url` (open question 4)."""
    return (flag or os.environ.get(API_URL_ENV) or settings.cli.api_url).rstrip("/")


def request_body(
    question: str,
    *,
    session: str | None,
    chapter: list[int] | None,
    code: bool | None,
    debug: bool,
    stream: bool,
) -> dict[str, Any]:
    filters = (
        {"chapter_no": list(chapter or []), "has_code": code}
        if chapter or code is not None
        else None
    )
    return {
        "question": question,
        "session_id": session,
        "filters": filters,
        "stream": stream,
        "debug": debug,
    }


def exit_code(status: str) -> int:
    return {"answered": EXIT_ANSWERED, "no_information": EXIT_NO_INFORMATION}.get(
        status, EXIT_ERROR
    )


def pages(source: dict[str, Any]) -> str:
    start, end = source.get("page_printed_start", ""), source.get("page_printed_end", "")
    return str(start) if start == end else f"{start}-{end}"


def label_order(source: dict[str, Any]) -> int:
    label = str(source.get("label", ""))
    return int(label[1:]) if label[1:].isdigit() else 0


def is_sse(response: httpx.Response) -> bool:
    return response.headers.get("content-type", "").lower().startswith(SSE_CONTENT_TYPE)


def sources_table(sources: list[dict[str, Any]]) -> list[str]:
    """One row per source in label order: label, chapter, section, printed pages, score."""
    rows = [
        (
            str(s.get("label", "")),
            f"{int(s.get('chapter_no', 0)):02d} {s.get('chapter_title', '')}",
            str(s.get("section_title", "")),
            pages(s),
            f"{float(s.get('score', 0.0)):.2f}",
        )
        for s in sorted(sources, key=label_order)
    ]
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return [
        "  ".join(cell.ljust(widths[i]) if i < 4 else cell for i, cell in enumerate(row))
        for row in rows
    ]


def render(body: dict[str, Any], *, debug: bool = False, color: bool = False) -> str:
    """The record as text (Part 3): the answer or the error text, the sources, the debug field,
    and the trace id as the last line. Color only when asked (a terminal)."""
    status = body.get("status", "error")
    answer = str(body.get("answer") or "")
    detail = body.get("detail")
    detail = None if detail is None else (detail if isinstance(detail, str) else json.dumps(detail))
    if status == "no_information":
        lines = [ABSTAIN_TEXT]
    elif status == "answered":
        lines = [answer, *sources_table(body.get("sources") or [])]
    else:
        lines = [answer or detail or "error"]
        if detail and detail != lines[0]:
            lines.append(f"detail: {detail}")
    if debug and body.get("debug") is not None:
        lines.append(json.dumps(body["debug"], indent=2, ensure_ascii=False, default=str))
    trace = f"trace_id={body.get('trace_id', '')}"
    lines.append(typer.style(trace, dim=True) if color else trace)
    return "\n".join(lines)


def sse_events(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """`event:` / `data:` pairs separated by a blank line, as Stage 9 writes them."""
    name, data = "", ""
    for line in lines:
        if line.startswith("event:"):
            name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data += line.partition(":")[2].strip()
        elif not line.strip() and name:
            yield name, json.loads(data or "{}")
            name, data = "", ""
    if name:
        yield name, json.loads(data or "{}")


def _print_stream(response: httpx.Response, *, debug: bool, color: bool) -> int:
    """Part 4: pieces as they arrive, then the record from the `done` event. A `done` that is
    not `answered` after text was printed is a withdrawn answer."""
    printed = False
    for name, data in sse_events(response.iter_lines()):
        if name == "token":
            typer.echo(data.get("text", ""), nl=False)
            printed = True
        elif name == "done":
            if printed:
                typer.echo()
            status = data.get("status", "error")
            if status != "answered" and printed:
                typer.echo(WITHDRAWN)
            if status == "answered":
                typer.echo(
                    "\n".join(
                        [
                            *sources_table(data.get("sources") or []),
                            f"trace_id={data.get('trace_id', '')}",
                        ]
                    )
                )
            else:
                typer.echo(render({**data, "answer": data.get("answer")}, debug=debug, color=color))
            return exit_code(status)
    typer.echo("The stream ended without a done event.", err=True)
    return EXIT_ERROR


def _post(client: httpx.Client, body: dict[str, Any], *, stream: bool) -> httpx.Response:
    if stream:
        response = client.send(client.build_request("POST", "/ask", json=body), stream=True)
        if is_sse(response):
            return response
        response.read()
        if response.status_code < 400:  # an error answer is plain JSON by design
            typer.echo("The API did not stream; showing the normal response.", err=True)
        return response
    return client.post("/ask", json=body)


@app.command()
def ask(
    question: str,
    session: Annotated[
        str | None, typer.Option(help="Session id for follow-up questions (used from Milestone 4).")
    ] = None,
    chapter: Annotated[
        list[int] | None, typer.Option(help="Only these chapter numbers. Repeatable.")
    ] = None,
    code: Annotated[
        bool | None, typer.Option("--code/--no-code", help="Only chunks with, or without, code.")
    ] = None,
    debug: Annotated[bool, typer.Option(help="Print the debug field of the response.")] = False,
    stream: Annotated[bool, typer.Option(help="Print the answer as it streams.")] = False,
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the raw Response record.")
    ] = False,
    retrieve_only: Annotated[
        bool, typer.Option("--retrieve-only", help="Local retrieval only; no API, no answer.")
    ] = False,
    api_url_flag: Annotated[
        str | None,
        typer.Option("--api-url", help=f"API address. Default: {API_URL_ENV} or cli.api_url."),
    ] = None,
) -> None:
    """Ask the book one question through the API. Exit 0 answered, 2 no information, 1 error."""
    if stream and json_out:
        typer.echo("--stream and --json cannot be combined: the stream has no record.", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    if stream and debug:
        typer.echo(
            "--debug prints nothing with --stream: the done event has no debug field.", err=True
        )
    if retrieve_only:
        typer.echo("retrieve-only: local retrieval, this mode does not answer", err=True)
        retrieve(question, chapter, code, None)
        return
    settings = load_settings()
    url = api_url(api_url_flag, settings)
    if not url.startswith(("http://", "https://")):
        typer.echo(f"Bad API address {url!r}: it must start with http:// or https://", err=True)
        raise typer.Exit(code=EXIT_ERROR)
    body = request_body(
        question, session=session, chapter=chapter, code=code, debug=debug, stream=stream
    )
    color = sys.stdout.isatty()
    try:
        with httpx.Client(
            base_url=url, timeout=settings.cli.timeout_s, transport=http_transport
        ) as client:
            response = _post(client, body, stream=stream)
            if stream and is_sse(response):
                try:
                    code_out = _print_stream(response, debug=debug, color=color)
                except ValueError:
                    typer.echo("The API sent a malformed stream.", err=True)
                    code_out = EXIT_ERROR
                finally:
                    response.close()
                raise typer.Exit(code=code_out)
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if not isinstance(payload, dict):
                typer.echo(
                    f"The API answered {response.status_code} without a JSON record.", err=True
                )
                raise typer.Exit(code=EXIT_ERROR)
    except httpx.InvalidURL as exc:
        typer.echo(f"Bad API address {url!r}: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None
    except httpx.ConnectError:
        typer.echo(f"The API is not running at {url}. Start it with make serve.", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None
    except httpx.TimeoutException:
        typer.echo(f"The API at {url} did not answer within {settings.cli.timeout_s} s.", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None
    except httpx.HTTPError as exc:
        typer.echo(f"The request to {url} failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_ERROR) from None
    if json_out:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(render(payload, debug=debug, color=color))
    status = payload.get("status", "error") if response.status_code < 400 else "error"
    raise typer.Exit(code=exit_code(status))


@app.command()
def retrieve(
    question: str,
    chapter: Annotated[
        list[int] | None, typer.Option(help="Only these chapter numbers. Repeatable.")
    ] = None,
    code: Annotated[
        bool | None, typer.Option("--code/--no-code", help="Only chunks with, or without, code.")
    ] = None,
    top_k: Annotated[
        int | None, typer.Option(help="How many chunks. Default retrieval.top_k_dense.")
    ] = None,
) -> None:
    """List the chunks that dense retrieval returns for one question (no answer)."""
    from flp_rag.graph.nodes import retrieve as retrieve_mod
    from flp_rag.stores.embedding_cache import EmbeddingCache

    settings = load_settings()
    tracing.configure_logging()
    tracing.configure_tracing(settings)
    filters = retrieve_mod.RetrievalFilters(chapter_no=tuple(chapter or ()), has_code=code)
    trace_id = ""
    try:
        with EmbeddingCache(query_cache_path(settings)) as cache:

            @tracing.stage("rag.request", kind="CHAIN")
            def request() -> list:
                nonlocal trace_id
                trace_id = format(tracing.current_span().get_span_context().trace_id, "032x")
                tracing.current_span().set_attribute("rag.question_raw", question[:1000])
                return retrieve_mod.retrieve_dense(
                    question, settings, filters=filters, top_k=top_k, cache=cache
                )

            hits = request()
        texts = retrieve_mod.display_texts(hits, index_dir(settings))
    finally:
        tracing.shutdown_tracing()

    if not hits:
        typer.echo(f"no chunks matched. trace_id={trace_id}")
        raise typer.Exit(code=2)
    for i, h in enumerate(hits, 1):
        p = h.payload
        pages = f"p. {p['page_printed_start']}" + (
            f"-{p['page_printed_end']}" if p["page_printed_end"] != p["page_printed_start"] else ""
        )
        typer.echo(
            f"{i}. {h.chunk_id} Chapter {int(p['chapter_no']):02d}: {p['chapter_title']} "
            f"{pages} score={h.fused_score:.3f}"
        )
        preview = texts.get(h.chunk_id, "(display text not found in the parent store)")
        preview = preview[:PREVIEW_CHARS].replace("\n", " ")
        typer.echo(f"   {preview}{'...' if len(texts.get(h.chunk_id, '')) > PREVIEW_CHARS else ''}")
    typer.echo(f"trace_id={trace_id}")


@app.command()
def ingest(
    only: Annotated[str | None, typer.Option(help="Run one stage only, e.g. parse.")] = None,
    force: Annotated[
        bool, typer.Option(help="Ignore an existing index with the same hashes.")
    ] = False,
    pdf: Annotated[Path, typer.Option(help="Path to the PDF.")] = run_mod.DEFAULT_PDF,
) -> None:
    """Run Stages 1-8 (parse only until issue #14)."""
    settings = load_settings()
    tracing.configure_logging()
    tracing.configure_tracing(settings)
    try:
        results = run_mod.run(settings, pdf_path=pdf, only=only, force=force)
    finally:
        tracing.shutdown_tracing()
    parse = results.get("parse")
    if parse is not None:
        if parse.already_indexed:
            typer.echo(
                f"already indexed: doc_id={parse.doc_id} index_version={parse.already_indexed}"
            )
            return
        typer.echo(
            f"PARSE OK: doc_id={parse.doc_id} pages={parse.pages_parsed}/{parse.pages_total} "
            f"words={parse.words_total} header_words_deleted={parse.header_words_deleted} "
            f"chapters={parse.chapters_found} pages_without_header={parse.pages_without_header} "
            f"glyphs_recovered={parse.glyphs_recovered} -> {parse.output_path}"
        )
    struct = results.get("structure")
    if struct is not None:
        typer.echo(
            f"STRUCTURE OK: blocks={struct.blocks_total} by_type={struct.blocks_by_type} "
            f"chapters={struct.chapters_found} code_merged={struct.code_blocks_merged_across_pages} "
            f"callouts_merged={struct.callouts_merged_across_pages} "
            f"paragraphs_merged={struct.paragraphs_merged_across_pages} "
            f"labels_dropped={struct.labels_dropped} -> {struct.output_path}"
        )
    cleaned = results.get("clean")
    if cleaned is not None:
        typer.echo(
            f"CLEAN OK: blocks={cleaned.blocks_total} changed={cleaned.blocks_changed} "
            f"rule_counts={cleaned.rule_counts} chars_removed={cleaned.chars_removed} "
            f"-> {cleaned.output_path}"
        )
    chunked = results.get("chunk")
    if chunked is not None:
        typer.echo(
            f"CHUNK OK: chunks={chunked.chunks_total} parents={chunked.parents_total} "
            f"tokens p50={chunked.tokens_p50} p95={chunked.tokens_p95} max={chunked.tokens_max} "
            f"with_code={chunked.chunks_with_code} over_max={chunked.chunks_over_max} "
            f"under_min={chunked.chunks_under_min} composition={chunked.composition} "
            f"-> {chunked.output_path}"
        )
        width = max(chunked.histogram.values(), default=1)
        for bucket, n in chunked.histogram.items():
            typer.echo(f"  {bucket:>9} | {'#' * (40 * n // width):<40} {n}")
    enriched = results.get("enrich")
    if enriched is not None:
        typer.echo(
            f"ENRICH OK: chunks={enriched.chunks_enriched} index_version={enriched.index_version} "
            f"llm_calls={enriched.llm_calls} cost_usd={enriched.cost_usd} -> {enriched.output_path}"
        )
    embedded = results.get("embed")
    if embedded is not None:
        typer.echo(
            f"EMBED OK: vectors={embedded.vectors_total} dim={embedded.dense_dim} "
            f"model={embedded.embed_model} sparse={embedded.sparse_model} "
            f"cache_hits={embedded.cache_hits} cache_misses={embedded.cache_misses} "
            f"batches={embedded.batches} self_retrieval_failures={embedded.self_retrieval_failures} "
            f"-> {embedded.output_path}"
        )
    indexed = results.get("index")
    if indexed is not None:
        typer.echo(
            f"INDEX OK: collection={indexed.collection} points={indexed.points_upserted} "
            f"parents={indexed.parents_stored} chunks={indexed.chunks_stored} "
            f"alias_switched={indexed.alias_switched} -> {indexed.manifest_path}"
        )
    verified = results.get("verify")
    if verified is not None:
        if verified.passed:
            typer.echo(
                f"VERIFY OK: {verified.smoke_hits}/{verified.smoke_total} smoke queries hit; "
                f"alias {settings.index.alias} -> {verified.collection}"
                + (
                    f" (previous {verified.previous_collection})"
                    if verified.previous_collection
                    else ""
                )
                + (f"; pruned {verified.pruned}" if verified.pruned else "")
            )
        else:
            typer.echo(
                f"VERIFY FAILED: {', '.join(verified.failed_checks)}; "
                f"smoke {verified.smoke_hits}/{verified.smoke_total}; alias unchanged -> {verified.manifest_path}",
                err=True,
            )
        for flag in verified.flags:
            typer.echo(f"  flag: {flag}")
    if not run_mod.verify_passed(results):
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_() -> None:
    """Run the evaluation harness. Milestone 6."""
    _not_yet(34)


@app.command()
def calibrate() -> None:
    """Threshold sweep of spec Section 8.4. Milestone 6."""
    _not_yet(35)
