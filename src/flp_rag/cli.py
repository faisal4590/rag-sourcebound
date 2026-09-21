# Typer CLI: flp ask, flp ingest, flp eval, flp calibrate. Spec Appendix D.
"""Command line entry point. Subcommands are stubs until their milestone lands."""

from pathlib import Path
from typing import Annotated

import typer

from flp_rag import tracing
from flp_rag.ingest import run as run_mod
from flp_rag.settings import Settings, load_settings

app = typer.Typer(no_args_is_help=True, help="Front Line PHP RAG.")


def _not_yet(issue: int) -> None:
    typer.echo(f"not implemented yet, see issue #{issue}", err=True)
    raise typer.Exit(code=2)


PREVIEW_CHARS = 200


def index_dir(settings: Settings) -> Path:
    """data/index next to config.yaml, where Stage 7 writes the parent stores."""
    return settings.config_path.parent / "data" / "index"


def query_cache_path(settings: Settings) -> Path:
    return settings.config_path.parent / settings.embed.cache_path


@app.command()
def ask(
    question: str,
    chapter: Annotated[list[int] | None, typer.Option(help="Only these chapter numbers. Repeatable.")] = None,
    code: Annotated[bool | None, typer.Option("--code/--no-code", help="Only chunks with, or without, code.")] = None,
    top_k: Annotated[int | None, typer.Option(help="How many chunks. Default retrieval.top_k_dense.")] = None,
) -> None:
    """Retrieve chunks for one question (Milestone 2: dense retrieval only, no answer yet)."""
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
                return retrieve_mod.retrieve_dense(question, settings, filters=filters, top_k=top_k, cache=cache)

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
    force: Annotated[bool, typer.Option(help="Ignore an existing index with the same hashes.")] = False,
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
                + (f" (previous {verified.previous_collection})" if verified.previous_collection else "")
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
