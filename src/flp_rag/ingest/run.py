# Runs Stages 1-8 in order. Writes data/<stage>/*.jsonl. Entry point for `make ingest`.
"""Ingestion runner.

Opens the root span `ingest.run` and calls the stages in order. Until issue #14 lands, only
Stage 1 is wired; `only="parse"` runs it alone.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from flp_rag.ingest import s01_parse, s02_structure
from flp_rag.settings import Settings
from flp_rag.tracing import current_span, git_sha, stage

log = structlog.get_logger()

STAGES: tuple[str, ...] = ("parse", "structure")  # later: clean, chunk, enrich, embed, index, verify

DEFAULT_PDF = Path("data/raw/front-line-php-revised-for-php-82.pdf")


@stage("ingest.run", kind="CHAIN")
def run(
    settings: Settings,
    *,
    pdf_path: Path = DEFAULT_PDF,
    only: str | None = None,
    force: bool = False,
    data_dir: Path = Path("data"),
) -> dict[str, Any]:
    """Run the ingestion stages. Returns one result record per stage that ran."""
    if only is not None and only not in STAGES:
        raise ValueError(f"unknown stage {only!r}; choose from {STAGES}")
    started = time.perf_counter()
    span = current_span()
    span.set_attribute("rag.git_sha", git_sha())
    span.set_attribute("rag.chunk_config_hash", settings.chunk_config_hash)
    span.set_attribute("rag.config_hash", settings.config_hash)

    results: dict[str, Any] = {}
    outcome = "ok"
    doc_id = s01_parse.compute_doc_id(pdf_path)
    span.set_attribute("rag.doc_id", doc_id)
    steps: dict[str, Callable[[], Any]] = {
        "parse": lambda: s01_parse.parse(
            pdf_path, settings, out_dir=data_dir / "parsed", index_dir=data_dir / "index",
            force=force,
        ),
        "structure": lambda: s02_structure.structure(
            doc_id, settings, in_dir=data_dir / "parsed", out_dir=data_dir / "blocks"
        ),
    }
    for name in STAGES:
        if only is not None and name != only:
            continue
        results[name] = steps[name]()
        if name == "parse" and results[name].already_indexed is not None:
            outcome = "already_indexed"
            break
    span.set_attribute("rag.duration_s", round(time.perf_counter() - started, 3))
    span.set_attribute("rag.result", outcome)
    return results
