# Stage 8 - Ingestion verification and alias switch. Spec Section 5, Stage 8.
"""Stage 8: prove that the index is complete before any request uses it.

Input: the new collection, the parent store, the enriched chunks, the chapter table, and the
smoke queries. Output: a `verify` block in the manifest and, when every check passes, the alias
switch.

Checks (spec Section 5, Stage 8):
1. Coverage: every chapter has at least one chunk, and the chunks of a chapter cover at least
   `verify.min_chapter_coverage` of its clean tokens (a block is covered when its normalized
   text sits inside a chunk of the chapter). Chapters under the ratio are flagged, not failed.
2. Duplicates: no two chunks share `display_text`.
3. Smoke retrieval: every query in `verify.smoke_queries` returns its expected chapter in the
   top 5 dense results of the new collection.
4. Parent integrity: every `parent_id` in the store's `chunks` table exists in `parents`.

If any check fails, the alias stays where it is and the run fails. On success the alias moves to
the new collection atomically, the previous collection stays for one rollback
(`index.keep_previous`), and older `flp_chunks_*` collections are deleted.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import tiktoken
from qdrant_client import QdrantClient, models

from flp_rag.contracts import Attrs, Block, Chapter, Chunk, read_jsonl
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.ingest.s06_s07_index import (
    COLLECTION_PREFIX,
    DENSE_VECTOR,
    MANIFEST_FILE,
    STORE_FILE,
    alias_target,
    qdrant_client,
)
from flp_rag.models import Embedder, get_embeddings
from flp_rag.settings import Settings
from flp_rag.stores.embedding_cache import EmbeddingCache
from flp_rag.stores.parent_store import ParentStore
from flp_rag.tracing import stage

log = structlog.get_logger()

SMOKE_TOP_K = 5


@dataclass(frozen=True)
class SmokeQuery:
    query: str
    expected_chapter: int


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifyResult:
    doc_id: str
    index_version: str
    collection: str
    passed: bool
    checks: list[CheckResult]
    smoke_hits: int
    smoke_total: int
    flags: list[str]
    alias_switched: bool
    previous_collection: str | None
    pruned: list[str]
    manifest_path: Path

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_attrs(self) -> Attrs:
        return {
            "rag.doc_id": self.doc_id,
            "rag.index_version": self.index_version,
            "rag.collection": self.collection,
            "rag.checks_passed": sum(1 for c in self.checks if c.passed),
            "rag.checks_failed": len(self.failed_checks),
            "rag.failed_checks": json.dumps(self.failed_checks),
            "rag.smoke_hits": self.smoke_hits,
            "rag.smoke_total": self.smoke_total,
            "rag.flags": json.dumps(self.flags),
            "rag.alias_switched": self.alias_switched,
            "rag.pruned": json.dumps(self.pruned),
        }


# --------------------------------------------------------------------------- checks


def load_smoke_queries(path: Path) -> list[SmokeQuery]:
    out: list[SmokeQuery] = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.append(SmokeQuery(query=str(row["query"]), expected_chapter=int(row["expected_chapter"])))
    if not out:
        raise ValueError(f"{path} holds no smoke queries")
    return out


def normalize_text(text: str) -> str:
    """Lowercase with whitespace runs collapsed: the form in which a block is looked up inside a
    chunk. Shared with eval/chunk_checks.py so both coverage numbers mean the same thing."""
    return " ".join(text.lower().split())


def check_coverage(
    chunks: Sequence[Chunk],
    blocks: Sequence[Block],
    chapters: Sequence[Chapter],
    min_coverage: float,
    enc: tiktoken.Encoding,
) -> CheckResult:
    """Every chapter has a chunk, and the chunks of a chapter cover at least `min_coverage` of
    its clean tokens. A block is covered when its normalized text is a substring of a normalized
    chunk of the same chapter; chapter titles are not chunk text and do not count. Chapters
    under the ratio are flagged in the details with their uncovered block ids, not failed."""
    chunk_texts: dict[int, list[str]] = {}
    for c in chunks:
        chunk_texts.setdefault(int(c.payload["chapter_no"]), []).append(normalize_text(c.display_text))
    empty = [ch.title for ch in chapters if not chunk_texts.get(ch.chapter_no)]
    thin: list[dict[str, Any]] = []
    for ch in chapters:
        texts = chunk_texts.get(ch.chapter_no, [])
        total = covered = 0
        uncovered: list[str] = []
        for b in blocks:
            if b.chapter_no != ch.chapter_no or b.type == "chapter_title":
                continue
            n = len(enc.encode(b.text))
            total += n
            if any(normalize_text(b.text) in t for t in texts):
                covered += n
            else:
                uncovered.append(b.block_id)
        ratio = covered / total if total else 1.0
        if ratio < min_coverage:
            thin.append({"chapter": ch.title, "coverage": round(ratio, 4), "tokens": total,
                         "uncovered_blocks": uncovered})
    return CheckResult(
        "coverage", passed=not empty,
        details={"chapters": len(chapters), "empty": empty, "under_min_coverage": thin},
    )


def check_duplicates(chunks: Sequence[Chunk]) -> CheckResult:
    seen: dict[str, str] = {}
    dupes: list[list[str]] = []
    for c in chunks:
        if c.display_text in seen:
            dupes.append([seen[c.display_text], c.chunk_id])
        else:
            seen[c.display_text] = c.chunk_id
    return CheckResult("duplicates", passed=not dupes, details={"pairs": dupes[:10], "count": len(dupes)})


def check_parents(store_path: Path) -> CheckResult:
    with ParentStore(store_path) as store:
        orphans = store.orphan_chunk_ids()
        counts = {"parents": store.count("parents"), "chunks": store.count("chunks")}
    return CheckResult("parents", passed=not orphans, details={**counts, "orphans": orphans[:10]})


def check_smoke(
    client: QdrantClient,
    collection: str,
    queries: Sequence[SmokeQuery],
    embedder: Embedder,
    cache: EmbeddingCache | None = None,
    top_k: int = SMOKE_TOP_K,
) -> CheckResult:
    """Each query's expected chapter appears among the top-k dense hits."""
    rows: list[dict[str, Any]] = []
    hits = 0
    for q in queries:
        vector = cache.get_query(embedder.model_name, q.query) if cache else None
        if vector is None:
            vector = embedder.embed_query(q.query)
            if cache:
                cache.put_query(embedder.model_name, q.query, vector)
        points = client.query_points(collection, query=vector, using=DENSE_VECTOR, limit=top_k).points
        chapters = [int(p.payload["chapter_no"]) for p in points]
        hit = q.expected_chapter in chapters
        hits += int(hit)
        rows.append({"query": q.query, "expected": q.expected_chapter, "top_chapters": chapters, "hit": hit})
    return CheckResult("smoke", passed=hits == len(queries), details={"hits": hits, "total": len(queries), "queries": rows})


# --------------------------------------------------------------------------- alias and pruning


def switch_alias(client: QdrantClient, alias: str, collection: str) -> str | None:
    """Point `alias` at `collection` in one atomic operation. Returns the previous target."""
    previous = alias_target(client, alias)
    if previous == collection:
        return previous
    ops: list[Any] = []
    if previous is not None:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=ops)
    return previous


_RUN_NUMBER = re.compile(r"^v(\d+)-")


def run_number(collection: str, prefix: str = COLLECTION_PREFIX) -> int:
    m = _RUN_NUMBER.match(collection.removeprefix(prefix))
    return int(m.group(1)) if m else -1


def prune_collections(
    client: QdrantClient, current: str, keep_previous: int, prefix: str = COLLECTION_PREFIX
) -> list[str]:
    """Delete old `prefix*` collections. Kept: the current one, the `keep_previous` newest
    others by run number, and every collection that any alias still points at."""
    in_use = {a.collection_name for a in client.get_aliases().aliases}
    others = sorted(
        (c.name for c in client.get_collections().collections
         if c.name.startswith(prefix) and c.name != current),
        key=run_number, reverse=True,
    )
    keep = {current} | in_use | set(others[:keep_previous])
    deleted = [name for name in others if name not in keep]
    for name in deleted:
        client.delete_collection(name)
    return sorted(deleted)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomic: a temp file next to the manifest, then rename."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- the stage


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@stage("ingest.verify", kind="CHAIN")
def verify(
    doc_id: str,
    settings: Settings,
    *,
    in_dir: Path = Path("data/enriched"),
    clean_dir: Path = Path("data/clean"),
    parsed_dir: Path = Path("data/parsed"),
    index_dir: Path = Path("data/index"),
    smoke_path: Path | None = None,
    cache_path: Path | None = None,
    client: QdrantClient | None = None,
    embedder: Embedder | None = None,
) -> VerifyResult:
    """Run Stage 8 on the index that Stage 7 built. Never raises for a failed check: the result
    says which checks failed, the manifest records it, and the alias stays put."""
    chunks = list(read_jsonl(in_dir / f"{doc_id}.jsonl", Chunk))
    chapters = list(read_jsonl(parsed_dir / f"{doc_id}.chapters.jsonl", Chapter))
    index_version = str(chunks[0].payload["index_version"])
    version_dir = index_dir / index_version
    manifest_path = version_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} is missing; Stage 7 (index) must run for {index_version} before verify"
        )
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    if "collection" not in manifest:
        raise ValueError(f"{manifest_path} has no `collection`; rerun Stage 7 for {index_version}")
    collection = str(manifest["collection"])
    blocks = list(read_jsonl(clean_dir / f"{doc_id}.jsonl", Block))

    client = client or qdrant_client(settings)
    embedder = embedder or get_embeddings(settings)
    root = settings.config_path.parent
    queries = load_smoke_queries(smoke_path or (root / settings.verify.smoke_queries))

    checks = [
        check_coverage(chunks, blocks, chapters, settings.verify.min_chapter_coverage, tokenizer(settings.chunk)),
        check_duplicates(chunks),
        check_parents(version_dir / STORE_FILE),
    ]
    with EmbeddingCache(cache_path or (root / settings.embed.cache_path)) as cache:
        smoke = check_smoke(client, collection, queries, embedder, cache)
    checks.append(smoke)
    passed = all(c.passed for c in checks)
    flags = [
        f"{t['chapter']}: coverage {t['coverage']} ({len(t['uncovered_blocks'])} blocks uncovered)"
        for t in checks[0].details["under_min_coverage"]
    ]
    for flag in flags:
        log.warning("chapter under min coverage", detail=flag)

    def report(*, switched: bool, previous: str | None, pruned: list[str]) -> dict[str, Any]:
        return {
            "passed": passed,
            "verified_at": _now(),
            "checks": {c.name: {"passed": c.passed, **c.details} for c in checks},
            "smoke_hits": smoke.details["hits"],
            "smoke_total": smoke.details["total"],
            "flags": flags,
            "alias": settings.index.alias,
            "alias_switched": switched,
            "previous_collection": previous,
            "pruned": pruned,
        }

    # Record the verdict before touching the alias. If the process dies between the two writes,
    # the manifest says passed but not switched; the next run re-verifies and the switch is
    # idempotent, and Stage 7 skips a collection that is already behind the alias.
    write_manifest(manifest_path, {**manifest, "verify": report(switched=False, previous=None, pruned=[])})

    previous: str | None = None
    switched = False
    pruned: list[str] = []
    if passed:
        previous = switch_alias(client, settings.index.alias, collection)
        switched = True
        pruned = prune_collections(client, collection, settings.index.keep_previous)
        write_manifest(manifest_path, {**manifest, "verify": report(switched=True, previous=previous, pruned=pruned)})
    else:
        log.error("verification failed; alias unchanged", failed=[c.name for c in checks if not c.passed])

    return VerifyResult(
        doc_id=doc_id,
        index_version=index_version,
        collection=collection,
        passed=passed,
        checks=checks,
        smoke_hits=smoke.details["hits"],
        smoke_total=smoke.details["total"],
        flags=flags,
        alias_switched=switched,
        previous_collection=previous,
        pruned=pruned,
        manifest_path=manifest_path,
    )
