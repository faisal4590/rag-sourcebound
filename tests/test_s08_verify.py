"""Done-when tests for Stage 8, verification and the alias switch. Spec Section 5, Stage 8. Issue #14.

Synthetic tests use QdrantClient(":memory:") and a fake embedder. The live test uses a throwaway
index version and a throwaway alias, so the production collection and alias are never touched.
"""

import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from qdrant_client import QdrantClient, models

from flp_rag import tracing
from flp_rag.contracts import Chapter, Chunk, Parent, read_jsonl, write_jsonl
from flp_rag.ingest import s01_parse as s1
from flp_rag.ingest import s06_s07_index as s7
from flp_rag.ingest import s08_verify as s8
from flp_rag.settings import load_settings
from flp_rag.stores.parent_store import ParentStore
from tests.test_s06_s07_index import (
    ENRICHED,
    PARENTS,
    VECTORS,
    FakeEmbedder,
    _qdrant_up,
)

ROOT = Path(__file__).resolve().parents[1]

needs_live = pytest.mark.skipif(
    not (ENRICHED.exists() and VECTORS.exists() and PARENTS.exists() and _qdrant_up()),
    reason="needs Stage 5 and 6 outputs in data/ and a Qdrant server at index.qdrant_url",
)

CHAPTERS = [
    Chapter(0, 5, "Property Promotion", "Part 1", "PHP, the Language", 65, 76),
    Chapter(1, 6, "Readonly Properties", "Part 1", "PHP, the Language", 77, 86),
]


def _chunk(i: int, text: str, chapter: Chapter, parent: str = "d:05:s000") -> Chunk:
    payload = {
        "doc_id": "d", "chunk_id": f"d:{chapter.chapter_no:02d}:{i:04d}", "parent_id": parent,
        "part": chapter.part, "chapter_no": chapter.chapter_no, "chapter_title": chapter.title,
        "section_title": "", "page_pdf_start": chapter.page_pdf_start, "page_pdf_end": chapter.page_pdf_start,
        "page_printed_start": chapter.page_pdf_start - 2, "page_printed_end": chapter.page_pdf_start - 2,
        "has_code": False, "code_lang": "", "code_group_id": None, "part_index": None,
        "php_symbols": [], "token_count": len(text.split()), "chunk_config_hash": "abcdef0123456789",
        "index_version": "v1-fake-abcdef01", "embed_model": "fake/embedder",
        "created_at": "2026-09-22T10:00:00Z", "chunk_kind": "prose_only",
    }
    return Chunk(payload["chunk_id"], parent, text, f"hdr {i}\n{text}", len(text.split()), payload)


def _parent(pid: str, chapter: Chapter) -> Parent:
    return Parent(pid, chapter.chapter_no, "", "section text", chapter.page_pdf_start - 2, chapter.page_pdf_end - 2, 20)


def _setup(tmp_path: Path, chunks: list[Chunk], parents: list[Parent], version: str = "v1-fake-abcdef01",
           smoke: list[dict] | None = None):
    """Stage 5, 6, 7 outputs for synthetic chunks, indexed into an in-memory Qdrant."""
    settings = load_settings()
    settings = settings.model_copy(update={"embed": settings.embed.model_copy(update={"dim": 8, "batch_size": 2, "retry_backoff_s": 0.0})})
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    chunks = [Chunk(c.chunk_id, c.parent_id, c.display_text, c.embedding_text, c.token_count,
                    {**c.payload, "index_version": version}) for c in chunks]
    write_jsonl(tmp_path / "enriched" / "d.jsonl", chunks)
    write_jsonl(tmp_path / "parents" / "d.jsonl", parents)
    write_jsonl(tmp_path / "parsed" / "d.chapters.jsonl", CHAPTERS)
    smoke_rows = smoke if smoke is not None else [
        {"query": chunks[0].embedding_text, "expected_chapter": chunks[0].payload["chapter_no"]},
        {"query": chunks[-1].embedding_text, "expected_chapter": chunks[-1].payload["chapter_no"]},
    ]
    (tmp_path / "smoke.jsonl").write_text("\n".join(json.dumps(r) for r in smoke_rows) + "\n")
    embedder = FakeEmbedder()
    s7.embed("d", settings, in_dir=tmp_path / "enriched", out_dir=tmp_path / "vectors",
             cache_path=tmp_path / "cache.sqlite", embedder=embedder)
    return settings, embedder


def _index_and_verify(tmp_path: Path, settings, embedder, client: QdrantClient):
    s7.index("d", settings, in_dir=tmp_path / "enriched", vectors_dir=tmp_path / "vectors",
             parents_dir=tmp_path / "parents", index_dir=tmp_path / "index", client=client)
    return s8.verify("d", settings, in_dir=tmp_path / "enriched", parsed_dir=tmp_path / "parsed",
                     index_dir=tmp_path / "index", smoke_path=tmp_path / "smoke.jsonl",
                     cache_path=tmp_path / "cache.sqlite", client=client, embedder=embedder)


GOOD_CHUNKS = [
    _chunk(0, "Constructor promotion removes boilerplate.", CHAPTERS[0]),
    _chunk(1, "Promoted properties may have defaults.", CHAPTERS[0]),
    _chunk(2, "Readonly properties are set once.", CHAPTERS[1], parent="d:06:s001"),
]
GOOD_PARENTS = [_parent("d:05:s000", CHAPTERS[0]), _parent("d:06:s001", CHAPTERS[1])]


# --------------------------------------------------------------------------- checks


def test_coverage_fails_on_an_empty_chapter_and_flags_thin_ones() -> None:
    ok = s8.check_coverage(GOOD_CHUNKS, CHAPTERS, min_per_page=0.5)
    assert ok.passed and ok.details["empty"] == []
    assert [t["chapter"] for t in ok.details["under_min_per_page"]] == ["Property Promotion", "Readonly Properties"]
    missing = s8.check_coverage(GOOD_CHUNKS[:2], CHAPTERS, min_per_page=0.0)
    assert not missing.passed and missing.details["empty"] == ["Readonly Properties"]


def test_duplicates_check() -> None:
    assert s8.check_duplicates(GOOD_CHUNKS).passed
    dup = s8.check_duplicates(GOOD_CHUNKS + [_chunk(9, GOOD_CHUNKS[0].display_text, CHAPTERS[0])])
    assert not dup.passed and dup.details["pairs"] == [["d:05:0000", "d:05:0009"]]


def test_parents_check_reads_the_store(tmp_path: Path) -> None:
    with ParentStore(tmp_path / "store.sqlite") as store:
        store.put_parents(GOOD_PARENTS)
        store.put_chunks(GOOD_CHUNKS + [_chunk(9, "lost", CHAPTERS[0], parent="d:99:s999")])
    result = s8.check_parents(tmp_path / "store.sqlite")
    assert not result.passed and result.details["orphans"] == ["d:05:0009"]
    assert result.details["parents"] == 2 and result.details["chunks"] == 4


def test_load_smoke_queries_rejects_empty_file(tmp_path: Path) -> None:
    (tmp_path / "s.jsonl").write_text("\n")
    with pytest.raises(ValueError, match="no smoke queries"):
        s8.load_smoke_queries(tmp_path / "s.jsonl")
    assert s8.load_smoke_queries(ROOT / "eval" / "smoke_queries.jsonl")[1] == s8.SmokeQuery(
        "How do readonly properties work?", 6)


# --------------------------------------------------------------------------- the stage


def test_verify_passes_switches_alias_and_writes_the_report(tmp_path: Path) -> None:
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS)
    client = QdrantClient(":memory:")

    result = _index_and_verify(tmp_path, settings, embedder, client)

    assert result.passed and result.failed_checks == []
    assert (result.smoke_hits, result.smoke_total) == (2, 2)
    assert result.alias_switched and result.previous_collection is None and result.pruned == []
    assert s7.alias_target(client, settings.index.alias) == "flp_chunks_v1-fake-abcdef01"
    assert result.flags == ["Property Promotion: 2 chunks over 12 pages", "Readonly Properties: 1 chunks over 10 pages"]
    manifest = json.loads(result.manifest_path.read_text())
    v = manifest["verify"]
    assert v["passed"] is True and v["alias_switched"] is True and v["smoke_hits"] == 2
    assert set(v["checks"]) == {"coverage", "duplicates", "parents", "smoke"}
    assert v["checks"]["smoke"]["queries"][0]["hit"] is True
    # Stage 1's already-indexed check now finds this run.
    assert s1.find_existing_index(tmp_path / "index", "d", "abcdef0123456789") == "v1-fake-abcdef01"


def test_verify_failure_leaves_the_alias_alone(tmp_path: Path) -> None:
    # Three chunks and top-5 means every chapter present is always a hit; expect one that is not.
    bad_smoke = [{"query": "What is the Rust borrow checker?", "expected_chapter": 99}]
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, smoke=bad_smoke)
    client = QdrantClient(":memory:")

    result = _index_and_verify(tmp_path, settings, embedder, client)

    assert not result.passed and result.failed_checks == ["smoke"]
    assert result.smoke_hits == 0 and not result.alias_switched and result.pruned == []
    assert s7.alias_target(client, settings.index.alias) is None
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["verify"]["passed"] is False
    assert s1.find_existing_index(tmp_path / "index", "d", "abcdef0123456789") is None


def test_second_version_switches_and_third_prunes(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version="v1-fake-abcdef01")
    first = _index_and_verify(tmp_path, settings, embedder, client)
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version="v2-fake-abcdef01")
    second = _index_and_verify(tmp_path, settings, embedder, client)
    assert second.previous_collection == first.collection and second.pruned == []
    assert s7.alias_target(client, settings.index.alias) == second.collection
    assert client.collection_exists(first.collection)  # kept for one rollback
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version="v3-fake-abcdef01")
    third = _index_and_verify(tmp_path, settings, embedder, client)
    assert third.previous_collection == second.collection
    assert third.pruned == [first.collection]
    assert not client.collection_exists(first.collection) and client.collection_exists(second.collection)


def test_prune_keeps_collections_that_any_alias_serves(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version="v1-fake-abcdef01")
    first = _index_and_verify(tmp_path, settings, embedder, client)
    # A second alias (staging, a rollback alias, a colleague) points at v1.
    client.update_collection_aliases(change_aliases_operations=[models.CreateAliasOperation(
        create_alias=models.CreateAlias(collection_name=first.collection, alias_name="flp_staging"))])
    for v in ("v2-fake-abcdef01", "v3-fake-abcdef01", "v4-fake-abcdef01"):
        settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version=v)
        last = _index_and_verify(tmp_path, settings, embedder, client)
    names = {c.name for c in client.get_collections().collections}
    assert first.collection in names  # served by flp_staging, never pruned
    assert "flp_chunks_v2-fake-abcdef01" not in names
    assert {"flp_chunks_v3-fake-abcdef01", "flp_chunks_v4-fake-abcdef01"} <= names
    assert last.pruned == ["flp_chunks_v2-fake-abcdef01"]


def test_keep_previous_retains_that_many_generations(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    for v in ("v1-fake-abcdef01", "v2-fake-abcdef01", "v3-fake-abcdef01", "v4-fake-abcdef01"):
        settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS, version=v)
        settings = settings.model_copy(update={"index": settings.index.model_copy(update={"keep_previous": 2})})
        last = _index_and_verify(tmp_path, settings, embedder, client)
    names = sorted(c.name for c in client.get_collections().collections)
    assert names == ["flp_chunks_v2-fake-abcdef01", "flp_chunks_v3-fake-abcdef01", "flp_chunks_v4-fake-abcdef01"]
    assert last.pruned == ["flp_chunks_v1-fake-abcdef01"]


def test_crash_between_switch_and_report_is_resumable(tmp_path: Path) -> None:
    """Simulate: verify passed and the alias moved, but the final manifest write never happened.
    The next full run must skip the rebuild in Stage 7 and finish the report in Stage 8."""
    client = QdrantClient(":memory:")
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS)
    result = _index_and_verify(tmp_path, settings, embedder, client)
    manifest = json.loads(result.manifest_path.read_text())
    manifest["verify"]["alias_switched"] = False  # the first (pre-switch) write survived, the second did not
    result.manifest_path.write_text(json.dumps(manifest))
    assert s1.find_existing_index(tmp_path / "index", "d", "abcdef0123456789") is None  # not "already indexed"

    again = s7.index("d", settings, in_dir=tmp_path / "enriched", vectors_dir=tmp_path / "vectors",
                     parents_dir=tmp_path / "parents", index_dir=tmp_path / "index", client=client)
    assert again.alias_switched is True and again.points_upserted == 3  # skipped, not rebuilt
    verified = s8.verify("d", settings, in_dir=tmp_path / "enriched", parsed_dir=tmp_path / "parsed",
                         index_dir=tmp_path / "index", smoke_path=tmp_path / "smoke.jsonl",
                         cache_path=tmp_path / "cache.sqlite", client=client, embedder=embedder)
    assert verified.passed and verified.alias_switched and verified.previous_collection == result.collection
    assert s1.find_existing_index(tmp_path / "index", "d", "abcdef0123456789") == "v1-fake-abcdef01"


def test_verify_without_a_manifest_names_stage_7(tmp_path: Path) -> None:
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS)
    with pytest.raises(FileNotFoundError, match="Stage 7"):
        s8.verify("d", settings, in_dir=tmp_path / "enriched", parsed_dir=tmp_path / "parsed",
                  index_dir=tmp_path / "index", smoke_path=tmp_path / "smoke.jsonl",
                  cache_path=tmp_path / "cache.sqlite", client=QdrantClient(":memory:"), embedder=embedder)


def test_switch_alias_is_idempotent(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    settings, embedder = _setup(tmp_path, GOOD_CHUNKS, GOOD_PARENTS)
    result = _index_and_verify(tmp_path, settings, embedder, client)
    assert s8.switch_alias(client, settings.index.alias, result.collection) == result.collection
    assert s7.alias_target(client, settings.index.alias) == result.collection


# --------------------------------------------------------------------------- live


@needs_live
def test_verify_the_real_book_under_a_throwaway_alias(tmp_path: Path) -> None:
    settings = load_settings()
    test_version = "v999-test-" + tmp_path.name[-8:].lower()
    test_alias = f"flp_test_alias_{tmp_path.name[-8:].lower()}"
    settings = settings.model_copy(update={"index": settings.index.model_copy(update={"alias": test_alias})})
    tracing.configure_tracing(settings, exporter=InMemorySpanExporter(), force=True)
    client = s7.qdrant_client(settings)
    chunks = list(read_jsonl(ENRICHED, Chunk))
    rewritten = [Chunk(c.chunk_id, c.parent_id, c.display_text, c.embedding_text, c.token_count,
                       {**c.payload, "index_version": test_version}) for c in chunks]
    write_jsonl(tmp_path / "enriched" / "bca146b9df2d0a2b.jsonl", rewritten)
    before = {c.name for c in client.get_collections().collections}

    indexed = s7.index("bca146b9df2d0a2b", settings, in_dir=tmp_path / "enriched", vectors_dir=VECTORS.parent,
                       parents_dir=PARENTS.parent, index_dir=tmp_path / "index", client=client)
    try:
        # Run the checks directly instead of verify(): verify() prunes every flp_chunks_*
        # collection that is not the new or previous one, which would remove the production one.
        cov = s8.check_coverage(rewritten, list(read_jsonl(ROOT / "data" / "parsed" / "bca146b9df2d0a2b.chapters.jsonl", Chapter)), settings.verify.min_chunks_per_page)
        dup = s8.check_duplicates(rewritten)
        par = s8.check_parents(indexed.store_path)
        from flp_rag.models import get_embeddings
        smoke = s8.check_smoke(client, indexed.collection, s8.load_smoke_queries(ROOT / "eval" / "smoke_queries.jsonl"), get_embeddings(settings))
        assert cov.passed and dup.passed and par.passed, (cov.details["empty"], dup.details, par.details["orphans"])
        assert [t["chapter"] for t in cov.details["under_min_per_page"]] == ["First-class callables"]
        assert smoke.passed, [q for q in smoke.details["queries"] if not q["hit"]]
        assert smoke.details["hits"] == 10
        previous = s8.switch_alias(client, test_alias, indexed.collection)
        assert previous is None and s7.alias_target(client, test_alias) == indexed.collection
    finally:
        try:
            client.update_collection_aliases(change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=test_alias))])
        finally:
            client.delete_collection(indexed.collection)
    assert before <= {c.name for c in client.get_collections().collections}
