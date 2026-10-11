# "Done when" test for the assemble node. Spec Section 6, Stage 15. Issue #20.
"""Stage 15 context assembly: selection, grouping, budget, order, labels, code fences, span.
Unit tests run on a synthetic parent store with no graph, no index, and no model; the graph
tests use stub nodes around the real gate, assemble, and respond."""

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import tracing
from flp_rag.contracts import Block, Candidate, Chunk, Parent
from flp_rag.graph import build
from flp_rag.graph.nodes import assemble as a
from flp_rag.graph.state import RagState, initial_state
from flp_rag.ingest.s04_chunk import tokenizer
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.parent_store import ParentStore
from tests.test_graph_build import stub_nodes

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (ROOT / "config.yaml").read_text()
CODE = (
    "class CustomerDTO\n{\n    public function __construct(\n        public string $name,\n    ) {}\n}"
)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="module")
def enc(settings):
    return tokenizer(settings.chunk)


def words(n: int, seed: str = "alpha") -> str:
    vocabulary = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
    start = vocabulary.index(seed) if seed in vocabulary else 0
    return " ".join(vocabulary[(start + i) % len(vocabulary)] for i in range(n)) + "."


# --------------------------------------------------------------------------- synthetic book


def block(
    bid: str,
    type_: str,
    text: str,
    *,
    chapter: int,
    title: str,
    section: str,
    page: int,
    index: int,
) -> Block:
    return Block(
        bid, type_, text, "Part 1", chapter, title, section, page + 2, page, index, type_ == "code"
    )  # type: ignore[arg-type]


def chunk(
    cid: str, parent_id: str, blocks: list[Block], *, chapter: int, title: str, section: str
) -> Chunk:
    text = "\n\n".join(b.text for b in blocks)
    pages = [b.page_printed for b in blocks]
    payload = {
        "parent_id": parent_id,
        "chapter_no": chapter,
        "chapter_title": title,
        "section_title": section,
        "page_printed_start": min(pages),
        "page_printed_end": max(pages),
        "block_ids": [b.block_id for b in blocks],
        "token_count": len(text.split()),
    }
    return Chunk(cid, parent_id, text, text, len(text.split()), payload)


def parent(pid: str, blocks: list[Block], *, chapter: int, section: str, enc) -> Parent:
    text = "\n\n".join(b.text for b in blocks)
    pages = [b.page_printed for b in blocks]
    return Parent(pid, chapter, section, text, min(pages), max(pages), len(enc.encode(text)))


@pytest.fixture(scope="module")
def book(tmp_path_factory, enc):
    """Chapter 5 'Constructor promotion' (two chunks, one parent, with code), Chapter 6 two
    sections, Chapter 30 one section. Returns (store_path, chunks by id, parents by id)."""
    b5 = [
        block(
            "d:066:000",
            "heading",
            "Constructor promotion",
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
            page=64,
            index=0,
        ),
        block(
            "d:066:001",
            "paragraph",
            words(60),
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
            page=64,
            index=1,
        ),
        block(
            "d:066:002",
            "code",
            CODE,
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
            page=64,
            index=2,
        ),
        block(
            "d:067:000",
            "paragraph",
            words(60, "beta"),
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
            page=65,
            index=0,
        ),
        block(
            "d:068:000",
            "paragraph",
            words(60, "sigma"),
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
            page=66,
            index=0,
        ),
    ]
    b6a = [
        block(
            "d:079:000",
            "heading",
            "Readonly and clone",
            chapter=6,
            title="Readonly Properties",
            section="Readonly and clone",
            page=77,
            index=0,
        ),
        block(
            "d:079:001",
            "paragraph",
            words(80, "gamma"),
            chapter=6,
            title="Readonly Properties",
            section="Readonly and clone",
            page=77,
            index=1,
        ),
    ]
    b6b = [
        block(
            "d:083:000",
            "heading",
            "Readonly classes",
            chapter=6,
            title="Readonly Properties",
            section="Readonly classes",
            page=81,
            index=0,
        ),
        block(
            "d:083:001",
            "paragraph",
            words(80, "delta"),
            chapter=6,
            title="Readonly Properties",
            section="Readonly classes",
            page=81,
            index=1,
        ),
    ]
    b30 = [
        block(
            "d:311:000",
            "paragraph",
            words(70, "omega"),
            chapter=30,
            title="PHP 8.2",
            section="PHP 8.2",
            page=309,
            index=0,
        )
    ]
    chunks = [
        chunk(
            "d:05:0038",
            "d:05:s006",
            b5[:3],
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
        ),
        chunk(
            "d:05:0039",
            "d:05:s006",
            b5[3:4],
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
        ),
        chunk(
            "d:05:0040",
            "d:05:s006",
            b5[4:],
            chapter=5,
            title="Property Promotion",
            section="Constructor promotion",
        ),
        chunk(
            "d:06:0046",
            "d:06:s010",
            b6a,
            chapter=6,
            title="Readonly Properties",
            section="Readonly and clone",
        ),
        chunk(
            "d:06:0047",
            "d:06:s011",
            b6b,
            chapter=6,
            title="Readonly Properties",
            section="Readonly classes",
        ),
        chunk("d:30:0200", "d:30:s090", b30, chapter=30, title="PHP 8.2", section="PHP 8.2"),
    ]
    parents = [
        parent("d:05:s006", b5, chapter=5, section="Constructor promotion", enc=enc),
        parent("d:06:s010", b6a, chapter=6, section="Readonly and clone", enc=enc),
        parent("d:06:s011", b6b, chapter=6, section="Readonly classes", enc=enc),
        parent("d:30:s090", b30, chapter=30, section="PHP 8.2", enc=enc),
    ]
    path = tmp_path_factory.mktemp("store") / "store.sqlite"
    with ParentStore(path) as store:
        store.put_parents(parents)
        store.put_chunks(chunks)
        store.put_blocks(b5 + b6a + b6b + b30, chunks)
    return path, {c.chunk_id: c for c in chunks}, {p.parent_id: p for p in parents}


def cand(chunks: dict[str, Chunk], cid: str, score: float) -> Candidate:
    c = chunks[cid]
    return Candidate(cid, score, score, ["dense"], c.payload)


# --------------------------------------------------------------------------- store


def test_store_keeps_block_boundaries_in_order(book) -> None:
    path, _, _ = book
    with ParentStore(path) as store:
        blocks = store.get_blocks("d:05:s006")
        assert [b.block_id for b in blocks] == [
            "d:066:000",
            "d:066:001",
            "d:066:002",
            "d:067:000",
            "d:068:000",
        ]
        assert [b.type for b in blocks] == [
            "heading",
            "paragraph",
            "code",
            "paragraph",
            "paragraph",
        ]
        assert blocks[2].text == CODE and blocks[2].chunk_id == "d:05:0038"
        assert [b.block_id for b in store.get_chunk_blocks("d:05:0039")] == ["d:067:000"]
        assert store.count("blocks") == 10 and store.get_blocks("nope") == []


# --------------------------------------------------------------------------- pure functions


def test_selection_keeps_scores_at_or_above_tau_pass_and_at_most_max_blocks(settings, book) -> None:
    _, chunks, _ = book
    cands = [
        cand(chunks, "d:05:0038", 0.9),
        cand(chunks, "d:06:0046", 0.35),
        cand(chunks, "d:06:0047", 0.3),
        cand(chunks, "d:30:0200", 0.5),
        cand(chunks, "d:05:0039", 0.6),
    ]
    picked = a.select_candidates(cands, settings.gate.tau_pass, 3)
    assert [c.chunk_id for c in picked] == ["d:05:0038", "d:05:0039", "d:30:0200"]
    assert [c.chunk_id for c in a.select_candidates(cands, settings.gate.tau_pass, 8)][
        -1
    ] == "d:06:0046"


def test_grouping_one_group_per_parent_scored_by_its_best_chunk(book) -> None:
    _, chunks, _ = book
    groups = a.group_by_parent(
        [
            cand(chunks, "d:05:0039", 0.6),
            cand(chunks, "d:30:0200", 0.7),
            cand(chunks, "d:05:0038", 0.9),
        ]
    )
    assert [(g.parent_id, g.score) for g in groups] == [("d:05:s006", 0.9), ("d:30:s090", 0.7)]
    assert [c.chunk_id for c in groups[0].chunks] == [
        "d:05:0038",
        "d:05:0039",
    ]  # book order inside the group


def test_render_fences_code_with_the_original_indentation(book) -> None:
    path, _, _ = book
    with ParentStore(path) as store:
        text = a.render_blocks(store.get_blocks("d:05:s006"))
    assert "```php\n" + CODE + "\n```" in text
    assert text.startswith("Constructor promotion\n\n") and "        public string $name," in text


def test_label_line_names_chapter_section_and_pages() -> None:
    assert (
        a.label_line("S1", 6, "Readonly Properties", "Readonly and clone", 77, 77)
        == "[S1] Chapter 06: Readonly Properties > Readonly and clone, p. 77"
    )
    assert (
        a.label_line("S2", 30, "PHP 8.2", "PHP 8.2", 309, 311)
        == "[S2] Chapter 30: PHP 8.2, p. 309-311"
    )


def test_two_hits_in_one_section_give_one_parent_block(settings, enc, book) -> None:
    path, chunks, _ = book
    with ParentStore(path) as store:
        result = a.assemble(
            [cand(chunks, "d:05:0038", 0.9), cand(chunks, "d:05:0039", 0.6)], store, settings, enc
        )
    assert len(result.blocks) == 1 and result.n_parents_used == 1 and result.truncated is False
    blk = result.blocks[0]
    assert blk.label == "S1" and blk.source_ids == ["d:05:0038", "d:05:0039"]
    assert blk.text.startswith(
        "[S1] Chapter 05: Property Promotion > Constructor promotion, p. 64-66\n"
    )
    assert "```php" in blk.text and blk.token_count == len(enc.encode(blk.text))
    assert result.tokens_used == blk.token_count


def test_blocks_are_in_book_order_and_labels_follow_position(settings, enc, book) -> None:
    path, chunks, _ = book
    with ParentStore(path) as store:
        result = a.assemble(
            [cand(chunks, "d:30:0200", 0.9), cand(chunks, "d:06:0046", 0.5)], store, settings, enc
        )
    assert [(b.label, b.chapter_no) for b in result.blocks] == [("S1", 6), ("S2", 30)]
    assert (
        result.label_map["S2"]["chunk_ids"] == ["d:30:0200"]
        and result.label_map["S1"]["section_title"] == "Readonly and clone"
    )
    assert set(result.label_map) == {"S1", "S2"}


def budget_settings(settings: Settings, max_tokens: int) -> Settings:
    return settings.model_copy(
        update={"context": settings.context.model_copy(update={"max_tokens": max_tokens})}
    )


def test_budget_falls_back_to_chunks_then_best_chunk_then_skips(settings, enc, book) -> None:
    path, chunks, parents = book
    p5 = parents["d:05:s006"].token_count  # the chapter 5 parent, about 150 tokens
    with ParentStore(path) as store:
        # Budget for the Chapter 6 parent (highest score) plus the Chapter 5 best chunk only.
        c6 = parents["d:06:s010"].token_count + 20
        best5 = len(enc.encode(a.render_blocks(store.get_chunk_blocks("d:05:0038")))) + 40
        small = budget_settings(settings, c6 + best5)
        result = a.assemble(
            [
                cand(chunks, "d:06:0046", 0.9),
                cand(chunks, "d:05:0038", 0.8),
                cand(chunks, "d:05:0039", 0.7),
            ],
            store,
            small,
            enc,
        )
        assert (
            result.n_parents_used == 1
            and result.n_chunk_fallbacks == 1
            and result.truncated is True
        )
        five = next(b for b in result.blocks if b.chapter_no == 5)
        assert five.source_ids == ["d:05:0038"] and "```php" in five.text
        # A budget with room for the parent of Chapter 6 only: the Chapter 5 group is skipped.
        tiny = budget_settings(settings, c6)
        result = a.assemble(
            [cand(chunks, "d:06:0046", 0.9), cand(chunks, "d:05:0038", 0.8)], store, tiny, enc
        )
        assert result.n_skipped == 1 and len(result.blocks) == 1 and result.truncated is True
        assert p5 > c6 - parents["d:06:s010"].token_count  # sanity: the parent really did not fit


def test_all_passing_chunks_fallback_comes_before_best_chunk(settings, enc, book) -> None:
    # Two of the three chunks of the Chapter 5 parent pass. Room for both chunks but not for the
    # parent (its third chunk adds 60 words): option 2 wins over option 3.
    path, chunks, _ = book
    with ParentStore(path) as store:
        both = a.render_blocks(
            store.get_chunk_blocks("d:05:0038") + store.get_chunk_blocks("d:05:0039")
        )
        label = a.label_line("S8", 5, "Property Promotion", "Constructor promotion", 64, 65)
        budget = len(enc.encode(f"{label}\n{both}")) + 5
        result = a.assemble(
            [cand(chunks, "d:05:0038", 0.9), cand(chunks, "d:05:0039", 0.8)],
            store,
            budget_settings(settings, budget),
            enc,
        )
        assert (
            result.n_parents_used == 0
            and result.n_chunk_fallbacks == 1
            and result.truncated is True
        )
        assert (
            result.blocks[0].source_ids == ["d:05:0038", "d:05:0039"]
            and result.label_map["S1"]["kind"] == "chunks"
        )


def test_a_candidate_whose_chunk_is_missing_from_the_store_is_an_error(settings, enc, book) -> None:
    path, chunks, _ = book
    ghost = Candidate(
        "d:99:0001", 0.9, 0.9, ["dense"], {**chunks["d:05:0038"].payload, "chapter_no": 99}
    )
    ghost_parentless = Candidate("d:99:0001", 0.9, 0.9, ["dense"], chunks["d:05:0038"].payload)
    with ParentStore(path) as store, pytest.raises(KeyError, match="d:99:0001"):
        a.assemble([ghost], store, settings, enc)
    with ParentStore(path) as store, pytest.raises(KeyError):
        a.assemble([ghost_parentless], store, settings, enc)


# --------------------------------------------------------------------------- configuration


def test_context_budget_must_exceed_the_largest_chunk(tmp_path: Path) -> None:
    from tests.test_settings import _load

    with pytest.raises(ValueError) as exc:
        _load(tmp_path, SHIPPED.replace("max_tokens: 6000", "max_tokens: 500"))
    assert (
        "500" in str(exc.value)
        and "900" in str(exc.value)
        and "context.max_tokens" in str(exc.value)
    )


def test_context_order_accepts_position_only(tmp_path: Path) -> None:
    from tests.test_settings import _load

    with pytest.raises(ValueError, match="order"):
        _load(tmp_path, SHIPPED.replace("order: position", "order: sandwich"))


# --------------------------------------------------------------------------- node in the graph


def test_node_writes_the_span_and_the_state(settings, book) -> None:
    path, chunks, _ = book
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    stubs = stub_nodes()
    nodes = build.default_nodes(build.Deps(settings=settings, store_path=path))

    def rerank(state: RagState) -> dict[str, Any]:
        return {
            "candidates": [
                cand(chunks, "d:05:0038", 0.9),
                cand(chunks, "d:05:0039", 0.6),
                cand(chunks, "d:30:0200", 0.7),
            ]
        }

    mixed = build.GraphNodes(
        **{**stubs.as_dict(), "rerank": rerank, "gate": nodes.gate, "assemble": nodes.assemble}
    )
    state = build.build_graph(mixed).invoke(
        initial_state("What is constructor property promotion?", request_id="r1")
    )
    attrs = {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}["context.assemble"]
    assert attrs["rag.n_blocks"] == 2 and attrs["rag.n_parents_used"] == 2
    assert attrs["rag.n_chunk_fallbacks"] == 0 and attrs["rag.n_skipped"] == 0
    assert attrs["rag.truncated"] is False and attrs["rag.tokens_used"] == sum(
        b.token_count for b in state["context"]
    )
    assert json.loads(attrs["rag.source_ids"]) == ["d:05:0038", "d:05:0039", "d:30:0200"]
    assert [b.label for b in state["context"]] == ["S1", "S2"] and set(state["label_map"]) == {
        "S1",
        "S2",
    }
    assert "generate" in state["visited"]


def test_node_without_a_store_path_names_the_problem(settings) -> None:
    nodes = build.default_nodes(build.Deps(settings=settings))
    state = initial_state("q", request_id="r1")
    state["candidates"] = [Candidate("d:05:0038", 0.9, 0.9, ["dense"], {"chapter_no": 5})]
    with pytest.raises(RuntimeError, match="store"):
        nodes.assemble(state)


# --------------------------------------------------------------------------- example 3 and 4


def sized_store(
    tmp_path: Path, enc, parent_tokens: list[int], chunk_tokens: int
) -> tuple[Path, list[Candidate]]:
    """Six parents in six chapters, each with two chunks: one passing chunk of `chunk_tokens`
    and one that completes the parent to `parent_tokens[i]`. Every block belongs to a chunk,
    as in the real store, so the parent renders from all of its blocks."""
    path = tmp_path / "store.sqlite"
    cands: list[Candidate] = []
    with ParentStore(path) as store:
        parents, chunks, blocks = [], [], []
        for i, n in enumerate(parent_tokens, start=1):
            pid, title = f"d:{i:02d}:s000", f"Chapter {i}"
            hit = block(
                f"d:{100 + i:03d}:000",
                "paragraph",
                words(chunk_tokens, "alpha"),
                chapter=i,
                title=title,
                section="S",
                page=i * 10,
                index=0,
            )
            rest = block(
                f"d:{100 + i:03d}:001",
                "paragraph",
                words(n - chunk_tokens, "beta"),
                chapter=i,
                title=title,
                section="S",
                page=i * 10,
                index=1,
            )
            c_hit = chunk(f"d:{i:02d}:0000", pid, [hit], chapter=i, title=title, section="S")
            c_rest = chunk(f"d:{i:02d}:0001", pid, [rest], chapter=i, title=title, section="S")
            chunks += [c_hit, c_rest]
            parents.append(parent(pid, [hit, rest], chapter=i, section="S", enc=enc))
            blocks += [hit, rest]
            cands.append(
                Candidate(c_hit.chunk_id, 1.0 - i / 100, 1.0 - i / 100, ["dense"], c_hit.payload)
            )
        store.put_parents(parents)
        store.put_chunks(chunks)
        store.put_blocks(blocks, chunks)
    return path, cands


def test_example_3_four_parents_two_chunk_fallbacks(tmp_path: Path, settings, enc) -> None:
    # Parents of about 1600, 1400, 1200, 1100, 900, 800 tokens; chunks of about 300. Labels and
    # separators add a few tokens per block, so the budget is 6000 like the issue.
    path, cands = sized_store(tmp_path, enc, [1600, 1400, 1200, 1100, 900, 800], 300)
    # The issue counts body tokens only; each block also carries a label line of about 20
    # tokens, so the budget gets that overhead for six blocks on top of the 6,000.
    budget = settings.context.max_tokens + 6 * 20
    with ParentStore(path) as store:
        result = a.assemble(cands, store, budget_settings(settings, budget), enc)
    assert len(result.blocks) == 6
    assert (result.n_parents_used, result.n_chunk_fallbacks, result.n_skipped) == (4, 2, 0)
    assert result.truncated is True and result.tokens_used <= budget
    assert [result.label_map[f"S{i}"]["kind"] for i in range(1, 7)] == ["parent"] * 4 + [
        "chunk" if False else "chunks"
    ] * 2


def test_example_4_a_group_that_does_not_fit_is_skipped(tmp_path: Path, settings, enc) -> None:
    path, cands = sized_store(tmp_path, enc, [1600, 1400, 1200, 1100, 900, 800, 1000], 420)
    with ParentStore(path) as store:
        result = a.assemble(cands, store, settings, enc)
    assert result.n_skipped >= 1 and result.truncated is True
    assert result.tokens_used <= settings.context.max_tokens
    assert len(result.blocks) == 7 - result.n_skipped


def test_label_map_carries_pages_and_chunk_ids(book, settings, enc) -> None:
    path, chunks, _ = book
    with ParentStore(path) as store:
        result = a.assemble([cand(chunks, "d:06:0046", 0.9)], store, settings, enc)
    entry = result.label_map["S1"]
    assert entry["chunk_ids"] == ["d:06:0046"] and (
        entry["page_printed_start"],
        entry["page_printed_end"],
    ) == (77, 77)
    assert entry["chapter_no"] == 6 and entry["kind"] == "parent"
    assert all(label in result.label_map for label in (b.label for b in result.blocks))


def test_store_without_blocks_falls_back_to_plain_text_and_says_so(
    tmp_path: Path, book, settings, enc
) -> None:
    _, chunks, parents = book
    path = tmp_path / "old.sqlite"
    with ParentStore(path) as store:
        store.put_parents(parents.values())
        store.put_chunks(chunks.values())
    with ParentStore.open_readonly(path) as store:
        result = a.assemble([cand(chunks, "d:05:0038", 0.9)], store, settings, enc)
    assert result.plain_text is True and "```php" not in result.blocks[0].text
    assert CODE in result.blocks[0].text  # the code is there, just not fenced


def test_read_only_open_refuses_a_missing_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="flp ingest"):
        ParentStore.open_readonly(tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()


def test_duplicate_block_ownership_is_an_error(tmp_path: Path, book) -> None:
    _, chunks, _ = book
    twin = Chunk(
        "d:05:0099",
        "d:05:s006",
        "x",
        "x",
        1,
        {**chunks["d:05:0038"].payload, "block_ids": ["d:066:001"]},
    )
    with (
        ParentStore(tmp_path / "dup.sqlite") as store,
        pytest.raises(ValueError, match="d:066:001"),
    ):
        store.put_blocks([], [chunks["d:05:0038"], twin])
