# Issue #37: contextual summaries in Stage 5 (spec Stage 5, rules 5 and 7; Milestone 7, item 1).
"""One small-model call per chunk writes a short summary that is prepended to `embedding_text`;
the summary is cached by `sha256(prompt_version + chunk_id + display_text)`, so a rerun costs
nothing; failures are retried, counted, and bounded; the enrichment prompt is versioned; the
chunk config hash covers the switch and the prompt, so an index with summaries is a new index."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flp_rag import models, tracing
from flp_rag.contracts import Chunk, read_jsonl, write_jsonl
from flp_rag.ingest import s05_enrich as s5
from flp_rag.ingest import s05_summary as summ
from flp_rag.ingest.s06_s07_index import build_manifest
from flp_rag.settings import INGEST_SOURCE_FILES, Settings, chunk_config_hash, load_settings
from flp_rag.stores.summary_cache import SummaryCache
from tests.test_generate import FakeChat
from tests.test_s05_enrich import _chunk

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "Defines the CustomerDTO class with typed public properties and a constructor."


def with_summaries(settings: Settings, **extra: Any) -> Settings:
    update = {"contextual_summary": True, "summary_pause_s": 0.0, **extra}
    return settings.model_copy(update={"enrich": settings.enrich.model_copy(update=update)})


class RateLimited(Exception):
    """Shaped like the OpenAI client's RateLimitError: a status and a Retry-After header."""

    status_code = 429

    def __init__(self, retry_after: str) -> None:
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"retry-after": retry_after}})()


def summarizer(
    settings: Settings, cache: SummaryCache, *outputs: Any, sleeps: list[float] | None = None
) -> tuple[summ.Summarizer, FakeChat]:
    chat = FakeChat(*outputs, model_name="fake:small")
    prompt = summ.load_summary_prompt(ROOT / settings.enrich.summary_prompt_file)
    made = summ.Summarizer(
        chat,
        prompt,
        settings.enrich,
        cache,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )
    return made, chat


# --------------------------------------------------------------------------- prompt and key


def test_prompt_file_splits_into_system_rules_and_a_chunk_template() -> None:
    settings = load_settings()
    path = ROOT / settings.enrich.summary_prompt_file
    prompt = summ.load_summary_prompt(path)
    assert prompt.version == hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    assert (
        prompt.system.startswith("You write a short search summary")
        and "Chunk:" not in prompt.system
    )
    user = summ.user_message(prompt, _chunk())
    assert user.startswith("Chunk:\nChapter 5: Property Promotion\nSection: Property Promotion")
    assert "Printed pages 64-64" in user and user.rstrip().endswith(
        "}"
    )  # the code is in the message
    assert "{text}" not in user and "{chapter_no}" not in user


def test_cache_key_covers_prompt_version_chunk_id_and_text() -> None:
    key = summ.summary_key("abcd1234", "d:05:0046", "text")
    assert key == hashlib.sha256(b"abcd1234d:05:0046text").hexdigest()
    assert key != summ.summary_key("abcd1235", "d:05:0046", "text")
    assert key != summ.summary_key("abcd1234", "d:05:0047", "text")
    assert key != summ.summary_key("abcd1234", "d:05:0046", "text.")


def test_check_summary_cleans_the_output_and_rejects_empty_or_long_text() -> None:
    assert summ.check_summary('  "Readonly  properties\nare set once."  ', 80) == (
        "Readonly properties are set once."
    )
    assert summ.check_summary("```text\nOne sentence.\n```", 80) == "One sentence."
    assert summ.check_summary("Summary: One sentence.", 80) == "One sentence."
    with pytest.raises(models.InvalidOutput):
        summ.check_summary("   ", 80)
    with pytest.raises(models.InvalidOutput, match="words"):
        summ.check_summary("word " * 81, 80)


# --------------------------------------------------------------------------- the summarizer


def test_summary_is_prepended_and_the_second_run_comes_from_the_cache(tmp_path: Path) -> None:
    settings = with_summaries(load_settings())
    chunk = _chunk()
    with SummaryCache(tmp_path / "s.sqlite") as cache:
        first, chat = summarizer(settings, cache, SUMMARY)
        out, tally = first.apply(chunk)
        assert out.embedding_text == f"{SUMMARY}\n{chunk.embedding_text}"
        assert out.display_text == chunk.display_text  # the display text never changes
        assert out.payload["context_summary"] == SUMMARY
        assert out.payload["enrich_prompt_version"] == first.prompt.version
        assert tally == summ.Tally(
            calls=1, input_tokens=1000, output_tokens=100, cost_usd=0.0045, failures=0, cache_hits=0
        )
        system, messages, temperature = chat.calls[0]
        assert (
            "search summary" in system
            and messages[0][0] == "user"
            and "CustomerDTO" in messages[0][1]
        )
        assert temperature == settings.enrich.summary_temperature
        assert chat.max_tokens == settings.enrich.summary_max_output_tokens
        assert cache.count() == 1

        second, chat2 = summarizer(settings, cache)  # no scripted output: a call would fail
        again, tally2 = second.apply(chunk)
        assert again.embedding_text == out.embedding_text and chat2.calls == []
        assert tally2 == summ.Tally(0, 0, 0, 0.0, 0, 1)

        # a changed text is a new key: the cached summary of the old text is not reused
        third, chat3 = summarizer(settings, cache, "Other summary.")
        changed, _ = third.apply(replace(chunk, display_text=chunk.display_text + " // edit"))
        assert changed.payload["context_summary"] == "Other summary." and len(chat3.calls) == 1
        # another model's row is a miss: the summary is written again by the new model
        key = summ.summary_key(first.prompt.version, chunk.chunk_id, chunk.display_text)
        assert cache.get(key, model="other:model") is None
        other = summ.Summarizer(
            FakeChat("By the other model.", model_name="other:model"),
            first.prompt,
            settings.enrich,
            cache,
            sleep=lambda _s: None,
        )
        swapped, _ = other.apply(chunk)
        assert swapped.payload["context_summary"] == "By the other model."
        assert cache.get(key, model="other:model") == "By the other model."
        assert cache.count() == 2  # the row of the new model replaced the old one


def test_transport_errors_and_invalid_outputs_are_retried_then_counted(tmp_path: Path) -> None:
    settings = with_summaries(load_settings(), summary_retry_backoff_s=1.5, summary_max_failures=1)
    sleeps: list[float] = []
    with SummaryCache(tmp_path / "s.sqlite") as cache:
        # a rate limit with Retry-After 7 s, then a too-long answer, then a good one
        made, _chat = summarizer(
            settings, cache, RateLimited("7"), "word " * 200, SUMMARY, sleeps=sleeps
        )
        out, tally = made.apply(_chunk())
        assert out.payload["context_summary"] == SUMMARY
        assert tally.calls == 3 and tally.failures == 0 and tally.input_tokens == 2000
        assert sleeps[0] >= 7 and sleeps[1] == pytest.approx(
            3.0
        )  # header wins, then the doubled backoff

        # three transport errors: the chunk is kept without a summary and the failure is counted
        broken, _ = summarizer(
            settings, cache, ConnectionError("a"), ConnectionError("b"), ConnectionError("c")
        )
        plain, tally = broken.apply(replace(_chunk(), chunk_id="d:05:0047"))
        assert "context_summary" in plain.payload and plain.payload["context_summary"] is None
        assert (
            plain.embedding_text == _chunk().embedding_text
            and tally.failures == 1
            and tally.calls == 3
        )

        # an auth or config error is not a transport error: it propagates
        fatal, _ = summarizer(settings, cache, models.ModelConfigError("bad model"))
        with pytest.raises(models.ModelConfigError):
            fatal.apply(replace(_chunk(), chunk_id="d:05:0048"))


def test_enrich_all_pauses_between_calls_and_bounds_the_failures(tmp_path: Path) -> None:
    settings = with_summaries(load_settings(), summary_pause_s=2.5)
    chunks = [replace(_chunk(), chunk_id=f"d:05:00{i}") for i in range(3)]
    sleeps: list[float] = []
    with SummaryCache(tmp_path / "s.sqlite") as cache:
        made, _ = summarizer(settings, cache, "One.", "Two.", "Three.", sleeps=sleeps)
        out, tally = s5.enrich_all(
            chunks,
            settings,
            index_version="v9-x-y",
            created_at="2026-11-22T10:00:00Z",
            summarizer=made,
        )
        assert [c.payload["context_summary"] for c in out] == ["One.", "Two.", "Three."]
        assert tally.calls == 3 and sleeps == [2.5, 2.5]  # a pause after each call but the last
        s5.validate_payload(out[0].payload)
    with SummaryCache(tmp_path / "t.sqlite") as cache:
        failing, chat = summarizer(settings, cache, *([ConnectionError("x")] * 3), "Two.", "Three.")
        with pytest.raises(
            RuntimeError, match=r"1 chunk\(s\) have no summary .*stopped at chunk 1 of 3"
        ):
            s5.enrich_all(
                chunks,
                settings,
                index_version="v9-x-y",
                created_at="2026-11-22T10:00:00Z",
                summarizer=failing,
            )
        assert len(chat.calls) == 3  # the run stops at the first chunk over the bound


def test_a_quota_rate_limit_and_a_cut_output_are_not_retried(tmp_path: Path) -> None:
    settings = with_summaries(load_settings(), summary_retry_after_max_s=30, summary_max_failures=2)
    sleeps: list[float] = []
    with SummaryCache(tmp_path / "s.sqlite") as cache:
        quota, chat = summarizer(settings, cache, RateLimited("3600"), SUMMARY, sleeps=sleeps)
        plain, tally = quota.apply(_chunk())
        assert plain.payload["context_summary"] is None
        assert tally == summ.Tally(calls=1, failures=1)
        assert sleeps == [] and len(chat.calls) == 1  # no hour-long sleep, no second call
        cut, chat = summarizer(settings, cache, ("", "length"), SUMMARY)
        plain, tally = cut.apply(replace(_chunk(), chunk_id="d:05:0047"))
        assert plain.payload["context_summary"] is None
        assert tally.calls == 1 and tally.failures == 1 and len(chat.calls) == 1


def test_enrich_chunks_without_the_switch_is_unchanged_and_hypothetical_questions_still_raise() -> (
    None
):
    settings = load_settings()
    out = s5.enrich_chunks(
        [_chunk()], settings, index_version="v", created_at="2026-09-21T10:00:00Z"
    )
    assert (
        "context_summary" not in out[0].payload and out[0].embedding_text == _chunk().embedding_text
    )
    on = settings.model_copy(
        update={"enrich": settings.enrich.model_copy(update={"hypothetical_questions": 3})}
    )
    with pytest.raises(NotImplementedError, match="#38"):
        s5.enrich_chunks([_chunk()], on, index_version="v", created_at="2026-09-21T10:00:00Z")


# --------------------------------------------------------------------------- the stage


def test_enrich_stage_writes_summaries_and_the_span_attributes(tmp_path: Path) -> None:
    settings = with_summaries(load_settings(), summary_cache_path=tmp_path / "summaries.sqlite")
    chunks = [replace(_chunk(), chunk_id=f"d:05:00{i}") for i in range(2)]
    (tmp_path / "chunks").mkdir()
    write_jsonl(tmp_path / "chunks" / "d.jsonl", chunks)
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(settings, exporter=exporter, force=True)
    chat = FakeChat("First.", "Second.", model_name="fake:small")
    result = s5.enrich(
        "d",
        settings,
        in_dir=tmp_path / "chunks",
        out_dir=tmp_path / "enriched",
        index_dir=tmp_path / "index",
        chat=chat,
    )
    out = list(read_jsonl(result.output_path, Chunk))
    assert [c.payload["context_summary"] for c in out] == ["First.", "Second."]
    assert all(c.embedding_text.startswith(c.payload["context_summary"] + "\n") for c in out)
    assert result.llm_calls == 2 and result.cache_hits == 0 and result.failures == 0
    assert (
        result.prompt_version
        == summ.load_summary_prompt(ROOT / settings.enrich.summary_prompt_file).version
    )
    span = next(s for s in exporter.get_finished_spans() if s.name == "ingest.enrich")
    a = span.attributes
    assert a["rag.llm_calls"] == 2 and a["gen_ai.usage.input_tokens"] == 2000
    assert a["rag.cost_usd"] == pytest.approx(0.009) and a["rag.failures"] == 0
    assert a["rag.cache_hits"] == 0 and a["rag.prompt_version"] == result.prompt_version
    # the rerun reads the cache: no model, no calls
    again = s5.enrich(
        "d",
        settings,
        in_dir=tmp_path / "chunks",
        out_dir=tmp_path / "enriched2",
        index_dir=tmp_path / "index",
        chat=FakeChat(model_name="fake:small"),  # the same model: another model would be a miss
    )
    assert again.llm_calls == 0 and again.cache_hits == 2
    tracing.configure_tracing(load_settings(), exporter=InMemorySpanExporter(), force=True)


def test_manifest_records_the_enrichment(tmp_path: Path) -> None:
    settings = load_settings()
    plain = s5.enrich_chunk(
        _chunk(), settings, index_version="v1-x-y", created_at="2026-11-22T10:00:00Z"
    )
    summarized = summ.with_summary(plain, SUMMARY, "abcd1234")
    kwargs: dict[str, Any] = {
        "settings": settings,
        "doc_id": "d",
        "index_version": "v1-x-y",
        "collection": "c",
        "parents": [],
        "vector_meta": {},
        "points": 1,
        "created_at": "2026-11-22T10:00:00Z",
    }
    assert build_manifest(chunks=[plain], **kwargs)["enrich"] == {
        "contextual_summary": False,
        "prompt_version": None,
        "chunks_with_summary": 0,
    }
    assert build_manifest(chunks=[summarized], **kwargs)["enrich"] == {
        "contextual_summary": True,
        "prompt_version": "abcd1234",
        "chunks_with_summary": 1,
    }


# --------------------------------------------------------------------------- versions and roles


def test_chunk_config_hash_covers_the_switch_and_the_prompt(tmp_path: Path) -> None:
    shipped = load_settings()
    stamped = chunk_config_hash(
        shipped.parse,
        shipped.structure,
        shipped.clean,
        shipped.chunk,
        shipped.enrich,
        prompt_root=ROOT,
    )
    assert stamped == shipped.chunk_config_hash
    # from here the switch is forced off, whatever config.yaml says during a live build
    s = shipped.model_copy(
        update={"enrich": shipped.enrich.model_copy(update={"contextual_summary": False})}
    )
    base = chunk_config_hash(s.parse, s.structure, s.clean, s.chunk, s.enrich, prompt_root=ROOT)
    on = with_summaries(s)
    on_hash = chunk_config_hash(s.parse, s.structure, s.clean, s.chunk, on.enrich, prompt_root=ROOT)
    assert on_hash != base
    # with the switch on, a new prompt version is a new hash; with it off, the prompt is not read
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "enrich_v1.md").write_text("other rules\n\nChunk:\n{text}\n")
    other = chunk_config_hash(
        s.parse, s.structure, s.clean, s.chunk, on.enrich, prompt_root=tmp_path
    )
    assert other != on_hash
    off = chunk_config_hash(s.parse, s.structure, s.clean, s.chunk, s.enrich, prompt_root=tmp_path)
    assert off == base
    # pauses, retries, and the cache path do not shape a summary: the hash ignores them
    tuned = with_summaries(
        s, summary_pause_s=9.0, summary_retry_attempts=5, summary_cache_path=tmp_path / "x.sqlite"
    )
    tuned_hash = chunk_config_hash(
        s.parse, s.structure, s.clean, s.chunk, tuned.enrich, prompt_root=ROOT
    )
    assert tuned_hash == on_hash
    model = with_summaries(s, summary_model="groq:openai/gpt-oss-120b")
    model_hash = chunk_config_hash(
        s.parse, s.structure, s.clean, s.chunk, model.enrich, prompt_root=ROOT
    )
    assert model_hash != on_hash
    assert "s05_summary.py" in INGEST_SOURCE_FILES  # the summary code shapes embedding_text


def test_the_enrich_role_names_the_summary_model() -> None:
    settings = load_settings()
    assert models._chat_spec(settings, "enrich") == (
        "enrich.summary_model",
        settings.enrich.summary_model,
    )
    assert settings.enrich.summary_model in settings.prices_usd_per_million_tokens
    assert json.loads((ROOT / "prompts" / "registry.json").read_text())["enrich_v1.md"]
