"""Done-when tests for settings.py. Spec Section 9. Issues #2 and #5."""

import re
from pathlib import Path

import pytest

from flp_rag.settings import ABSTAIN_TEXT, load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_abstain_text_is_exact() -> None:
    assert ABSTAIN_TEXT == "No information found"


def test_loads_shipped_config() -> None:
    s = load_settings()
    assert s.abstain_text == ABSTAIN_TEXT
    assert s.trace.exporter_endpoint == "http://localhost:6006/v1/traces"
    assert s.trace.capture_content is True
    assert s.trace.retention_days == 30


def test_config_hash_is_eight_hex_chars() -> None:
    s = load_settings()
    assert re.fullmatch(r"[0-9a-f]{8}", s.config_hash)
    assert s.config_hash == load_settings().config_hash


def test_config_hash_tracks_file_bytes(tmp_path: Path) -> None:
    src = (ROOT / "config.yaml").read_text()
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(src)
    b.write_text(src.replace("retention_days: 30", "retention_days: 31"))
    assert load_settings(a).config_hash != load_settings(b).config_hash
    assert load_settings(b).trace.retention_days == 31


def test_missing_required_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("abstain_text: 'No information found'\n")
    with pytest.raises(ValueError, match="trace"):
        load_settings(p)


@pytest.mark.parametrize(
    "line",
    ["openai_api_key: sk-abc", "anthropic_key: sk-ant", "gen:\n  token: x", "db_password: pw",
     "cohere_credentials: c"],
)
def test_secret_looking_keys_are_rejected(tmp_path: Path, line: str) -> None:
    src = (ROOT / "config.yaml").read_text()
    p = tmp_path / "leaky.yaml"
    p.write_text(src + "\n" + line + "\n")
    with pytest.raises(ValueError, match="secret-looking"):
        load_settings(p)


def test_token_limits_are_not_secrets() -> None:
    # max_tokens, tokenizer, prices_usd_per_million_tokens all load.
    assert load_settings().trace.content_max_chars == 20_000
