# Loads config.yaml into pydantic models and computes config_hash. Spec Section 9.
"""Configuration loader.

One file, `config.yaml`, holds every tunable value. API keys never live there; they come from
environment variables. `config_hash = sha256(file bytes)[:8]` goes on every trace.

Issue #2 types the `trace` section. Issue #5 types the remaining sections; until then they pass
through as plain dicts under `extra="allow"`.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ABSTAIN_TEXT = "No information found"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_SECRET_KEY = re.compile(r"(^|_)(api_?key|secret|token|password)(_|$)")


class TraceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    exporter_endpoint: str
    capture_content: bool = True
    retention_days: int = 30


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    abstain_text: str = ABSTAIN_TEXT
    trace: TraceConfig
    config_hash: str = Field(pattern=r"^[0-9a-f]{8}$")
    config_path: Path

    @model_validator(mode="before")
    @classmethod
    def _reject_secrets(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaked = sorted(_find_secret_keys(data))
            if leaked:
                raise ValueError(
                    f"config.yaml holds secret-looking keys {leaked}; "
                    "move them to environment variables"
                )
        return data

    @model_validator(mode="after")
    def _abstain_text_is_exact(self) -> "Settings":
        if self.abstain_text != ABSTAIN_TEXT:
            raise ValueError(f"abstain_text must be exactly {ABSTAIN_TEXT!r}")
        return self


def _find_secret_keys(node: Any, prefix: str = "") -> list[str]:
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if _SECRET_KEY.search(str(key).lower()):
            found.append(path)
        found.extend(_find_secret_keys(value, path))
    return found


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Read `config.yaml`, validate it, and stamp it with its hash."""
    config_path = Path(path)
    raw_bytes = config_path.read_bytes()
    data = yaml.safe_load(raw_bytes) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must hold a mapping at the top level")
    return Settings(
        **data,
        config_hash=hashlib.sha256(raw_bytes).hexdigest()[:8],
        config_path=config_path,
    )
