# Request and response models for the API.
"""Wire types for Stage 9. `AskResponse` mirrors the `Response` contract (spec Section 10)."""

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from flp_rag.contracts import Response


class FiltersIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter_no: list[int] | None = None
    has_code: bool | None = None


class AskRequest(BaseModel):
    """Length limits are Stage 10's job (issue #18), with its exact 400 message."""

    model_config = ConfigDict(extra="forbid")

    question: str
    session_id: str | None = None
    filters: FiltersIn | None = None
    stream: bool = False
    debug: bool = False


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    rating: Literal["up", "down"]
    comment: str | None = None


class SourceOut(BaseModel):
    label: str
    chunk_id: str
    chapter_no: int
    chapter_title: str
    section_title: str
    page_printed_start: int
    page_printed_end: int
    score: float


class AskResponse(BaseModel):
    answer: str
    status: Literal["answered", "no_information", "error"]
    sources: list[SourceOut]
    trace_id: str
    timings_ms: dict[str, int]
    model: dict[str, str]
    index_version: str
    prompt_version: str
    debug: dict[str, Any] | None = None

    @classmethod
    def from_contract(
        cls, r: Response, *, index_version: str, prompt_version: str
    ) -> "AskResponse":
        return cls(
            answer=r.answer,
            status=r.status,
            sources=[SourceOut(**asdict(s)) for s in r.sources],
            trace_id=r.trace_id,
            timings_ms=r.timings_ms,
            model=r.model,
            index_version=r.index_version or index_version,
            prompt_version=r.prompt_version or prompt_version,
            debug=r.debug,
        )


class ErrorResponse(BaseModel):
    """Every error carries the trace id (spec Stage 9 rule 3)."""

    status: Literal["error"] = "error"
    answer: str = ""
    trace_id: str
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_version: str | None
    alias: str
    collection: str | None
    models: dict[str, str]
    prompt_version: str
    config_hash: str
    backends: dict[str, bool]
    trace_id: str
