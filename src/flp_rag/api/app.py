# FastAPI: POST /ask, POST /feedback, GET /health. Spec Section 6, Stage 9.
"""Stage 9: accept a question, start the trace, and return a typed response.

Rules (spec Section 6, Stage 9):
1. `request_id` is a UUIDv7. The root span `rag.request` opens before any other work.
2. The root span carries `rag.request_id`, `rag.session_id`, `rag.index_version`,
   `rag.prompt_version`, `rag.config_hash`.
3. Every response carries `trace_id`, error responses included. A middleware opens the root
   span for every request, so even a validation error has one, and sets `X-Trace-Id`.
4. A request over `api.timeout_s` is cancelled and answered with HTTP 504 and the `trace_id`.
   Every error that leaves the graph early (400, 504, 500, 501) answers with the same
   `Response` shape as a normal request, built by Stage 18's builder: status `error`, the fixed
   `api.error_text`, no sources, plus a `detail` line. Stage 18 also writes the one summary
   log line for these requests, so every request has exactly one.
   LangGraph runs sync nodes in a thread pool, and a Python thread cannot be interrupted: the
   node that is running when the deadline passes finishes on its own, but no later node starts
   and its output is dropped. Each node therefore bounds its own calls (Qdrant client timeout,
   model call timeouts from Stage 16 on) so nothing outlives the deadline by much.
5. With `stream: true` the answer goes out as Server-Sent Events; the last event is `done`
   with `status` and `sources`.

The graph is compiled once at startup from `Deps` built from settings. Tests pass their own
deps and graph through `create_app`.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog
import uuid_utils
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.trace import Status, StatusCode

from flp_rag import tracing
from flp_rag.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FeedbackRequest,
    HealthResponse,
)
from flp_rag.contracts import Response
from flp_rag.graph.build import Deps, compile_default
from flp_rag.graph.nodes import generate as gen_mod
from flp_rag.graph.nodes import respond as respond_mod
from flp_rag.graph.nodes.guard_input import InputRejected
from flp_rag.graph.nodes.retrieve import RetrievalFilters
from flp_rag.graph.state import RagState, initial_state
from flp_rag.ingest.s06_s07_index import (
    COLLECTION_PREFIX,
    MANIFEST_FILE,
    STORE_FILE,
    alias_target,
    qdrant_client,
)
from flp_rag.models import get_embeddings, get_sparse_embeddings
from flp_rag.settings import Settings, load_settings
from flp_rag.stores.embedding_cache import EmbeddingCache

log = structlog.get_logger()

GraphFactory = Callable[[Deps], CompiledStateGraph]


# --------------------------------------------------------------------------- versions


def prompt_version(settings: Settings) -> str:
    """`sha256(prompt file)[:8]` of the answer prompt (spec Stage 16 rule 5)."""
    path = settings.config_path.parent / settings.gen.prompt_file
    return gen_mod.prompt_version(path) if path.exists() else ""


def current_index_version(settings: Settings, deps: Deps) -> tuple[str | None, str | None]:
    """(index_version, collection) behind the alias, or (None, None) when no alias exists."""
    try:
        client = deps.client or qdrant_client(settings)
        collection = (
            alias_target(client, settings.index.alias)
            if deps.collection is None
            else deps.collection
        )
    except (httpx.HTTPError, OSError, ValueError) as exc:
        log.warning("qdrant unreachable", error=str(exc))
        return None, None
    if collection is None:
        return None, None
    version = collection.removeprefix(COLLECTION_PREFIX)
    manifest = settings.config_path.parent / "data" / "index" / version / MANIFEST_FILE
    if manifest.exists():
        with contextlib.suppress(ValueError, OSError):
            version = str(json.loads(manifest.read_text()).get("index_version", version))
    return version, collection


# --------------------------------------------------------------------------- app state


@dataclass
class AppState:
    settings: Settings
    deps: Deps
    graph: CompiledStateGraph
    index_version: str | None
    collection: str | None
    prompt_version: str


def build_deps(settings: Settings) -> Deps:
    cache = EmbeddingCache(settings.config_path.parent / settings.embed.cache_path)
    return Deps(
        settings=settings,
        client=qdrant_client(settings),
        embedder=get_embeddings(settings),
        sparse=get_sparse_embeddings(settings),
        cache=cache,
    )


def create_app(
    settings: Settings | None = None,
    *,
    deps: Deps | None = None,
    graph_factory: GraphFactory = compile_default,
    configure_tracing: bool = True,
) -> FastAPI:
    settings = settings or load_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if configure_tracing:
            tracing.configure_logging()
            tracing.configure_tracing(settings)
        owns_deps = deps is None
        d = deps or build_deps(settings)
        version, collection = current_index_version(settings, d)
        if version is not None and d.store_path is None:
            d.store_path = default_index_dir(settings) / version / STORE_FILE
        app.state.rag = AppState(
            settings=settings,
            deps=d,
            graph=graph_factory(d),
            index_version=version,
            collection=collection,
            prompt_version=prompt_version(settings),
        )
        log.info("api ready", index_version=version, collection=collection, port=settings.api.port)
        yield
        if owns_deps and d.cache is not None:
            d.cache.close()
        if configure_tracing:
            tracing.shutdown_tracing()

    app = FastAPI(title="flp-rag", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_span(request: Request, call_next: Callable[[Request], Any]) -> Any:
        """One root span per request. The trace id is on `request.state` for every handler.

        Unhandled errors are answered here, inside the span, so the exception lands on the
        trace and the body still carries the trace id (spec Stage 9 rule 3).
        """
        st: AppState | None = getattr(request.app.state, "rag", None)
        tracer = tracing.get_tracer()
        with tracer.start_as_current_span(
            "rag.request", record_exception=False, set_status_on_exception=False
        ) as span:
            request_id = str(uuid_utils.uuid7())
            trace_id = format(span.get_span_context().trace_id, "032x")
            request.state.request_id, request.state.trace_id = request_id, trace_id
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("rag.request_id", request_id)
            span.set_attribute("http.route", request.url.path)
            if st is not None:
                span.set_attribute("rag.index_version", st.index_version or "")
                span.set_attribute("rag.prompt_version", st.prompt_version)
                span.set_attribute("rag.config_hash", st.settings.config_hash)
            started = time.perf_counter()
            try:
                response = await call_next(request)
            except Exception as exc:  # noqa: BLE001 - every failure must answer with a trace id
                response = _error_response(exc, trace_id, request.url.path)
            span.set_attribute("rag.latency_ms", int((time.perf_counter() - started) * 1000))
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 400:
                span.set_attribute("rag.status", "error")
            response.headers["X-Trace-Id"] = trace_id
            return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(
            trace_id=getattr(request.state, "trace_id", ""), detail=_first_error(exc)
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={400: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
    )
    async def ask(request: Request, body: AskRequest) -> Any:
        st: AppState = request.app.state.rag
        span = tracing.current_span()
        span.set_attribute("rag.session_id", body.session_id or "")
        span.set_attribute("rag.question_raw", body.question[:1000])
        filters = (
            RetrievalFilters(
                chapter_no=tuple(body.filters.chapter_no or ()), has_code=body.filters.has_code
            )
            if body.filters
            else None
        )
        state = initial_state(
            body.question,
            request_id=request.state.request_id,
            trace_id=request.state.trace_id,
            session_id=body.session_id,
            filters=filters,
            stream=body.stream,
            debug=body.debug,
            index_version=st.index_version or "",
        )
        try:
            final = await asyncio.wait_for(
                st.graph.ainvoke(state), timeout=st.settings.api.timeout_s
            )
        except InputRejected as exc:
            # Stage 10 rule 1: the exact message, HTTP 400, trace id included.
            return _early_exit(state, st, exc, 400, str(exc), reason="guard.rejected")
        except TimeoutError as exc:
            log.error(
                "request timed out",
                timeout_s=st.settings.api.timeout_s,
                trace_id=request.state.trace_id,
            )
            detail = f"request exceeded {st.settings.api.timeout_s} s"
            return _early_exit(state, st, exc, 504, detail, reason="timeout")
        except Exception as exc:  # noqa: BLE001 - a failed stage still answers in the one shape
            status, detail = error_status(exc, request.state.trace_id)
            _record_error(exc, request.url.path, request.state.trace_id)
            return _early_exit(state, st, exc, status, detail, reason=None)
        response = _response_from_state(final, st)
        span.set_attribute("rag.status", response.status)
        span.set_attribute("rag.abstain_reason", final.get("abstain_reason") or "")
        span.set_attribute("rag.cost_usd", float(final.get("cost_usd", 0.0)))
        if body.stream:
            return StreamingResponse(_sse(response), media_type="text/event-stream")
        return response

    @app.post("/feedback")
    async def feedback(request: Request, body: FeedbackRequest) -> dict[str, Any]:
        """Milestone 3 stub: accepted and logged. Milestone 5 (#31) writes the score to the trace."""
        log.info("feedback", trace_id=body.trace_id, rating=body.rating, comment=body.comment or "")
        return {"ok": True, "trace_id": request.state.trace_id}

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        st: AppState = request.app.state.rag
        s = st.settings
        bound = s.api.health_timeout_s
        try:
            qdrant_ok = await asyncio.wait_for(asyncio.to_thread(_qdrant_reachable, st), bound)
        except TimeoutError:
            log.warning("qdrant health check timed out", timeout_s=bound)
            qdrant_ok = False
        phoenix_ok = await _phoenix_reachable(s.trace.exporter_endpoint, bound)
        healthy = qdrant_ok and phoenix_ok and bool(st.index_version)
        return HealthResponse(
            status="ok" if healthy else "degraded",
            index_version=st.index_version,
            alias=s.index.alias,
            collection=st.collection,
            models={
                "embed": s.embed.model,
                "sparse": s.embed.sparse_model,
                "rerank": s.rerank.model,
                "gen": s.gen.model,
                "query": s.query.model,
                "judge": s.output.judge_model,
            },
            prompt_version=st.prompt_version,
            config_hash=s.config_hash,
            backends={"qdrant": qdrant_ok, "phoenix": phoenix_ok},
            trace_id=request.state.trace_id,
        )

    return app


# --------------------------------------------------------------------------- helpers


def _first_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request"
    e = errors[0]
    loc = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
    return f"{loc}: {e.get('msg', 'invalid')}"


def _record_error(exc: Exception, path: str, trace_id: str) -> None:
    """The exception on the root span and one ERROR line (spec 7.8 rule 5); never twice."""
    span = tracing.current_span()
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"[:200]))
    log.error("request failed", error=f"{type(exc).__name__}: {exc}", path=path, trace_id=trace_id)


def error_status(exc: Exception, trace_id: str) -> tuple[int, str]:
    """500 for an unexpected failure, 501 for a stage that a later issue implements. The
    detail is safe to show: it never carries the exception message of an unexpected failure."""
    if isinstance(exc, NotImplementedError):
        # A placeholder names its issue. That message is ours and safe to show.
        return 501, f"NotImplementedError: {exc}"
    # The full error is in the log under this trace id; the client gets no internals.
    return 500, f"internal error, see trace {trace_id}"


def _error_response(exc: Exception, trace_id: str, path: str) -> JSONResponse:
    """A failure outside `/ask` (no graph state): the short error body with the trace id."""
    _record_error(exc, path, trace_id)
    tracing.current_span().set_attribute("rag.status", "error")
    status, detail = error_status(exc, trace_id)
    body = ErrorResponse(trace_id=trace_id, detail=detail)
    return JSONResponse(status_code=status, content=body.model_dump())


def _early_exit(
    state: RagState, st: AppState, exc: Exception, status: int, detail: str, *, reason: str | None
) -> JSONResponse:
    """A request that left the graph before Stage 18: the same `Response` shape through the
    Stage 18 builder, the one summary log line, and the root span attributes. The record and
    the line hold the totals known before the error (the state the graph started from), and
    the debug field names only the exception class and the safe detail."""
    contract = respond_mod.build_response(
        state, st.settings, error=f"{type(exc).__name__}: {detail}"
    )
    span = tracing.current_span()
    span.set_attribute("rag.status", "error")
    if reason:
        span.set_attribute("rag.abstain_reason", reason)
    summary = respond_mod.summarize({**state, "abstain_reason": reason}, contract, st.settings)
    respond_mod.log_request(summary)
    out = AskResponse.from_contract(
        contract,
        index_version=st.index_version or "",
        prompt_version=st.prompt_version,
        detail=detail,
    )
    return JSONResponse(status_code=status, content=out.model_dump())


def _response_from_state(final: RagState, st: AppState) -> AskResponse:
    contract: Response | None = final.get("response")
    if contract is None:  # a graph whose respond node did not build the record (test stubs)
        contract = respond_mod.build_response(final, st.settings)
    return AskResponse.from_contract(
        contract, index_version=st.index_version or "", prompt_version=st.prompt_version
    )


async def _sse(response: AskResponse) -> AsyncIterator[bytes]:
    """Milestone 3 streaming: the finished answer in word-sized `token` events, then `done`.
    Stage 16 (#21) replaces the token source with the model stream and the 40-char abstain buffer."""
    if response.status == "answered":
        words = response.answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            yield _event("token", {"text": token})
    done = {
        "status": response.status,
        "sources": [s.model_dump() for s in response.sources],
        "trace_id": response.trace_id,
        "answer": response.answer if response.status != "answered" else None,
    }
    yield _event("done", done)


def _event(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _qdrant_reachable(st: AppState) -> bool:
    try:
        client = st.deps.client or qdrant_client(st.settings)
        client.get_collections()
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable" here
        log.warning("qdrant health check failed", error=str(exc))
        return False


async def _phoenix_reachable(exporter_endpoint: str, timeout_s: float) -> bool:
    base = exporter_endpoint.split("/v1/")[0]
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{base}/healthz")
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def default_index_dir(settings: Settings) -> Path:
    return settings.config_path.parent / "data" / "index"


app = create_app()
