# The span registry of issue #29: loads span_registry.yaml and checks spans against it.
"""One file lists every span (`span_registry.yaml`); this module reads it and applies the rules.

`load_registry()` parses the file once. `Registry.expected_spans(path)` names the spans a
request path must have. `check_span` compares one span with its entry: kind, parent, required
attributes, types, the sentinel rule (never null), the text and JSON size limits, and the
content switch. `check_trace` does it for a whole trace and adds the span set comparison.
`cost_mismatch` checks the cost sum on the root. `classify_path` reads the path of a live
trace from its root attributes and span names, for `eval/trace_coverage.py`.

`SpanView` is the one shape both readers produce: `from_readable` for the in-memory exporter
of the tests, `from_phoenix` for the rows of the Phoenix REST API. `tracing.py` reads the
sentinels and the content keys from the same file, so the rules have one source.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FILE = Path(__file__).resolve().parent / "span_registry.yaml"
KINDS = ("CHAIN", "RETRIEVER", "RERANKER", "LLM", "TOOL", "EMBEDDING")
CONTENT_VALUES = frozenset({"input.value", "output.value"})
ROOT_REQUEST, ROOT_INGEST = "rag.request", "ingest.run"
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "int": (int,),
    "float": (float, int),
    "number": (int, float),
    "bool": (bool,),
    "json": (str,),
}


@dataclass(frozen=True)
class AttrSpec:
    type: str
    sentinel: str | int | None = None


@dataclass(frozen=True)
class SpanSpec:
    name: str
    kind: str
    parent: str | None
    paths: frozenset[str] | None  # None = every path
    required: tuple[str, ...]
    conditional: dict[str, str]
    content: tuple[str, ...]
    documents: tuple[str, ...]
    documents_when: str | None
    optional: bool
    when: str | None
    enabled_by: str | None  # the configuration switch of an optional span: cache, sparse
    runtime: bool  # an optional span decided per request (check.code, the judge, a batch)
    source: str

    def on_path(self, path: str) -> bool:
        return self.paths is None or path in self.paths


@dataclass(frozen=True)
class SpanView:
    """One span as both readers hand it over: the parent by name, the kind resolved."""

    name: str
    parent: str | None
    kind: str | None
    status: str  # OK, ERROR, UNSET
    attributes: Mapping[str, Any]
    trace_id: str = ""


@dataclass(frozen=True)
class Registry:
    version: int
    text_chars: int
    json_chars: int
    truncated_marker: str
    sentinel_number: int
    sentinel_text: str
    content_keys: frozenset[str]
    content_suffixes: tuple[str, ...]
    paths: dict[str, str]
    attributes: dict[str, AttrSpec]
    request: dict[str, SpanSpec]
    ingest: dict[str, SpanSpec]
    _by_name: dict[str, SpanSpec] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name.update(self.request)
        self._by_name.update(self.ingest)

    def spec(self, name: str) -> SpanSpec | None:
        return self._by_name.get(name)

    def is_content(self, key: str) -> bool:
        return key in self.content_keys or key.endswith(self.content_suffixes)

    def sentinel_for(self, key: str) -> str | int | None:
        spec = self.attributes.get(key)
        return spec.sentinel if spec is not None else None

    def expected_spans(
        self, path: str, *, cache_enabled: bool = False, sparse_enabled: bool = True
    ) -> set[str]:
        """The spans a request on `path` must have. Optional spans join only when their
        configuration switch is on; runtime-conditional ones (check.code, check.faithfulness)
        never join: they are checked when present."""
        if path not in self.paths:
            raise KeyError(f"unknown request path {path!r}; one of {sorted(self.paths)}")
        switches = {"cache": cache_enabled or path == "cache_hit", "sparse": sparse_enabled}
        out: set[str] = set()
        for name, spec in self.request.items():
            if not spec.on_path(path):
                continue
            if spec.optional:
                if spec.enabled_by is not None and switches.get(spec.enabled_by):
                    out.add(name)
                continue
            out.add(name)
        return out


def _span_spec(name: str, raw: Mapping[str, Any]) -> SpanSpec:
    paths = raw.get("paths", "all")
    return SpanSpec(
        name=name,
        kind=str(raw["kind"]),
        parent=raw.get("parent"),
        paths=None if paths == "all" else frozenset(str(p) for p in paths),
        required=tuple(raw.get("required", ())),
        conditional=dict(raw.get("conditional", {}) or {}),
        content=tuple(raw.get("content", ())),
        documents=tuple(raw.get("documents", ())),
        documents_when=raw.get("documents_when"),
        optional=bool(raw.get("optional", False)),
        when=raw.get("when"),
        enabled_by=raw.get("enabled_by"),
        runtime=bool(raw.get("runtime", False)),
        source=str(raw.get("source", "")),
    )


@functools.cache
def load_registry(path: Path = REGISTRY_FILE) -> Registry:
    data = yaml.safe_load(path.read_text())
    limits, sentinels = data["limits"], data["sentinels"]
    registry = Registry(
        version=int(data["version"]),
        text_chars=int(limits["text_chars"]),
        json_chars=int(limits["json_chars"]),
        truncated_marker=str(limits["truncated_marker"]),
        sentinel_number=int(sentinels["number"]),
        sentinel_text=str(sentinels["text"]),
        content_keys=frozenset(data["content_keys"]),
        content_suffixes=tuple(data["content_suffixes"]),
        paths={str(k): str(v) for k, v in data["paths"].items()},
        attributes={k: AttrSpec(**v) for k, v in data["attributes"].items()},
        request={k: _span_spec(k, v) for k, v in data["request"].items()},
        ingest={k: _span_spec(k, v) for k, v in data["ingest"].items()},
    )
    _validate(registry)
    return registry


def _validate(registry: Registry) -> None:
    for table in (registry.request, registry.ingest):
        for name, spec in table.items():
            if spec.kind not in KINDS:
                raise ValueError(f"{name}: unknown kind {spec.kind!r}")
            if spec.parent is not None and spec.parent not in table:
                raise ValueError(f"{name}: parent {spec.parent!r} is not a span of the same tree")
            for key in (*spec.required, *spec.conditional, *spec.content):
                if key not in registry.attributes:
                    raise ValueError(f"{name}: {key} is not in the attribute catalog")
            if spec.paths is not None and not spec.paths <= set(registry.paths):
                raise ValueError(f"{name}: unknown path in {sorted(spec.paths)}")
            if spec.optional and not spec.runtime and spec.enabled_by not in ("cache", "sparse"):
                raise ValueError(f"{name}: an optional span needs enabled_by or runtime: true")
    for key, spec in registry.attributes.items():
        if spec.type not in _TYPES:
            raise ValueError(f"{key}: unknown type {spec.type!r}")


# --------------------------------------------------------------------------- readers


def from_readable(spans: Iterable[Any]) -> list[SpanView]:
    """Views of OpenTelemetry SDK spans (the in-memory exporter), parents resolved by id."""
    spans = list(spans)
    by_id = {s.context.span_id: s for s in spans}
    out = []
    for s in sorted(spans, key=lambda s: s.start_time):
        parent = by_id.get(s.parent.span_id) if s.parent is not None else None
        attrs = dict(s.attributes or {})
        out.append(
            SpanView(
                name=s.name,
                parent=parent.name if parent is not None else None,
                kind=attrs.get("openinference.span.kind"),
                status=s.status.status_code.name,
                attributes=attrs,
                trace_id=format(s.context.trace_id, "032x"),
            )
        )
    return out


def from_phoenix(rows: Iterable[Mapping[str, Any]]) -> list[SpanView]:
    """Views of Phoenix REST span rows (`GET /v1/projects/<p>/spans`). Phoenix keeps the dotted
    keys but parses the JSON string of `*.document.metadata` into one key per field; those are
    folded back into the one JSON string the registry names."""
    rows = list(rows)
    by_id = {str(r["context"]["span_id"]): r for r in rows}
    out = []
    for r in sorted(rows, key=lambda r: str(r.get("start_time", ""))):
        parent = by_id.get(str(r.get("parent_id") or ""))
        attrs = _fold_metadata(_flatten(r.get("attributes") or {}))
        out.append(
            SpanView(
                name=str(r["name"]),
                parent=str(parent["name"]) if parent is not None else None,
                kind=attrs.get("openinference.span.kind")
                or (r.get("span_kind") if r.get("span_kind") != "UNKNOWN" else None),
                status=str(r.get("status_code") or "UNSET"),
                attributes=attrs,
                trace_id=str(r["context"]["trace_id"]),
            )
        )
    return out


_METADATA = ".document.metadata."


def _fold_metadata(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """`x.document.metadata.chapter_no: 6` ... -> `x.document.metadata: '{"chapter_no": 6}'`."""
    out: dict[str, Any] = {}
    folded: dict[str, dict[str, Any]] = {}
    for key, value in attrs.items():
        head, sep, field = key.rpartition(_METADATA)
        if sep and "." not in field:
            folded.setdefault(head + _METADATA[:-1], {})[field] = value
        else:
            out[key] = value
    for key, record in folded.items():
        out[key] = json.dumps(record, ensure_ascii=False)
    return out


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """`{"rag": {"status": "x"}}` -> `{"rag.status": "x"}`; document lists stay indexed."""
    out: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value):
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}.{i}"))
    else:
        out[prefix] = value
    return out


# --------------------------------------------------------------------------- checks


def _type_ok(spec: AttrSpec, value: Any) -> bool:
    if spec.type == "bool":
        return isinstance(value, bool)
    if spec.type in ("int", "number", "float") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[spec.type])


def attribute_violations(registry: Registry, key: str, value: Any) -> list[str]:
    """The rules of Part 2 for one attribute: never null, the right type, text at most
    `text_chars` (longer values end with the marker), JSON at most `json_chars` and parseable."""
    if value is None:
        return [f"{key} is null (the registry sentinel is required)"]
    if key in CONTENT_VALUES:  # the stage input and output: `trace.content_max_chars` rules them
        return []
    spec = registry.attributes.get(key)
    problems: list[str] = []
    if spec is not None and not _type_ok(spec, value):
        problems.append(f"{key} has type {type(value).__name__}, the registry says {spec.type}")
    if isinstance(value, str):
        is_json = (spec is not None and spec.type == "json") or (
            spec is None and not registry.is_content(key) and value.startswith(("[", "{"))
        )
        limit = registry.json_chars if is_json else registry.text_chars
        if len(value) > limit:
            problems.append(f"{key} holds {len(value)} characters, the limit is {limit}")
        elif is_json and not value.endswith(registry.truncated_marker):
            try:
                json.loads(value)
            except ValueError:
                problems.append(f"{key} is not valid JSON")
    return problems


def check_span(registry: Registry, span: SpanView, *, capture_content: bool) -> list[str]:
    """Every problem with one span against its registry entry. An unknown span name is a
    problem by itself. A span with status ERROR is checked for kind and parent only."""
    spec = registry.spec(span.name)
    if spec is None:
        return [f"{span.name}: not in the registry"]
    problems: list[str] = []
    if span.kind != spec.kind:
        problems.append(f"{span.name}: kind {span.kind!r}, the registry says {spec.kind!r}")
    if span.parent != spec.parent:
        problems.append(f"{span.name}: parent {span.parent!r}, the registry says {spec.parent!r}")
    if span.status == "ERROR":
        return problems
    attrs = span.attributes
    for key in spec.required:
        if key not in attrs:
            problems.append(f"{span.name}: required attribute {key} is missing")
    for key, value in attrs.items():
        if key == "openinference.span.kind":
            continue
        if not capture_content and registry.is_content(key):
            problems.append(f"{span.name}: content attribute {key} recorded with the switch off")
        problems.extend(f"{span.name}: {p}" for p in attribute_violations(registry, key, value))
    if spec.documents and _documents_due(spec.documents_when, attrs):
        for prefix in spec.documents:
            for leaf in ("id", "score", "metadata"):
                key = f"{prefix}.0.document.{leaf}"
                if key not in attrs:
                    problems.append(f"{span.name}: document attribute {key} is missing")
    return problems


def _documents_due(condition: str | None, attrs: Mapping[str, Any]) -> bool:
    """`documents_when` is `<attribute> > 0`; documents are due when that count is positive."""
    if not condition:
        return True
    key, _, threshold = condition.partition(">")
    value = attrs.get(key.strip())
    return isinstance(value, int | float) and value > float(threshold or 0)


def check_trace(
    registry: Registry,
    spans: Sequence[SpanView],
    path: str,
    *,
    capture_content: bool,
    cache_enabled: bool = False,
    sparse_enabled: bool = True,
) -> list[str]:
    """Part 7: the exact span set of the path (missing and unexpected spans), then every span
    against its entry. Repeated spans (two llm.generate on the repair path) are one name."""
    expected = registry.expected_spans(
        path, cache_enabled=cache_enabled, sparse_enabled=sparse_enabled
    )
    names = {s.name for s in spans}
    roots = [s for s in spans if s.parent is None]
    problems = [] if len(roots) == 1 else [f"{len(roots)} root spans, the trace needs one"]
    problems += [f"missing span {n} on path {path}" for n in sorted(expected - names)]
    allowed = expected | {n for n, s in registry.request.items() if s.optional and s.on_path(path)}
    problems += [f"unexpected span {n} on path {path}" for n in sorted(names - allowed)]
    for span in spans:
        problems.extend(check_span(registry, span, capture_content=capture_content))
    return problems


def cost_mismatch(spans: Sequence[SpanView], *, tolerance: float = 1e-9) -> float | None:
    """Part 4: the root `rag.cost_usd` against the sum over the LLM spans (every model call,
    repairs and the judge included). None when they agree."""
    root = next((s for s in spans if s.parent is None), None)
    if root is None:
        return None
    total = sum(float(s.attributes.get("rag.cost_usd", 0.0)) for s in spans if s.kind == "LLM")
    root_cost = float(root.attributes.get("rag.cost_usd", 0.0))
    diff = abs(root_cost - total)
    return diff if diff > tolerance else None


def classify_path(root: Mapping[str, Any], span_names: Sequence[str]) -> str | None:
    """The Part 7 path of a live trace, from the root attributes and the span names. None for
    a trace that fits no listed path (a model abstain, an invalid output after repair)."""
    status, reason = root.get("rag.status"), root.get("rag.abstain_reason") or "none"
    if root.get("cache.hit") is True:
        return "cache_hit"
    if status == "answered":
        return "repair" if list(span_names).count("llm.generate") > 1 else "answered"
    if status == "no_information":
        if str(reason).startswith("gate."):
            return "gate_abstain"
        if reason == "query.smalltalk":
            return "smalltalk"
        if reason in ("output.unfaithful", "output.judge_unavailable"):
            return "unfaithful"  # the judge ran (or failed) after the fixed checks: same tree
        if reason in ("gen.model_abstained", "gen.invalid_output"):
            return "generate_abstain"  # Stage 16 abstained; the output check stops early
        if reason == "output.invalid_after_repair":
            return "repair"  # two generations, the second one failed a fixed check
        return None
    if status == "error":
        if reason == "guard.rejected":
            return "guard_reject"
        if "llm.generate" in span_names:
            return "generate_error"
    return None
