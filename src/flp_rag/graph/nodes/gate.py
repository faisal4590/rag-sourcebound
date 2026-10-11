# Stage 14 - Relevance gate: decision table on tau_low, tau_pass, tau_high. Spec Section 6, Stage 14
"""Stage 14: decide `No information found` before any money goes to generation.

The decision is a plain function of the rerank scores and the three thresholds in `gate.*` of
`config.yaml` (spec Section 6, Stage 14; issue #19). No model call. Each zone includes its lower
edge:

    no candidates              -> abstain, gate.no_candidates
    top < tau_low              -> abstain, gate.low_top_score     (off-topic)
    tau_low  <= top < tau_pass -> abstain, gate.no_passing_chunk  (in-domain miss)
    tau_pass <= top < tau_high -> pass,    gate.borderline        (borderline: strict prompt)
    top >= tau_high            -> pass,    gate.pass

The node returns exactly `pass` or `abstain` for the conditional edge; `borderline` is a flag on
the pass path. The scores must be reranker scores in 0 to 1; anything else is an error, not a
decision. The node writes `rag.gate_decision`, `rag.score_top1`, `rag.n_pass`, `rag.tau_*`,
`rag.borderline` on every request, `rag.abstain_reason` on the abstain path, and
`rag.gate_reason` (the zone) always.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from flp_rag.contracts import Candidate
from flp_rag.graph.state import RagState
from flp_rag.settings import ABSTAIN_TEXT, GateConfig, Settings
from flp_rag.tracing import current_span

Decision = Literal["pass", "abstain"]


class ScoreOutOfRange(ValueError):
    """A rerank score outside 0 to 1: the thresholds belong to the reranker's scale."""


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    reason: str
    score_top1: float | None  # None when there are no candidates
    n_pass: int

    @property
    def borderline(self) -> bool:
        return self.reason == "gate.borderline"


def decide(scores: Sequence[float], gate: GateConfig) -> GateDecision:
    """The decision table on the best score. `n_pass` counts scores at or above `tau_pass`."""
    for s in scores:
        if not 0.0 <= s <= 1.0:  # also rejects NaN
            raise ScoreOutOfRange(
                f"rerank score {s} is outside 0 to 1; the gate needs reranker scores"
            )
    if not scores:
        return GateDecision("abstain", "gate.no_candidates", None, 0)
    top = max(scores)
    n_pass = sum(1 for s in scores if s >= gate.tau_pass)
    if top < gate.tau_low:
        return GateDecision("abstain", "gate.low_top_score", top, n_pass)
    if top < gate.tau_pass:
        return GateDecision("abstain", "gate.no_passing_chunk", top, n_pass)
    if top < gate.tau_high:
        return GateDecision("pass", "gate.borderline", top, n_pass)
    return GateDecision("pass", "gate.pass", top, n_pass)


def rerank_scores(candidates: Sequence[Candidate]) -> list[float]:
    """Every candidate must carry a rerank score (Stage 13 ran). A fused or dense score is on
    another scale and would make the gate stop every request or none."""
    scores: list[float] = []
    for c in candidates:
        if c.rerank_score is None:
            raise ValueError(
                f"candidate {c.chunk_id} has no rerank_score; Stage 13 must run before the gate"
            )
        scores.append(float(c.rerank_score))
    return scores


def make_gate(settings: Settings) -> Callable[[RagState], dict[str, Any]]:
    gate = settings.gate

    def gate_node(state: RagState) -> dict[str, Any]:
        candidates = list(state.get("candidates", []))
        scores = rerank_scores(candidates)
        d = decide(scores, gate)
        top = d.score_top1 if d.score_top1 is not None else 0.0
        span = current_span()
        for key, value in (
            ("rag.gate_decision", d.decision),
            ("rag.gate_reason", d.reason),
            ("rag.score_top1", top),
            ("rag.n_pass", d.n_pass),
            ("rag.tau_low", gate.tau_low),
            ("rag.tau_pass", gate.tau_pass),
            ("rag.tau_high", gate.tau_high),
            ("rag.borderline", d.borderline),
        ):
            span.set_attribute(key, value)
        out: dict[str, Any] = {
            "gate_decision": d.decision,
            "borderline": d.borderline,
            "top_score": top,
            "n_pass": d.n_pass,
        }
        if state.get("debug"):
            out["debug_info"] = {
                **state.get("debug_info", {}),
                "gate": {
                    "decision": d.decision,
                    "reason": d.reason,
                    "score_top1": top,
                    "n_pass": d.n_pass,
                },
            }
        if d.decision == "abstain":
            span.set_attribute("rag.abstain_reason", d.reason)
            return {
                **out,
                "status": "no_information",
                "answer": ABSTAIN_TEXT,
                "abstain_reason": d.reason,
                "sources": [],
            }
        passing = [c for c, s in zip(candidates, scores, strict=True) if s >= gate.tau_pass]
        return {**out, "candidates": passing}

    return gate_node
