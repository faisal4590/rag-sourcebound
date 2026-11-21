# Issue #32: the Grafana dashboard and alert provisioning under docs/dashboards/ are well formed.
"""The dashboard holds the five panels of spec 7.9 (plus their sub-panels), every PromQL refers
to a metric of `metrics.py`, and the alert file holds the five rules of 7.10 with the
thresholds of config.yaml."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from flp_rag.settings import load_settings

DOCS = Path(__file__).resolve().parents[1] / "docs" / "dashboards"
METRICS = {
    "rag_requests",
    "rag_stage_latency_ms",
    "rag_tokens",
    "rag_cost_usd",
    "rag_rerank_score_top1",
    "rag_retrieval_overlap_ratio",
    "rag_errors",
    "rag_user_feedback",
    "flp_eval_metric",  # the documented exporter gauge of panel 5b
}
LABELS = {  # the attribute names of metrics.py after the OTel to Prometheus conversion
    "status",
    "abstain_reason",
    "cache_hit",
    "http_route",
    "span_name",
    "gen_ai_request_model",
    "direction",
    "exception_type",
    "rating",
    "le",
    "metric",
    "index_version",
    "prompt_version",
}


def test_dashboard_json_has_the_five_panels_on_known_metrics() -> None:
    data = json.loads((DOCS / "grafana-flp-rag.json").read_text())
    titles = [p["title"] for p in data["panels"]]
    assert [t[:2] for t in titles] == ["1.", "2.", "3.", "3b", "3c", "4.", "4b", "4c", "5.", "5b"]
    variables = {v["name"]: v for v in data["templating"]["list"]}
    assert variables["DS_PROMETHEUS"]["type"] == "datasource"  # import and file provisioning alike
    for panel in data["panels"]:
        assert panel["gridPos"]["w"] <= 24 and panel["targets"]
        assert panel["datasource"]["uid"] == "${DS_PROMETHEUS}"
        if panel["type"] == "heatmap":
            assert panel["options"]["rowsFrame"]["layout"] == "le"
            assert all(t["format"] == "heatmap" for t in panel["targets"])
        for target in panel["targets"]:
            check_promql(target["expr"])


def check_promql(expr: str) -> None:
    names = set(re.findall(r"\b(rag_[a-z_0-9]+|flp_eval_metric)\b", expr))
    assert names, expr
    assert {n.removesuffix("_bucket") for n in names} <= METRICS, names
    for selector in re.findall(r"\{([^}]*)\}", expr):
        labels = {m.group(1) for m in re.finditer(r"([a-z_]+)\s*(?:=|!=|=~|!~)", selector)}
        assert labels <= LABELS, selector
    # every request counter is read for /ask only: the health and page roots have their own route
    for selector in re.findall(r"rag_requests\{([^}]*)\}", expr):
        assert 'http_route="/ask"' in selector, expr
    assert "rag_requests[" not in expr and "rag_requests " not in expr, expr


def test_alert_rules_match_the_spec_and_the_config_thresholds() -> None:
    data = yaml.safe_load((DOCS / "alerts.yaml").read_text())
    rules = data["groups"][0]["rules"]
    assert [r["uid"] for r in rules] == [
        "flp-abstain-rate",
        "flp-p95-latency",
        "flp-errors",
        "flp-faithfulness",
        "flp-cost",
    ]
    cfg = load_settings().alerts
    by_uid = {r["uid"]: r for r in rules}
    thresholds = {
        "flp-p95-latency": int(cfg.p95_latency_max_s * 1000),
        "flp-errors": cfg.errors_max,
        "flp-faithfulness": pytest.approx(cfg.faithfulness_fail_rate_max),
        "flp-cost": cfg.cost_max_usd_per_day,
    }
    for uid, value in thresholds.items():
        (node,) = [d for d in by_uid[uid]["data"] if d["model"].get("type") == "threshold"]
        assert node["model"]["conditions"][0]["evaluator"] == {"type": "gt", "params": [value]}
    (math,) = [d for d in by_uid["flp-abstain-rate"]["data"] if d["model"].get("type") == "math"]
    expected = f"$C > {cfg.abstain_rate_max:.2f} && $D > {cfg.abstain_min_requests}"
    assert math["model"]["expression"] == expected
    for rule in rules:
        refs = {d["refId"] for d in rule["data"]}
        assert rule["condition"] in refs
        assert rule["noDataState"] == "OK" and rule["execErrState"] == "Error"
        for node in rule["data"]:
            model = node["model"]
            assert model["refId"] == node["refId"]
            if node["datasourceUid"] == "__expr__":
                assert model["type"] in {"reduce", "threshold", "math"}
                if model["type"] != "math":  # a reduce or threshold node reads one input
                    assert model["expression"] in refs
            else:
                assert node["datasourceUid"] == "${DS_PROMETHEUS}"
                assert node["relativeTimeRange"]["from"] > 0 and model["instant"] is True
                check_promql(model["expr"])
