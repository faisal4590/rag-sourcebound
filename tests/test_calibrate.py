# Issue #35: the gate threshold sweep of spec 8.4 (eval/calibrate.py) on synthetic scores; no model.
"""The F1 sweep over tau_pass, the percentile rules for tau_low and tau_high with the clamp that
keeps the order, the degenerate inputs, the score sources, and the config and manifest writes."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from flp_rag.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import calibrate

CFG = load_settings().eval.calibration


def scored(case_id: str, abstain: bool, scores: list[float]) -> calibrate.ScoredCase:
    return calibrate.ScoredCase(case_id=case_id, expected_abstain=abstain, scores=scores)


CASES = [
    scored("g1", False, [0.95, 0.4]),
    scored("g2", False, [0.72, 0.1]),
    scored("g3", False, [0.60, 0.2]),
    scored("g4", False, [0.42]),
    scored("g5", False, [0.20]),  # the low tail every real set has: one weakly scored answer
    scored("u1", True, [0.30, 0.1]),
    scored("u2", True, [0.05]),
    scored("o1", True, [0.01]),
    scored("o2", True, []),  # no candidates at all: always an abstain
]


def test_sweep_applies_the_rule_no_score_at_or_above_tau_means_abstain() -> None:
    rows = calibrate.sweep(CASES, taus=[0.05, 0.35, 0.40, 0.65])
    by_tau = {r["tau_pass"]: r for r in rows}
    # tau 0.35: u1 (0.30), u2, o1, o2 abstain and so does g5 (0.20) -> precision 4/5, recall 1
    assert by_tau[0.35]["precision"] == pytest.approx(0.8) and by_tau[0.35]["recall"] == 1.0
    assert by_tau[0.35]["f1"] == pytest.approx(8 / 9)
    # tau 0.40: the same five abstain (g4 at 0.42 still answers)
    assert by_tau[0.40]["n_abstain"] == 5 and by_tau[0.40]["f1"] == pytest.approx(8 / 9)
    # tau 0.05: only o1 (0.01) and o2 (no candidates) abstain -> precision 1, recall 2/4
    assert by_tau[0.05]["recall"] == 0.5 and by_tau[0.05]["precision"] == 1.0
    assert by_tau[0.65]["n_abstain"] == 7


def test_f1_is_exact_so_equal_ratios_tie() -> None:
    # 2 right of 4 abstains with 4 expected, and 4 right of 8 abstains with 8 expected: both 0.5
    a = calibrate.sweep(
        [scored("u", True, [0.0])] * 2
        + [scored("g", False, [0.0])] * 2
        + [scored("x", True, [0.9])] * 2,
        taus=[0.5],
    )
    b = calibrate.sweep(
        [scored("u", True, [0.0])] * 4
        + [scored("g", False, [0.0])] * 4
        + [scored("x", True, [0.9])] * 4,
        taus=[0.5],
    )
    assert a[0]["f1"] == b[0]["f1"] == 0.5


def test_best_tau_is_the_highest_f1_and_the_middle_of_the_longest_plateau_on_a_tie() -> None:
    rows = calibrate.sweep(CASES, taus=calibrate.taus_of(CFG))
    best = calibrate.best_tau(rows)
    assert best["f1"] == pytest.approx(8 / 9) and best["tau_pass"] == 0.40  # plateau 0.35, 0.40
    plateau = [
        {"tau_pass": t, "f1": f}
        for t, f in [(0.30, 0.8), (0.35, 1.0), (0.40, 1.0), (0.45, 1.0), (0.50, 1.0), (0.55, 0.9)]
    ]
    assert calibrate.best_tau(plateau)["tau_pass"] == 0.45  # four taus -> the upper middle
    split = [
        {"tau_pass": t, "f1": f}
        for t, f in [(0.1, 1.0), (0.2, 0.5), (0.3, 1.0), (0.4, 1.0), (0.5, 1.0)]
    ]
    assert calibrate.best_tau(split)["tau_pass"] == 0.4  # the longest run, not the first tau


def test_taus_run_the_configured_grid() -> None:
    taus = calibrate.taus_of(CFG)
    assert taus[0] == 0.05 and taus[-1] == 0.95 and len(taus) == 19
    assert all(round(b - a, 2) == 0.05 for a, b in itertools.pairwise(taus))


def test_tau_low_and_tau_high_are_percentiles_when_the_order_holds() -> None:
    chosen = calibrate.choose(CASES, CFG)
    best = sorted([0.95, 0.72, 0.60, 0.42, 0.20])  # answerable score_top1
    assert chosen.tau_pass == 0.40
    assert chosen.tau_low == pytest.approx(calibrate.percentile(best, 0.05), abs=1e-4)
    assert chosen.tau_high == pytest.approx(calibrate.percentile(best, 0.50), abs=1e-4)
    assert chosen.ordered and not chosen.clamped
    # passing answers: 0.95, 0.72, 0.60, 0.42; below tau_high 0.60: only 0.42 -> 1 of 4
    assert chosen.borderline_share == pytest.approx(0.25)


def test_clean_separation_clamps_tau_low_next_to_tau_pass() -> None:
    """Spec 8.4 puts tau_low at p05 of the answerable scores; when every answer scores high,
    that is above the F1-best tau_pass. The clamp keeps the gate's order and says so."""
    cases = [
        scored("g1", False, [0.9]),
        scored("g2", False, [0.95]),
        scored("g3", False, [0.8]),
        scored("u1", True, [0.1]),
        scored("o1", True, [0.02]),
    ]
    chosen = calibrate.choose(cases, CFG)
    assert chosen.f1 == 1.0 and 0.1 < chosen.tau_pass < 0.8
    assert (
        chosen.tau_low == pytest.approx(chosen.tau_pass - 0.05)
        and chosen.clamped
        and chosen.ordered
    )
    assert chosen.tau_high == pytest.approx(0.9)  # p50 of 0.8, 0.9, 0.95, unclamped


def test_degenerate_inputs_raise() -> None:
    with pytest.raises(ValueError, match="both groups"):
        calibrate.choose([scored("g1", False, [0.9])], CFG)
    with pytest.raises(ValueError, match="F1 0 everywhere"):
        calibrate.choose([scored("g1", False, [0.0]), scored("u1", True, [0.99])], CFG)
    with pytest.raises(ValueError, match="empty"):
        calibrate.percentile([], 0.5)


def test_write_config_refuses_unordered_values(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text((ROOT / "config.yaml").read_text())
    bad = calibrate.Chosen(
        tau_low=0.5,
        tau_pass=0.4,
        tau_high=0.9,
        f1=1.0,
        clamped=False,
        borderline_share=0.0,
        sweep=[],
    )
    with pytest.raises(ValueError, match="ordered"):
        calibrate.write_config(cfg, bad, reranker="r")
    assert cfg.read_text() == (ROOT / "config.yaml").read_text()


def test_write_config_replaces_only_the_three_gate_values_and_names_the_reranker(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "config.yaml"
    text = (ROOT / "config.yaml").read_text().replace("tau_pass: 0.35", "tau_pass: .35   # keep me")
    cfg.write_text(text)
    chosen = calibrate.Chosen(
        tau_low=0.123,
        tau_pass=0.45,
        tau_high=0.789,
        f1=0.97,
        clamped=True,
        borderline_share=0.2,
        sweep=[],
    )
    calibrate.write_config(cfg, chosen, reranker="BAAI/bge-reranker-base")
    after = cfg.read_text()
    assert (
        "tau_low: 0.123" in after
        and "tau_pass: 0.45   # keep me" in after
        and "tau_high: 0.789" in after
    )
    gate_block = after.split("gate:")[1].split("context:")[0]
    assert "BAAI/bge-reranker-base" in gate_block
    assert text.split("gate:")[0] == after.split("gate:")[0]
    assert text.split("context:", 1)[1] == after.split("context:", 1)[1]
    s = load_settings(cfg)
    assert (s.gate.tau_low, s.gate.tau_pass, s.gate.tau_high) == (0.123, 0.45, 0.789)
    assert not list(tmp_path.glob("*.tmp.yaml"))


def test_write_config_fails_loudly_without_a_gate_block_or_a_key(tmp_path: Path) -> None:
    chosen = calibrate.Chosen(
        tau_low=0.1,
        tau_pass=0.3,
        tau_high=0.6,
        f1=0.9,
        clamped=False,
        borderline_share=0.0,
        sweep=[],
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text("api:\n  timeout_s: 20\n")
    with pytest.raises(ValueError, match="no gate: block"):
        calibrate.write_config(cfg, chosen, reranker="r")
    cfg.write_text("gate:\n  tau_low: 0.2\n  tau_pass: 0.35\n")
    with pytest.raises(ValueError, match="tau_high not found"):
        calibrate.write_config(cfg, chosen, reranker="r")


def test_write_manifest_adds_the_gate_block(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"index_version": "v3", "models": {"rerank_model": "x"}}))
    chosen = calibrate.Chosen(
        tau_low=0.1,
        tau_pass=0.3,
        tau_high=0.6,
        f1=0.9,
        clamped=False,
        borderline_share=0.0,
        sweep=[],
    )
    calibrate.write_manifest(manifest, chosen, reranker="x", n_cases=100)
    data = json.loads(manifest.read_text())
    assert (
        data["gate"]["tau_pass"] == 0.3
        and data["gate"]["reranker"] == "x"
        and data["gate"]["clamped"] is False
    )
    assert (
        data["gate"]["n_cases"] == 100
        and data["gate"]["calibrated_at"]
        and data["index_version"] == "v3"
    )


def test_scores_from_a_harness_run_file(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    rows: list[dict[str, Any]] = [
        {"case_id": "g1", "expected_abstain": False, "scores": [0.9, 0.2], "status": "answered"},
        {"case_id": "o1", "expected_abstain": True, "scores": [], "status": "no_information"},
        {"case_id": "g2", "expected_abstain": False, "scores": [], "status": "error"},
    ]
    run.write_text(json.dumps({"cases": rows}))
    cases = calibrate.scores_from_run(run, max_empty_share=0.5)
    assert [c.case_id for c in cases] == ["g1", "o1"]  # an error row has no scores to calibrate on
    assert cases[0].scores == [0.9, 0.2] and cases[1].expected_abstain is True
    with pytest.raises(ValueError, match="no scores"):
        calibrate.scores_from_run(run, max_empty_share=0.1)  # 1 of 2 empty is over the limit
    run.write_text(
        json.dumps({"cases": [{"case_id": "g1", "expected_abstain": False, "status": "answered"}]})
    )
    with pytest.raises(ValueError, match="has no `scores`"):
        calibrate.scores_from_run(run, max_empty_share=0.5)


def test_report_file_and_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    chosen = calibrate.choose(CASES, CFG)
    out = tmp_path / "calibrate.json"
    calibrate.write_report(out, chosen, cases=CASES, reranker="r", source="test")
    data = json.loads(out.read_text())
    assert data["chosen"]["tau_pass"] == 0.40 and data["chosen"]["ordered"] is True
    assert len(data["sweep"]) == 19 and data["n_cases"] == 9
    calibrate.print_table(chosen)
    text = capsys.readouterr().out
    assert "<- chosen" in text and "tau_low" in text and "borderline share" in text


def test_run_dry_and_write_from_a_run_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text((ROOT / "config.yaml").read_text())
    settings = load_settings(cfg)
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "index_version": "v9",
                "cases": [
                    {
                        "case_id": c.case_id,
                        "expected_abstain": c.expected_abstain,
                        "scores": c.scores,
                        "status": "answered",
                    }
                    for c in CASES
                    if c.scores  # o2 has no candidates: over max_error_share in nine rows
                ],
            }
        )
    )
    (tmp_path / "data" / "index" / "v9").mkdir(parents=True)
    (tmp_path / "data" / "index" / "v9" / "manifest.json").write_text(
        json.dumps({"index_version": "v9"})
    )
    assert calibrate.run(settings, run_file=run, out=tmp_path / "report.json") == 0
    assert (
        "dry run" in capsys.readouterr().out
        and cfg.read_text() == (ROOT / "config.yaml").read_text()
    )
    assert calibrate.run(settings, run_file=run, write=True, out=tmp_path / "report.json") == 0
    assert load_settings(cfg).gate.tau_pass == 0.40
    assert (
        json.loads((tmp_path / "data" / "index" / "v9" / "manifest.json").read_text())["gate"][
            "tau_pass"
        ]
        == 0.40
    )
    with pytest.raises(ValueError, match="--run"):
        calibrate.run(settings)
