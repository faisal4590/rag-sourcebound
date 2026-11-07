# Threshold sweep of Section 8.4. Writes tau_low, tau_pass, tau_high to config.yaml.
"""Calibrate the three gate thresholds from reranker scores (issue #35, spec 8.4).

    make calibrate                                   # = uv run flp calibrate --local (dry run)
    uv run python eval/calibrate.py --local [--write]
    uv run python eval/calibrate.py --run eval/runs/<harness run>.json [--write]

Scores come from one of two places. `--run` reads the `scores` of every case row of a harness
run file (the reranker scores of the kept candidates, which the debug field carries also when
the gate abstained, so no run with the gate switched off is needed). `--local` runs hybrid
retrieval and the reranker on the live index for every golden case with the raw question (no
paraphrases, no model call, about three minutes). Then, per spec 8.4 with the grid and the two
percentiles from `eval.calibration`: sweep `tau_pass` under the rule "no score at or above
tau_pass -> abstain", keep the tau with the highest abstain F1 (the middle of the longest
plateau on a tie), set `tau_low` to the 5th and `tau_high` to the 50th percentile of the best
score over the answerable cases. The gate needs tau_low < tau_pass < tau_high, and p05 sits
above the chosen tau_pass whenever the two groups separate cleanly; the values are then clamped
to one grid step away from tau_pass and the report says so (`clamped`). The table, the three
values, and the share of answers the new tau_high makes borderline are printed and written to
`eval/runs/calibrate_<stamp>.json`; `--write` puts the values into the `gate:` block of
`config.yaml` (comments kept, the reranker named, the file checked with `load_settings` before
it replaces the old one) and into the manifest of the index. Rerun `make eval` afterwards:
`config_hash` changes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flp_rag.contracts import EvalCase, read_jsonl
from flp_rag.settings import CalibrationConfig, Settings, load_settings

sys.path.insert(0, str(Path(__file__).resolve().parent))

GATE_KEYS = ("tau_low", "tau_pass", "tau_high")


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    expected_abstain: bool
    scores: list[float]  # reranker scores of the kept candidates, best first; [] = no candidates

    @property
    def best(self) -> float | None:
        return max(self.scores) if self.scores else None


@dataclass(frozen=True)
class Chosen:
    tau_low: float
    tau_pass: float
    tau_high: float
    f1: float
    clamped: bool  # tau_low or tau_high moved next to tau_pass to keep the order
    borderline_share: float  # answered answerable cases with tau_pass <= best < tau_high
    sweep: list[dict[str, Any]]

    @property
    def ordered(self) -> bool:
        return self.tau_low < self.tau_pass < self.tau_high


# --------------------------------------------------------------------------- the sweep


def taus_of(cfg: CalibrationConfig) -> list[float]:
    n = round((cfg.tau_stop - cfg.tau_start) / cfg.tau_step) + 1
    return [round(cfg.tau_start + i * cfg.tau_step, 4) for i in range(n)]


def percentile(values: Sequence[float], p: float) -> float:
    """Inclusive linear interpolation, like numpy's default."""
    if not values:
        raise ValueError("percentile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = p * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def abstains_at(case: ScoredCase, tau: float) -> bool:
    """Spec 8.4 rule 3: abstain when no score is at or above tau_pass."""
    return case.best is None or case.best < tau


def sweep(cases: Sequence[ScoredCase], *, taus: Sequence[float]) -> list[dict[str, Any]]:
    expected = sum(1 for c in cases if c.expected_abstain)
    rows: list[dict[str, Any]] = []
    for tau in taus:
        abstained = [c for c in cases if abstains_at(c, tau)]
        right = sum(1 for c in abstained if c.expected_abstain)
        precision = right / len(abstained) if abstained else None
        recall = right / expected if expected else None
        denominator = len(abstained) + expected
        f1 = 2 * right / denominator if denominator else 0.0  # exact, no float tie-splitting
        rows.append(
            {
                "tau_pass": tau,
                "n_abstain": len(abstained),
                "true_abstain": right,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def best_tau(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The highest F1. On a tie, the middle of the longest run of tied taus: the threshold with
    the most room on both sides, so a small drift in scores does not flip a decision."""
    top = max(r["f1"] for r in rows)
    runs: list[list[dict[str, Any]]] = []
    for r in rows:
        if r["f1"] == top:
            if runs and runs[-1][-1] is rows[rows.index(r) - 1]:
                runs[-1].append(r)
            else:
                runs.append([r])
    longest = max(runs, key=len)
    return longest[len(longest) // 2]


def choose(cases: Sequence[ScoredCase], cfg: CalibrationConfig) -> Chosen:
    answerable = [c for c in cases if not c.expected_abstain]
    expected = [c for c in cases if c.expected_abstain]
    if not answerable or not expected:
        raise ValueError(
            f"calibration needs both groups: {len(answerable)} answerable, "
            f"{len(expected)} expected abstains"
        )
    rows = sweep(cases, taus=taus_of(cfg))
    best = best_tau(rows)
    if best["f1"] == 0.0:
        raise ValueError("no tau_pass separates the groups at all (F1 0 everywhere)")
    tau_pass = float(best["tau_pass"])
    answerable_best = [c.best for c in answerable if c.best is not None]
    p_low = round(percentile(answerable_best, cfg.low_percentile), 4)
    p_high = round(percentile(answerable_best, cfg.high_percentile), 4)
    tau_low = min(p_low, round(tau_pass - cfg.tau_step, 4))
    tau_high = max(p_high, round(tau_pass + cfg.tau_step, 4))
    passing = [b for b in answerable_best if b >= tau_pass]
    borderline = sum(1 for b in passing if b < tau_high)
    return Chosen(
        tau_low=tau_low,
        tau_pass=tau_pass,
        tau_high=tau_high,
        f1=float(best["f1"]),
        clamped=(tau_low != p_low) or (tau_high != p_high),
        borderline_share=borderline / len(passing) if passing else 0.0,
        sweep=rows,
    )


# --------------------------------------------------------------------------- score sources


def scores_from_run(path: Path, *, max_empty_share: float) -> list[ScoredCase]:
    """The `scores` column of a harness run file. Error rows carry none and are skipped; a row
    without the column is a run made before the column existed (or without `debug`) and is an
    error; too many empty lists (no candidates at all) is one too."""
    data = json.loads(path.read_text())
    out: list[ScoredCase] = []
    for row in data.get("cases") or []:
        if row.get("status") == "error":
            continue
        if "scores" not in row:
            raise ValueError(
                f"{path}: case {row.get('case_id')} has no `scores`; rerun `make eval`"
            )
        out.append(
            ScoredCase(
                case_id=str(row["case_id"]),
                expected_abstain=bool(row["expected_abstain"]),
                scores=[float(s) for s in row["scores"]],
            )
        )
    empty = sum(1 for c in out if not c.scores)
    if out and empty / len(out) > max_empty_share:
        raise ValueError(f"{path}: {empty} of {len(out)} cases have no scores; not a scorable run")
    return out


def scores_local(settings: Settings, cases: Sequence[EvalCase]) -> tuple[list[ScoredCase], str]:
    """Hybrid retrieval plus the reranker on the live index, raw question, no model call.
    Returns the scored cases and the index version they came from."""
    from rerank_eval import run_case

    from flp_rag.api.app import build_deps, current_index_version, default_index_dir
    from flp_rag.ingest.s06_s07_index import STORE_FILE
    from flp_rag.stores.parent_store import ParentStore

    if not settings.rerank.enabled:
        raise RuntimeError("rerank.enabled is false; the gate has no reranker scores to calibrate")
    deps = build_deps(settings)
    out: list[ScoredCase] = []
    try:
        version, _ = current_index_version(settings, deps)
        if version is None:
            raise RuntimeError("no index behind the alias; run `make ingest`")
        store_path = default_index_dir(settings) / str(version) / STORE_FILE
        deps.reranker.warm_up()  # type: ignore[union-attr]
        with ParentStore.open_readonly(store_path) as store:
            for case in cases:
                row = run_case(case, settings, deps, store)
                out.append(ScoredCase(case.case_id, case.expected_abstain, list(row["scores"])))
                print(f"{case.case_id} best {max(row['scores'], default=0.0):.3f}")
    finally:
        if deps.cache is not None:
            deps.cache.close()
        if deps.client is not None:
            deps.client.close()
    return out, str(version)


# --------------------------------------------------------------------------- writes


def write_config(path: Path, chosen: Chosen, *, reranker: str) -> None:
    """Replace the three values inside the `gate:` block, keep every other byte, name the
    reranker in the comment of `tau_low`, and check the result with `load_settings` on a
    temporary copy before it replaces the file."""
    if not chosen.ordered:
        raise ValueError(
            "thresholds are not ordered (need tau_low < tau_pass < tau_high): "
            f"{chosen.tau_low} / {chosen.tau_pass} / {chosen.tau_high}; nothing written"
        )
    text = path.read_text()
    match = re.search(r"^gate:\n((?:[ \t]+.*\n)+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{path} has no gate: block")
    block = match.group(1)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    for key in GATE_KEYS:
        value = getattr(chosen, key)
        pattern = re.compile(rf"^(\s+{key}:\s*)[0-9.]+(\s*#.*)?$", flags=re.MULTILINE)
        found = pattern.search(block)
        if found is None:
            raise ValueError(f"{path}: gate.{key} not found")
        if key == "tau_low":
            comment = f"  # calibrated {stamp} for {reranker} (spec 8.4, `make calibrate`)"
        else:
            comment = found.group(2) or ""  # the existing comment, if any, stays
        block = (
            block[: found.start()] + f"{found.group(1)}{value:g}{comment}" + block[found.end() :]
        )
    new_text = text[: match.start(1)] + block + text[match.end(1) :]
    tmp = path.with_suffix(".calibrate.tmp.yaml")
    tmp.write_text(new_text)
    try:
        checked = load_settings(tmp)
        got = (checked.gate.tau_low, checked.gate.tau_pass, checked.gate.tau_high)
        if got != (chosen.tau_low, chosen.tau_pass, chosen.tau_high):
            raise ValueError(f"{path}: the written gate block reads back as {got}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write_manifest(path: Path, chosen: Chosen, *, reranker: str, n_cases: int) -> None:
    data = json.loads(path.read_text())
    data["gate"] = {
        "tau_low": chosen.tau_low,
        "tau_pass": chosen.tau_pass,
        "tau_high": chosen.tau_high,
        "reranker": reranker,
        "f1_at_tau_pass": chosen.f1,
        "clamped": chosen.clamped,
        "n_cases": n_cases,
        "calibrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_report(
    path: Path, chosen: Chosen, *, cases: Sequence[ScoredCase], reranker: str, source: str
) -> None:
    record = {
        "calibrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "reranker": reranker,
        "n_cases": len(cases),
        "n_answerable": sum(1 for c in cases if not c.expected_abstain),
        "chosen": {k: v for k, v in asdict(chosen).items() if k != "sweep"}
        | {"ordered": chosen.ordered},
        "sweep": chosen.sweep,
        "cases": [asdict(c) for c in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def print_table(chosen: Chosen) -> None:
    print(f"{'tau_pass':>8} {'abstains':>8} {'precision':>9} {'recall':>7} {'f1':>6}")
    for r in chosen.sweep:
        p = "-" if r["precision"] is None else f"{r['precision']:.3f}"
        rc = "-" if r["recall"] is None else f"{r['recall']:.3f}"
        mark = "  <- chosen" if r["tau_pass"] == chosen.tau_pass else ""
        print(f"{r['tau_pass']:>8.2f} {r['n_abstain']:>8} {p:>9} {rc:>7} {r['f1']:>6.3f}{mark}")
    print(
        f"tau_low {chosen.tau_low:g}, tau_pass {chosen.tau_pass:g} (F1 {chosen.f1:.3f}), "
        f"tau_high {chosen.tau_high:g}; clamped: {chosen.clamped}; "
        f"borderline share of the passing answers at this tau_high: {chosen.borderline_share:.0%}"
    )


# --------------------------------------------------------------------------- the run


def run(
    settings: Settings,
    *,
    run_file: Path | None = None,
    local: bool = False,
    write: bool = False,
    out: Path | None = None,
) -> int:
    root = settings.config_path.parent
    cfg = settings.eval.calibration
    reranker = settings.rerank.model
    version: str | None
    if run_file is not None:
        cases = scores_from_run(run_file, max_empty_share=settings.eval.max_error_share)
        source = str(run_file)
        version = json.loads(run_file.read_text()).get("index_version") or None
    elif local:
        golden = list(read_jsonl(root / settings.eval.golden_file, EvalCase))
        cases, version = scores_local(settings, golden)
        source = "local: hybrid retrieval + reranker on the live index, raw question"
    else:
        raise ValueError("pass --run <harness run file> or --local")
    if not cases:
        raise ValueError(f"{source} holds no scored cases")
    chosen = choose(cases, cfg)
    print_table(chosen)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out_path = out or root / settings.eval.runs_dir / f"calibrate_{stamp}.json"
    write_report(out_path, chosen, cases=cases, reranker=reranker, source=source)
    print(f"report -> {out_path}")
    if not write:
        print("dry run: add --write to put the values into config.yaml and the manifest")
        return 0
    write_config(settings.config_path, chosen, reranker=reranker)
    print(f"config.yaml gate: block updated ({settings.config_path}); rerun `make eval`")
    if version:
        from flp_rag.api.app import default_index_dir

        manifest = default_index_dir(settings) / version / "manifest.json"
        if manifest.exists():
            write_manifest(manifest, chosen, reranker=reranker, n_cases=len(cases))
            print(f"manifest updated ({manifest})")
        else:
            print(f"no manifest at {manifest}; config.yaml only")
    else:
        print("the run file names no index version; manifest skipped, config.yaml only")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", type=Path, default=None, help="a harness run file (wins over --local)"
    )
    parser.add_argument("--local", action="store_true", help="score the golden cases locally")
    parser.add_argument("--write", action="store_true", help="write config.yaml and the manifest")
    parser.add_argument("--out", type=Path, default=None, help="report path")
    args = parser.parse_args(argv)
    return run(load_settings(), run_file=args.run, local=args.local, write=args.write, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
