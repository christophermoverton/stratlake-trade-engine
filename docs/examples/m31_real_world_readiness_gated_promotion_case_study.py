from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.metrics import compute_performance_metrics  # noqa: E402
from src.research.promotion import evaluate_promotion_gates, write_promotion_gate_artifact  # noqa: E402
from src.research.registry import build_review_metadata, canonicalize_value  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("docs/examples/output/m31_real_world_readiness_gated_promotion_case_study")
GATE_CONFIG_PATH = REPO_ROOT / "configs" / "statistical_readiness_promotion_gates_example.yml"
SUMMARY_FILENAME = "summary.json"
PRIOR_PATTERN_NOTE = (
    "Reuses the real_world_campaign_case_study.py artifact-first pattern: a runnable docs/example script, "
    "deterministic artifacts under docs/examples/output, stitched campaign/review/candidate summaries, "
    "and local fixture-backed validation. Unlike the full real-world campaign example, this companion uses "
    "a pinned market-shaped fixture so tests do not require live data or repository-local feature partitions."
)
READINESS_FIELDS = (
    "effective_n",
    "p_value",
    "hit_rate_p_value",
    "autocorr_lag1",
    "split_mean_diff_p",
    "sharpe_stability_ratio",
)


@dataclass(frozen=True)
class M31RealWorldCaseStudyArtifacts:
    output_root: Path
    summary_path: Path
    summary: dict[str, Any]
    run_dirs: dict[str, Path]


def run_case_study(
    *,
    output_root: Path | None = None,
    verbose: bool = True,
) -> M31RealWorldCaseStudyArtifacts:
    resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    gate_config = _load_gate_config()
    preflight = _preflight(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    registry_entries: list[dict[str, Any]] = []
    scenario_summaries: dict[str, dict[str, Any]] = {}
    run_dirs: dict[str, Path] = {}
    for scenario_id, frame in sorted(_build_market_snapshots().items()):
        run_id = f"m31_real_world_{scenario_id}"
        run_dir = resolved_output_root / "runs" / scenario_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs[scenario_id] = run_dir

        metrics = compute_performance_metrics(frame)
        evaluation = evaluate_promotion_gates(
            run_type="strategy",
            config=gate_config,
            sources={"metrics": metrics},
        )
        if evaluation is None:
            raise AssertionError("Expected M31 readiness promotion gate evaluation.")

        market_data_path = run_dir / "market_data.csv"
        returns_path = run_dir / "strategy_returns.csv"
        metrics_path = run_dir / "metrics.json"
        manifest_path = run_dir / "manifest.json"
        promotion_gate_path = write_promotion_gate_artifact(run_dir, evaluation)
        if promotion_gate_path is None:
            raise AssertionError("Expected promotion_gates.json to be written.")

        _write_csv(market_data_path, _market_data_columns(frame))
        _write_csv(returns_path, _strategy_return_columns(frame))
        _write_json(metrics_path, _readiness_metric_excerpt(metrics))

        promotion_summary = evaluation.summary()
        review_metadata = build_review_metadata(
            promotion_status=evaluation.promotion_status,
            promotion_gate_summary=promotion_summary,
        )
        run_manifest = canonicalize_value(
            {
                "run_id": run_id,
                "run_type": "strategy",
                "artifact_files": sorted(
                    [
                        "market_data.csv",
                        "metrics.json",
                        "promotion_gates.json",
                        "strategy_returns.csv",
                        "manifest.json",
                    ]
                ),
                "data_source": "pinned_market_shaped_fixture",
                "promotion_status": evaluation.promotion_status,
                "review_status": review_metadata["status"],
                "promotion_gate_summary": promotion_summary,
            }
        )
        _write_json(manifest_path, run_manifest)

        entry = canonicalize_value(
            {
                "run_id": run_id,
                "run_type": "strategy",
                "artifact_path": _relative_to_output(run_dir, resolved_output_root),
                "market_data_path": _relative_to_output(market_data_path, resolved_output_root),
                "metrics_path": _relative_to_output(metrics_path, resolved_output_root),
                "promotion_gates_path": _relative_to_output(promotion_gate_path, resolved_output_root),
                "promotion_status": evaluation.promotion_status,
                "review_status": review_metadata["status"],
                "review_metadata": review_metadata,
                "promotion_gate_summary": promotion_summary,
                "readiness_metrics": _readiness_metric_excerpt(metrics),
            }
        )
        registry_entries.append(entry)
        scenario_summaries[scenario_id] = entry

    _write_registry_jsonl(resolved_output_root / "registry.jsonl", registry_entries)
    review_summary = _build_review_summary(registry_entries)
    campaign_summary = _build_campaign_summary(scenario_summaries, preflight)
    candidate_review_summary = _build_candidate_review_summary(scenario_summaries)
    manifest = _build_manifest(resolved_output_root, scenario_summaries, preflight)

    _write_json(resolved_output_root / "review_summary.json", review_summary)
    _write_json(resolved_output_root / "campaign_summary.json", campaign_summary)
    _write_json(resolved_output_root / "candidate_review_summary.json", candidate_review_summary)
    _write_json(resolved_output_root / "manifest.json", manifest)
    summary = _build_case_study_summary(scenario_summaries, preflight)
    summary_path = resolved_output_root / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    if verbose:
        print("M31 real-world readiness-gated promotion case study")
        print(f"output_dir: {resolved_output_root.as_posix()}")
        print("data_source: pinned_market_shaped_fixture")
        for scenario_id in sorted(scenario_summaries):
            promotion_summary = scenario_summaries[scenario_id]["promotion_gate_summary"]
            print(
                f"{scenario_id}: promotion_status={promotion_summary['promotion_status']} "
                f"highest_severity={promotion_summary['highest_severity']} "
                f"reason_codes={','.join(promotion_summary['decision_reason_codes']) or 'none'}"
            )

    return M31RealWorldCaseStudyArtifacts(
        output_root=resolved_output_root,
        summary_path=summary_path,
        summary=summary,
        run_dirs=run_dirs,
    )


def _load_gate_config() -> dict[str, Any]:
    if not GATE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "M31 readiness gate config is unavailable. Expected "
            f"{GATE_CONFIG_PATH.relative_to(REPO_ROOT).as_posix()}."
        )
    payload = yaml.safe_load(GATE_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("promotion_gates"), dict):
        raise AssertionError("Expected promotion_gates mapping in M31 readiness gate config.")
    return dict(payload["promotion_gates"])


def _preflight(output_root: Path) -> dict[str, Any]:
    return canonicalize_value(
        {
            "status": "passed",
            "data_source": "pinned_market_shaped_fixture",
            "gate_config": "configs/statistical_readiness_promotion_gates_example.yml",
            "live_market_data_required": False,
            "network_required": False,
            "output_root": _portable_output_root(output_root),
            "prior_case_study_pattern": PRIOR_PATTERN_NOTE,
        }
    )


def _build_market_snapshots() -> dict[str, pd.DataFrame]:
    high_conviction = ([0.004, 0.003, 0.001, 0.005, 0.002, 0.004, 0.003, 0.005, -0.001, 0.004] * 8)
    return {
        "broad_market_momentum": _market_fixture_frame(
            symbol="M31_BROAD",
            returns=high_conviction,
            signals=_alternating_signals(len(high_conviction)),
            start="2025-01-02",
        ),
        "balanced_rotation_review": _market_fixture_frame(
            symbol="M31_ROTATION",
            returns=[
                0.004,
                0.003,
                -0.002,
                0.005,
                0.001,
                -0.003,
                0.006,
                0.002,
                -0.001,
                0.004,
                0.003,
                -0.002,
                0.005,
                0.002,
                -0.004,
                0.006,
                0.001,
                -0.002,
                0.004,
                0.003,
                -0.001,
                0.005,
                0.002,
                -0.003,
                0.004,
                0.002,
                -0.002,
                0.005,
                0.001,
                -0.004,
                0.006,
                0.002,
                -0.001,
                0.004,
                0.003,
                -0.002,
            ],
            signals=[
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ],
            start="2025-03-03",
        ),
        "short_history_breakout": _market_fixture_frame(
            symbol="M31_BREAKOUT",
            returns=high_conviction[:20],
            signals=_alternating_signals(20),
            start="2025-05-01",
        ),
    }


def _market_fixture_frame(
    *,
    symbol: str,
    returns: list[float],
    signals: list[float],
    start: str,
) -> pd.DataFrame:
    equity = pd.Series(returns, dtype="float64").add(1.0).cumprod()
    close = equity.mul(100.0).round(6)
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range(start, periods=len(returns), freq="B", tz="UTC"),
            "symbol": [symbol] * len(returns),
            "timeframe": ["1D"] * len(returns),
            "close": close,
            "signal": signals,
            "strategy_return": returns,
            "equity_curve": equity,
        }
    )


def _alternating_signals(count: int) -> list[float]:
    return [1.0 if index % 2 else 0.0 for index in range(count)]


def _market_data_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, ["ts_utc", "symbol", "close", "signal"]].copy(deep=True)
    output["ts_utc"] = pd.to_datetime(output["ts_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output


def _strategy_return_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, ["ts_utc", "symbol", "strategy_return", "equity_curve"]].copy(deep=True)
    output["ts_utc"] = pd.to_datetime(output["ts_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return output


def _readiness_metric_excerpt(metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {field: metrics.get(field) for field in READINESS_FIELDS}
    payload.update(
        {
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "total_return": metrics.get("total_return"),
            "trade_count": metrics.get("trade_count"),
        }
    )
    return canonicalize_value(payload)


def _build_review_summary(registry_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return canonicalize_value(
        {
            "run_type": "m31_real_world_readiness_review_example",
            "entry_count": len(registry_entries),
            "promotion_status_counts": _count_values(entry["promotion_status"] for entry in registry_entries),
            "review_status_counts": _count_values(entry["review_status"] for entry in registry_entries),
            "decision_reason_codes": _sorted_reason_codes(registry_entries),
            "entries": registry_entries,
        }
    )


def _build_campaign_summary(
    scenario_summaries: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    scenario_matrix = [
        {
            "scenario_id": scenario_id,
            "promotion_status": entry["promotion_status"],
            "review_status": entry["review_status"],
            "highest_severity": entry["promotion_gate_summary"]["highest_severity"],
            "severity_counts": entry["promotion_gate_summary"]["severity_counts"],
            "decision_reason_codes": entry["promotion_gate_summary"]["decision_reason_codes"],
        }
        for scenario_id, entry in sorted(scenario_summaries.items())
    ]
    blocking_summary = scenario_summaries["short_history_breakout"]["promotion_gate_summary"]
    return canonicalize_value(
        {
            "run_type": "research_campaign",
            "campaign_run_id": "m31_real_world_readiness_gated_promotion_case_study",
            "status": "completed",
            "preflight": preflight,
            "scenario_matrix": scenario_matrix,
            "final_outcomes": {
                "promotion_status_counts": _count_values(
                    entry["promotion_status"] for entry in scenario_summaries.values()
                ),
                "review_status_counts": _count_values(entry["review_status"] for entry in scenario_summaries.values()),
                "review_promotion_status": blocking_summary["promotion_status"],
                "review_promotion_gate_status": blocking_summary["evaluation_status"],
                "review_promotion_highest_severity": blocking_summary["highest_severity"],
                "review_promotion_severity_counts": blocking_summary["severity_counts"],
                "review_promotion_decision_reason_codes": blocking_summary["decision_reason_codes"],
                "review_promotion_gate_summary": blocking_summary,
            },
        }
    )


def _build_candidate_review_summary(scenario_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    portfolio_summary = scenario_summaries["broad_market_momentum"]["promotion_gate_summary"]
    return canonicalize_value(
        {
            "run_type": "candidate_selection_review",
            "candidate_selection_run_id": "m31_real_world_candidate_selection_example",
            "portfolio_run_id": "m31_real_world_portfolio_example",
            "promotion_context": {
                "candidate_promotion_status_counts": _count_values(
                    entry["promotion_status"] for entry in scenario_summaries.values()
                ),
                "candidate_review_status_counts": _count_values(
                    entry["review_status"] for entry in scenario_summaries.values()
                ),
                "portfolio_promotion_gate_summary": portfolio_summary,
            },
        }
    )


def _build_manifest(
    output_root: Path,
    scenario_summaries: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    run_files = []
    for scenario_id in sorted(scenario_summaries):
        run_files.extend(
            [
                f"runs/{scenario_id}/market_data.csv",
                f"runs/{scenario_id}/metrics.json",
                f"runs/{scenario_id}/promotion_gates.json",
                f"runs/{scenario_id}/strategy_returns.csv",
                f"runs/{scenario_id}/manifest.json",
            ]
        )
    return canonicalize_value(
        {
            "run_type": "m31_real_world_readiness_gated_promotion_case_study",
            "artifact_files": sorted(
                [
                    "candidate_review_summary.json",
                    "campaign_summary.json",
                    "manifest.json",
                    "registry.jsonl",
                    "review_summary.json",
                    SUMMARY_FILENAME,
                    *run_files,
                ]
            ),
            "data_source": "pinned_market_shaped_fixture",
            "example_config": "configs/statistical_readiness_promotion_gates_example.yml",
            "live_market_data_required": False,
            "network_required": False,
            "output_dir": _portable_output_root(output_root),
            "preflight": preflight,
            "prior_case_study_pattern": PRIOR_PATTERN_NOTE,
            "scenario_count": len(scenario_summaries),
        }
    )


def _build_case_study_summary(
    scenario_summaries: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "case_study": "m31_real_world_readiness_gated_promotion_case_study",
            "data_source": "pinned_market_shaped_fixture",
            "preflight": preflight,
            "prior_case_study_pattern": PRIOR_PATTERN_NOTE,
            "promotion_status_counts": _count_values(entry["promotion_status"] for entry in scenario_summaries.values()),
            "review_status_counts": _count_values(entry["review_status"] for entry in scenario_summaries.values()),
            "decision_reason_codes": _sorted_reason_codes(list(scenario_summaries.values())),
            "scenarios": scenario_summaries,
            "threshold_note": (
                "The readiness thresholds come from the example M31 gate config and are illustrative policy "
                "defaults, not universal statistical truth or live-trading readiness claims."
            ),
        }
    )


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _sorted_reason_codes(entries: list[dict[str, Any]]) -> list[str]:
    codes: set[str] = set()
    for entry in entries:
        summary = entry.get("promotion_gate_summary", {})
        if isinstance(summary, dict):
            codes.update(str(code) for code in summary.get("decision_reason_codes", []) if str(code))
    return sorted(codes)


def _relative_to_output(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def _portable_output_root(output_root: Path) -> str:
    if output_root.is_absolute():
        return "."
    return output_root.as_posix()


def _write_registry_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for entry in entries),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def main() -> int:
    run_case_study()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
