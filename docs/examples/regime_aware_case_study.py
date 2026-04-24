from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research.regimes import (  # noqa: E402
    REGIME_DIMENSIONS,
    RegimeClassificationConfig,
    RegimeConditionalConfig,
    RegimeTransitionConfig,
    align_regimes_to_alpha_windows,
    align_regimes_to_strategy_timeseries,
    analyze_alpha_regime_transitions,
    analyze_strategy_regime_transitions,
    classify_market_regimes,
    compare_regime_results,
    evaluate_all_dimensions,
    load_regime_review_bundle,
    render_artifact_inventory_markdown,
    render_attribution_summary_markdown,
    render_comparison_summary_markdown,
    render_transition_highlights_markdown,
    summarize_regime_attribution,
    summarize_transition_attribution,
    write_regime_artifacts,
    write_regime_attribution_artifacts,
    write_regime_conditional_artifacts_multi_dimension,
    write_regime_transition_artifacts,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "regime_aware_case_study"
SUMMARY_FILENAME = "summary.json"
INTERPRETATION_FILENAME = "final_interpretation.md"


@dataclass(frozen=True)
class CaseStudyArtifacts:
    summary: dict[str, Any]
    summary_path: Path
    output_root: Path
    regime_dir: Path
    strategy_dir: Path
    alpha_baseline_dir: Path
    alpha_defensive_dir: Path


def _normalize_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        timestamp = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        return path.as_posix()
    return "." if str(relative) == "." else relative.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _preview(frame: pd.DataFrame, *, columns: list[str], limit: int = 5) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    preview = frame.loc[:, available].head(limit).copy(deep=True)
    for column in preview.columns:
        if column == "ts_utc" or column.endswith("ts_utc"):
            preview[column] = pd.to_datetime(preview[column], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


def build_market_data() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=24, freq="D", tz="UTC")
    returns = [
        (0.0020, 0.0015, 0.0025),
        (0.0030, 0.0020, 0.0035),
        (0.0025, 0.0020, 0.0030),
        (0.0010, 0.0005, 0.0015),
        (-0.0100, -0.0120, -0.0110),
        (-0.0150, -0.0170, -0.0160),
        (-0.0200, -0.0210, -0.0190),
        (-0.0100, -0.0110, -0.0090),
        (-0.0050, -0.0040, -0.0060),
        (0.0080, 0.0070, 0.0090),
        (0.0100, 0.0090, 0.0110),
        (0.0120, 0.0110, 0.0130),
        (0.0040, -0.0060, 0.0070),
        (-0.0030, 0.0050, -0.0040),
        (0.0060, -0.0080, 0.0050),
        (0.0110, 0.0100, 0.0120),
        (0.0130, 0.0120, 0.0140),
        (-0.0090, -0.0100, -0.0080),
        (-0.0120, -0.0130, -0.0110),
        (-0.0140, -0.0160, -0.0150),
        (0.0070, 0.0060, 0.0080),
        (0.0090, 0.0080, 0.0100),
        (0.0020, 0.0010, 0.0030),
        (-0.0010, -0.0020, 0.0000),
    ]

    rows: list[dict[str, Any]] = []
    for ts_utc, symbol_returns in zip(timestamps, returns, strict=True):
        for symbol, asset_return in zip(("AAA", "BBB", "CCC"), symbol_returns, strict=True):
            rows.append({"ts_utc": ts_utc, "symbol": symbol, "asset_return": asset_return})
    return pd.DataFrame(rows)


def build_strategy_timeseries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=24, freq="D", tz="UTC"),
            "strategy_return": [
                0.0010, 0.0020, 0.0030, 0.0010, -0.0060, -0.0090, -0.0120, -0.0060,
                -0.0020, 0.0060, 0.0070, 0.0080, -0.0010, 0.0000, -0.0020, 0.0090,
                0.0100, -0.0040, -0.0060, -0.0070, 0.0050, 0.0060, 0.0020, -0.0010,
            ],
        }
    )


def build_alpha_windows(flavor: str) -> pd.DataFrame:
    if flavor == "baseline":
        ic = [
            0.010, 0.020, 0.030, 0.020, -0.010, -0.020, -0.030, -0.020,
            -0.010, 0.040, 0.050, 0.060, 0.010, -0.010, 0.000, 0.050,
            0.060, -0.010, -0.020, -0.030, 0.030, 0.040, 0.020, 0.000,
        ]
        rank_ic = [
            0.020, 0.030, 0.040, 0.030, 0.000, -0.010, -0.020, -0.010,
            0.000, 0.050, 0.060, 0.070, 0.020, 0.000, 0.010, 0.060,
            0.070, 0.000, -0.010, -0.020, 0.040, 0.050, 0.030, 0.010,
        ]
    elif flavor == "defensive":
        ic = [
            0.015, 0.018, 0.020, 0.015, 0.020, 0.030, 0.035, 0.025,
            0.020, 0.010, 0.012, 0.015, 0.030, 0.025, 0.028, 0.020,
            0.018, 0.030, 0.035, 0.040, 0.018, 0.020, 0.015, 0.012,
        ]
        rank_ic = [
            0.022, 0.025, 0.028, 0.022, 0.030, 0.040, 0.045, 0.035,
            0.030, 0.018, 0.020, 0.024, 0.038, 0.034, 0.036, 0.026,
            0.024, 0.040, 0.045, 0.050, 0.024, 0.026, 0.021, 0.018,
        ]
    else:
        raise ValueError(f"Unsupported alpha flavor: {flavor}")

    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=24, freq="D", tz="UTC"),
            "ic": ic,
            "rank_ic": rank_ic,
            "sample_size": [48] * 24,
            "alpha_variant": [flavor] * 24,
        }
    )


def _write_notebook_review_markdown(
    output_root: Path,
    *,
    strategy_bundle_dir: Path,
    alpha_baseline_bundle_dir: Path,
) -> dict[str, str]:
    review_dir = output_root / "notebook_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    strategy_bundle = load_regime_review_bundle(strategy_bundle_dir)
    alpha_bundle = load_regime_review_bundle(alpha_baseline_bundle_dir)

    files = {
        "strategy_inventory": review_dir / "strategy_inventory.md",
        "strategy_attribution_summary": review_dir / "strategy_attribution_summary.md",
        "strategy_transition_highlights": review_dir / "strategy_transition_highlights.md",
        "alpha_baseline_inventory": review_dir / "alpha_baseline_inventory.md",
        "alpha_baseline_attribution_summary": review_dir / "alpha_baseline_attribution_summary.md",
        "alpha_baseline_transition_highlights": review_dir / "alpha_baseline_transition_highlights.md",
        "alpha_baseline_comparison_summary": review_dir / "alpha_baseline_comparison_summary.md",
    }
    _write_text(files["strategy_inventory"], render_artifact_inventory_markdown(strategy_bundle))
    _write_text(files["strategy_attribution_summary"], render_attribution_summary_markdown(strategy_bundle))
    _write_text(files["strategy_transition_highlights"], render_transition_highlights_markdown(strategy_bundle))
    _write_text(files["alpha_baseline_inventory"], render_artifact_inventory_markdown(alpha_bundle))
    _write_text(files["alpha_baseline_attribution_summary"], render_attribution_summary_markdown(alpha_bundle))
    _write_text(files["alpha_baseline_transition_highlights"], render_transition_highlights_markdown(alpha_bundle))
    _write_text(files["alpha_baseline_comparison_summary"], render_comparison_summary_markdown(alpha_bundle))
    return {key: _relative_to_output(path, output_root) for key, path in files.items()}


def _build_interpretation_markdown(summary: dict[str, Any]) -> str:
    strategy_best = summary["strategy"]["attribution"]["best_regime"]
    strategy_worst = summary["strategy"]["attribution"]["worst_regime"]
    strategy_transition = summary["strategy"]["transition"]["most_adverse_category"]
    stress_onset = summary["strategy"]["transition"]["stress_onset_summary"]
    alpha_baseline_best = summary["alpha"]["baseline"]["attribution"]["best_regime"]
    alpha_defensive_best = summary["alpha"]["defensive"]["attribution"]["best_regime"]
    alpha_comparison = summary["alpha"]["comparison"]

    lines = [
        "# Final Interpretation Notes",
        "",
        "## Strategy Surface",
        (
            f"- The strongest observed strategy regime in this deterministic study was "
            f"`{strategy_best['regime_label']}` with `{summary['strategy']['attribution']['primary_metric']}` "
            f"of `{strategy_best['metric_value']}`."
        ),
        (
            f"- The weakest observed strategy regime was `{strategy_worst['regime_label']}` "
            f"with `{summary['strategy']['attribution']['primary_metric']}` of `{strategy_worst['metric_value']}`."
        ),
        (
            f"- The most adverse transition category in the strategy transition summary was "
            f"`{strategy_transition['transition_category']}` with "
            f"`{summary['strategy']['transition']['primary_metric']}` of `{strategy_transition['metric_value']}`."
        ),
        (
            f"- The dedicated `stress_onset` summary reported `present={str(stress_onset['present']).lower()}` "
            f"with metric value `{stress_onset.get('primary_metric_value')}`."
        ),
        (
            f"- Fragility flag: `{str(summary['strategy']['attribution']['fragility_flag']).lower()}`. "
            f"Reason: {summary['strategy']['attribution']['fragility_reason']}."
        ),
        "",
        "## Alpha Surface",
        (
            f"- The baseline alpha was strongest in `{alpha_baseline_best['regime_label']}`, while the defensive alpha "
            f"was strongest in `{alpha_defensive_best['regime_label']}`."
        ),
        (
            f"- The comparison surface ranked `{alpha_comparison['best_run']['run_id']}` first in the best-supported "
            f"regime slice and flagged `{alpha_comparison['warning_row_count']}` warning-bearing rows."
        ),
        (
            "- The baseline alpha remains more cyclical in this example, while the defensive alpha gives up some calm-regime "
            "upside in exchange for steadier stressed-regime evidence."
        ),
        "",
        "## Caution Boundaries",
        "- These outputs are descriptive summaries of one deterministic case-study fixture, not causal claims about why the regimes occurred.",
        "- Sparse or empty regime slices should be treated as evidence limits, not as proof that a strategy or alpha is robust to missing conditions.",
        "- The case study is meant to validate the end-to-end artifact and review workflow; it is not a live trading recommendation.",
    ]
    return "\n".join(lines)


def _artifact_tree(output_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in output_root.rglob("*")
        if path.is_file()
    )


def _write_bundle(
    bundle_dir: Path,
    *,
    labels: pd.DataFrame,
    regime_metadata: dict[str, Any],
    conditional_results: dict[str, Any],
    transition_result: Any | None,
    attribution: Any,
    transition_attribution: Any | None,
    comparison: Any | None,
    run_id: str,
    metadata: dict[str, Any],
) -> None:
    write_regime_artifacts(bundle_dir, labels, metadata=regime_metadata)
    write_regime_conditional_artifacts_multi_dimension(
        bundle_dir,
        conditional_results,
        run_id=run_id,
        extra_metadata=metadata,
    )
    if transition_result is not None:
        write_regime_transition_artifacts(
            bundle_dir,
            transition_result,
            run_id=run_id,
            extra_metadata=metadata,
        )
    write_regime_attribution_artifacts(
        bundle_dir,
        attribution,
        transition=transition_attribution,
        comparison=comparison,
        run_id=run_id,
        extra_metadata=metadata,
    )


def _best_run_summary(comparison: Any) -> dict[str, Any]:
    table = comparison.comparison_table.copy(deep=True)
    if table.empty:
        return {"run_id": None, "winner_count": 0}
    winners = table.loc[table["rank_within_regime"] == 1, ["run_id"]].copy()
    if winners.empty:
        return {"run_id": None, "winner_count": 0}
    counts = winners["run_id"].value_counts().sort_index()
    best_run_id = counts.sort_values(ascending=False, kind="stable").index[0]
    return {"run_id": str(best_run_id), "winner_count": int(counts.loc[best_run_id])}


def run_case_study(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reset_output: bool = True,
    verbose: bool = True,
) -> CaseStudyArtifacts:
    output_root = Path(output_root)
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    regime_dir = output_root / "regime_bundle"
    strategy_dir = output_root / "strategy_bundle"
    alpha_baseline_dir = output_root / "alpha_baseline_bundle"
    alpha_defensive_dir = output_root / "alpha_defensive_bundle"
    for directory in (regime_dir, strategy_dir, alpha_baseline_dir, alpha_defensive_dir):
        directory.mkdir(parents=True, exist_ok=True)

    market_data = build_market_data()
    strategy_frame = build_strategy_timeseries()
    alpha_baseline_frame = build_alpha_windows("baseline")
    alpha_defensive_frame = build_alpha_windows("defensive")

    classification_config = RegimeClassificationConfig(
        return_column="asset_return",
        volatility_window=3,
        trend_window=3,
        trend_return_threshold=0.01,
        drawdown_threshold=0.04,
        near_peak_drawdown_threshold=0.01,
        stress_window=3,
        stress_correlation_threshold=0.60,
        stress_dispersion_quantile=0.70,
        metadata={"case_study": "m24_7_regime_aware_case_study"},
    )
    classification = classify_market_regimes(market_data, config=classification_config)
    labels = classification.labels

    regime_metadata = {
        **classification.metadata,
        "case_study": "m24_7_regime_aware_case_study",
        "input_surface": "market_fixture",
    }
    write_regime_artifacts(regime_dir, labels, metadata=regime_metadata)

    strategy_aligned = align_regimes_to_strategy_timeseries(strategy_frame, labels)
    alpha_baseline_aligned = align_regimes_to_alpha_windows(alpha_baseline_frame, labels)
    alpha_defensive_aligned = align_regimes_to_alpha_windows(alpha_defensive_frame, labels)

    conditional_config = RegimeConditionalConfig(
        min_observations=2,
        metadata={"case_study": "m24_7_regime_aware_case_study"},
    )
    strategy_results = evaluate_all_dimensions(
        strategy_aligned,
        surface="strategy",
        return_column="strategy_return",
        config=conditional_config,
    )
    alpha_baseline_results = evaluate_all_dimensions(
        alpha_baseline_aligned,
        surface="alpha",
        config=conditional_config,
    )
    alpha_defensive_results = evaluate_all_dimensions(
        alpha_defensive_aligned,
        surface="alpha",
        config=conditional_config,
    )

    transition_config = RegimeTransitionConfig(
        pre_event_rows=1,
        post_event_rows=1,
        min_observations=2,
        metadata={"case_study": "m24_7_regime_aware_case_study"},
    )
    strategy_transition = analyze_strategy_regime_transitions(
        strategy_aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=transition_config,
    )
    alpha_baseline_transition = analyze_alpha_regime_transitions(
        alpha_baseline_aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=transition_config,
    )
    alpha_defensive_transition = analyze_alpha_regime_transitions(
        alpha_defensive_aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=transition_config,
    )

    strategy_attribution = summarize_regime_attribution(
        strategy_results["composite"],
        run_id="strategy",
        metadata={"case_study_surface": "strategy"},
    )
    strategy_transition_attribution = summarize_transition_attribution(
        strategy_transition,
        run_id="strategy",
        metadata={"case_study_surface": "strategy"},
    )
    alpha_baseline_attribution = summarize_regime_attribution(
        alpha_baseline_results["composite"],
        run_id="alpha_baseline",
        metadata={"case_study_surface": "alpha_baseline"},
    )
    alpha_baseline_transition_attribution = summarize_transition_attribution(
        alpha_baseline_transition,
        run_id="alpha_baseline",
        metadata={"case_study_surface": "alpha_baseline"},
    )
    alpha_defensive_attribution = summarize_regime_attribution(
        alpha_defensive_results["composite"],
        run_id="alpha_defensive",
        metadata={"case_study_surface": "alpha_defensive"},
    )
    alpha_defensive_transition_attribution = summarize_transition_attribution(
        alpha_defensive_transition,
        run_id="alpha_defensive",
        metadata={"case_study_surface": "alpha_defensive"},
    )
    alpha_comparison = compare_regime_results(
        {
            "alpha_baseline": alpha_baseline_results["composite"],
            "alpha_defensive": alpha_defensive_results["composite"],
        },
        surface="alpha",
        dimension="composite",
        metadata={"case_study": "m24_7_regime_aware_case_study"},
    )
    alpha_comparison_best_run = _best_run_summary(alpha_comparison)
    alpha_warning_row_count = int(alpha_comparison.comparison_table["warning_flag"].astype("bool").sum())

    _write_bundle(
        strategy_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "bundle": "strategy"},
        conditional_results=strategy_results,
        transition_result=strategy_transition,
        attribution=strategy_attribution,
        transition_attribution=strategy_transition_attribution,
        comparison=None,
        run_id="strategy_bundle",
        metadata={"surface": "strategy"},
    )
    _write_bundle(
        alpha_baseline_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "bundle": "alpha_baseline"},
        conditional_results=alpha_baseline_results,
        transition_result=alpha_baseline_transition,
        attribution=alpha_baseline_attribution,
        transition_attribution=alpha_baseline_transition_attribution,
        comparison=alpha_comparison,
        run_id="alpha_baseline_bundle",
        metadata={"surface": "alpha", "alpha_variant": "baseline"},
    )
    _write_bundle(
        alpha_defensive_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "bundle": "alpha_defensive"},
        conditional_results=alpha_defensive_results,
        transition_result=alpha_defensive_transition,
        attribution=alpha_defensive_attribution,
        transition_attribution=alpha_defensive_transition_attribution,
        comparison=None,
        run_id="alpha_defensive_bundle",
        metadata={"surface": "alpha", "alpha_variant": "defensive"},
    )

    notebook_review_files = _write_notebook_review_markdown(
        output_root,
        strategy_bundle_dir=strategy_dir,
        alpha_baseline_bundle_dir=alpha_baseline_dir,
    )

    summary: dict[str, Any] = {
        "case_study": {
            "milestone": "M24.7",
            "name": "canonical_regime_aware_case_study",
            "description": "Deterministic end-to-end regime-aware workflow covering classification, artifacts, conditional evaluation, transitions, attribution, comparison, and notebook review.",
        },
        "commands": [
            "python docs/examples/regime_aware_case_study.py",
            "python docs/examples/regime_aware_case_study.py --output-root docs/examples/output/regime_aware_case_study",
        ],
        "inputs": {
            "market_data_rows": int(len(market_data)),
            "market_timestamps": int(market_data["ts_utc"].nunique()),
            "market_symbols": sorted(market_data["symbol"].unique().tolist()),
            "strategy_rows": int(len(strategy_frame)),
            "alpha_rows_per_variant": int(len(alpha_baseline_frame)),
        },
        "paths": {
            "output_root": ".",
            "regime_bundle": _relative_to_output(regime_dir, output_root),
            "strategy_bundle": _relative_to_output(strategy_dir, output_root),
            "alpha_baseline_bundle": _relative_to_output(alpha_baseline_dir, output_root),
            "alpha_defensive_bundle": _relative_to_output(alpha_defensive_dir, output_root),
            "notebook_review_dir": "notebook_review",
            "interpretation_markdown": INTERPRETATION_FILENAME,
        },
        "classification": {
            "defined_row_count": int(labels["is_defined"].sum()),
            "undefined_row_count": int((~labels["is_defined"]).sum()),
            "label_preview": _preview(
                labels,
                columns=[
                    "ts_utc",
                    "volatility_state",
                    "trend_state",
                    "drawdown_recovery_state",
                    "stress_state",
                    "regime_label",
                ],
            ),
            "state_distribution": {
                "volatility": labels["volatility_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "trend": labels["trend_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "drawdown_recovery": labels["drawdown_recovery_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "stress": labels["stress_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
            },
        },
        "strategy": {
            "alignment_preview": _preview(
                strategy_aligned,
                columns=["ts_utc", "strategy_return", "regime_label", "regime_alignment_status"],
            ),
            "conditional_dimensions": sorted(strategy_results),
            "conditional_summary": strategy_results["composite"].alignment_summary,
            "transition": {
                "event_count": int(len(strategy_transition.events)),
                "stress_transition_count": int(strategy_transition.events["is_stress_transition"].astype("bool").sum()),
                "primary_metric": strategy_transition_attribution.summary["primary_metric"],
                "most_adverse_category": strategy_transition_attribution.summary["worst_transition_category"],
                "stress_onset_summary": strategy_transition_attribution.summary["stress_onset_summary"],
            },
            "attribution": {
                "primary_metric": strategy_attribution.summary["primary_metric"],
                "best_regime": strategy_attribution.summary["best_regime"],
                "worst_regime": strategy_attribution.summary["worst_regime"],
                "fragility_flag": strategy_attribution.summary["fragility_flag"],
                "fragility_reason": strategy_attribution.summary["fragility_reason"],
            },
        },
        "alpha": {
            "baseline": {
                "conditional_dimensions": sorted(alpha_baseline_results),
                "attribution": {
                    "primary_metric": alpha_baseline_attribution.summary["primary_metric"],
                    "best_regime": alpha_baseline_attribution.summary["best_regime"],
                    "worst_regime": alpha_baseline_attribution.summary["worst_regime"],
                    "fragility_flag": alpha_baseline_attribution.summary["fragility_flag"],
                    "fragility_reason": alpha_baseline_attribution.summary["fragility_reason"],
                },
            },
            "defensive": {
                "conditional_dimensions": sorted(alpha_defensive_results),
                "attribution": {
                    "primary_metric": alpha_defensive_attribution.summary["primary_metric"],
                    "best_regime": alpha_defensive_attribution.summary["best_regime"],
                    "worst_regime": alpha_defensive_attribution.summary["worst_regime"],
                    "fragility_flag": alpha_defensive_attribution.summary["fragility_flag"],
                    "fragility_reason": alpha_defensive_attribution.summary["fragility_reason"],
                },
            },
            "comparison": {
                "best_run": alpha_comparison_best_run,
                "robust_run_ids": alpha_comparison.summary["robust_run_ids"],
                "regime_winners": alpha_comparison.summary["regime_winners"],
                "warning_row_count": alpha_warning_row_count,
            },
        },
        "notebook_review": notebook_review_files,
        "artifact_tree": _artifact_tree(output_root),
        "limitations": [
            "The market, strategy, and alpha inputs are deterministic repository fixtures designed for validation and documentation.",
            "The case study demonstrates strategy and alpha surfaces directly; a portfolio surface is deferred to keep the canonical example concise and fast to validate.",
            "Interpretation remains descriptive and evidence-based, not causal.",
        ],
    }

    interpretation_markdown = _build_interpretation_markdown(summary)
    _write_text(output_root / INTERPRETATION_FILENAME, interpretation_markdown)
    summary["artifact_tree"] = _artifact_tree(output_root)

    summary_path = output_root / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    if verbose:
        print(json.dumps(summary["paths"], indent=2, sort_keys=True))
        print(f"summary_path={summary_path.as_posix()}")

    return CaseStudyArtifacts(
        summary=summary,
        summary_path=summary_path,
        output_root=output_root,
        regime_dir=regime_dir,
        strategy_dir=strategy_dir,
        alpha_baseline_dir=alpha_baseline_dir,
        alpha_defensive_dir=alpha_defensive_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical M24.7 regime-aware case study.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the deterministic case-study artifacts should be written.",
    )
    parser.add_argument(
        "--no-reset-output",
        action="store_true",
        help="Keep any existing output root instead of clearing it before writing artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_case_study(output_root=args.output_root, reset_output=not args.no_reset_output, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
