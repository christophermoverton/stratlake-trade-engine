from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.run_portfolio import run_cli as run_portfolio_cli  # noqa: E402
from src.cli.run_strategy import run_cli as run_strategy_cli  # noqa: E402
from src.data.load_features import FeaturePaths, load_features  # noqa: E402
from src.research.metrics import aggregate_strategy_returns  # noqa: E402
from src.research.regimes import (  # noqa: E402
    GMM_LABEL_COLUMNS,
    REGIME_AUDIT_COLUMNS,
    REGIME_CALIBRATION_FILENAME,
    REGIME_DIMENSIONS,
    RegimeClassificationConfig,
    RegimeConditionalConfig,
    RegimeTransitionConfig,
    align_regimes_to_portfolio_windows,
    align_regimes_to_strategy_timeseries,
    analyze_portfolio_regime_transitions,
    analyze_strategy_regime_transitions,
    apply_regime_calibration,
    apply_regime_policy,
    builtin_regime_calibration_profiles,
    classify_market_regimes,
    classify_regime_shifts_with_gmm,
    compare_regime_results,
    evaluate_all_dimensions,
    summarize_regime_attribution,
    summarize_transition_attribution,
    write_regime_artifacts,
    write_regime_attribution_artifacts,
    write_regime_calibration_artifacts,
    write_regime_conditional_artifacts_multi_dimension,
    write_regime_gmm_artifacts,
    write_regime_policy_artifacts,
    write_regime_transition_artifacts,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "full_year_regime_calibration_case_study"
SUMMARY_FILENAME = "summary.json"
INTERPRETATION_FILENAME = "final_interpretation.md"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
DATA_COVERAGE_FILENAME = "data_coverage_summary.json"
DATASET_NAME = "features_daily"
REQUESTED_START_DATE = "2025-01-01"
REQUESTED_END_DATE = "2025-12-31"
REQUESTED_END_EXCLUSIVE = "2026-01-01"
TIMEFRAME = "1D"
MARKET_BASKET_SIZE = 12
STRATEGY_NAMES = ("momentum_v1", "mean_reversion_v1")
PORTFOLIO_NAME = "full_year_regime_static_portfolio_2025"
DOWNSTREAM_CALIBRATION_PROFILE = "baseline"


@dataclass(frozen=True)
class DataCoverageResult:
    summary: dict[str, Any]
    summary_path: Path
    market_symbols: list[str]
    features_root: Path


@dataclass(frozen=True)
class StrategySurfaceArtifacts:
    name: str
    run_id: str
    metrics: dict[str, Any]
    returns_frame: pd.DataFrame


@dataclass(frozen=True)
class CaseStudyArtifacts:
    summary: dict[str, Any]
    summary_path: Path
    output_root: Path
    data_coverage_path: Path
    interpretation_path: Path
    artifact_manifest_path: Path


class RealDataCoverageError(RuntimeError):
    """Raised when the required real 2025 feature coverage is unavailable."""


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        return path.as_posix()
    return "." if str(relative) == "." else relative.as_posix()


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _preview(frame: pd.DataFrame, *, columns: list[str], limit: int = 5) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    preview = frame.loc[:, available].head(limit).copy(deep=True)
    for column in preview.columns:
        if column == "ts_utc" or column.endswith("ts_utc"):
            preview[column] = pd.to_datetime(preview[column], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return _normalize_json_value(preview.to_dict(orient="records"))


def _rename_timestamp_column(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    if "ts_utc" not in normalized.columns and source in normalized.columns:
        normalized = normalized.rename(columns={source: "ts_utc"})
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["ts_utc"]).sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return normalized


@contextmanager
def example_environment(features_root: Path) -> Iterator[None]:
    previous_features_root = os.environ.get("FEATURES_ROOT")
    previous_cwd = Path.cwd()
    os.environ["FEATURES_ROOT"] = str(features_root)
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if previous_features_root is None:
            os.environ.pop("FEATURES_ROOT", None)
        else:
            os.environ["FEATURES_ROOT"] = previous_features_root


def _resolve_features_root(features_root: str | Path | None) -> Path:
    if features_root is None:
        return REPO_ROOT / "data"
    return Path(features_root)


def _dataset_root(features_root: Path) -> Path:
    return FeaturePaths(root=features_root).dataset_root(DATASET_NAME)


def _discover_2025_symbols(dataset_root: Path) -> list[str]:
    detected: set[str] = set()
    for symbol_dir in sorted(dataset_root.glob("symbol=*")):
        if not symbol_dir.is_dir():
            continue
        year_dir = symbol_dir / "year=2025"
        if not year_dir.is_dir():
            continue
        parquet_files = sorted(path for path in year_dir.glob("**/*.parquet") if path.is_file())
        if not parquet_files:
            continue
        detected.add(symbol_dir.name.split("symbol=", 1)[1])
    return sorted(detected)


def validate_real_2025_data_coverage(
    *,
    output_root: Path,
    features_root: str | Path | None = None,
    allow_real_data_fixture: bool = False,
) -> DataCoverageResult:
    resolved_features_root = _resolve_features_root(features_root)
    dataset_root = _dataset_root(resolved_features_root)
    coverage_path = output_root / DATA_COVERAGE_FILENAME
    detected_symbols = _discover_2025_symbols(dataset_root)

    frame = load_features(
        DATASET_NAME,
        start=REQUESTED_START_DATE,
        end=REQUESTED_END_EXCLUSIVE,
        paths=FeaturePaths(root=resolved_features_root),
        validate_integrity=False,
    )

    has_ts_utc = "ts_utc" in frame.columns
    has_date = "date" in frame.columns
    observed_timestamps = (
        pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        if has_ts_utc
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    if has_ts_utc:
        observed_dates = observed_timestamps.dt.normalize().dt.date
    elif has_date:
        observed_dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    else:
        observed_dates = pd.Series(dtype="object")

    valid_dates = observed_dates.dropna().sort_values(kind="stable")
    unique_dates = sorted(valid_dates.unique().tolist())
    unique_timestamps = observed_timestamps.dropna().sort_values(kind="stable")
    date_index = pd.Series(pd.to_datetime(unique_dates), dtype="datetime64[ns]")
    month_presence = {
        f"{month:02d}": bool((date_index.dt.month == month).any()) if not date_index.empty else False
        for month in range(1, 13)
    }

    gap_rows: list[dict[str, Any]] = []
    max_gap_days = 0
    if len(unique_dates) >= 2:
        for left, right in zip(unique_dates[:-1], unique_dates[1:], strict=False):
            gap_days = int((pd.Timestamp(right) - pd.Timestamp(left)).days)
            max_gap_days = max(max_gap_days, gap_days)
            if gap_days > 7:
                gap_rows.append({"from_date": str(left), "to_date": str(right), "gap_days": gap_days})

    symbol_counts = (
        frame["symbol"].astype("string").value_counts(dropna=True).sort_index().astype("int64")
        if "symbol" in frame.columns
        else pd.Series(dtype="int64")
    )
    per_symbol_coverage = (
        frame.assign(ts_utc=observed_timestamps)
        .dropna(subset=["symbol", "ts_utc"])
        .groupby("symbol", sort=True)["ts_utc"]
        .agg(["min", "max", "count"])
        .reset_index()
        if has_ts_utc and "symbol" in frame.columns and not frame.empty
        else pd.DataFrame(columns=["symbol", "min", "max", "count"])
    )
    if not per_symbol_coverage.empty:
        per_symbol_coverage["min"] = pd.to_datetime(per_symbol_coverage["min"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        per_symbol_coverage["max"] = pd.to_datetime(per_symbol_coverage["max"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resolved_features_root.resolve().relative_to((REPO_ROOT / "tests").resolve())
    except ValueError:
        is_fixture_root = False
    else:
        is_fixture_root = True
    data_source_type = "real_data_fixture" if is_fixture_root else "downloaded_real_data"
    fixture_mode_enabled = bool(is_fixture_root and allow_real_data_fixture)
    fixture_scope = "tests-only explicit fixture root" if is_fixture_root else None
    fixture_data_source_notes = (
        "Explicit real-data fixture roots are accepted only when allow_real_data_fixture=True."
        if is_fixture_root
        else "Production case study expects repository-local downloaded 2025 features_daily coverage."
    )

    failure_reasons: list[str] = []
    if not dataset_root.exists():
        failure_reasons.append(f"Dataset root does not exist: {_relative_to_repo(dataset_root)}")
    if not has_ts_utc and not has_date:
        failure_reasons.append("features_daily is missing both 'ts_utc' and 'date' columns.")
    if frame.empty:
        failure_reasons.append("No rows were loaded for the requested 2025 window.")
    if "symbol" not in frame.columns or frame["symbol"].dropna().empty:
        failure_reasons.append("No symbol coverage was found in the loaded frame.")
    if not detected_symbols:
        failure_reasons.append("No symbol=.../year=2025 parquet partitions were found.")
    if is_fixture_root and not allow_real_data_fixture:
        failure_reasons.append("Real-data fixture roots are allowed only when allow_real_data_fixture=True.")
    if unique_timestamps.empty:
        failure_reasons.append("No valid timestamps were available after parsing ts_utc/date columns.")
    if not unique_timestamps.empty and unique_timestamps.min().date() > pd.Timestamp("2025-01-10").date():
        failure_reasons.append("Observed 2025 data starts too late to support a full-year calendar study.")
    if not unique_timestamps.empty and unique_timestamps.max().date() < pd.Timestamp("2025-12-20").date():
        failure_reasons.append("Observed 2025 data ends too early to support a full-year calendar study.")
    if not all(month_presence.values()):
        missing_months = [month for month, present in month_presence.items() if not present]
        failure_reasons.append(f"Observed 2025 data is missing one or more calendar months: {missing_months}.")
    if len(unique_dates) < 200:
        failure_reasons.append(
            f"Observed 2025 coverage includes only {len(unique_dates)} unique dates; expected full-year trading coverage."
        )
    if max_gap_days > 7:
        failure_reasons.append(
            f"Observed 2025 coverage contains at least one date gap greater than 7 days (max_gap_days={max_gap_days})."
        )

    summary = {
        "artifact_type": "data_coverage_summary",
        "dataset": DATASET_NAME,
        "data_source": {
            "features_root": _relative_to_repo(resolved_features_root),
            "dataset_root": _relative_to_repo(dataset_root),
            "type": data_source_type,
            "downloaded_real_data": not is_fixture_root,
            "real_data_fixture": is_fixture_root,
            "real_data_used": True,
            "mock_or_synthetic_data": False,
            "allow_real_data_fixture": bool(allow_real_data_fixture),
            "fixture_mode_enabled": fixture_mode_enabled,
            "fixture_scope": fixture_scope,
            "fixture_data_source_notes": fixture_data_source_notes,
        },
        "requested_window": {
            "start_date": REQUESTED_START_DATE,
            "end_date": REQUESTED_END_DATE,
            "end_exclusive": REQUESTED_END_EXCLUSIVE,
        },
        "observed_window": {
            "min_timestamp": None if unique_timestamps.empty else unique_timestamps.min(),
            "max_timestamp": None if unique_timestamps.empty else unique_timestamps.max(),
            "min_date": None if valid_dates.empty else str(valid_dates.iloc[0]),
            "max_date": None if valid_dates.empty else str(valid_dates.iloc[-1]),
        },
        "counts": {
            "row_count": int(len(frame)),
            "symbol_count": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
            "detected_symbol_partition_count": int(len(detected_symbols)),
            "unique_date_count": int(len(unique_dates)),
        },
        "schema": {
            "columns": [str(column) for column in frame.columns],
            "has_ts_utc": has_ts_utc,
            "has_date": has_date,
        },
        "diagnostics": {
            "month_presence": month_presence,
            "max_gap_days": int(max_gap_days),
            "large_gap_examples": gap_rows[:10],
            "rows_per_symbol_summary": {
                "min": None if symbol_counts.empty else int(symbol_counts.min()),
                "median": None if symbol_counts.empty else float(symbol_counts.median()),
                "max": None if symbol_counts.empty else int(symbol_counts.max()),
            },
            "per_symbol_coverage_preview": _normalize_json_value(per_symbol_coverage.head(10).to_dict(orient="records")),
        },
        "market_basket_symbols": detected_symbols[:MARKET_BASKET_SIZE],
        "coverage_status": {
            "passed": len(failure_reasons) == 0,
            "failure_reasons": failure_reasons,
        },
    }
    _write_json(coverage_path, summary)
    if failure_reasons:
        raise RealDataCoverageError(
            "Full-year 2025 real-data coverage validation failed. "
            f"See {_relative_to_output(coverage_path, output_root)} for details."
        )
    return DataCoverageResult(
        summary=summary,
        summary_path=coverage_path,
        market_symbols=detected_symbols[:MARKET_BASKET_SIZE],
        features_root=resolved_features_root,
    )


def _load_market_data(features_root: Path, *, market_symbols: list[str]) -> pd.DataFrame:
    frame = load_features(
        DATASET_NAME,
        start=REQUESTED_START_DATE,
        end=REQUESTED_END_EXCLUSIVE,
        symbols=market_symbols,
        paths=FeaturePaths(root=features_root),
    )
    market = frame.loc[:, ["ts_utc", "symbol", "close"]].copy(deep=True)
    market["close"] = pd.to_numeric(market["close"], errors="coerce").astype("float64")
    market = market.dropna(subset=["ts_utc", "symbol", "close"]).sort_values(["ts_utc", "symbol"], kind="stable")
    return market.reset_index(drop=True)


def _run_strategy_surfaces() -> dict[str, StrategySurfaceArtifacts]:
    outputs: dict[str, StrategySurfaceArtifacts] = {}
    for strategy_name in STRATEGY_NAMES:
        result = run_strategy_cli(
            [
                "--strategy",
                strategy_name,
                "--start",
                REQUESTED_START_DATE,
                "--end",
                REQUESTED_END_EXCLUSIVE,
            ]
        )
        returns_frame = aggregate_strategy_returns(result.results_df)
        returns_frame["ts_utc"] = pd.to_datetime(returns_frame["ts_utc"], utc=True, errors="coerce")
        returns_frame = returns_frame.dropna(subset=["ts_utc"]).sort_values("ts_utc", kind="stable").reset_index(drop=True)
        outputs[strategy_name] = StrategySurfaceArtifacts(
            name=strategy_name,
            run_id=result.run_id,
            metrics=dict(result.metrics),
            returns_frame=returns_frame,
        )
    return outputs


def _run_static_portfolio(output_root: Path, strategy_surfaces: dict[str, StrategySurfaceArtifacts]) -> tuple[Any, pd.DataFrame]:
    portfolio_output_root = output_root / "native_artifacts" / "portfolios"
    portfolio_output_root.mkdir(parents=True, exist_ok=True)
    portfolio_result = run_portfolio_cli(
        [
            "--run-ids",
            *[strategy_surfaces[name].run_id for name in STRATEGY_NAMES],
            "--portfolio-name",
            PORTFOLIO_NAME,
            "--output-dir",
            portfolio_output_root.as_posix(),
            "--timeframe",
            TIMEFRAME,
        ]
    )
    portfolio_frame = pd.read_csv(portfolio_result.experiment_dir / "portfolio_returns.csv")
    return portfolio_result, _rename_timestamp_column(portfolio_frame, source="ts")


def _write_surface_bundle(
    bundle_dir: Path,
    *,
    labels: pd.DataFrame,
    regime_metadata: dict[str, Any],
    conditional_results: dict[str, Any],
    transition_result: Any,
    attribution: Any,
    transition_attribution: Any,
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


def _evaluate_strategy_surface(
    *,
    strategy_surface: StrategySurfaceArtifacts,
    labels: pd.DataFrame,
    output_dir: Path,
    regime_metadata: dict[str, Any],
    comparison: Any | None,
) -> dict[str, Any]:
    aligned = align_regimes_to_strategy_timeseries(strategy_surface.returns_frame, labels)
    conditional_results = evaluate_all_dimensions(
        aligned,
        surface="strategy",
        return_column="strategy_return",
        config=RegimeConditionalConfig(
            min_observations=5,
            metadata={"case_study": "full_year_regime_calibration_case_study"},
        ),
    )
    transition_result = analyze_strategy_regime_transitions(
        aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=RegimeTransitionConfig(
            pre_event_rows=3,
            post_event_rows=3,
            min_observations=5,
            metadata={"case_study": "full_year_regime_calibration_case_study"},
        ),
    )
    attribution = summarize_regime_attribution(conditional_results["composite"], run_id=strategy_surface.name)
    transition_attribution = summarize_transition_attribution(transition_result, run_id=strategy_surface.name)
    _write_surface_bundle(
        output_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "surface": "strategy", "strategy_name": strategy_surface.name},
        conditional_results=conditional_results,
        transition_result=transition_result,
        attribution=attribution,
        transition_attribution=transition_attribution,
        comparison=comparison,
        run_id=f"{strategy_surface.name}_bundle",
        metadata={"surface": "strategy", "strategy_name": strategy_surface.name},
    )
    return {
        "aligned": aligned,
        "conditional_results": conditional_results,
        "transition_result": transition_result,
        "attribution": attribution,
        "transition_attribution": transition_attribution,
    }


def _evaluate_portfolio_surface(
    *,
    portfolio_frame: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
    regime_metadata: dict[str, Any],
    comparison: Any | None = None,
) -> dict[str, Any]:
    aligned = align_regimes_to_portfolio_windows(portfolio_frame, labels)
    conditional_results = evaluate_all_dimensions(
        aligned,
        surface="portfolio",
        return_column="portfolio_return",
        config=RegimeConditionalConfig(
            min_observations=5,
            metadata={"case_study": "full_year_regime_calibration_case_study"},
        ),
    )
    transition_result = analyze_portfolio_regime_transitions(
        aligned,
        labels,
        dimensions=list(REGIME_DIMENSIONS),
        config=RegimeTransitionConfig(
            pre_event_rows=3,
            post_event_rows=3,
            min_observations=5,
            metadata={"case_study": "full_year_regime_calibration_case_study"},
        ),
    )
    attribution = summarize_regime_attribution(conditional_results["composite"], run_id="static_portfolio")
    transition_attribution = summarize_transition_attribution(transition_result, run_id="static_portfolio")
    _write_surface_bundle(
        output_dir,
        labels=labels,
        regime_metadata={**regime_metadata, "surface": "portfolio", "portfolio_name": PORTFOLIO_NAME},
        conditional_results=conditional_results,
        transition_result=transition_result,
        attribution=attribution,
        transition_attribution=transition_attribution,
        comparison=comparison,
        run_id="static_portfolio_bundle",
        metadata={"surface": "portfolio", "portfolio_name": PORTFOLIO_NAME},
    )
    return {
        "aligned": aligned,
        "conditional_results": conditional_results,
        "transition_result": transition_result,
        "attribution": attribution,
        "transition_attribution": transition_attribution,
    }


def _policy_config() -> dict[str, Any]:
    return {
        "regime_aliases": {
            "risk_on": {"match": {"trend": "uptrend", "stress": "normal_stress"}},
            "risk_off": {"match": {"stress_state": "correlation_stress"}},
            "drawdown_watch": {"match": {"drawdown_recovery": "drawdown"}},
        },
        "regime_policy": {
            "default": {
                "signal_scale": 1.0,
                "allocation_scale": 1.0,
                "alpha_weight_multiplier": 1.0,
                "volatility_target": 0.10,
                "gross_exposure_cap": 1.0,
                "max_component_weight": 1.0,
                "rebalance_enabled": True,
                "optimizer_override": None,
                "allocation_rule_override": None,
                "fallback_policy": "baseline",
            },
            "confidence": {
                "min_confidence": 0.60,
                "low_confidence_fallback": "neutral",
                "ambiguous_fallback": "reduce_exposure",
                "unsupported_fallback": "baseline",
            },
            "regimes": {
                "risk_on": {
                    "signal_scale": 1.05,
                    "allocation_scale": 1.05,
                    "alpha_weight_multiplier": 1.0,
                    "volatility_target": 0.11,
                    "gross_exposure_cap": 1.0,
                    "max_component_weight": 1.0,
                    "rebalance_enabled": True,
                    "optimizer_override": "equal_weight",
                    "allocation_rule_override": None,
                    "fallback_policy": "baseline",
                },
                "risk_off": {
                    "signal_scale": 0.65,
                    "allocation_scale": 0.65,
                    "alpha_weight_multiplier": 1.0,
                    "volatility_target": 0.07,
                    "gross_exposure_cap": 0.65,
                    "max_component_weight": 0.65,
                    "rebalance_enabled": True,
                    "optimizer_override": "equal_weight",
                    "allocation_rule_override": None,
                    "fallback_policy": "reduce_exposure",
                },
                "drawdown_watch": {
                    "signal_scale": 0.85,
                    "allocation_scale": 0.85,
                    "alpha_weight_multiplier": 1.0,
                    "volatility_target": 0.08,
                    "gross_exposure_cap": 0.85,
                    "max_component_weight": 0.85,
                    "rebalance_enabled": True,
                    "optimizer_override": "equal_weight",
                    "allocation_rule_override": None,
                    "fallback_policy": "reduce_exposure",
                },
            },
        },
    }


def _build_interpretation(summary: dict[str, Any]) -> str:
    raw_portfolio = summary["raw_regime_evaluation"]["portfolio"]
    calibrated_portfolio = summary["calibrated_regime_evaluation"]["portfolio"]
    adaptive = summary["adaptive_policy"]["comparison_metrics"]
    calibration_changed = summary["calibrated_regime_evaluation"]["interpretation_change"]["portfolio_best_regime_changed"]
    gmm = summary["gmm_confidence"]
    adaptive_delta = adaptive["adaptive_total_return"] - adaptive["baseline_total_return"]
    adaptive_direction = "improved" if adaptive_delta > 0 else "degraded" if adaptive_delta < 0 else "matched"
    lines = [
        "# Full-Year 2025 Regime Calibration Case Study",
        "",
        "## Static Baseline",
        (
            f"- The static portfolio baseline recorded total_return={adaptive['baseline_total_return']} and "
            f"sharpe={adaptive['baseline_sharpe']} before any regime-aware overlay."
        ),
        (
            f"- Raw M24 attribution identified `{raw_portfolio['best_regime']['regime_label']}` as the strongest "
            f"portfolio regime and `{raw_portfolio['worst_regime']['regime_label']}` as the weakest."
        ),
        "",
        "## Calibration And Confidence",
        (
            f"- Downstream calibration used the `{summary['calibrated_regime_evaluation']['downstream_profile']}` profile; "
            f"portfolio best-regime interpretation changed={str(calibration_changed).lower()}."
        ),
        (
            f"- The calibrated portfolio surface now highlights "
            f"`{calibrated_portfolio['best_regime']['regime_label']}` as the strongest regime."
        ),
        (
            f"- GMM confidence flagged {gmm['low_confidence_row_count']} low-confidence rows and "
            f"{gmm['shift_event_count']} shift events, which were treated as a diagnostic layer rather than a taxonomy replacement."
        ),
        "",
        "## Adaptive Portfolio",
        (
            f"- The adaptive overlay {adaptive_direction} the static baseline on total return by {adaptive_delta} "
            f"while moving Sharpe from {adaptive['baseline_sharpe']} to {adaptive['adaptive_sharpe']}."
        ),
        (
            f"- Static remained competitive when the fallback path was active on {adaptive['fallback_row_count']} rows, "
            f"which limited how aggressively the adaptive policy could respond."
        ),
        "",
        "## Follow-Up Questions",
        "- Should downstream policy use a more conservative calibration profile when GMM low-confidence share rises?",
        "- Would symbol-level confidence or portfolio-component confidence improve adaptation timing versus market-level confidence only?",
        "- Are the most adverse transition categories better handled by risk caps, allocation scaling, or explicit rebalance suppression?",
    ]
    return "\n".join(lines)


def _artifact_tree(output_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(output_root)).replace("\\", "/")
        for path in output_root.rglob("*")
        if path.is_file()
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


def _build_policy_input(portfolio_frame: pd.DataFrame, calibrated_labels: pd.DataFrame) -> pd.DataFrame:
    aligned = align_regimes_to_portfolio_windows(
        portfolio_frame,
        calibrated_labels,
        include_metric_columns=True,
    )
    rename_map = {
        "regime_volatility_state": "volatility_state",
        "regime_trend_state": "trend_state",
        "regime_drawdown_recovery_state": "drawdown_recovery_state",
        "regime_stress_state": "stress_state",
        "regime_label": "regime_label",
        "regime_is_defined": "is_defined",
        "regime_volatility_metric": "volatility_metric",
        "regime_trend_metric": "trend_metric",
        "regime_drawdown_metric": "drawdown_metric",
        "regime_stress_correlation_metric": "stress_correlation_metric",
        "regime_stress_dispersion_metric": "stress_dispersion_metric",
    }
    policy_input = aligned.rename(columns=rename_map)
    required_columns = [
        "ts_utc",
        "portfolio_return",
        "volatility_state",
        "trend_state",
        "drawdown_recovery_state",
        "stress_state",
        "regime_label",
        "is_defined",
        "volatility_metric",
        "trend_metric",
        "drawdown_metric",
        "stress_correlation_metric",
        "stress_dispersion_metric",
    ]
    return policy_input.loc[:, [column for column in required_columns if column in policy_input.columns]].copy(deep=True)


def run_case_study(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    features_root: str | Path | None = None,
    reset_output: bool = True,
    verbose: bool = True,
    allow_real_data_fixture: bool = False,
) -> CaseStudyArtifacts:
    resolved_output_root = Path(output_root)
    if reset_output and resolved_output_root.exists():
        shutil.rmtree(resolved_output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    coverage = validate_real_2025_data_coverage(
        output_root=resolved_output_root,
        features_root=features_root,
        allow_real_data_fixture=allow_real_data_fixture,
    )

    raw_regime_dir = resolved_output_root / "regime_bundle"
    raw_strategy_dirs = {name: resolved_output_root / f"strategy_{name}_bundle" for name in STRATEGY_NAMES}
    raw_portfolio_dir = resolved_output_root / "static_portfolio_bundle"
    calibrated_portfolio_dir = resolved_output_root / "calibrated_portfolio_bundle"
    gmm_dir = resolved_output_root / "gmm_confidence"
    calibration_root = resolved_output_root / "calibration_profiles"
    adaptive_policy_dir = resolved_output_root / "adaptive_policy"
    for directory in [raw_regime_dir, *raw_strategy_dirs.values(), raw_portfolio_dir, calibrated_portfolio_dir, gmm_dir, calibration_root, adaptive_policy_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    with example_environment(coverage.features_root):
        market_data = _load_market_data(coverage.features_root, market_symbols=coverage.market_symbols)
        strategy_surfaces = _run_strategy_surfaces()
        portfolio_result, portfolio_frame = _run_static_portfolio(resolved_output_root, strategy_surfaces)

    classification = classify_market_regimes(
        market_data,
        config=RegimeClassificationConfig(
            price_column="close",
            metadata={
                "case_study": "full_year_regime_calibration_case_study",
                "source_dataset": DATASET_NAME,
                "market_symbols": coverage.market_symbols,
            },
        ),
    )
    labels = classification.labels
    raw_regime_metadata = {
        **classification.metadata,
        "case_study": "full_year_regime_calibration_case_study",
        "date_window": {"start": REQUESTED_START_DATE, "end_exclusive": REQUESTED_END_EXCLUSIVE},
        "source_dataset": DATASET_NAME,
        "market_symbols": coverage.market_symbols,
    }
    write_regime_artifacts(raw_regime_dir, labels, metadata=raw_regime_metadata)

    gmm_feature_frame = labels.dropna(subset=list(REGIME_AUDIT_COLUMNS)).reset_index(drop=True)
    gmm_result = classify_regime_shifts_with_gmm(
        gmm_feature_frame.loc[:, ["ts_utc", *list(REGIME_AUDIT_COLUMNS)]],
        config={
            "n_components": 3,
            "random_state": 42,
            "min_observations": 30,
            "feature_columns": REGIME_AUDIT_COLUMNS,
            "metadata": {"case_study": "full_year_regime_calibration_case_study"},
        },
    )
    gmm_manifest = write_regime_gmm_artifacts(
        gmm_dir,
        gmm_result,
        run_id="full_year_2025_gmm",
        source_regime_artifact_references={"regime_manifest": "../regime_bundle/regime_manifest.json"},
        extra_metadata={"case_study": "full_year_regime_calibration_case_study"},
    )

    labels_with_confidence = labels.merge(
        gmm_result.labels.loc[:, ["ts_utc", "gmm_confidence_score"]],
        on="ts_utc",
        how="left",
        sort=False,
    )

    calibration_results: dict[str, Any] = {}
    calibration_profile_summary: dict[str, Any] = {}
    for profile_name in sorted(builtin_regime_calibration_profiles()):
        result = apply_regime_calibration(
            labels_with_confidence,
            profile=profile_name,
            confidence_column="gmm_confidence_score",
            low_confidence_threshold=0.60,
            metadata={"case_study": "full_year_regime_calibration_case_study"},
        )
        calibration_results[profile_name] = result
        write_regime_calibration_artifacts(
            calibration_root / profile_name,
            result,
            run_id=f"calibration_{profile_name}",
            source_regime_artifact_references={"regime_manifest": "../regime_bundle/regime_manifest.json"},
            taxonomy_metadata={"taxonomy_version": classification.taxonomy_version},
            extra_metadata={"case_study": "full_year_regime_calibration_case_study"},
        )
        calibration_profile_summary[profile_name] = {
            "profile_flags": result.profile_flags,
            "stability_metrics": result.stability_metrics,
            "warning_count": int(len(result.warnings)),
            "warnings": list(result.warnings),
        }

    raw_strategy_conditionals = {}
    raw_strategy_summaries: dict[str, Any] = {}
    for strategy_name, strategy_surface in strategy_surfaces.items():
        strategy_aligned = align_regimes_to_strategy_timeseries(strategy_surface.returns_frame, labels)
        raw_strategy_conditionals[strategy_name] = evaluate_all_dimensions(
            strategy_aligned,
            surface="strategy",
            return_column="strategy_return",
            config=RegimeConditionalConfig(
                min_observations=5,
                metadata={"case_study": "full_year_regime_calibration_case_study"},
            ),
        )["composite"]
    strategy_comparison = compare_regime_results(
        raw_strategy_conditionals,
        surface="strategy",
        dimension="composite",
        metadata={"case_study": "full_year_regime_calibration_case_study"},
    )

    for strategy_name, strategy_surface in strategy_surfaces.items():
        evaluation = _evaluate_strategy_surface(
            strategy_surface=strategy_surface,
            labels=labels,
            output_dir=raw_strategy_dirs[strategy_name],
            regime_metadata=raw_regime_metadata,
            comparison=strategy_comparison if strategy_name == STRATEGY_NAMES[0] else None,
        )
        raw_strategy_summaries[strategy_name] = {
            "run_id": strategy_surface.run_id,
            "metrics": strategy_surface.metrics,
            "best_regime": evaluation["attribution"].summary["best_regime"],
            "worst_regime": evaluation["attribution"].summary["worst_regime"],
            "fragility_flag": evaluation["attribution"].summary["fragility_flag"],
        }

    raw_portfolio_evaluation = _evaluate_portfolio_surface(
        portfolio_frame=portfolio_frame,
        labels=labels,
        output_dir=raw_portfolio_dir,
        regime_metadata=raw_regime_metadata,
    )

    downstream_calibration = calibration_results[DOWNSTREAM_CALIBRATION_PROFILE]
    calibrated_labels = downstream_calibration.labels
    calibrated_portfolio_evaluation = _evaluate_portfolio_surface(
        portfolio_frame=portfolio_frame,
        labels=calibrated_labels,
        output_dir=calibrated_portfolio_dir,
        regime_metadata={**raw_regime_metadata, "calibration_profile": DOWNSTREAM_CALIBRATION_PROFILE},
    )

    confidence_frame = gmm_result.labels.loc[
        :,
        [column for column in ("ts_utc", "confidence_score", "confidence_bucket", "fallback_flag", "fallback_reason", "predicted_label") if column in gmm_result.labels.columns],
    ].copy(deep=True)
    adaptive_policy_result = apply_regime_policy(
        _build_policy_input(portfolio_frame, calibrated_labels),
        config=_policy_config(),
        confidence_frame=confidence_frame,
        baseline_frame=portfolio_frame,
        surface="portfolio",
        calibration_profile=DOWNSTREAM_CALIBRATION_PROFILE,
        is_unstable_profile=bool(downstream_calibration.profile_flags.get("is_unstable_profile", False)),
        eligible_for_downstream_decisioning=not bool(downstream_calibration.profile_flags.get("is_unstable_profile", False)),
        metadata={"case_study": "full_year_regime_calibration_case_study"},
    )
    adaptive_policy_manifest = write_regime_policy_artifacts(
        adaptive_policy_dir,
        adaptive_policy_result,
        run_id="adaptive_portfolio_policy",
        source_regime_artifact_references={"calibrated_portfolio_bundle": "../calibrated_portfolio_bundle"},
        calibration_profile=DOWNSTREAM_CALIBRATION_PROFILE,
        confidence_artifact_references={"gmm_manifest": "../gmm_confidence/regime_gmm_manifest.json"},
        extra_metadata={"case_study": "full_year_regime_calibration_case_study"},
    )

    adaptive_comparison_row = adaptive_policy_result.comparison.iloc[0].to_dict() if not adaptive_policy_result.comparison.empty else {}
    raw_best_regime = raw_portfolio_evaluation["attribution"].summary["best_regime"]
    calibrated_best_regime = calibrated_portfolio_evaluation["attribution"].summary["best_regime"]

    summary: dict[str, Any] = {
        "case_study": {
            "milestone": "M25",
            "name": "full_year_regime_calibration_case_study",
            "objective": "Full-year 2025 real-data regime calibration, confidence, and adaptive portfolio case study.",
        },
        "input_path": {
            "dataset": DATASET_NAME,
            "features_root": _relative_to_repo(coverage.features_root),
            "features_dataset_root": _relative_to_repo(_dataset_root(coverage.features_root)),
            "requested_window": {
                "start_date": REQUESTED_START_DATE,
                "end_date": REQUESTED_END_DATE,
                "end_exclusive": REQUESTED_END_EXCLUSIVE,
            },
            "market_symbols": coverage.market_symbols,
            "strategy_names": list(STRATEGY_NAMES),
            "portfolio_name": PORTFOLIO_NAME,
        },
        "paths": {
            "output_root": ".",
            "data_coverage_summary": DATA_COVERAGE_FILENAME,
            "regime_bundle": _relative_to_output(raw_regime_dir, resolved_output_root),
            "gmm_confidence": _relative_to_output(gmm_dir, resolved_output_root),
            "calibration_profiles": _relative_to_output(calibration_root, resolved_output_root),
            "static_portfolio_bundle": _relative_to_output(raw_portfolio_dir, resolved_output_root),
            "calibrated_portfolio_bundle": _relative_to_output(calibrated_portfolio_dir, resolved_output_root),
            "adaptive_policy": _relative_to_output(adaptive_policy_dir, resolved_output_root),
            "final_interpretation": INTERPRETATION_FILENAME,
            "artifact_manifest": ARTIFACT_MANIFEST_FILENAME,
        },
        "data_coverage": coverage.summary,
        "static_baseline": {
            "strategy_runs": raw_strategy_summaries,
            "portfolio_run": {
                "run_id": portfolio_result.run_id,
                "component_count": int(portfolio_result.component_count),
                "metrics": dict(portfolio_result.metrics),
                "portfolio_return_preview": _preview(portfolio_frame, columns=["ts_utc", "portfolio_return"], limit=5),
            },
            "strategy_comparison": {
                "best_run": _best_run_summary(strategy_comparison),
                "robust_run_ids": strategy_comparison.summary["robust_run_ids"],
                "warning_row_count": int(strategy_comparison.comparison_table["warning_flag"].notna().sum()),
            },
        },
        "classification": {
            "defined_row_count": int(labels["is_defined"].sum()),
            "undefined_row_count": int((~labels["is_defined"]).sum()),
            "label_preview": _preview(labels, columns=["ts_utc", "volatility_state", "trend_state", "drawdown_recovery_state", "stress_state", "regime_label"], limit=6),
            "state_distribution": {
                "volatility": labels["volatility_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "trend": labels["trend_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "drawdown_recovery": labels["drawdown_recovery_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
                "stress": labels["stress_state"].astype("string").value_counts(dropna=False).sort_index().to_dict(),
            },
        },
        "raw_regime_evaluation": {
            "portfolio": {
                "best_regime": raw_portfolio_evaluation["attribution"].summary["best_regime"],
                "worst_regime": raw_portfolio_evaluation["attribution"].summary["worst_regime"],
                "fragility_flag": raw_portfolio_evaluation["attribution"].summary["fragility_flag"],
                "transition_worst_category": raw_portfolio_evaluation["transition_attribution"].summary["worst_transition_category"],
            },
            "strategies": raw_strategy_summaries,
        },
        "calibration_profile_evaluation": {
            "profiles": calibration_profile_summary,
            "downstream_profile": DOWNSTREAM_CALIBRATION_PROFILE,
        },
        "gmm_confidence": {
            "summary": gmm_result.summary,
            "manifest": gmm_manifest,
            "label_column_contract": list(GMM_LABEL_COLUMNS),
            "low_confidence_row_count": int(gmm_result.labels["gmm_is_low_confidence"].astype("bool").sum()),
            "shift_event_count": int(len(gmm_result.shift_events)),
            "low_confidence_preview": _preview(
                gmm_result.labels.loc[gmm_result.labels["gmm_is_low_confidence"].astype("bool")],
                columns=["ts_utc", "gmm_confidence_score", "confidence_bucket", "fallback_reason"],
                limit=8,
            ),
        },
        "calibrated_regime_evaluation": {
            "downstream_profile": DOWNSTREAM_CALIBRATION_PROFILE,
            "portfolio": {
                "best_regime": calibrated_portfolio_evaluation["attribution"].summary["best_regime"],
                "worst_regime": calibrated_portfolio_evaluation["attribution"].summary["worst_regime"],
                "fragility_flag": calibrated_portfolio_evaluation["attribution"].summary["fragility_flag"],
                "transition_worst_category": calibrated_portfolio_evaluation["transition_attribution"].summary["worst_transition_category"],
            },
            "interpretation_change": {
                "portfolio_best_regime_changed": raw_best_regime["regime_label"] != calibrated_best_regime["regime_label"],
                "portfolio_worst_regime_changed": (
                    raw_portfolio_evaluation["attribution"].summary["worst_regime"]["regime_label"]
                    != calibrated_portfolio_evaluation["attribution"].summary["worst_regime"]["regime_label"]
                ),
            },
            "downstream_calibration_artifact": {
                "profile_name": downstream_calibration.profile.name,
                "manifest_path": f"calibration_profiles/{DOWNSTREAM_CALIBRATION_PROFILE}/{REGIME_CALIBRATION_FILENAME}",
                "is_unstable_profile": bool(downstream_calibration.profile_flags.get("is_unstable_profile", False)),
            },
        },
        "adaptive_policy": {
            "policy_summary": adaptive_policy_result.summary,
            "policy_artifact_manifest": adaptive_policy_manifest,
            "comparison_metrics": adaptive_comparison_row,
        },
        "limitations": [
            "The regime surface is derived from a deterministic basket of repository-available features_daily symbols rather than an external benchmark index.",
            "GMM confidence is a diagnostic overlay and does not replace the Milestone 24 taxonomy labels.",
            "Adaptive comparison is a deterministic research diagnostic that rescales baseline returns; it is not live trading logic.",
            "Policy parameters are intentionally conservative and illustrative rather than tuned to force adaptive outperformance.",
        ],
    }

    interpretation_path = resolved_output_root / INTERPRETATION_FILENAME
    _write_text(interpretation_path, _build_interpretation(summary))

    summary_path = resolved_output_root / SUMMARY_FILENAME
    summary["artifact_tree"] = _artifact_tree(resolved_output_root)
    _write_json(summary_path, summary)

    artifact_manifest_path = resolved_output_root / ARTIFACT_MANIFEST_FILENAME
    _write_json(
        artifact_manifest_path,
        {"artifact_type": "full_year_regime_calibration_case_study", "file_inventory": _artifact_tree(resolved_output_root)},
    )
    summary["artifact_tree"] = _artifact_tree(resolved_output_root)
    _write_json(summary_path, summary)

    if verbose:
        print(json.dumps(summary["paths"], indent=2, sort_keys=True))
        print(f"summary_path={summary_path.as_posix()}")

    return CaseStudyArtifacts(
        summary=summary,
        summary_path=summary_path,
        output_root=resolved_output_root,
        data_coverage_path=coverage.summary_path,
        interpretation_path=interpretation_path,
        artifact_manifest_path=artifact_manifest_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full-year 2025 regime calibration case study.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory where deterministic example artifacts should be written.")
    parser.add_argument("--features-root", type=Path, default=None, help="Optional features root override. Defaults to repository data.")
    parser.add_argument("--allow-real-data-fixture", action="store_true", help="Allow an explicit real-data fixture root for test execution.")
    parser.add_argument("--no-reset-output", action="store_true", help="Keep any existing output root instead of clearing it before writing artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_case_study(
        output_root=args.output_root,
        features_root=args.features_root,
        reset_output=not args.no_reset_output,
        verbose=True,
        allow_real_data_fixture=bool(args.allow_real_data_fixture),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
