from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from src.research.metrics import compute_performance_metrics
from src.research.registry import load_registry
from src.research.strategy_qa import LOW_DATA_THRESHOLD

_TOLERANCE = 1e-6
_REQUIRED_ARTIFACTS = (
    "metrics.json",
    "equity_curve.csv",
    "signal_diagnostics.json",
    "qa_summary.json",
)
_OPTIONAL_METRIC_KEYS = (
    "total_return",
    "cumulative_return",
    "max_drawdown",
)
_REGISTRY_METRIC_KEYS = (
    "cumulative_return",
    "total_return",
    "volatility",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "hit_rate",
    "profit_factor",
    "turnover",
    "total_turnover",
    "average_turnover",
    "trade_count",
    "rebalance_count",
    "percent_periods_traded",
    "average_trade_size",
    "total_transaction_cost",
    "total_slippage_cost",
    "total_execution_friction",
    "average_execution_friction_per_trade",
    "exposure_pct",
)


class ConsistencyError(ValueError):
    """Raised when run artifacts disagree with one another."""


def validate_features_to_signals_consistency(
    features_df: pd.DataFrame,
    signals: pd.Series,
) -> None:
    """Ensure strategy signals preserve deterministic feature-row identity."""

    if not isinstance(signals, pd.Series):
        raise ConsistencyError("ConsistencyError: strategy signals must be provided as a pandas Series.")

    _validate_unique_index(features_df, owner="feature dataset")
    if not signals.index.is_unique:
        duplicate_label = signals.index[signals.index.duplicated()][0]
        raise ConsistencyError(
            "ConsistencyError: strategy signal output contains duplicate index labels. "
            f"First duplicate label: {duplicate_label!r}."
        )

    if len(signals) != len(features_df):
        raise ConsistencyError(
            "ConsistencyError: feature-to-signal row count mismatch "
            f"(features={len(features_df)} vs signals={len(signals)})."
        )
    if not signals.index.equals(features_df.index):
        raise ConsistencyError(
            "ConsistencyError: feature-to-signal index mismatch. "
            "Strategy signals must be aligned exactly with the input DataFrame index without drops, duplicates, or reordering."
        )

    _validate_frame_key_identity(
        features_df,
        features_df.assign(signal=signals.to_numpy(copy=False)),
        left_name="features",
        right_name="signals",
    )


def validate_signals_to_backtest_consistency(
    signal_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    *,
    return_column: str,
) -> None:
    """Ensure backtest execution adds columns without changing row identity."""

    if return_column not in signal_df.columns:
        raise ConsistencyError(
            f"ConsistencyError: backtest input is missing expected return column {return_column!r}."
        )
    if return_column not in backtest_df.columns:
        raise ConsistencyError(
            f"ConsistencyError: backtest output is missing expected return column {return_column!r}."
        )
    if "signal" not in signal_df.columns or "signal" not in backtest_df.columns:
        raise ConsistencyError("ConsistencyError: signal-to-backtest validation requires a 'signal' column.")

    _validate_unique_index(signal_df, owner="signal DataFrame")
    _validate_unique_index(backtest_df, owner="backtest output")

    if len(signal_df) != len(backtest_df):
        raise ConsistencyError(
            "ConsistencyError: signal-to-backtest row count mismatch "
            f"(signals={len(signal_df)} vs backtest={len(backtest_df)})."
        )
    if not backtest_df.index.equals(signal_df.index):
        raise ConsistencyError(
            "ConsistencyError: signal-to-backtest index mismatch. "
            "Backtest execution must preserve the signal frame index exactly."
        )

    _validate_frame_key_identity(
        signal_df,
        backtest_df,
        left_name="signals",
        right_name="backtest",
    )

    signal_values = pd.to_numeric(signal_df["signal"], errors="coerce").astype("float64")
    backtest_signal_values = pd.to_numeric(backtest_df["signal"], errors="coerce").astype("float64")
    if not signal_values.equals(backtest_signal_values):
        raise ConsistencyError(
            "ConsistencyError: backtest output altered the 'signal' column. "
            "Backtest execution may add columns, but it must preserve input signals."
        )

    executed_signal = pd.to_numeric(backtest_df.get("executed_signal"), errors="coerce")
    if executed_signal.isna().any():
        first_bad = executed_signal.index[executed_signal.isna()][0]
        raise ConsistencyError(
            "ConsistencyError: backtest output contains invalid executed_signal values. "
            f"First failing index: {first_bad!r}."
        )


def validate_strategy_artifact_payload_consistency(
    *,
    results_df: pd.DataFrame,
    signals_frame: pd.DataFrame,
    equity_curve_frame: pd.DataFrame,
    metrics: dict[str, Any],
    signal_diagnostics: dict[str, Any],
    qa_summary: dict[str, Any],
) -> None:
    """Ensure in-memory strategy artifacts match the validated run outputs."""

    _validate_frame_key_identity(
        results_df,
        signals_frame,
        left_name="strategy results",
        right_name="signals artifact",
    )

    if len(signals_frame) != len(results_df):
        raise ConsistencyError(
            "ConsistencyError: signals.parquet row count mismatch "
            f"(results={len(results_df)} vs artifact={len(signals_frame)})."
        )
    if len(equity_curve_frame) != len(results_df):
        raise ConsistencyError(
            "ConsistencyError: equity_curve.csv row count mismatch "
            f"(results={len(results_df)} vs artifact={len(equity_curve_frame)})."
        )

    _validate_frame_key_identity(
        results_df,
        equity_curve_frame,
        left_name="strategy results",
        right_name="equity curve artifact",
    )
    _validate_metrics_against_equity(metrics, equity_curve_frame, [])
    metrics_errors: list[str] = []
    _validate_metrics_against_equity(metrics, equity_curve_frame, metrics_errors)
    if metrics_errors:
        raise ConsistencyError("; ".join(metrics_errors))

    if signal_diagnostics.get("total_rows") != len(results_df):
        raise ConsistencyError(
            "ConsistencyError: signal_diagnostics total_rows mismatch "
            f"(diagnostics={signal_diagnostics.get('total_rows')} vs results={len(results_df)})."
        )
    qa_errors: list[str] = []
    _validate_qa_summary(qa_summary, metrics, signal_diagnostics, equity_curve_frame, qa_errors)
    if qa_errors:
        raise ConsistencyError("; ".join(qa_errors))


def validate_walk_forward_consistency(
    split_results: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
    metrics_by_split: pd.DataFrame,
) -> None:
    """Ensure split-level walk-forward outputs match the aggregate summary."""

    split_ids = [str(split_result["split_id"]) for split_result in split_results]
    duplicate_split_ids = sorted({split_id for split_id in split_ids if split_ids.count(split_id) > 1})
    if duplicate_split_ids:
        raise ConsistencyError(
            "ConsistencyError: walk-forward split ids must be unique. "
            f"Duplicate split ids: {duplicate_split_ids}."
        )

    csv_split_ids = metrics_by_split.get("split_id", pd.Series(dtype="string")).astype("string").tolist()
    if split_ids != csv_split_ids:
        raise ConsistencyError(
            "ConsistencyError: metrics_by_split.csv split universe mismatch "
            f"(executed={split_ids} vs csv={csv_split_ids})."
        )

    concatenated = pd.concat(
        [pd.DataFrame(split_result["results_df"]) for split_result in split_results],
        ignore_index=True,
    ) if split_results else pd.DataFrame(columns=["strategy_return", "equity_curve"])

    for split_result in split_results:
        split_id = str(split_result["split_id"])
        row = metrics_by_split.loc[metrics_by_split["split_id"].astype("string") == split_id]
        if row.empty:
            raise ConsistencyError(f"ConsistencyError: metrics_by_split.csv is missing split_id '{split_id}'.")
        record = row.iloc[0]
        results_df = pd.DataFrame(split_result["results_df"])
        split_metadata = dict(split_result["split_metadata"])
        if split_metadata.get("split_id") != split_id:
            raise ConsistencyError(
                "ConsistencyError: walk-forward split metadata mismatch "
                f"(split_metadata={split_metadata.get('split_id')!r} vs split_id={split_id!r})."
            )
        if int(split_result["test_rows"]) != len(results_df):
            raise ConsistencyError(
                "ConsistencyError: walk-forward split row count mismatch "
                f"(split_id={split_id}, declared_test_rows={split_result['test_rows']} vs results={len(results_df)})."
            )
        for column in ("mode", "train_start", "train_end", "test_start", "test_end"):
            if column in record and str(record[column]) != str(split_metadata.get(column)):
                raise ConsistencyError(
                    "ConsistencyError: metrics_by_split metadata mismatch "
                    f"(split_id={split_id}, field={column}, csv={record[column]!r}, metadata={split_metadata.get(column)!r})."
                )

    if int(aggregate_summary.get("split_count", len(split_results))) != len(split_results):
        raise ConsistencyError(
            "ConsistencyError: walk-forward aggregate split_count mismatch "
            f"(summary={aggregate_summary.get('split_count')} vs executed={len(split_results)})."
        )

    if split_results:
        aggregate_metrics = compute_performance_metrics(concatenated)
        for key in _OPTIONAL_METRIC_KEYS:
            if key in aggregate_summary and key in aggregate_metrics:
                errors: list[str] = []
                _compare_numeric(
                    name=f"walk_forward.aggregate.{key}",
                    expected=aggregate_summary[key],
                    actual=aggregate_metrics[key],
                    errors=errors,
                )
                if errors:
                    raise ConsistencyError("; ".join(errors))


def validate_portfolio_sleeve_aggregation_consistency(
    returns_wide: pd.DataFrame,
    weights_wide: pd.DataFrame,
    portfolio_output: pd.DataFrame,
) -> None:
    """Ensure portfolio output rows and sleeve math match the aligned sleeve inputs."""

    if len(returns_wide) != len(portfolio_output):
        raise ConsistencyError(
            "ConsistencyError: portfolio output row count mismatch "
            f"(sleeve_returns={len(returns_wide)} vs portfolio_output={len(portfolio_output)})."
        )

    portfolio_ts = pd.to_datetime(portfolio_output["ts_utc"], utc=True, errors="coerce")
    if portfolio_ts.isna().any():
        raise ConsistencyError("ConsistencyError: portfolio output contains unparsable ts_utc values.")
    if not pd.DatetimeIndex(portfolio_ts, name="ts_utc").equals(returns_wide.index):
        raise ConsistencyError(
            "ConsistencyError: portfolio timeline mismatch. "
            "Portfolio output ts_utc must exactly match aligned sleeve return timestamps."
        )

    return_columns = sorted(column for column in portfolio_output.columns if column.startswith("strategy_return__"))
    weight_columns = sorted(column for column in portfolio_output.columns if column.startswith("weight__"))
    expected_return_columns = [f"strategy_return__{column}" for column in returns_wide.columns]
    expected_weight_columns = [f"weight__{column}" for column in weights_wide.columns]
    if return_columns != expected_return_columns:
        raise ConsistencyError(
            "ConsistencyError: portfolio sleeve universe mismatch in strategy_return__ columns "
            f"(expected={expected_return_columns} vs actual={return_columns})."
        )
    if weight_columns != expected_weight_columns:
        raise ConsistencyError(
            "ConsistencyError: portfolio sleeve universe mismatch in weight__ columns "
            f"(expected={expected_weight_columns} vs actual={weight_columns})."
        )

    expected_gross = (returns_wide * weights_wide).sum(axis=1).astype("float64").reset_index(drop=True)
    actual_gross = pd.to_numeric(portfolio_output["gross_portfolio_return"], errors="coerce").reset_index(drop=True)
    gross_mismatch = (expected_gross - actual_gross).abs() > 1e-10
    if gross_mismatch.any():
        bad_index = int(gross_mismatch[gross_mismatch].index[0])
        raise ConsistencyError(
            "ConsistencyError: portfolio weighted return mismatch "
            f"at row index {bad_index} (expected={expected_gross.iloc[bad_index]} vs actual={actual_gross.iloc[bad_index]})."
        )


def validate_portfolio_artifact_payload_consistency(
    *,
    portfolio_output: pd.DataFrame,
    weights_frame: pd.DataFrame,
    returns_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
    metrics: dict[str, Any],
    qa_summary: dict[str, Any],
    config: dict[str, Any],
    components: list[dict[str, Any]],
) -> None:
    """Ensure in-memory portfolio artifacts match the validated portfolio output."""

    from src.portfolio.metrics import compute_portfolio_metrics

    if len(weights_frame) != len(portfolio_output):
        raise ConsistencyError(
            "ConsistencyError: weights.csv row count mismatch "
            f"(portfolio_output={len(portfolio_output)} vs artifact={len(weights_frame)})."
        )
    if len(returns_frame) != len(portfolio_output):
        raise ConsistencyError(
            "ConsistencyError: portfolio_returns.csv row count mismatch "
            f"(portfolio_output={len(portfolio_output)} vs artifact={len(returns_frame)})."
        )
    if len(equity_frame) != len(portfolio_output):
        raise ConsistencyError(
            "ConsistencyError: portfolio_equity_curve.csv row count mismatch "
            f"(portfolio_output={len(portfolio_output)} vs artifact={len(equity_frame)})."
        )

    expected_metrics = compute_portfolio_metrics(
        portfolio_output,
        str(config.get("timeframe")),
        validation_config=config.get("validation"),
        risk_config=config.get("risk"),
    )
    errors: list[str] = []
    for key in sorted(set(expected_metrics) | set(metrics)):
        _compare_numeric(
            name=f"portfolio.metrics.{key}",
            expected=metrics.get(key),
            actual=expected_metrics.get(key),
            errors=errors,
        )
    if errors:
        raise ConsistencyError("; ".join(errors))

    if qa_summary.get("row_count") != len(portfolio_output):
        raise ConsistencyError(
            "ConsistencyError: portfolio qa_summary row_count mismatch "
            f"(qa_summary={qa_summary.get('row_count')} vs portfolio_output={len(portfolio_output)})."
        )
    if qa_summary.get("strategy_count") != len(components):
        raise ConsistencyError(
            "ConsistencyError: portfolio qa_summary strategy_count mismatch "
            f"(qa_summary={qa_summary.get('strategy_count')} vs components={len(components)})."
        )


def validate_portfolio_walk_forward_consistency(
    split_results: list[dict[str, Any]],
    metrics_by_split: pd.DataFrame,
    aggregate_metrics: dict[str, Any],
) -> None:
    """Ensure portfolio walk-forward split artifacts and aggregate summaries are consistent."""

    split_ids = [str(split_result["split_id"]) for split_result in split_results]
    duplicate_split_ids = sorted({split_id for split_id in split_ids if split_ids.count(split_id) > 1})
    if duplicate_split_ids:
        raise ConsistencyError(
            "ConsistencyError: portfolio walk-forward split ids must be unique. "
            f"Duplicate split ids: {duplicate_split_ids}."
        )

    csv_split_ids = metrics_by_split.get("split_id", pd.Series(dtype="string")).astype("string").tolist()
    if split_ids != csv_split_ids:
        raise ConsistencyError(
            "ConsistencyError: portfolio metrics_by_split split universe mismatch "
            f"(executed={split_ids} vs csv={csv_split_ids})."
        )

    aggregate_split_ids = [str(value) for value in aggregate_metrics.get("split_ids", [])]
    if split_ids != aggregate_split_ids:
        raise ConsistencyError(
            "ConsistencyError: portfolio aggregate split universe mismatch "
            f"(executed={split_ids} vs aggregate={aggregate_split_ids})."
        )
    if int(aggregate_metrics.get("split_count", len(split_results))) != len(split_results):
        raise ConsistencyError(
            "ConsistencyError: portfolio aggregate split_count mismatch "
            f"(aggregate={aggregate_metrics.get('split_count')} vs executed={len(split_results)})."
        )

    metric_summary = aggregate_metrics.get("metric_summary")
    if not isinstance(metric_summary, dict):
        raise ConsistencyError("ConsistencyError: aggregate portfolio metrics must include a metric_summary object.")
    for split_result in split_results:
        split_id = str(split_result["split_id"])
        row = metrics_by_split.loc[metrics_by_split["split_id"].astype("string") == split_id]
        if row.empty:
            raise ConsistencyError(f"ConsistencyError: portfolio metrics_by_split.csv is missing split_id '{split_id}'.")
        record = row.iloc[0]
        metadata = dict(split_result["split_metadata"])
        if int(metadata.get("row_count", split_result["row_count"])) != int(split_result["row_count"]):
            raise ConsistencyError(
                "ConsistencyError: portfolio split row_count mismatch "
                f"(split_id={split_id}, metadata={metadata.get('row_count')} vs actual={split_result['row_count']})."
            )
        for column in ("mode", "train_start", "train_end", "test_start", "test_end", "start", "end", "row_count"):
            if column in record and str(record[column]) != str(metadata.get(column)):
                raise ConsistencyError(
                    "ConsistencyError: portfolio metrics_by_split metadata mismatch "
                    f"(split_id={split_id}, field={column}, csv={record[column]!r}, metadata={metadata.get(column)!r})."
                )

    for metric_name, aggregate_value in metric_summary.items():
        if metric_name not in metrics_by_split.columns:
            continue
        series = pd.to_numeric(metrics_by_split[metric_name], errors="coerce").dropna()
        if series.empty:
            continue
        errors: list[str] = []
        _compare_numeric(
            name=f"portfolio.walk_forward.aggregate.{metric_name}",
            expected=aggregate_value,
            actual=float(series.mean()),
            errors=errors,
        )
        if errors:
            raise ConsistencyError("; ".join(errors))


def validate_run_consistency(
    run_dir: Path,
    *,
    strict: bool = True,
    validate_registry: bool = True,
) -> None:
    """Validate that one persisted run directory is internally consistent."""

    resolved_run_dir = Path(run_dir)
    manifest = _load_json(resolved_run_dir / "manifest.json")

    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, list):
        raise ConsistencyError("ConsistencyError: manifest.json must contain an artifact_files list.")
    manifest_files = {str(value) for value in artifact_files}
    actual_files = {
        path.relative_to(resolved_run_dir).as_posix()
        for path in resolved_run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }

    metrics = _load_json(resolved_run_dir / "metrics.json")
    equity_curve = pd.read_csv(resolved_run_dir / "equity_curve.csv")
    diagnostics = _load_json(resolved_run_dir / "signal_diagnostics.json")
    qa_summary = _load_json(resolved_run_dir / "qa_summary.json")

    errors: list[str] = []
    warnings_list: list[str] = []

    _validate_manifest(resolved_run_dir, manifest_files, actual_files, errors)
    _validate_metrics_against_equity(metrics, equity_curve, errors)
    _validate_qa_summary(qa_summary, metrics, diagnostics, equity_curve, errors)
    _validate_signal_diagnostics(diagnostics, warnings_list)
    if validate_registry:
        _validate_registry_entry(resolved_run_dir, manifest, metrics, errors)
    _validate_walk_forward(resolved_run_dir, manifest, metrics, errors, warnings_list)

    for message in warnings_list:
        warnings.warn(message, stacklevel=2)

    if errors and strict:
        raise ConsistencyError("; ".join(errors))

    if errors:
        for message in errors:
            warnings.warn(message, stacklevel=2)


def _validate_manifest(
    run_dir: Path,
    manifest_files: set[str],
    actual_files: set[str],
    errors: list[str],
) -> None:
    missing_from_disk = sorted(path for path in manifest_files if path not in actual_files)
    if missing_from_disk:
        errors.append(
            "ConsistencyError: manifest.json references missing files: "
            + ", ".join(missing_from_disk)
        )

    missing_from_manifest = sorted(path for path in actual_files if path not in manifest_files)
    if missing_from_manifest:
        errors.append(
            "ConsistencyError: manifest.json is missing artifact entries for: "
            + ", ".join(missing_from_manifest)
        )

    required = set(_REQUIRED_ARTIFACTS)
    if "signals.parquet" in manifest_files or (run_dir / "signals.parquet").exists():
        required.add("signals.parquet")

    missing_required = sorted(path for path in required if path not in actual_files)
    if missing_required:
        errors.append(
            "ConsistencyError: required artifacts missing from run directory: "
            + ", ".join(missing_required)
        )


def _validate_metrics_against_equity(
    metrics: dict[str, Any],
    equity_curve: pd.DataFrame,
    errors: list[str],
) -> None:
    equity = _numeric_series(equity_curve, "equity", errors, artifact_name="equity_curve.csv")
    if equity is None or equity.empty:
        return

    returns = _numeric_series(equity_curve, "strategy_return", [], artifact_name="equity_curve.csv")
    if returns is not None and not returns.empty:
        recomputed_metrics = compute_performance_metrics(equity_curve)
        recomputed_total_return = float(recomputed_metrics["total_return"])
        recomputed_max_drawdown = float(recomputed_metrics["max_drawdown"])
    else:
        first_equity = float(equity.iloc[0])
        last_equity = float(equity.iloc[-1])
        if abs(first_equity) <= _TOLERANCE:
            errors.append("ConsistencyError: equity_curve.csv starts at zero, cannot recompute total_return.")
            return

        recomputed_total_return = float((last_equity / first_equity) - 1.0)
        recomputed_max_drawdown = float((1.0 - (equity / equity.cummax())).max())

    for key in ("total_return", "cumulative_return"):
        if key in metrics:
            _compare_numeric(
                name=key,
                expected=metrics[key],
                actual=recomputed_total_return,
                errors=errors,
            )

    if "max_drawdown" in metrics:
        _compare_numeric(
            name="max_drawdown",
            expected=metrics["max_drawdown"],
            actual=recomputed_max_drawdown,
            errors=errors,
        )


def _validate_qa_summary(
    qa_summary: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    equity_curve: pd.DataFrame,
    errors: list[str],
) -> None:
    expected_row_count = int(len(equity_curve))
    if qa_summary.get("row_count") != expected_row_count:
        errors.append(
            "ConsistencyError: qa_summary.json row_count mismatch "
            f"(qa_summary={qa_summary.get('row_count')} vs equity_curve={expected_row_count})"
        )

    qa_signal = qa_summary.get("signal")
    if not isinstance(qa_signal, dict):
        errors.append("ConsistencyError: qa_summary.json must contain a signal object.")
        return

    if qa_signal.get("total_trades") != diagnostics.get("total_trades"):
        errors.append(
            "ConsistencyError: qa_summary.json total_trades mismatch "
            f"(qa_summary={qa_signal.get('total_trades')} vs signal_diagnostics={diagnostics.get('total_trades')})"
        )

    for key in ("pct_long", "pct_short", "pct_flat", "turnover"):
        if key in diagnostics and key in qa_signal:
            _compare_numeric(
                name=f"qa_summary.signal.{key}",
                expected=qa_signal[key],
                actual=diagnostics[key],
                errors=errors,
            )

    qa_metrics = qa_summary.get("metrics")
    if isinstance(qa_metrics, dict):
        metric_pairs = (
            ("total_return", "total_return"),
            ("sharpe", "sharpe_ratio"),
            ("max_drawdown", "max_drawdown"),
        )
        for qa_key, metric_key in metric_pairs:
            if qa_key in qa_metrics and metric_key in metrics:
                _compare_numeric(
                    name=f"qa_summary.metrics.{qa_key}",
                    expected=qa_metrics[qa_key],
                    actual=metrics[metric_key],
                    errors=errors,
                )

    expected_status = _expected_qa_status(qa_summary)
    actual_status = qa_summary.get("overall_status")
    if actual_status != expected_status:
        errors.append(
            "ConsistencyError: qa_summary.json overall_status mismatch "
            f"(qa_summary={actual_status!r} vs expected={expected_status!r})"
        )


def _validate_signal_diagnostics(
    diagnostics: dict[str, Any],
    warnings_list: list[str],
) -> None:
    pct_long = _coerce_float(diagnostics.get("pct_long"))
    pct_short = _coerce_float(diagnostics.get("pct_short"))
    pct_flat = _coerce_float(diagnostics.get("pct_flat"))
    exposure_pct = _coerce_float(diagnostics.get("exposure_pct"))
    turnover = _coerce_float(diagnostics.get("turnover"))
    total_rows = int(diagnostics.get("total_rows", 0) or 0)

    if total_rows > 0 and None not in (pct_long, pct_short, pct_flat):
        total = pct_long + pct_short + pct_flat
        if abs(total - 1.0) > _TOLERANCE:
            warnings_list.append(
                "ConsistencyWarning: signal_diagnostics.json percentages should sum to 1.0 "
                f"(pct_long + pct_short + pct_flat = {total})."
            )

    if total_rows > 0 and None not in (pct_long, pct_short, exposure_pct):
        expected_exposure = pct_long + pct_short
        if abs(exposure_pct - expected_exposure) > _TOLERANCE:
            warnings_list.append(
                "ConsistencyWarning: signal_diagnostics.json exposure_pct mismatch "
                f"(exposure_pct={exposure_pct} vs pct_long + pct_short={expected_exposure})."
            )

    if turnover is not None and not (0.0 - _TOLERANCE <= turnover <= 1.0 + _TOLERANCE):
        warnings_list.append(
            "ConsistencyWarning: signal_diagnostics.json turnover should be within [0, 1] "
            f"(turnover={turnover})."
        )


def _validate_registry_entry(
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    errors: list[str],
) -> None:
    registry_path = run_dir.parent / "registry.jsonl"
    if not registry_path.exists():
        errors.append(f"ConsistencyError: registry file not found at {registry_path.as_posix()}.")
        return

    run_id = manifest.get("run_id") or run_dir.name
    matching_entries = [entry for entry in load_registry(registry_path) if entry.get("run_id") == run_id]
    if not matching_entries:
        errors.append(f"ConsistencyError: registry.jsonl has no entry for run_id '{run_id}'.")
        return

    entry = matching_entries[-1]
    if entry.get("strategy_name") != manifest.get("strategy_name"):
        errors.append(
            "ConsistencyError: registry strategy_name mismatch "
            f"(registry={entry.get('strategy_name')!r} vs manifest={manifest.get('strategy_name')!r})"
        )

    artifact_path = entry.get("artifact_path")
    expected_paths = {run_dir.as_posix(), run_dir.resolve().as_posix()}
    if artifact_path not in expected_paths:
        errors.append(
            "ConsistencyError: registry artifact_path mismatch "
            f"(registry={artifact_path!r} vs run_dir={run_dir.as_posix()!r})"
        )

    metrics_summary = entry.get("metrics_summary")
    if not isinstance(metrics_summary, dict):
        errors.append("ConsistencyError: registry.jsonl entry must contain a metrics_summary object.")
        return

    for key in _REGISTRY_METRIC_KEYS:
        if key in metrics_summary and key in metrics:
            _compare_numeric(
                name=f"registry.metrics_summary.{key}",
                expected=metrics_summary[key],
                actual=metrics[key],
                errors=errors,
            )


def _validate_walk_forward(
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    errors: list[str],
    warnings_list: list[str],
) -> None:
    split_count = manifest.get("split_count")
    if manifest.get("evaluation_mode") != "walk_forward" and split_count is None and not (run_dir / "splits").exists():
        return

    splits_dir = run_dir / "splits"
    metrics_by_split_path = run_dir / "metrics_by_split.csv"
    if not splits_dir.exists():
        errors.append("ConsistencyError: walk-forward run is missing the splits directory.")
        return
    if not metrics_by_split_path.exists():
        errors.append("ConsistencyError: walk-forward run is missing metrics_by_split.csv.")
        return

    metrics_by_split = pd.read_csv(metrics_by_split_path)
    split_dirs = sorted(path for path in splits_dir.iterdir() if path.is_dir())
    expected_count = int(split_count) if split_count is not None else len(metrics_by_split)

    if len(split_dirs) != expected_count:
        errors.append(
            "ConsistencyError: split directory count mismatch "
            f"(manifest={expected_count} vs disk={len(split_dirs)})"
        )
    if len(metrics_by_split) != expected_count:
        errors.append(
            "ConsistencyError: metrics_by_split.csv row count mismatch "
            f"(manifest={expected_count} vs csv={len(metrics_by_split)})"
        )

    if metrics.get("split_count") is not None and int(metrics["split_count"]) != expected_count:
        errors.append(
            "ConsistencyError: metrics.json split_count mismatch "
            f"(metrics={metrics.get('split_count')} vs manifest={expected_count})"
        )

    split_id_order = [str(value) for value in metrics_by_split.get("split_id", pd.Series(dtype="string")).tolist()]
    actual_split_ids = [path.name for path in split_dirs]
    if split_id_order and actual_split_ids != split_id_order:
        errors.append(
            "ConsistencyError: split directory order/name mismatch "
            f"(metrics_by_split={split_id_order} vs disk={actual_split_ids})"
        )

    aggregate_frames: list[pd.DataFrame] = []
    for split_dir in split_dirs:
        split_metrics = _load_json(split_dir / "metrics.json")
        split_manifest = _load_json(split_dir / "split.json")
        if split_manifest.get("split_id") != split_dir.name:
            errors.append(
                "ConsistencyError: split.json split_id mismatch "
                f"(split.json={split_manifest.get('split_id')!r} vs directory={split_dir.name!r})"
            )

        split_equity = pd.read_csv(split_dir / "equity_curve.csv")
        aggregate_frames.append(split_equity)

        split_row = metrics_by_split.loc[metrics_by_split["split_id"].astype("string") == split_dir.name]
        if split_row.empty:
            errors.append(f"ConsistencyError: metrics_by_split.csv is missing split_id '{split_dir.name}'.")
            continue

        row = split_row.iloc[0].to_dict()
        for key in _OPTIONAL_METRIC_KEYS:
            if key in split_metrics and key in row:
                _compare_numeric(
                    name=f"metrics_by_split.{split_dir.name}.{key}",
                    expected=row[key],
                    actual=split_metrics[key],
                    errors=errors,
                )

    if not aggregate_frames:
        return

    aggregate_equity = pd.concat(aggregate_frames, ignore_index=True)
    aggregate_metrics = compute_performance_metrics(aggregate_equity)
    for key in _OPTIONAL_METRIC_KEYS:
        if key in metrics and key in aggregate_metrics:
            _compare_numeric(
                name=f"walk_forward.aggregate.{key}",
                expected=metrics[key],
                actual=aggregate_metrics[key],
                errors=errors,
            )

    if split_count == 0 and not metrics_by_split.empty:
        warnings_list.append("ConsistencyWarning: walk-forward run declares zero splits but metrics_by_split.csv is not empty.")


def _validate_unique_index(df: pd.DataFrame, *, owner: str) -> None:
    if not df.index.is_unique:
        duplicate_label = df.index[df.index.duplicated()][0]
        raise ConsistencyError(
            f"ConsistencyError: {owner} contains duplicate index labels. First duplicate label: {duplicate_label!r}."
        )


def _validate_frame_key_identity(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> None:
    left_keys = _row_identity_frame(left)
    right_keys = _row_identity_frame(right)
    if left_keys is None or right_keys is None:
        return
    shared_columns = [column for column in left_keys.columns if column in right_keys.columns]
    if not shared_columns:
        return
    left_keys = left_keys.loc[:, shared_columns]
    right_keys = right_keys.loc[:, shared_columns]

    if len(left_keys) != len(right_keys):
        raise ConsistencyError(
            "ConsistencyError: cross-layer key count mismatch "
            f"({left_name}={len(left_keys)} vs {right_name}={len(right_keys)})."
        )
    if not left_keys.equals(right_keys):
        mismatch_index = next(
            position
            for position, (left_row, right_row) in enumerate(
                zip(left_keys.itertuples(index=False), right_keys.itertuples(index=False), strict=False)
            )
            if left_row != right_row
        )
        raise ConsistencyError(
            "ConsistencyError: cross-layer key mismatch "
            f"between {left_name} and {right_name} at row {mismatch_index} "
            f"(expected={left_keys.iloc[mismatch_index].to_dict()} vs actual={right_keys.iloc[mismatch_index].to_dict()})."
        )


def _validate_duplicate_keys_for_frame(
    df: pd.DataFrame,
    *,
    key_columns: list[str],
    owner: str,
) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if not duplicate_mask.any():
        return
    duplicate_row = df.loc[duplicate_mask, key_columns].iloc[0].to_dict()
    raise ConsistencyError(
        f"ConsistencyError: {owner} contains duplicate key rows for columns {key_columns}. "
        f"First duplicate key: {duplicate_row}."
    )


def _row_identity_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    if "ts_utc" in df.columns or "date" in df.columns:
        frame = pd.DataFrame(index=df.index)
        if "symbol" in df.columns:
            frame["symbol"] = df["symbol"].astype("string")
        if "timeframe" in df.columns:
            frame["timeframe"] = df["timeframe"].astype("string")
        if "ts_utc" in df.columns:
            timestamps = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        else:
            timestamps = pd.to_datetime(df["date"], utc=True, errors="coerce")
        frame["row_ts_utc"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        key_columns = frame.columns.tolist()
        _validate_duplicate_keys_for_frame(frame, key_columns=key_columns, owner="cross-layer frame")
        return frame.reset_index(drop=True)

    split_columns = [
        column
        for column in ("split_id", "mode", "train_start", "train_end", "test_start", "test_end")
        if column in df.columns
    ]
    if split_columns:
        frame = df.loc[:, split_columns].copy()
        for column in split_columns:
            frame[column] = frame[column].astype("string")
        _validate_duplicate_keys_for_frame(frame, key_columns=split_columns, owner="cross-layer frame")
        return frame.reset_index(drop=True)
    return None


def _expected_qa_status(qa_summary: dict[str, Any]) -> str:
    execution = qa_summary.get("execution") or {}
    flags = qa_summary.get("flags") or {}
    row_count = qa_summary.get("row_count")
    low_data = flags.get("low_data")
    if low_data is None and isinstance(row_count, int):
        low_data = row_count < LOW_DATA_THRESHOLD

    failure = (
        bool(flags.get("no_data"))
        or
        not bool(execution.get("valid_returns"))
        or not bool(execution.get("equity_curve_present"))
        or bool(qa_summary.get("integrity_failure", False))
    )
    if failure:
        return "fail"
    if any(
        bool(flags.get(key))
        for key in ("degenerate_signal", "no_trades", "high_turnover")
    ) or bool(low_data):
        return "warn"
    return "pass"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConsistencyError(f"ConsistencyError: required artifact not found: {path.name}")
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ConsistencyError(f"Expected JSON object in {path.as_posix()}.")
    return payload


def _numeric_series(
    df: pd.DataFrame,
    column: str,
    errors: list[str],
    *,
    artifact_name: str,
) -> pd.Series | None:
    if column not in df.columns:
        errors.append(f"ConsistencyError: {artifact_name} is missing required column '{column}'.")
        return None

    series = pd.to_numeric(df[column], errors="coerce")
    non_numeric = series.isna() & df[column].notna()
    if non_numeric.any():
        errors.append(f"ConsistencyError: {artifact_name} contains non-numeric values in '{column}'.")
        return None
    return series.astype("float64")


def _compare_numeric(
    *,
    name: str,
    expected: Any,
    actual: Any,
    errors: list[str],
) -> None:
    left = _coerce_float(expected)
    right = _coerce_float(actual)
    if left is None and right is None:
        if _is_empty_like(expected) and _is_empty_like(actual):
            return
        if expected == actual:
            return
        errors.append(f"ConsistencyError: {name} mismatch (expected={expected!r} vs actual={actual!r})")
        return
    if left is None or right is None:
        if expected == actual:
            return
        errors.append(f"ConsistencyError: {name} mismatch (expected={expected!r} vs actual={actual!r})")
        return
    if abs(left - right) > _TOLERANCE:
        errors.append(f"ConsistencyError: {name} mismatch (expected={left} vs actual={right})")


def _coerce_float(value: Any) -> float | None:
    if value is None or _is_empty_like(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_empty_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False
