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
    "exposure_pct",
)


class ConsistencyError(ValueError):
    """Raised when run artifacts disagree with one another."""


def validate_run_consistency(
    run_dir: Path,
    *,
    strict: bool = True,
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
        return
    if left is None or right is None:
        errors.append(f"ConsistencyError: {name} mismatch (expected={expected!r} vs actual={actual!r})")
        return
    if abs(left - right) > _TOLERANCE:
        errors.append(f"ConsistencyError: {name} mismatch (expected={left} vs actual={right})")


def _coerce_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
