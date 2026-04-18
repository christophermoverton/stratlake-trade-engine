from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.execution import ExecutionConfig
from src.research.position_constructors import (
    normalize_position_constructor_config,
    position_constructor_metadata_payload,
)
from src.research.backtest_runner import run_backtest
from src.research.metrics import compute_performance_metrics
from src.research.signal_semantics import attach_signal_metadata, extract_signal_metadata

_SLEEVE_RETURNS_FILENAME = "sleeve_returns.csv"
_SLEEVE_EQUITY_CURVE_FILENAME = "sleeve_equity_curve.csv"
_SLEEVE_METRICS_FILENAME = "sleeve_metrics.json"
_MANIFEST_FILENAME = "manifest.json"


class AlphaSleeveError(ValueError):
    """Raised when one deterministic alpha sleeve cannot be generated safely."""


@dataclass(frozen=True)
class AlphaSleeveResult:
    """Structured deterministic sleeve outputs derived from mapped alpha signals."""

    signals: pd.DataFrame
    backtest_results: pd.DataFrame
    sleeve_returns: pd.DataFrame
    sleeve_equity_curve: pd.DataFrame
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def generate_alpha_sleeve(
    *,
    signals: pd.DataFrame,
    dataset: pd.DataFrame,
    execution_config: ExecutionConfig | None = None,
    price_column: str | None = None,
    realized_return_column: str | None = None,
    position_constructor: dict[str, Any] | None = None,
    alpha_name: str | None = None,
    run_id: str | None = None,
) -> AlphaSleeveResult:
    """Convert mapped alpha exposures into one backtestable sleeve return stream."""

    resolved_signals = _validate_signals(signals, position_constructor=position_constructor)
    prepared_dataset, return_column_name, return_source = _prepare_dataset_returns(
        dataset,
        price_column=price_column,
        realized_return_column=realized_return_column,
    )
    backtest_input = _build_backtest_input(resolved_signals, prepared_dataset, return_column_name=return_column_name)
    backtest_results = run_backtest(backtest_input, execution_config=execution_config)
    sleeve_returns = _aggregate_sleeve_returns(backtest_results)
    sleeve_equity_curve = _build_sleeve_equity_curve(sleeve_returns)
    metrics = _build_sleeve_metrics(
        backtest_results,
        sleeve_returns=sleeve_returns,
        return_source=return_source,
        alpha_name=alpha_name,
        run_id=run_id,
    )
    backtest_signal_semantics = backtest_results.attrs.get("backtest_signal_semantics", {})
    return AlphaSleeveResult(
        signals=resolved_signals.reset_index(drop=True),
        backtest_results=backtest_results.reset_index(drop=True),
        sleeve_returns=sleeve_returns,
        sleeve_equity_curve=sleeve_equity_curve,
        metrics=metrics,
        metadata={
            "alpha_name": alpha_name,
            "run_id": run_id,
            "return_column": return_column_name,
            "return_source": return_source,
            "signal_row_count": int(len(resolved_signals)),
            "symbol_count": int(resolved_signals["symbol"].astype("string").nunique()),
            "timestamp_count": int(pd.to_datetime(resolved_signals["ts_utc"], utc=True, errors="coerce").nunique()),
            "constructor_id": backtest_signal_semantics.get("constructor_id"),
            "constructor_params": dict(backtest_signal_semantics.get("constructor_params", {})),
        },
    )


def write_alpha_sleeve_artifacts(
    output_dir: str | Path,
    sleeve_result: AlphaSleeveResult,
    *,
    update_manifest: bool = True,
) -> dict[str, Any]:
    """Persist deterministic alpha sleeve artifacts and optionally augment the manifest."""

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(resolved_output_dir / _SLEEVE_RETURNS_FILENAME, sleeve_result.sleeve_returns)
    _write_csv(resolved_output_dir / _SLEEVE_EQUITY_CURVE_FILENAME, sleeve_result.sleeve_equity_curve)
    _write_json(resolved_output_dir / _SLEEVE_METRICS_FILENAME, _normalize_mapping(sleeve_result.metrics))

    manifest_payload: dict[str, Any] = {}
    if update_manifest:
        manifest_payload = augment_alpha_manifest_with_sleeve(
            resolved_output_dir,
            sleeve_result=sleeve_result,
        )
    return manifest_payload


def augment_alpha_manifest_with_sleeve(
    output_dir: str | Path,
    *,
    sleeve_result: AlphaSleeveResult,
) -> dict[str, Any]:
    """Update one alpha manifest in place with sleeve artifact references."""

    resolved_output_dir = Path(output_dir)
    manifest_path = resolved_output_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Alpha manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = payload.get("artifact_files", [])
    if not isinstance(artifact_files, list):
        artifact_files = []
    artifact_files = sorted(
        set(
            [
                *[str(item) for item in artifact_files],
                _SLEEVE_RETURNS_FILENAME,
                _SLEEVE_EQUITY_CURVE_FILENAME,
                _SLEEVE_METRICS_FILENAME,
            ]
        )
    )
    payload["artifact_files"] = artifact_files

    artifact_paths = payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        artifact_paths = {}
    artifact_paths.update(
        {
            "sleeve_returns": _SLEEVE_RETURNS_FILENAME,
            "sleeve_equity_curve": _SLEEVE_EQUITY_CURVE_FILENAME,
            "sleeve_metrics": _SLEEVE_METRICS_FILENAME,
        }
    )
    payload["artifact_paths"] = _normalize_mapping(artifact_paths)
    payload["sleeve_returns_path"] = _SLEEVE_RETURNS_FILENAME
    payload["sleeve_equity_curve_path"] = _SLEEVE_EQUITY_CURVE_FILENAME
    payload["sleeve_metrics_path"] = _SLEEVE_METRICS_FILENAME
    payload["constructor_id"] = sleeve_result.metadata.get("constructor_id")
    payload["constructor_params"] = _normalize_json_value(
        sleeve_result.metadata.get("constructor_params", {})
    )
    payload["sleeve"] = _normalize_mapping(
        {
            "enabled": True,
            "artifact_paths": {
                "sleeve_returns": _SLEEVE_RETURNS_FILENAME,
                "sleeve_equity_curve": _SLEEVE_EQUITY_CURVE_FILENAME,
                "sleeve_metrics": _SLEEVE_METRICS_FILENAME,
            },
            "constructor_id": sleeve_result.metadata.get("constructor_id"),
            "constructor_params": sleeve_result.metadata.get("constructor_params", {}),
            "metric_summary": sleeve_result.metrics,
            "row_count": int(len(sleeve_result.sleeve_returns)),
            "timestamp_count": int(len(sleeve_result.sleeve_equity_curve)),
        }
    )

    artifact_groups = payload.get("artifact_groups")
    if not isinstance(artifact_groups, dict):
        artifact_groups = {}
    alpha_group = artifact_groups.get("alpha_evaluation", [])
    if not isinstance(alpha_group, list):
        alpha_group = []
    artifact_groups["alpha_evaluation"] = sorted(
        set(
            [
                *[str(item) for item in alpha_group],
                _SLEEVE_RETURNS_FILENAME,
                _SLEEVE_EQUITY_CURVE_FILENAME,
                _SLEEVE_METRICS_FILENAME,
            ]
        )
    )
    artifact_groups["sleeve"] = [
        _SLEEVE_EQUITY_CURVE_FILENAME,
        _SLEEVE_METRICS_FILENAME,
        _SLEEVE_RETURNS_FILENAME,
    ]
    payload["artifact_groups"] = {
        key: value if not isinstance(value, list) else sorted(value)
        for key, value in sorted(artifact_groups.items())
    }
    payload["files_written"] = len(artifact_files)

    normalized = _normalize_mapping(payload)
    manifest_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    return normalized


def _validate_signals(
    signals: pd.DataFrame,
    *,
    position_constructor: dict[str, Any] | None,
) -> pd.DataFrame:
    metadata = extract_signal_metadata(signals)
    required_columns = ("symbol", "ts_utc", "signal")
    missing = [column for column in required_columns if column not in signals.columns]
    if missing:
        formatted = ", ".join(sorted(missing))
        raise AlphaSleeveError(f"Sleeve generation requires mapped signals with columns: {formatted}.")

    normalized = signals.copy(deep=True)
    normalized.attrs = {}
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="raise")
    normalized["signal"] = pd.to_numeric(normalized["signal"], errors="raise").astype("float64")
    if normalized["signal"].isna().any():
        raise AlphaSleeveError("Sleeve generation signal column must not contain NaN values.")
    sort_columns = [column for column in ("symbol", "ts_utc", "timeframe") if column in normalized.columns]
    normalized = normalized.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    if normalized.duplicated(subset=[column for column in ("symbol", "ts_utc", "timeframe") if column in normalized.columns]).any():
        raise AlphaSleeveError("Sleeve generation requires duplicate-free mapped signal keys.")
    constructor_config = normalize_position_constructor_config(position_constructor)
    if constructor_config is not None:
        metadata = dict(metadata or {})
        metadata.update(
            position_constructor_metadata_payload(
                name=str(constructor_config["name"]),
                params=dict(constructor_config["params"]),
            )
        )
    if metadata is not None:
        attach_signal_metadata(normalized, metadata)
    return normalized


def _prepare_dataset_returns(
    dataset: pd.DataFrame,
    *,
    price_column: str | None,
    realized_return_column: str | None,
) -> tuple[pd.DataFrame, str, str]:
    normalized = dataset.copy(deep=True)
    normalized.attrs = {}
    if "symbol" not in normalized.columns or "ts_utc" not in normalized.columns:
        raise AlphaSleeveError("Sleeve generation dataset must include 'symbol' and 'ts_utc' columns.")

    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="raise")
    sort_columns = [column for column in ("symbol", "ts_utc", "timeframe") if column in normalized.columns]
    normalized = normalized.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    return_column_name = "asset_return"
    if realized_return_column is not None:
        if realized_return_column not in normalized.columns:
            raise AlphaSleeveError(
                f"Sleeve generation realized return column {realized_return_column!r} was not found."
            )
        normalized[return_column_name] = pd.to_numeric(
            normalized[realized_return_column],
            errors="coerce",
        ).astype("float64")
        return normalized, return_column_name, f"realized_return_column:{realized_return_column}"

    if price_column is None:
        raise AlphaSleeveError("Sleeve generation requires either price_column or realized_return_column.")
    if price_column not in normalized.columns:
        raise AlphaSleeveError(f"Sleeve generation price column {price_column!r} was not found.")

    prices = pd.to_numeric(normalized[price_column], errors="coerce").astype("float64")
    normalized[return_column_name] = (
        prices.groupby(normalized["symbol"], sort=False, dropna=False).pct_change().astype("float64")
    )
    return normalized, return_column_name, f"price_column:{price_column}"


def _build_backtest_input(
    signals: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    return_column_name: str,
) -> pd.DataFrame:
    merge_keys = [column for column in ("symbol", "ts_utc", "timeframe") if column in signals.columns and column in dataset.columns]
    if merge_keys != ["symbol", "ts_utc", "timeframe"] and merge_keys != ["symbol", "ts_utc"]:
        raise AlphaSleeveError("Sleeve generation could not resolve canonical merge keys.")

    selected_columns = [*merge_keys, return_column_name]
    merged = signals.merge(
        dataset.loc[:, selected_columns],
        on=merge_keys,
        how="inner",
        sort=False,
    )
    if merged.empty:
        raise AlphaSleeveError("Sleeve generation produced no rows after joining signals to returns.")
    merged = merged.sort_values(merge_keys, kind="stable").reset_index(drop=True)
    if merged[return_column_name].isna().all():
        raise AlphaSleeveError("Sleeve generation produced no usable realized returns.")
    metadata = extract_signal_metadata(signals)
    if metadata is not None:
        attach_signal_metadata(merged, metadata)
    return merged


def _aggregate_sleeve_returns(backtest_results: pd.DataFrame) -> pd.DataFrame:
    time_column = "ts_utc" if "ts_utc" in backtest_results.columns else "date"
    if time_column not in backtest_results.columns:
        raise AlphaSleeveError("Sleeve backtest results must include 'ts_utc' or 'date'.")

    frame = backtest_results.copy(deep=True)
    frame.attrs = {}
    if time_column == "ts_utc":
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="raise")
    else:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    numeric_columns = [
        column
        for column in (
            "gross_strategy_return",
            "transaction_cost",
            "slippage_cost",
            "execution_friction",
            "net_strategy_return",
            "strategy_return",
        )
        if column in frame.columns
    ]
    grouped = (
        frame.groupby(time_column, sort=False, as_index=False)
        .agg(
            **{column: (column, "sum") for column in numeric_columns},
            active_symbol_count=("symbol", "nunique"),
        )
    )
    if "timeframe" in frame.columns:
        grouped["timeframe"] = frame.groupby(time_column, sort=False)["timeframe"].first().reset_index(drop=True)

    renamed = grouped.rename(
        columns={
            "gross_strategy_return": "gross_sleeve_return",
            "transaction_cost": "sleeve_transaction_cost",
            "slippage_cost": "sleeve_slippage_cost",
            "execution_friction": "sleeve_execution_friction",
            "net_strategy_return": "net_sleeve_return",
            "strategy_return": "sleeve_return",
        }
    )
    ordered_columns = [
        column
        for column in (
            time_column,
            "timeframe",
            "active_symbol_count",
            "gross_sleeve_return",
            "sleeve_transaction_cost",
            "sleeve_slippage_cost",
            "sleeve_execution_friction",
            "net_sleeve_return",
            "sleeve_return",
        )
        if column in renamed.columns
    ]
    return _csv_ready_frame(renamed.loc[:, ordered_columns].copy(), time_column=time_column)


def _build_sleeve_equity_curve(sleeve_returns: pd.DataFrame) -> pd.DataFrame:
    time_column = "ts_utc" if "ts_utc" in sleeve_returns.columns else "date"
    frame = sleeve_returns.loc[:, [column for column in (time_column, "timeframe", "sleeve_return") if column in sleeve_returns.columns]].copy()
    frame["sleeve_equity_curve"] = (
        1.0 + pd.to_numeric(sleeve_returns["sleeve_return"], errors="coerce").fillna(0.0).astype("float64")
    ).cumprod()
    return _csv_ready_frame(
        frame.loc[:, [column for column in (time_column, "timeframe", "sleeve_equity_curve") if column in frame.columns]].copy(),
        time_column=time_column,
    )


def _build_sleeve_metrics(
    backtest_results: pd.DataFrame,
    *,
    sleeve_returns: pd.DataFrame,
    return_source: str,
    alpha_name: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    metrics = dict(compute_performance_metrics(backtest_results))
    metrics.update(
        {
            "alpha_name": alpha_name,
            "run_id": run_id,
            "return_source": return_source,
            "sleeve_row_count": int(len(sleeve_returns)),
            "timestamp_count": int(len(sleeve_returns)),
            "symbol_count": int(backtest_results["symbol"].astype("string").nunique()) if "symbol" in backtest_results.columns else 0,
            "active_symbol_count_mean": float(pd.to_numeric(sleeve_returns.get("active_symbol_count"), errors="coerce").mean())
            if "active_symbol_count" in sleeve_returns.columns
            else None,
            "constructor_id": backtest_results.attrs.get("backtest_signal_semantics", {}).get("constructor_id"),
            "constructor_params": backtest_results.attrs.get("backtest_signal_semantics", {}).get("constructor_params", {}),
        }
    )
    return _normalize_mapping(metrics)


def _csv_ready_frame(df: pd.DataFrame, *, time_column: str) -> pd.DataFrame:
    frame = df.copy()
    if time_column == "ts_utc" and "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        frame = frame.sort_values("ts_utc", kind="stable").reset_index(drop=True)
        return frame
    if time_column == "date" and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.strftime("%Y-%m-%d")
        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
        return frame
    return frame.reset_index(drop=True)


def _normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_json_value(mapping[key]) for key in sorted(mapping)}


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
    if isinstance(value, pd.Series):
        return [_normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "AlphaSleeveError",
    "AlphaSleeveResult",
    "augment_alpha_manifest_with_sleeve",
    "generate_alpha_sleeve",
    "write_alpha_sleeve_artifacts",
]
