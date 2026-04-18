from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from src.research.consistency import (
    validate_run_consistency,
    validate_strategy_artifact_payload_consistency,
    validate_walk_forward_consistency,
)
from src.research.metrics import (
    broadcast_strategy_equity_curve,
    compute_performance_metrics,
    infer_position_series,
)
from src.research.promotion import (
    DEFAULT_PROMOTION_ARTIFACT_FILENAME,
    evaluate_promotion_gates,
    write_promotion_gate_artifact,
)
from src.research.registry import (
    build_review_metadata,
    default_registry_path,
    serialize_canonical_json,
    upsert_registry_entry,
)
from src.research.signal_semantics import extract_signal_metadata
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.strategy_qa import generate_strategy_qa_summary

ARTIFACTS_ROOT = Path("artifacts") / "strategies"
ROBUSTNESS_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "robustness"
_BACKTEST_COLUMNS = ("strategy_return", "equity_curve")
_METRIC_KEYS = (
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
    "benchmark_total_return",
    "excess_return",
    "benchmark_correlation",
    "relative_drawdown",
    "sanity_issue_count",
    "sanity_warning_count",
    "flagged_split_count",
)
_METRICS_SUMMARY_KEYS = _METRIC_KEYS
_SIGNAL_PRIORITY_COLUMNS = (
    "ts_utc",
    "date",
    "symbol",
    "signal",
    "executed_signal",
    "position",
    "delta_position",
    "abs_delta_position",
    "turnover",
    "trade_event",
    "gross_strategy_return",
    "transaction_cost",
    "slippage_cost",
    "execution_friction",
    "net_strategy_return",
)
_SIGNAL_SEMANTICS_FILENAME = "signal_semantics.json"
_SPLIT_METADATA_COLUMNS = (
    "split_id",
    "mode",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "split_rows",
    "train_rows",
    "test_rows",
)


def _sanitize_strategy_name(strategy_name: str) -> str:
    """Return a filesystem-friendly strategy name component for experiment paths."""

    cleaned = "".join(char if char.isalnum() else "_" for char in strategy_name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "strategy"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a dictionary to disk using stable JSON formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_timestamp_from_run_id(run_id: str) -> str:
    """Return a stable pseudo-timestamp derived from a deterministic run identifier."""

    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    year = 2000 + int(digest[0:2], 16) % 25
    month = (int(digest[2:4], 16) % 12) + 1
    day = (int(digest[4:6], 16) % 28) + 1
    hour = int(digest[6:8], 16) % 24
    minute = int(digest[8:10], 16) % 60
    second = int(digest[10:12], 16) % 60
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"


def _first_non_empty_string(series: pd.Series) -> str | None:
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return None


def _normalize_timeframe(timeframe: str | None) -> str | None:
    if timeframe is None:
        return None

    normalized = timeframe.strip().lower()
    if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
        return "1Min"
    if normalized in {"1d", "1day", "day", "daily"}:
        return "1D"
    return timeframe


def _infer_timeframe(results_df: pd.DataFrame, config: dict[str, Any]) -> str | None:
    if "timeframe" in results_df.columns:
        timeframe_value = _first_non_empty_string(results_df["timeframe"])
        if timeframe_value is not None:
            return _normalize_timeframe(timeframe_value)

    evaluation = config.get("evaluation")
    if isinstance(evaluation, dict):
        timeframe_value = evaluation.get("timeframe")
        if timeframe_value is not None:
            return _normalize_timeframe(str(timeframe_value))

    dataset = config.get("dataset")
    if isinstance(dataset, str):
        normalized = dataset.strip().lower()
        if normalized.endswith("1m"):
            return "1Min"
        if normalized.endswith("daily"):
            return "1D"

    return None


def _resolve_data_range(
    results_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    evaluation_mode: str,
) -> dict[str, str | None]:
    if evaluation_mode == "walk_forward":
        evaluation = config.get("evaluation")
        if isinstance(evaluation, dict):
            start = evaluation.get("start") or evaluation.get("train_start")
            end = evaluation.get("end") or evaluation.get("test_end")
            return {
                "start": None if start is None else str(start),
                "end": None if end is None else str(end),
            }

    if "date" in results_df.columns:
        date_series = results_df["date"].astype("string")
        if not date_series.dropna().empty:
            return {
                "start": str(date_series.min()),
                "end": str(date_series.max()),
            }

    if "ts_utc" in results_df.columns:
        timestamps = pd.to_datetime(results_df["ts_utc"], utc=True, errors="coerce").dropna()
        if not timestamps.empty:
            return {
                "start": timestamps.min().isoformat().replace("+00:00", "Z"),
                "end": timestamps.max().isoformat().replace("+00:00", "Z"),
            }

    start = config.get("start")
    end = config.get("end")
    return {
        "start": None if start is None else str(start),
        "end": None if end is None else str(end),
    }


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in _METRICS_SUMMARY_KEYS if key in metrics}


def _drop_none_values(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {key: value for key, value in payload.items() if value is not None}


def _constructor_payload(signal_semantics: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
    if signal_semantics is None:
        return None, {}
    constructor_id = signal_semantics.get("constructor_id")
    constructor_params = signal_semantics.get("constructor_params", {})
    if not isinstance(constructor_params, dict):
        constructor_params = {}
    return (
        None if constructor_id is None else str(constructor_id),
        dict(constructor_params),
    )


def _build_registry_entry(
    experiment_dir: Path,
    strategy_name: str,
    results_df: pd.DataFrame,
    metrics_summary: dict[str, Any],
    config: dict[str, Any],
    *,
    evaluation_mode: str,
    evaluation_config: dict[str, Any] | None,
    split_count: int | None,
    promotion_evaluation: Any | None = None,
) -> dict[str, Any]:
    run_id = experiment_dir.name
    review_metadata = build_review_metadata(
        review=config.get("review") if isinstance(config.get("review"), dict) else None,
        promotion_status=None if promotion_evaluation is None else promotion_evaluation.promotion_status,
        promotion_gate_summary=None if promotion_evaluation is None else promotion_evaluation.summary(),
    )
    signal_semantics = _signal_semantics_payload(results_df)
    constructor_id, constructor_params = _constructor_payload(signal_semantics)
    return {
        "run_id": run_id,
        "timestamp": _utc_timestamp_from_run_id(run_id),
        "strategy_name": strategy_name,
        "dataset": config.get("dataset"),
        "strategy_params": dict(config.get("parameters") or {}),
        "evaluation_mode": evaluation_mode,
        "evaluation_config": _drop_none_values(evaluation_config),
        "execution": _drop_none_values(config.get("execution")),
        "data_range": _resolve_data_range(results_df, config, evaluation_mode=evaluation_mode),
        "timeframe": _infer_timeframe(results_df, config),
        "metrics_summary": _metrics_summary(metrics_summary),
        "signal_type": None if signal_semantics is None else signal_semantics.get("signal_type"),
        "signal_version": None if signal_semantics is None else signal_semantics.get("version"),
        "constructor_id": constructor_id,
        "constructor_params": constructor_params,
        "signal_semantics": signal_semantics,
        "promotion_status": None if promotion_evaluation is None else promotion_evaluation.promotion_status,
        "review_status": review_metadata["status"],
        "review_metadata": review_metadata,
        "promotion_gate_summary": (
            None if promotion_evaluation is None else promotion_evaluation.summary()
        ),
        "artifact_path": experiment_dir.as_posix(),
        "split_count": split_count,
        "evaluation_config_path": config.get("evaluation_config_path"),
        "strict_mode": config.get("strict_mode"),
    }


def _write_registry_entry(entry: dict[str, Any]) -> None:
    """Append one completed run to the shared strategy registry."""

    upsert_registry_entry(default_registry_path(ARTIFACTS_ROOT), entry)


def _normalize_run_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return [
            _normalize_run_payload(record)
            for record in _normalize_artifact_frame(value).to_dict(orient="records")
        ]
    if isinstance(value, str | bytes | bool | int | float):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if isinstance(value, pd.Series):
        return [_normalize_run_payload(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_normalize_run_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_run_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_run_payload(value[key]) for key in sorted(value)}
    if pd.isna(value):
        return None
    return value


def _build_deterministic_run_id(
    strategy_name: str,
    *,
    evaluation_mode: str,
    config: dict[str, Any],
    results_df: pd.DataFrame | None = None,
) -> str:
    payload = {
        "strategy_name": strategy_name,
        "evaluation_mode": evaluation_mode,
        "config": _normalize_run_payload(config),
        "results": None if results_df is None else _normalize_run_payload(results_df),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_strategy_name(strategy_name)}_{evaluation_mode}_{digest}"


def create_experiment_dir(
    strategy_name: str,
    *,
    evaluation_mode: str,
    config: dict[str, Any],
    results_df: pd.DataFrame | None = None,
) -> Path:
    """Create and return the deterministic experiment directory for a strategy run."""

    experiment_dir = resolve_experiment_dir(
        strategy_name,
        evaluation_mode=evaluation_mode,
        config=config,
        results_df=results_df,
    )
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def resolve_experiment_dir(
    strategy_name: str,
    *,
    evaluation_mode: str,
    config: dict[str, Any],
    results_df: pd.DataFrame | None = None,
) -> Path:
    """Return the deterministic experiment directory path without creating it."""

    return ARTIFACTS_ROOT / _build_deterministic_run_id(
        strategy_name,
        evaluation_mode=evaluation_mode,
        config=config,
        results_df=results_df,
    )


def _build_robustness_run_id(
    strategy_name: str,
    *,
    config: dict[str, Any],
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    payload = {
        "strategy_name": strategy_name,
        "evaluation_mode": "robustness",
        "config": _normalize_run_payload(config),
        "variant_metrics": _normalize_run_payload(variant_metrics),
        "stability_metrics": _normalize_run_payload(stability_metrics),
        "summary": _normalize_run_payload(summary),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize_strategy_name(strategy_name)}_robustness_{digest}"


def resolve_robustness_experiment_dir(
    strategy_name: str,
    *,
    config: dict[str, Any],
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    """Return the deterministic experiment directory path for a robustness run."""

    run_id = _build_robustness_run_id(
        strategy_name,
        config=config,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        summary=summary,
    )
    return ROBUSTNESS_ARTIFACTS_ROOT / run_id


def _normalize_artifact_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return frame.reset_index(drop=True)

    sort_keys: dict[str, pd.Series] = {}
    if "ts_utc" in frame.columns:
        timestamps = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        sort_keys["ts_utc"] = timestamps
        frame["ts_utc"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "date" in frame.columns:
        date_values = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        sort_keys["date"] = date_values
        if date_values.notna().any():
            frame["date"] = date_values.dt.strftime("%Y-%m-%d")

    sort_columns = [column for column in ("symbol", "ts_utc", "date", "timeframe") if column in frame.columns]
    if sort_columns:
        ordering_frame = frame.assign(
            **{
                f"_sort_{column}": sort_keys[column]
                for column in ("ts_utc", "date")
                if column in sort_keys
            }
        )
        by = [
            "_sort_ts_utc" if column == "ts_utc" and "_sort_ts_utc" in ordering_frame.columns else
            "_sort_date" if column == "date" and "_sort_date" in ordering_frame.columns else
            column
            for column in sort_columns
        ]
        frame = ordering_frame.sort_values(by, kind="mergesort", na_position="last")
        frame = frame.drop(
            columns=[column for column in ("_sort_ts_utc", "_sort_date") if column in frame.columns]
        )

    return frame.reset_index(drop=True)


def _standardize_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "ts_utc" in result.columns:
        timestamps = pd.to_datetime(result["ts_utc"], utc=True, errors="coerce")
        result["ts_utc"] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif "date" in result.columns:
        timestamps = pd.to_datetime(result["date"], utc=True, errors="coerce")
        result.insert(0, "ts_utc", timestamps.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    else:
        result.insert(0, "ts_utc", pd.Series([None] * len(result), index=result.index, dtype="object"))

    return result


def _sorted_by_time(df: pd.DataFrame) -> pd.DataFrame:
    standardized = _standardize_timestamp_columns(df)
    if standardized.empty or "ts_utc" not in standardized.columns:
        return standardized.reset_index(drop=True)

    timestamps = pd.to_datetime(standardized["ts_utc"], utc=True, errors="coerce")
    ordered = standardized.assign(_sort_ts=timestamps).sort_values(
        by=["_sort_ts"],
        kind="stable",
        na_position="last",
    )
    return ordered.drop(columns="_sort_ts").reset_index(drop=True)


def _signals_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the signal-engine portion of an experiment output DataFrame."""

    signal_columns = [column for column in results_df.columns if column not in _BACKTEST_COLUMNS]
    frame = results_df.loc[:, signal_columns].copy()

    if "position" not in frame.columns:
        frame["position"] = infer_position_series(results_df)

    ordered_columns = [
        column
        for column in (*_SIGNAL_PRIORITY_COLUMNS, *frame.columns)
        if column in frame.columns
    ]
    deduped_columns = list(dict.fromkeys(ordered_columns))
    return _normalize_artifact_frame(_sorted_by_time(frame.loc[:, deduped_columns]))


def _equity_curve_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the standardized equity curve artifact for experiment inspection."""

    missing_columns = [column for column in _BACKTEST_COLUMNS if column not in results_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Experiment results must include backtest columns: {missing}.")

    frame = _normalize_artifact_frame(_sorted_by_time(results_df))
    strategy_equity = broadcast_strategy_equity_curve(frame)
    payload = pd.DataFrame(index=frame.index)
    payload["ts_utc"] = frame["ts_utc"] if "ts_utc" in frame.columns else None
    if "symbol" in frame.columns:
        payload["symbol"] = frame["symbol"]
    payload["equity"] = strategy_equity["equity"].to_numpy(copy=True)
    if "strategy_return" in frame.columns:
        payload["strategy_return"] = strategy_equity["strategy_return"].to_numpy(copy=True)
    if "signal" in frame.columns:
        payload["signal"] = frame["signal"]
    if "position" in frame.columns:
        payload["position"] = frame["position"]
    else:
        payload["position"] = infer_position_series(frame)
    return payload


def _legacy_equity_curve_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        column for column in ("signal", *_BACKTEST_COLUMNS) if column in results_df.columns
    ]
    return _normalize_artifact_frame(results_df.loc[:, preferred_columns].copy())


def _resolve_time_values(df: pd.DataFrame) -> pd.Series:
    if "ts_utc" in df.columns:
        return pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    if "date" in df.columns:
        return pd.to_datetime(df["date"], utc=True, errors="coerce")
    return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")


def _trades_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "strategy_return" not in results_df.columns:
        return pd.DataFrame(
            columns=["entry_ts_utc", "exit_ts_utc", "symbol", "direction", "return"]
        )

    frame = _normalize_artifact_frame(_sorted_by_time(results_df))
    position = infer_position_series(frame).astype("float64")
    returns = frame["strategy_return"].fillna(0.0).astype("float64")
    timestamps = _resolve_time_values(frame)
    symbols = frame["symbol"] if "symbol" in frame.columns else pd.Series([None] * len(frame), index=frame.index)

    trades: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for idx in range(len(frame)):
        current_position = float(position.iloc[idx])
        has_next = idx + 1 < len(frame)
        next_position = float(position.iloc[idx + 1]) if has_next else 0.0

        if current_position != 0.0 and current is None:
            current = {
                "entry_ts_utc": _format_timestamp(timestamps.iloc[idx]),
                "exit_ts_utc": None,
                "symbol": None if pd.isna(symbols.iloc[idx]) else symbols.iloc[idx],
                "direction": "long" if current_position > 0.0 else "short",
                "return": 1.0,
            }

        if current_position != 0.0 and current is not None:
            current["return"] *= 1.0 + float(returns.iloc[idx])
            if current.get("symbol") is None and not pd.isna(symbols.iloc[idx]):
                current["symbol"] = symbols.iloc[idx]

        trade_closes = (
            current is not None
            and current_position != 0.0
            and has_next
            and next_position != current_position
        )
        if trade_closes:
            current["exit_ts_utc"] = _format_timestamp(timestamps.iloc[idx])
            current["return"] = float(current["return"] - 1.0)
            trades.append(current)
            current = None

    return pd.DataFrame(trades, columns=["entry_ts_utc", "exit_ts_utc", "symbol", "direction", "return"])


def _format_timestamp(value: pd.Timestamp | Any) -> str | None:
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _metrics_by_split_frame(metrics_rows: list[dict[str, Any]]) -> pd.DataFrame:
    ordered_columns = [*_SPLIT_METADATA_COLUMNS, *_METRIC_KEYS]
    frame = pd.DataFrame(metrics_rows)
    if frame.empty:
        return pd.DataFrame(columns=ordered_columns)

    for column in ordered_columns:
        if column not in frame.columns:
            frame[column] = None

    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame.loc[:, [*ordered_columns, *remaining_columns]]


def _write_run_outputs(
    output_dir: Path,
    payloads: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = payloads.get("config")
    split_metadata = payloads.get("split_metadata")
    if split_metadata is None:
        _write_json(output_dir / "config.json", dict(config or {}))
    else:
        _write_json(output_dir / "split.json", dict(split_metadata))

    payloads["signals_frame"].to_parquet(output_dir / "signals.parquet")
    payloads["equity_curve_frame"].to_csv(output_dir / "equity_curve.csv", index=False)
    payloads["legacy_equity_curve_frame"].to_parquet(output_dir / "equity_curve.parquet")
    _write_json(output_dir / "metrics.json", dict(payloads["metrics"]))
    _write_json(output_dir / "signal_diagnostics.json", dict(payloads["signal_diagnostics"]))
    _write_json(output_dir / "qa_summary.json", dict(payloads["qa_summary"]))
    if payloads.get("signal_semantics") is not None:
        _write_json(output_dir / _SIGNAL_SEMANTICS_FILENAME, dict(payloads["signal_semantics"]))
    write_promotion_gate_artifact(output_dir, payloads.get("promotion_evaluation"))

    written = [
        "signals.parquet",
        "equity_curve.csv",
        "equity_curve.parquet",
        "metrics.json",
        "signal_diagnostics.json",
        "qa_summary.json",
    ]
    if payloads.get("signal_semantics") is not None:
        written.append(_SIGNAL_SEMANTICS_FILENAME)
    if payloads.get("promotion_evaluation") is not None:
        written.append(DEFAULT_PROMOTION_ARTIFACT_FILENAME)
    if split_metadata is None:
        written.insert(0, "config.json")
    else:
        written.append("split.json")

    trades_frame = payloads["trades_frame"]
    if not trades_frame.empty:
        trades_frame.to_parquet(output_dir / "trades.parquet")
        written.append("trades.parquet")

    return written


def _prepare_run_outputs(
    output_dir: Path,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    run_id: str,
    strategy_name: str,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    annotated_results_df = _normalize_artifact_frame(_sorted_by_time(results_df))
    annotated_results_df.attrs = dict(results_df.attrs)
    if annotated_results_df.attrs.get("dataset") is None and config.get("dataset") is not None:
        annotated_results_df.attrs["dataset"] = config["dataset"]
    evaluation = config.get("evaluation")
    if (
        annotated_results_df.attrs.get("timeframe") is None
        and isinstance(evaluation, dict)
        and evaluation.get("timeframe") is not None
    ):
        annotated_results_df.attrs["timeframe"] = evaluation["timeframe"]

    signals_frame = _signals_frame(annotated_results_df)
    equity_curve_frame = _equity_curve_frame(annotated_results_df)
    trades_frame = _trades_frame(annotated_results_df)
    signal_diagnostics = _signal_diagnostics_payload(annotated_results_df)
    signal_semantics = _signal_semantics_payload(annotated_results_df)
    qa_summary = generate_strategy_qa_summary(
        annotated_results_df,
        annotated_results_df["signal"] if "signal" in annotated_results_df.columns else pd.Series(dtype="float64"),
        signal_diagnostics,
        metrics,
        strategy_name=strategy_name,
        run_id=run_id,
    )

    validate_strategy_artifact_payload_consistency(
        results_df=annotated_results_df,
        signals_frame=signals_frame,
        equity_curve_frame=equity_curve_frame,
        metrics=metrics,
        signal_diagnostics=signal_diagnostics,
        qa_summary=qa_summary,
    )
    promotion_evaluation = None
    if split_metadata is None:
        split_metric_rows = None
        if "split_id" in annotated_results_df.columns:
            split_metric_rows = _split_metric_rows(annotated_results_df)
        promotion_evaluation = evaluate_promotion_gates(
            run_type="strategy",
            config=_promotion_gate_config(config),
            sources={
                "metrics": dict(metrics),
                "qa_summary": dict(qa_summary),
                "config": dict(config),
                "split_metrics": split_metric_rows,
            },
        )

    return {
        "config": dict(config),
        "split_metadata": None if split_metadata is None else dict(split_metadata),
        "signals_frame": signals_frame,
        "equity_curve_frame": equity_curve_frame,
        "legacy_equity_curve_frame": _legacy_equity_curve_frame(annotated_results_df),
        "trades_frame": trades_frame,
        "metrics": dict(metrics),
        "signal_diagnostics": dict(signal_diagnostics),
        "signal_semantics": None if signal_semantics is None else dict(signal_semantics),
        "qa_summary": dict(qa_summary),
        "promotion_evaluation": promotion_evaluation,
    }


def _signal_diagnostics_payload(results_df: pd.DataFrame) -> dict[str, Any]:
    diagnostics = results_df.attrs.get("signal_diagnostics")
    if isinstance(diagnostics, dict):
        return diagnostics
    if "signal" not in results_df.columns:
        return compute_signal_diagnostics(pd.Series(dtype="float64"), results_df)
    return compute_signal_diagnostics(results_df["signal"], results_df)


def _signal_semantics_payload(results_df: pd.DataFrame) -> dict[str, Any] | None:
    payload = extract_signal_metadata(results_df)
    if payload is None:
        return None
    return dict(payload)


def _manifest_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = _metrics_summary(metrics)
    return dict(summary) if summary else dict(metrics)


def _promotion_gate_config(config: dict[str, Any]) -> dict[str, Any] | None:
    payload = config.get("promotion_gates")
    return dict(payload) if isinstance(payload, dict) else None


def _split_metric_rows(results_df: pd.DataFrame) -> list[dict[str, Any]] | None:
    if "split_id" not in results_df.columns:
        return None
    rows: list[dict[str, Any]] = []
    for split_id, split_frame in results_df.groupby("split_id", sort=False):
        metrics = compute_performance_metrics(split_frame)
        metrics["split_id"] = split_id
        rows.append(metrics)
    return rows


def _build_manifest(
    *,
    experiment_dir: Path,
    strategy_name: str,
    evaluation_mode: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    signal_semantics: dict[str, Any] | None,
    split_count: int | None,
    promotion_evaluation: Any | None = None,
) -> dict[str, Any]:
    artifact_files = sorted(
        path.relative_to(experiment_dir).as_posix()
        for path in experiment_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    constructor_id, constructor_params = _constructor_payload(signal_semantics)
    return {
        "run_id": experiment_dir.name,
        "timestamp": _utc_timestamp_from_run_id(experiment_dir.name),
        "strategy_name": strategy_name,
        "evaluation_mode": evaluation_mode,
        "evaluation_config_path": config.get("evaluation_config_path"),
        "strict_mode": config.get("strict_mode"),
        "artifact_files": artifact_files,
        "signal_semantics_path": _SIGNAL_SEMANTICS_FILENAME if _SIGNAL_SEMANTICS_FILENAME in artifact_files else None,
        "signal_type": None if signal_semantics is None else signal_semantics.get("signal_type"),
        "signal_version": None if signal_semantics is None else signal_semantics.get("version"),
        "constructor_id": constructor_id,
        "constructor_params": constructor_params,
        "signal_semantics": signal_semantics,
        "split_count": split_count,
        "primary_metric": "sharpe_ratio",
        "metric_summary": _manifest_metric_summary(metrics),
        "promotion_gate_summary": (
            None if promotion_evaluation is None else promotion_evaluation.summary()
        ),
    }


def _write_manifest(
    experiment_dir: Path,
    strategy_name: str,
    evaluation_mode: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    signal_semantics: dict[str, Any] | None,
    split_count: int | None,
    promotion_evaluation: Any | None = None,
) -> None:
    _write_json(
        experiment_dir / "manifest.json",
        _build_manifest(
            experiment_dir=experiment_dir,
            strategy_name=strategy_name,
            evaluation_mode=evaluation_mode,
            config=config,
            metrics=metrics,
            signal_semantics=signal_semantics,
            split_count=split_count,
            promotion_evaluation=promotion_evaluation,
        ),
    )


def save_experiment_outputs(
    experiment_dir: Path,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    strategy_name: str | None = None,
) -> None:
    """Write the standard single-run experiment artifacts into an existing directory."""

    resolved_strategy_name = str(strategy_name or config.get("strategy_name") or experiment_dir.name)
    payloads = _prepare_run_outputs(
        experiment_dir,
        results_df,
        metrics,
        config,
        run_id=experiment_dir.name,
        strategy_name=resolved_strategy_name,
    )
    _write_run_outputs(experiment_dir, payloads)
    _write_manifest(
        experiment_dir,
        resolved_strategy_name,
        "single",
        config,
        metrics,
        signal_semantics=payloads.get("signal_semantics"),
        split_count=None,
        promotion_evaluation=payloads.get("promotion_evaluation"),
    )


def save_walk_forward_experiment(
    strategy_name: str,
    split_results: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Persist walk-forward artifacts under the standard strategy artifact root."""

    aggregate_results_df = pd.concat(
        [split_result["results_df"] for split_result in split_results],
        ignore_index=True,
    ) if split_results else pd.DataFrame(columns=["ts_utc", "signal", "strategy_return", "equity_curve"])
    aggregate_payloads = _prepare_run_outputs(
        ARTIFACTS_ROOT,
        aggregate_results_df,
        aggregate_summary,
        config,
        run_id="aggregate",
        strategy_name=strategy_name,
    )
    aggregate_payloads["promotion_evaluation"] = evaluate_promotion_gates(
        run_type="strategy",
        config=_promotion_gate_config(config),
        sources={
            "metrics": dict(aggregate_summary),
            "qa_summary": dict(aggregate_payloads["qa_summary"]),
            "config": dict(config),
            "split_metrics": [dict(split_result["metrics"]) for split_result in split_results],
        },
    )
    metrics_rows: list[dict[str, Any]] = []
    split_payloads: list[dict[str, Any]] = []
    for split_result in split_results:
        split_payload = _prepare_run_outputs(
            ARTIFACTS_ROOT,
            split_result["results_df"],
            split_result["metrics"],
            config,
            run_id=str(split_result["split_id"]),
            strategy_name=strategy_name,
            split_metadata=split_result["split_metadata"],
        )
        split_payloads.append(split_payload)
        metrics_rows.append(
            {
                **split_result["split_metadata"],
                "split_rows": split_result["split_rows"],
                "train_rows": split_result["train_rows"],
                "test_rows": split_result["test_rows"],
                **split_result["metrics"],
            }
        )

    metrics_by_split = _metrics_by_split_frame(metrics_rows)
    validate_walk_forward_consistency(split_results, aggregate_summary, metrics_by_split)
    experiment_dir = resolve_experiment_dir(
        strategy_name,
        evaluation_mode="walk_forward",
        config=config,
        results_df=aggregate_results_df,
    )

    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    aggregate_payloads["config"] = dict(config)
    splits_dir = experiment_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_run_outputs(experiment_dir, aggregate_payloads)

    for split_result, split_payload in zip(split_results, split_payloads, strict=False):
        split_id = str(split_result["split_id"])
        split_dir = splits_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=False)
        _write_run_outputs(split_dir, split_payload)

    metrics_by_split.to_csv(experiment_dir / "metrics_by_split.csv", index=False)
    _write_manifest(
        experiment_dir,
        strategy_name,
        "walk_forward",
        config,
        aggregate_summary,
        signal_semantics=aggregate_payloads.get("signal_semantics"),
        split_count=len(split_results),
        promotion_evaluation=aggregate_payloads.get("promotion_evaluation"),
    )
    validate_run_consistency(experiment_dir, validate_registry=False)
    _write_registry_entry(
        _build_registry_entry(
            experiment_dir,
            strategy_name,
            aggregate_results_df,
            aggregate_summary,
            config,
            evaluation_mode="walk_forward",
            evaluation_config=dict(config.get("evaluation") or {}),
            split_count=len(split_results),
            promotion_evaluation=aggregate_payloads.get("promotion_evaluation"),
        )
    )
    validate_run_consistency(experiment_dir)
    return experiment_dir


def save_robustness_experiment(
    strategy_name: str,
    *,
    config: dict[str, Any],
    variants: list[Any],
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    neighbor_metrics: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    """Persist robustness artifacts under a dedicated deterministic artifact root."""

    experiment_dir = resolve_robustness_experiment_dir(
        strategy_name,
        config=config,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        summary=summary,
    )
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    _write_json(experiment_dir / "config.json", dict(config))
    _write_json(
        experiment_dir / "variants.json",
        {
            "variants": [
                variant.to_record() if hasattr(variant, "to_record") else dict(variant)
                for variant in variants
            ]
        },
    )
    _write_json(experiment_dir / "summary.json", dict(summary))
    variant_metrics.to_csv(experiment_dir / "metrics_by_variant.csv", index=False)

    artifact_files = {
        "config.json",
        "variants.json",
        "summary.json",
        "metrics_by_variant.csv",
    }
    if not stability_metrics.empty:
        stability_metrics.to_csv(experiment_dir / "stability_metrics.csv", index=False)
        artifact_files.add("stability_metrics.csv")
    if not neighbor_metrics.empty:
        neighbor_metrics.to_csv(experiment_dir / "neighbor_metrics.csv", index=False)
        artifact_files.add("neighbor_metrics.csv")

    _write_json(
        experiment_dir / "manifest.json",
        {
            "run_id": experiment_dir.name,
            "timestamp": _utc_timestamp_from_run_id(experiment_dir.name),
            "strategy_name": strategy_name,
            "evaluation_mode": "robustness",
            "artifact_files": sorted(artifact_files),
            "primary_metric": str(summary.get("ranking_metric", "sharpe_ratio")),
            "metric_summary": dict(summary),
            "variant_count": int(summary.get("variant_count", len(variant_metrics))),
            "split_count": summary.get("split_count"),
            "stability_mode": summary.get("stability_mode"),
        },
    )
    return experiment_dir


def save_experiment(
    strategy_name: str,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """
    Persist a strategy experiment's artifacts under the research artifact directory.

    Args:
        strategy_name: Strategy identifier associated with the experiment run.
        results_df: Backtest results containing signal data and equity curve outputs.
        metrics: Computed strategy performance metrics.
        config: Strategy configuration used for the experiment run.

    Returns:
        The created experiment directory path.

    Raises:
        ValueError: If ``results_df`` does not include ``strategy_return`` and
            ``equity_curve`` columns required for the equity curve artifact.
    """

    experiment_dir = resolve_experiment_dir(
        strategy_name,
        evaluation_mode="single",
        config=config,
        results_df=results_df,
    )
    payloads = _prepare_run_outputs(
        experiment_dir,
        results_df,
        metrics,
        config,
        run_id=experiment_dir.name,
        strategy_name=strategy_name,
    )
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    _write_run_outputs(experiment_dir, payloads)
    _write_manifest(
        experiment_dir,
        strategy_name,
        "single",
        config,
        metrics,
        signal_semantics=payloads.get("signal_semantics"),
        split_count=None,
        promotion_evaluation=payloads.get("promotion_evaluation"),
    )
    validate_run_consistency(experiment_dir, validate_registry=False)
    _write_registry_entry(
        _build_registry_entry(
            experiment_dir,
            strategy_name,
            results_df,
            metrics,
            config,
            evaluation_mode="single",
            evaluation_config=None,
            split_count=None,
            promotion_evaluation=payloads.get("promotion_evaluation"),
        )
    )
    validate_run_consistency(experiment_dir)
    return experiment_dir
