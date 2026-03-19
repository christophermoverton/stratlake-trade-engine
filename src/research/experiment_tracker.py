from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.registry import append_registry_entry, default_registry_path

ARTIFACTS_ROOT = Path("artifacts") / "strategies"
_BACKTEST_COLUMNS = ("strategy_return", "equity_curve")
_METRICS_SUMMARY_KEYS = (
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


def _sanitize_strategy_name(strategy_name: str) -> str:
    """Return a filesystem-friendly strategy name component for experiment paths."""

    cleaned = "".join(char if char.isalnum() else "_" for char in strategy_name.strip().lower())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    return normalized or "strategy"


def _build_run_id(strategy_name: str) -> str:
    """Return a unique experiment run identifier using the current UTC timestamp."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{_sanitize_strategy_name(strategy_name)}"


def _signals_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the signal-engine portion of an experiment output DataFrame."""

    signal_columns = [column for column in results_df.columns if column not in _BACKTEST_COLUMNS]
    return results_df.loc[:, signal_columns].copy()


def _equity_curve_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the backtest artifact subset required for experiment inspection."""

    missing_columns = [column for column in _BACKTEST_COLUMNS if column not in results_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Experiment results must include backtest columns: {missing}.")

    preferred_columns = [
        column for column in ("signal", *_BACKTEST_COLUMNS) if column in results_df.columns
    ]
    return results_df.loc[:, preferred_columns].copy()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a dictionary to disk using stable JSON formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_timestamp_from_run_id(run_id: str) -> str:
    """Return an ISO8601 UTC timestamp derived from the run identifier when possible."""

    timestamp_component = run_id.split("_", 1)[0]
    try:
        created_at = datetime.strptime(timestamp_component, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=UTC)
    except ValueError:
        created_at = datetime.now(UTC)
    return created_at.isoformat().replace("+00:00", "Z")


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
) -> dict[str, Any]:
    run_id = experiment_dir.name
    return {
        "run_id": run_id,
        "timestamp": _utc_timestamp_from_run_id(run_id),
        "strategy_name": strategy_name,
        "dataset": config.get("dataset"),
        "strategy_params": dict(config.get("parameters") or {}),
        "evaluation_mode": evaluation_mode,
        "evaluation_config": _drop_none_values(evaluation_config),
        "data_range": _resolve_data_range(results_df, config, evaluation_mode=evaluation_mode),
        "timeframe": _infer_timeframe(results_df, config),
        "metrics_summary": _metrics_summary(metrics_summary),
        "artifact_path": experiment_dir.as_posix(),
        "split_count": split_count,
        "evaluation_config_path": config.get("evaluation_config_path"),
    }


def _write_registry_entry(entry: dict[str, Any]) -> None:
    """Append one completed run to the shared strategy registry."""

    append_registry_entry(default_registry_path(ARTIFACTS_ROOT), entry)


def create_experiment_dir(strategy_name: str) -> Path:
    """Create and return a new experiment directory for a strategy run."""

    experiment_dir = ARTIFACTS_ROOT / _build_run_id(strategy_name)
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def save_experiment_outputs(
    experiment_dir: Path,
    results_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Write the standard single-run experiment artifacts into an existing directory."""

    _signals_frame(results_df).to_parquet(experiment_dir / "signals.parquet")
    _equity_curve_frame(results_df).to_parquet(experiment_dir / "equity_curve.parquet")
    _write_json(experiment_dir / "metrics.json", metrics)
    _write_json(experiment_dir / "config.json", config)


def save_walk_forward_experiment(
    strategy_name: str,
    split_results: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    """Persist walk-forward artifacts under the standard strategy artifact root."""

    experiment_dir = create_experiment_dir(strategy_name)
    _write_json(experiment_dir / "metrics.json", aggregate_summary)
    _write_json(experiment_dir / "config.json", config)

    metrics_rows: list[dict[str, Any]] = []
    splits_dir = experiment_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_result in split_results:
        split_id = str(split_result["split_id"])
        split_dir = splits_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=False)

        results_df = split_result["results_df"]
        metrics = split_result["metrics"]

        _signals_frame(results_df).to_parquet(split_dir / "signals.parquet")
        _equity_curve_frame(results_df).to_parquet(split_dir / "equity_curve.parquet")
        _write_json(split_dir / "metrics.json", metrics)
        _write_json(split_dir / "split.json", split_result["split_metadata"])

        metrics_rows.append(
            {
                **split_result["split_metadata"],
                "split_rows": split_result["split_rows"],
                "train_rows": split_result["train_rows"],
                "test_rows": split_result["test_rows"],
                **metrics,
            }
        )

    pd.DataFrame(metrics_rows).to_csv(experiment_dir / "metrics_by_split.csv", index=False)
    aggregate_results_df = pd.concat(
        [split_result["results_df"] for split_result in split_results], ignore_index=True
    )
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
        )
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

    experiment_dir = create_experiment_dir(strategy_name)
    save_experiment_outputs(experiment_dir, results_df, metrics, config)
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
        )
    )
    return experiment_dir
