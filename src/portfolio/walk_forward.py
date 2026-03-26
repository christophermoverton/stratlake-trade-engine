from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.config.evaluation import EvaluationConfig, load_evaluation_config
from src.research import experiment_tracker
from src.research.registry import (
    STRATEGY_RUN_TYPE,
    canonicalize_value,
    default_registry_path,
    filter_by_run_type,
    generate_portfolio_run_id,
    load_registry,
    register_portfolio_run,
)
from src.research.splits import EvaluationSplit, generate_evaluation_splits

from .allocators import BaseAllocator
from .artifacts import (
    _normalize_components,
    _normalize_mapping,
    _normalize_portfolio_config,
    _portfolio_equity_curve_frame,
    _portfolio_returns_frame,
    _utc_timestamp_from_run_id,
    _weights_frame,
    _write_csv,
    _write_json,
)
from .constructor import construct_portfolio
from .loaders import build_aligned_return_matrix, load_strategy_run_returns
from .metrics import compute_portfolio_metrics
from .qa import run_portfolio_qa

PORTFOLIO_WALK_FORWARD_METRIC_KEYS: tuple[str, ...] = (
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
_SPLIT_METADATA_COLUMNS: tuple[str, ...] = (
    "split_id",
    "mode",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "start",
    "end",
    "row_count",
)


class PortfolioWalkForwardError(ValueError):
    """Raised when a portfolio walk-forward run cannot be executed safely."""


def run_portfolio_walk_forward(
    component_run_ids: list[str],
    evaluation_config_path: str | Path,
    allocator: BaseAllocator,
    timeframe: str,
    output_dir: str | Path,
    *,
    portfolio_name: str,
    initial_capital: float = 1.0,
    alignment_policy: str = "intersection",
    execution_config: ExecutionConfig | None = None,
) -> dict[str, Any]:
    """Construct and score one portfolio independently for each evaluation split."""

    normalized_portfolio_name = _normalize_required_string(portfolio_name, field_name="portfolio_name")
    normalized_timeframe = _normalize_timeframe(timeframe)
    normalized_component_run_ids = _normalize_component_run_ids(component_run_ids)
    evaluation_path = Path(evaluation_config_path)
    evaluation_config = load_evaluation_config(evaluation_path)
    _validate_timeframe_compatibility(normalized_timeframe, evaluation_config)

    splits = generate_evaluation_splits(evaluation_config)
    if not splits:
        raise PortfolioWalkForwardError("Walk-forward portfolio evaluation did not produce any splits.")

    components, strategy_returns = _resolve_component_returns(normalized_component_run_ids)
    root_config = _build_root_config(
        portfolio_name=normalized_portfolio_name,
        allocator_name=allocator.name,
        timeframe=normalized_timeframe,
        initial_capital=initial_capital,
        alignment_policy=alignment_policy,
        evaluation_path=evaluation_path,
        execution_config=execution_config,
    )
    run_id = generate_portfolio_run_id(
        portfolio_name=normalized_portfolio_name,
        allocator_name=allocator.name,
        component_run_ids=normalized_component_run_ids,
        timeframe=normalized_timeframe,
        start_ts=_run_start_ts(splits),
        end_ts=_run_end_ts(splits),
        config=root_config,
        evaluation_config_path=evaluation_path,
    )
    experiment_dir = Path(output_dir) / run_id
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    split_results = [
        _execute_portfolio_split(
            strategy_returns=strategy_returns,
            split=split,
            allocator=allocator,
            timeframe=normalized_timeframe,
            initial_capital=float(initial_capital),
            portfolio_name=normalized_portfolio_name,
            execution_config=execution_config,
        )
        for split in splits
    ]
    aggregate_metrics = _build_aggregate_metrics(split_results, timeframe=normalized_timeframe)
    metrics_by_split = _metrics_by_split_frame(split_results)

    _write_json(experiment_dir / "config.json", _normalize_portfolio_config(root_config, components=components))
    _write_json(experiment_dir / "components.json", {"components": _normalize_components(components)})
    _write_csv(experiment_dir / "metrics_by_split.csv", metrics_by_split)
    _write_json(experiment_dir / "aggregate_metrics.json", aggregate_metrics)

    split_artifact_dirs: list[str] = []
    splits_dir = experiment_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_result in split_results:
        split_dir = splits_dir / split_result["split_id"]
        split_dir.mkdir(parents=True, exist_ok=False)
        _write_split_artifacts(split_dir=split_dir, split_result=split_result)
        split_artifact_dirs.append(Path("splits", split_result["split_id"]).as_posix())

    manifest = _build_manifest(
        experiment_dir=experiment_dir,
        config=root_config,
        components=components,
        split_results=split_results,
        split_artifact_dirs=split_artifact_dirs,
        aggregate_metrics=aggregate_metrics,
    )
    _write_json(experiment_dir / "manifest.json", manifest)

    register_portfolio_run(
        registry_path=default_registry_path(Path(output_dir)),
        run_id=run_id,
        config=root_config,
        components=components,
        metrics=dict(aggregate_metrics["metric_summary"]),
        artifact_path=experiment_dir.as_posix(),
        metadata={
            "portfolio_name": normalized_portfolio_name,
            "allocator_name": allocator.name,
            "timeframe": normalized_timeframe,
            "start_ts": _run_start_ts(splits),
            "end_ts": _run_end_ts(splits),
            "evaluation_config_path": evaluation_path.as_posix(),
            "split_count": len(split_results),
            "aggregate_metrics": aggregate_metrics,
        },
    )

    return {
        "portfolio_name": normalized_portfolio_name,
        "run_id": run_id,
        "allocator_name": allocator.name,
        "timeframe": normalized_timeframe,
        "component_count": len(components),
        "components": _normalize_components(components),
        "config": _normalize_portfolio_config(root_config, components=components),
        "evaluation_config_path": evaluation_path.as_posix(),
        "split_count": len(split_results),
        "metrics": dict(aggregate_metrics["metric_summary"]),
        "aggregate_metrics": aggregate_metrics,
        "metrics_by_split": metrics_by_split,
        "experiment_dir": experiment_dir,
        "splits": split_results,
        "manifest": manifest,
    }


def _resolve_component_returns(component_run_ids: Sequence[str]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    entries = filter_by_run_type(
        load_registry(default_registry_path(experiment_tracker.ARTIFACTS_ROOT)),
        STRATEGY_RUN_TYPE,
    )
    registry_by_run_id = {
        str(entry["run_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("run_id") is not None
    }

    component_frames: list[pd.DataFrame] = []
    components: list[dict[str, Any]] = []
    for run_id in component_run_ids:
        registry_entry = registry_by_run_id.get(run_id)
        run_dir = (
            Path(str(registry_entry["artifact_path"]))
            if registry_entry is not None and registry_entry.get("artifact_path") is not None
            else experiment_tracker.ARTIFACTS_ROOT / run_id
        )
        if not run_dir.exists():
            raise PortfolioWalkForwardError(
                f"Component strategy run '{run_id}' could not be resolved to an artifact directory."
            )
        component_frame = load_strategy_run_returns(run_dir)
        strategy_names = sorted(component_frame["strategy_name"].astype("string").unique().tolist())
        if len(strategy_names) != 1:
            raise PortfolioWalkForwardError(
                f"Strategy run '{run_id}' resolved to an invalid component strategy set: {strategy_names}."
            )
        strategy_name = str(strategy_names[0])
        component_frames.append(component_frame)
        components.append(
            {
                "strategy_name": strategy_name,
                "run_id": run_id,
                "source_artifact_path": run_dir.as_posix(),
            }
        )

    return _normalize_components(components), pd.concat(component_frames, ignore_index=True)


def _execute_portfolio_split(
    *,
    strategy_returns: pd.DataFrame,
    split: EvaluationSplit,
    allocator: BaseAllocator,
    timeframe: str,
    initial_capital: float,
    portfolio_name: str,
    execution_config: ExecutionConfig | None,
) -> dict[str, Any]:
    split_returns = _slice_strategy_returns_for_split(strategy_returns, split)
    aligned_returns = build_aligned_return_matrix(split_returns)
    if aligned_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' produced an empty aligned return matrix after intersection alignment."
        )

    portfolio_output = construct_portfolio(
        aligned_returns,
        allocator,
        initial_capital=initial_capital,
        execution_config=execution_config,
    )
    metrics = compute_portfolio_metrics(portfolio_output, timeframe)
    split_metadata = _split_metadata(split, row_count=len(portfolio_output))
    return {
        "split_id": split.split_id,
        "split_metadata": split_metadata,
        "portfolio_output": portfolio_output,
        "metrics": metrics,
        "row_count": int(len(portfolio_output)),
        "portfolio_name": portfolio_name,
        "allocator_name": allocator.name,
        "timeframe": timeframe,
        "initial_capital": float(initial_capital),
        "execution": (
            resolve_execution_config().to_dict()
            if execution_config is None
            else execution_config.to_dict()
        ),
    }


def _slice_strategy_returns_for_split(
    strategy_returns: pd.DataFrame,
    split: EvaluationSplit,
) -> pd.DataFrame:
    date_values = pd.to_datetime(strategy_returns["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    mask = (date_values >= split.test_start) & (date_values < split.test_end)
    split_returns = strategy_returns.loc[mask].reset_index(drop=True)
    if split_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' has no component strategy data in test window [{split.test_start}, {split.test_end})."
        )

    expected_strategies = sorted(strategy_returns["strategy_name"].astype("string").unique().tolist())
    observed_strategies = sorted(split_returns["strategy_name"].astype("string").unique().tolist())
    missing = sorted(set(expected_strategies) - set(observed_strategies))
    if missing:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' is missing component returns for strategies {missing} "
            f"in test window [{split.test_start}, {split.test_end})."
        )
    return split_returns


def _metrics_by_split_frame(split_results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for split_result in split_results:
        rows.append(
            {
                **dict(split_result["split_metadata"]),
                **dict(split_result["metrics"]),
            }
        )

    ordered_columns = [*_SPLIT_METADATA_COLUMNS, *PORTFOLIO_WALK_FORWARD_METRIC_KEYS]
    frame = pd.DataFrame(rows)
    for column in ordered_columns:
        if column not in frame.columns:
            frame[column] = None
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame.loc[:, [*ordered_columns, *sorted(remaining_columns)]]


def _build_aggregate_metrics(
    split_results: Sequence[Mapping[str, Any]],
    *,
    timeframe: str,
) -> dict[str, Any]:
    if not split_results:
        raise PortfolioWalkForwardError("Walk-forward portfolio evaluation requires at least one executed split.")

    first_split = split_results[0]["split_metadata"]
    last_split = split_results[-1]["split_metadata"]
    metric_statistics: dict[str, dict[str, float | None]] = {}
    metric_summary: dict[str, float | None] = {}
    for metric_name in PORTFOLIO_WALK_FORWARD_METRIC_KEYS:
        metric_values = [
            split_result["metrics"].get(metric_name)
            for split_result in split_results
            if split_result["metrics"].get(metric_name) is not None
        ]
        if not metric_values:
            metric_statistics[metric_name] = {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
            }
            metric_summary[metric_name] = None
            continue

        series = pd.Series(metric_values, dtype="float64")
        stats = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
        metric_statistics[metric_name] = stats
        metric_summary[metric_name] = stats["mean"]

    return {
        "aggregation_method": "descriptive statistics computed across split-level portfolio metrics in split order",
        "mode": first_split["mode"],
        "split_count": len(split_results),
        "timeframe": timeframe,
        "first_train_start": first_split["train_start"],
        "last_test_end": last_split["test_end"],
        "split_ids": [str(split_result["split_id"]) for split_result in split_results],
        "metric_summary": metric_summary,
        "metric_statistics": metric_statistics,
    }


def _write_split_artifacts(*, split_dir: Path, split_result: Mapping[str, Any]) -> None:
    portfolio_output = split_result["portfolio_output"]
    split_config = {
        "portfolio_name": split_result["portfolio_name"],
        "allocator": split_result["allocator_name"],
        "initial_capital": float(split_result["initial_capital"]),
        "execution": dict(split_result.get("execution", {})),
        "timeframe": split_result["timeframe"],
    }
    _write_json(split_dir / "split.json", dict(split_result["split_metadata"]))
    _write_csv(split_dir / "weights.csv", _weights_frame(portfolio_output))
    _write_csv(split_dir / "portfolio_returns.csv", _portfolio_returns_frame(portfolio_output))
    _write_csv(split_dir / "portfolio_equity_curve.csv", _portfolio_equity_curve_frame(portfolio_output))
    _write_json(split_dir / "metrics.json", _normalize_mapping(dict(split_result["metrics"]), owner="metrics"))
    run_portfolio_qa(
        portfolio_output,
        dict(split_result["metrics"]),
        split_config,
        artifacts_dir=split_dir,
        run_id=split_dir.name,
        strict=True,
    )


def _build_manifest(
    *,
    experiment_dir: Path,
    config: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    split_results: Sequence[Mapping[str, Any]],
    split_artifact_dirs: Sequence[str],
    aggregate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_files = sorted(
        path.relative_to(experiment_dir).as_posix()
        for path in experiment_dir.rglob("*")
        if path.is_file()
    )
    if "manifest.json" not in artifact_files:
        artifact_files.append("manifest.json")
        artifact_files.sort()
    return {
        "run_id": experiment_dir.name,
        "timestamp": _utc_timestamp_from_run_id(experiment_dir.name),
        "portfolio_name": config["portfolio_name"],
        "allocator": config["allocator"],
        "evaluation_mode": "walk_forward",
        "evaluation_config_path": config["evaluation_config_path"],
        "component_count": len(components),
        "split_count": len(split_results),
        "split_artifact_dirs": list(split_artifact_dirs),
        "artifact_files": artifact_files,
        "aggregate_metric_summary": canonicalize_value(dict(aggregate_metrics["metric_summary"])),
        "aggregate_metrics_path": "aggregate_metrics.json",
        "metrics_by_split_path": "metrics_by_split.csv",
    }


def _build_root_config(
    *,
    portfolio_name: str,
    allocator_name: str,
    timeframe: str,
    initial_capital: float,
    alignment_policy: str,
    evaluation_path: Path,
    execution_config: ExecutionConfig | None,
) -> dict[str, Any]:
    return {
        "portfolio_name": portfolio_name,
        "allocator": allocator_name,
        "initial_capital": float(initial_capital),
        "alignment_policy": _normalize_required_string(alignment_policy, field_name="alignment_policy"),
        "execution": (
            resolve_execution_config().to_dict()
            if execution_config is None
            else execution_config.to_dict()
        ),
        "timeframe": timeframe,
        "evaluation_config_path": evaluation_path.as_posix(),
    }


def _split_metadata(split: EvaluationSplit, *, row_count: int) -> dict[str, Any]:
    return {
        "split_id": split.split_id,
        "mode": split.mode,
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "start": split.test_start,
        "end": split.test_end,
        "row_count": int(row_count),
    }


def _run_start_ts(splits: Sequence[EvaluationSplit]) -> str:
    return min(split.train_start for split in splits)


def _run_end_ts(splits: Sequence[EvaluationSplit]) -> str:
    return max(split.test_end for split in splits)


def _normalize_component_run_ids(component_run_ids: Sequence[str]) -> list[str]:
    normalized = sorted(
        _normalize_required_string(run_id, field_name="component_run_ids")
        for run_id in component_run_ids
    )
    if not normalized:
        raise PortfolioWalkForwardError("component_run_ids must contain at least one run id.")
    return normalized


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PortfolioWalkForwardError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise PortfolioWalkForwardError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_timeframe(timeframe: str) -> str:
    normalized = _normalize_required_string(timeframe, field_name="timeframe").lower()
    if normalized in {"1d", "1day", "day", "daily"}:
        return "1D"
    if normalized in {"1m", "1min", "1minute", "minute", "minutes"}:
        return "1Min"
    raise PortfolioWalkForwardError("timeframe must be one of: 1D, 1Min.")


def _validate_timeframe_compatibility(timeframe: str, evaluation_config: EvaluationConfig) -> None:
    evaluation_timeframe = _normalize_timeframe(evaluation_config.timeframe)
    if timeframe != evaluation_timeframe:
        raise PortfolioWalkForwardError(
            "Portfolio timeframe and evaluation timeframe must match. "
            f"Received portfolio timeframe {timeframe!r} and evaluation timeframe {evaluation_config.timeframe!r}."
        )


__all__ = ["PORTFOLIO_WALK_FORWARD_METRIC_KEYS", "PortfolioWalkForwardError", "run_portfolio_walk_forward"]
