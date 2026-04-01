from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config.execution import ExecutionConfig
from src.config.evaluation import EvaluationConfig, load_evaluation_config
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.config.sanity import SanityCheckConfig, resolve_sanity_check_config
from src.research.consistency import validate_portfolio_artifact_payload_consistency, validate_portfolio_walk_forward_consistency
from src.research import experiment_tracker
from src.research.registry import (
    STRATEGY_RUN_TYPE,
    canonicalize_value,
    default_registry_path,
    filter_by_run_type,
    generate_portfolio_run_id,
    load_registry,
)
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.sanity import SanityCheckError, validate_portfolio_output_sanity
from src.research.splits import EvaluationSplit, generate_evaluation_splits
from src.research.metrics import MINUTE_PERIODS_PER_YEAR, TRADING_DAYS_PER_YEAR
from src.research.promotion import (
    DEFAULT_PROMOTION_ARTIFACT_FILENAME,
    evaluate_promotion_gates,
    write_promotion_gate_artifact,
)

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
    register_validated_portfolio_run,
)
from .constructor import construct_portfolio
from .loaders import build_aligned_return_matrix, load_strategy_run_returns
from .metrics import compute_portfolio_metrics
from .qa import run_portfolio_qa

PORTFOLIO_WALK_FORWARD_METRIC_KEYS: tuple[str, ...] = (
    "cumulative_return",
    "gross_cumulative_return",
    "total_return",
    "gross_total_return",
    "net_total_return",
    "execution_drag_total_return",
    "volatility",
    "annualized_return",
    "gross_annualized_return",
    "annualized_volatility",
    "rolling_volatility_window",
    "rolling_volatility_latest",
    "rolling_volatility_mean",
    "rolling_volatility_max",
    "target_volatility",
    "volatility_targeting_enabled",
    "estimated_pre_target_volatility",
    "estimated_post_target_volatility",
    "volatility_scaling_factor",
    "realized_volatility",
    "latest_rolling_volatility",
    "volatility_target_scale",
    "volatility_target_scale_capped",
    "sharpe_ratio",
    "max_drawdown",
    "current_drawdown",
    "max_drawdown_duration",
    "current_drawdown_duration",
    "value_at_risk",
    "value_at_risk_confidence_level",
    "conditional_value_at_risk",
    "conditional_value_at_risk_confidence_level",
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
    "total_fixed_fee",
    "total_slippage_cost",
    "total_execution_friction",
    "average_execution_friction_per_trade",
    "average_fixed_fee_per_trade",
    "exposure_pct",
    "average_gross_exposure",
    "max_gross_exposure",
    "average_net_exposure",
    "min_net_exposure",
    "max_net_exposure",
    "average_leverage",
    "max_leverage",
    "max_single_weight",
    "max_weight_sum_deviation",
    "validation_issue_count",
    "sanity_issue_count",
    "sanity_warning_count",
    "flagged_split_count",
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
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    validation_config: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
    volatility_targeting_config: Mapping[str, Any] | None = None,
    promotion_gates: Mapping[str, Any] | None = None,
    sanity_config: SanityCheckConfig | dict[str, Any] | None = None,
    strict_mode: bool = False,
) -> dict[str, Any]:
    """Construct and score one portfolio independently for each evaluation split."""

    normalized_portfolio_name = _normalize_required_string(portfolio_name, field_name="portfolio_name")
    normalized_timeframe = _normalize_timeframe(timeframe)
    normalized_component_run_ids = _normalize_component_run_ids(component_run_ids)
    evaluation_path = Path(evaluation_config_path)
    evaluation_config = load_evaluation_config(evaluation_path)
    _validate_timeframe_compatibility(normalized_timeframe, evaluation_config)
    resolved_runtime = runtime_config or resolve_runtime_config(
        {
            "execution": None if execution_config is None else execution_config.to_dict(),
            "validation": validation_config,
            "risk": risk_config,
            "sanity": None if sanity_config is None else resolve_sanity_check_config(sanity_config).to_dict(),
        },
        cli_strict=strict_mode,
    )

    splits = generate_evaluation_splits(evaluation_config)
    if not splits:
        raise PortfolioWalkForwardError("Walk-forward portfolio evaluation did not produce any splits.")

    components, strategy_returns = _resolve_component_returns(normalized_component_run_ids)
    root_config = _build_root_config(
        portfolio_name=normalized_portfolio_name,
        allocator_name=allocator.name,
        optimizer_config=(
            allocator.optimizer_config.to_dict()
            if hasattr(allocator, "optimizer_config")
            else {"method": allocator.name}
        ),
        timeframe=normalized_timeframe,
        initial_capital=initial_capital,
        alignment_policy=alignment_policy,
        evaluation_path=evaluation_path,
        runtime_config=resolved_runtime,
        volatility_targeting_config=volatility_targeting_config,
        promotion_gates=promotion_gates,
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
    split_results = [
        _execute_portfolio_split(
            strategy_returns=strategy_returns,
            split=split,
            allocator=allocator,
            timeframe=normalized_timeframe,
            initial_capital=float(initial_capital),
            portfolio_name=normalized_portfolio_name,
            execution_config=resolved_runtime.execution,
            validation_config=resolved_runtime.portfolio_validation.to_dict(),
            risk_config=resolved_runtime.risk.to_dict(),
            volatility_targeting_config=volatility_targeting_config,
            sanity_config=resolved_runtime.sanity.to_dict(),
            strict_mode=resolved_runtime.strict_mode.enabled,
        )
        for split in splits
    ]
    aggregate_metrics = _build_aggregate_metrics(split_results, timeframe=normalized_timeframe)
    metrics_by_split = _metrics_by_split_frame(split_results)
    validate_portfolio_walk_forward_consistency(split_results, metrics_by_split, aggregate_metrics)

    experiment_dir = Path(output_dir) / run_id
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=False)

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
    promotion_evaluation = evaluate_promotion_gates(
        run_type="portfolio",
        config=_promotion_gate_config(root_config),
        sources={
            "metrics": dict(aggregate_metrics["metric_summary"]),
            "aggregate_metrics": dict(aggregate_metrics),
            "config": dict(root_config),
            "split_metrics": [dict(split_result["metrics"]) for split_result in split_results],
        },
    )
    if promotion_evaluation is not None:
        manifest["promotion_gate_summary"] = promotion_evaluation.summary()
        if DEFAULT_PROMOTION_ARTIFACT_FILENAME not in manifest["artifact_files"]:
            manifest["artifact_files"] = sorted([*manifest["artifact_files"], DEFAULT_PROMOTION_ARTIFACT_FILENAME])
        qa_group = manifest.get("artifact_groups", {}).get("qa", [])
        if isinstance(qa_group, list) and DEFAULT_PROMOTION_ARTIFACT_FILENAME not in qa_group:
            manifest["artifact_groups"]["qa"] = sorted([*qa_group, DEFAULT_PROMOTION_ARTIFACT_FILENAME])
        core_group = manifest.get("artifact_groups", {}).get("core", [])
        if isinstance(core_group, list) and DEFAULT_PROMOTION_ARTIFACT_FILENAME not in core_group:
            manifest["artifact_groups"]["core"] = sorted([*core_group, DEFAULT_PROMOTION_ARTIFACT_FILENAME])
    _write_json(experiment_dir / "manifest.json", manifest)
    write_promotion_gate_artifact(experiment_dir, promotion_evaluation)

    register_validated_portfolio_run(
        registry_path=default_registry_path(Path(output_dir)),
        run_id=run_id,
        config=root_config,
        components=components,
        metrics=dict(aggregate_metrics["metric_summary"]),
        artifact_path=experiment_dir.as_posix(),
        manifest=manifest,
        start_ts=_run_start_ts(splits),
        end_ts=_run_end_ts(splits),
        split_count=len(split_results),
        extra_metadata={
            "aggregate_metrics": aggregate_metrics,
            "promotion_gate_summary": (
                None if promotion_evaluation is None else promotion_evaluation.summary()
            ),
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
    validation_config: Mapping[str, Any] | None,
    risk_config: Mapping[str, Any] | None,
    volatility_targeting_config: Mapping[str, Any] | None,
    sanity_config: Mapping[str, Any] | None,
    strict_mode: bool,
) -> dict[str, Any]:
    train_returns, split_returns = _slice_strategy_returns_for_split(strategy_returns, split)
    optimization_returns = build_aligned_return_matrix(train_returns)
    aligned_returns = build_aligned_return_matrix(split_returns)
    if optimization_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' produced an empty aligned optimization matrix in train window."
        )
    if aligned_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' produced an empty aligned return matrix after intersection alignment."
        )

    try:
        portfolio_output = construct_portfolio(
            aligned_returns,
            allocator,
            initial_capital=initial_capital,
            execution_config=execution_config,
            validation_config=validation_config,
            optimization_returns=optimization_returns,
            risk_config=risk_config,
            volatility_targeting_config=volatility_targeting_config,
            periods_per_year=_periods_per_year_from_timeframe(timeframe),
        )
    except ValueError as exc:
        try:
            raise_research_validation_error(
                validator="portfolio_validation",
                scope=f"portfolio_walk_forward_split:{split.split_id}",
                exc=exc,
                strict_mode=strict_mode,
            )
        except ResearchStrictModeError as strict_exc:
            raise PortfolioWalkForwardError(str(strict_exc)) from strict_exc
    metrics = compute_portfolio_metrics(
        portfolio_output,
        timeframe,
        validation_config=validation_config,
        risk_config=dict(risk_config or {}),
    )
    try:
        sanity_report = validate_portfolio_output_sanity(
            portfolio_output,
            metrics,
            sanity_config,
            initial_capital=initial_capital,
            scope=f"portfolio_walk_forward_split:{split.split_id}",
        )
    except SanityCheckError as exc:
        try:
            raise_research_validation_error(
                validator="sanity",
                scope=f"portfolio_walk_forward_split:{split.split_id}",
                exc=exc,
                strict_mode=strict_mode,
            )
        except ResearchStrictModeError as strict_exc:
            raise PortfolioWalkForwardError(str(strict_exc)) from strict_exc
    metrics = sanity_report.apply_to_metrics(metrics)
    portfolio_output.attrs["sanity_check"] = sanity_report.to_dict()
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
            resolve_runtime_config().execution.to_dict()
            if execution_config is None
            else execution_config.to_dict()
        ),
        "validation": dict(validation_config or {}),
        "risk": dict(risk_config or {}),
        "volatility_targeting": dict(volatility_targeting_config or {}),
        "optimizer": portfolio_output.attrs.get("portfolio_constructor", {}).get("optimizer"),
        "sanity": dict(resolve_sanity_check_config(sanity_config).to_dict()),
    }


def _slice_strategy_returns_for_split(
    strategy_returns: pd.DataFrame,
    split: EvaluationSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_values = pd.to_datetime(strategy_returns["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    train_mask = (date_values >= split.train_start) & (date_values < split.train_end)
    test_mask = (date_values >= split.test_start) & (date_values < split.test_end)
    train_returns = strategy_returns.loc[train_mask].reset_index(drop=True)
    split_returns = strategy_returns.loc[test_mask].reset_index(drop=True)
    if train_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' has no component strategy data in train window [{split.train_start}, {split.train_end})."
        )
    if split_returns.empty:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' has no component strategy data in test window [{split.test_start}, {split.test_end})."
        )

    expected_strategies = sorted(strategy_returns["strategy_name"].astype("string").unique().tolist())
    observed_train_strategies = sorted(train_returns["strategy_name"].astype("string").unique().tolist())
    observed_strategies = sorted(split_returns["strategy_name"].astype("string").unique().tolist())
    missing_train = sorted(set(expected_strategies) - set(observed_train_strategies))
    missing = sorted(set(expected_strategies) - set(observed_strategies))
    if missing_train:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' is missing component returns for strategies {missing_train} "
            f"in train window [{split.train_start}, {split.train_end})."
        )
    if missing:
        raise PortfolioWalkForwardError(
            f"Split '{split.split_id}' is missing component returns for strategies {missing} "
            f"in test window [{split.test_start}, {split.test_end})."
        )
    return train_returns, split_returns


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

    flagged_splits = [
        str(split_result["split_id"])
        for split_result in split_results
        if float(split_result["metrics"].get("sanity_issue_count", 0.0) or 0.0) > 0.0
    ]
    failed_splits = [
        str(split_result["split_id"])
        for split_result in split_results
        if split_result["metrics"].get("sanity_status") == "fail"
    ]
    metric_summary["flagged_split_count"] = float(len(flagged_splits))
    metric_statistics["flagged_split_count"] = {
        "mean": float(len(flagged_splits)),
        "median": float(len(flagged_splits)),
        "std": 0.0,
        "min": float(len(flagged_splits)),
        "max": float(len(flagged_splits)),
    }

    return {
        "aggregation_method": "descriptive statistics computed across split-level portfolio metrics in split order",
        "mode": first_split["mode"],
        "split_count": len(split_results),
        "timeframe": timeframe,
        "first_train_start": first_split["train_start"],
        "last_test_end": last_split["test_end"],
        "split_ids": [str(split_result["split_id"]) for split_result in split_results],
        "sanity": {
            "status": "fail" if failed_splits else ("warn" if flagged_splits else "pass"),
            "flagged_split_count": len(flagged_splits),
            "failed_split_count": len(failed_splits),
            "flagged_splits": flagged_splits,
            "failed_splits": failed_splits,
        },
        "metric_summary": metric_summary,
        "metric_statistics": metric_statistics,
    }


def _write_split_artifacts(*, split_dir: Path, split_result: Mapping[str, Any]) -> None:
    portfolio_output = split_result["portfolio_output"]
    normalized_metrics = _normalize_mapping(dict(split_result["metrics"]), owner="metrics")
    split_config = {
        "portfolio_name": split_result["portfolio_name"],
        "allocator": split_result["allocator_name"],
        "optimizer": split_result.get("optimizer"),
        "initial_capital": float(split_result["initial_capital"]),
        "execution": dict(split_result.get("execution", {})),
        "timeframe": split_result["timeframe"],
        "validation": dict(split_result.get("validation", {})),
        "risk": dict(split_result.get("risk", {})),
        "volatility_targeting": dict(split_result.get("volatility_targeting", {})),
    }
    weights_frame = _weights_frame(portfolio_output)
    returns_frame = _portfolio_returns_frame(portfolio_output)
    equity_frame = _portfolio_equity_curve_frame(portfolio_output)
    qa_summary = run_portfolio_qa(
        portfolio_output,
        dict(split_result["metrics"]),
        split_config,
        run_id=split_dir.name,
        strict=True,
    )
    validate_portfolio_artifact_payload_consistency(
        portfolio_output=portfolio_output,
        weights_frame=weights_frame,
        returns_frame=returns_frame,
        equity_frame=equity_frame,
        metrics=normalized_metrics,
        qa_summary=qa_summary,
        config=split_config,
        components=[
            {
                "strategy_name": column.removeprefix("weight__"),
                "run_id": column.removeprefix("weight__"),
            }
            for column in weights_frame.columns
            if column.startswith("weight__")
        ],
    )
    _write_json(split_dir / "split.json", dict(split_result["split_metadata"]))
    _write_csv(split_dir / "weights.csv", weights_frame)
    _write_csv(split_dir / "portfolio_returns.csv", returns_frame)
    _write_csv(split_dir / "portfolio_equity_curve.csv", equity_frame)
    _write_json(split_dir / "metrics.json", normalized_metrics)
    _write_json(split_dir / "qa_summary.json", qa_summary)


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
        "optimizer": config.get("optimizer"),
        "optimizer_method": None
        if not isinstance(config.get("optimizer"), dict)
        else config["optimizer"].get("method"),
        "evaluation_mode": "walk_forward",
        "evaluation_config_path": config["evaluation_config_path"],
        "strict_mode": config.get("strict_mode"),
        "component_count": len(components),
        "split_count": len(split_results),
        "split_artifact_dirs": list(split_artifact_dirs),
        "artifact_files": artifact_files,
        "artifact_groups": {
            "core": sorted(
                [
                    "aggregate_metrics.json",
                    "components.json",
                    "config.json",
                    "manifest.json",
                    "metrics_by_split.csv",
                ]
            ),
            "metrics": ["aggregate_metrics.json", "metrics_by_split.csv"],
            "qa": sorted(
                f"{split_artifact_dir}/qa_summary.json"
                for split_artifact_dir in split_artifact_dirs
            ),
            "risk": [],
            "simulation": [],
            "walk_forward": list(split_artifact_dirs),
        },
        "aggregate_metric_summary": canonicalize_value(dict(aggregate_metrics["metric_summary"])),
        "aggregate_metrics_path": "aggregate_metrics.json",
        "metrics_by_split_path": "metrics_by_split.csv",
        "config_snapshot": {
            "execution": config.get("execution"),
            "risk": config.get("risk"),
            "runtime": config.get("runtime"),
            "sanity": config.get("sanity"),
            "strict_mode": config.get("strict_mode"),
            "validation": config.get("validation"),
            "volatility_targeting": config.get("volatility_targeting"),
        },
        "execution": {
            "config": config.get("execution"),
            "summary": {
                "gross_total_return": aggregate_metrics["metric_summary"].get("gross_total_return"),
                "net_total_return": aggregate_metrics["metric_summary"].get("net_total_return"),
                "execution_drag_total_return": aggregate_metrics["metric_summary"].get("execution_drag_total_return"),
                "total_transaction_cost": aggregate_metrics["metric_summary"].get("total_transaction_cost"),
                "total_fixed_fee": aggregate_metrics["metric_summary"].get("total_fixed_fee"),
                "total_slippage_cost": aggregate_metrics["metric_summary"].get("total_slippage_cost"),
                "total_execution_friction": aggregate_metrics["metric_summary"].get("total_execution_friction"),
            },
        },
        "risk": {
            "config": config.get("risk"),
            "summary": {
                "conditional_value_at_risk": aggregate_metrics["metric_summary"].get("conditional_value_at_risk"),
                "conditional_value_at_risk_confidence_level": aggregate_metrics["metric_summary"].get(
                    "conditional_value_at_risk_confidence_level"
                ),
                "estimated_post_target_volatility": aggregate_metrics["metric_summary"].get(
                    "estimated_post_target_volatility"
                ),
                "estimated_pre_target_volatility": aggregate_metrics["metric_summary"].get(
                    "estimated_pre_target_volatility"
                ),
                "max_drawdown": aggregate_metrics["metric_summary"].get("max_drawdown"),
                "realized_volatility": aggregate_metrics["metric_summary"].get("realized_volatility"),
                "target_volatility": aggregate_metrics["metric_summary"].get("target_volatility"),
                "value_at_risk": aggregate_metrics["metric_summary"].get("value_at_risk"),
                "value_at_risk_confidence_level": aggregate_metrics["metric_summary"].get(
                    "value_at_risk_confidence_level"
                ),
                "volatility_scaling_factor": aggregate_metrics["metric_summary"].get(
                    "volatility_scaling_factor"
                ),
                "volatility_targeting_enabled": aggregate_metrics["metric_summary"].get(
                    "volatility_targeting_enabled"
                ),
            },
        },
        "simulation": {
            "artifact_path": None,
            "enabled": False,
            "method": None,
            "num_paths": None,
            "path_length": None,
            "probability_of_loss": None,
            "seed": None,
            "summary_path": None,
        },
    }


def _build_root_config(
    *,
    portfolio_name: str,
    allocator_name: str,
    optimizer_config: Mapping[str, Any],
    timeframe: str,
    initial_capital: float,
    alignment_policy: str,
    evaluation_path: Path,
    runtime_config: RuntimeConfig,
    volatility_targeting_config: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "portfolio_name": portfolio_name,
        "allocator": allocator_name,
        "initial_capital": float(initial_capital),
        "alignment_policy": _normalize_required_string(alignment_policy, field_name="alignment_policy"),
        "optimizer": dict(optimizer_config),
        "execution": runtime_config.execution.to_dict(),
        "timeframe": timeframe,
        "evaluation_config_path": evaluation_path.as_posix(),
        "validation": runtime_config.portfolio_validation.to_dict(),
        "risk": runtime_config.risk.to_dict(),
        "volatility_targeting": None if volatility_targeting_config is None else dict(volatility_targeting_config),
        "sanity": runtime_config.sanity.to_dict(),
        "strict_mode": runtime_config.strict_mode.to_dict(),
        "runtime": runtime_config.to_dict(),
    }
    if promotion_gates is not None:
        payload["promotion_gates"] = dict(promotion_gates)
    return payload


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


def _promotion_gate_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = config.get("promotion_gates")
    return dict(payload) if isinstance(payload, Mapping) else None


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


def _periods_per_year_from_timeframe(timeframe: str) -> int:
    normalized = _normalize_timeframe(timeframe)
    if normalized == "1D":
        return TRADING_DAYS_PER_YEAR
    if normalized == "1Min":
        return MINUTE_PERIODS_PER_YEAR
    raise PortfolioWalkForwardError(f"Unsupported portfolio timeframe: {timeframe!r}.")


def _validate_timeframe_compatibility(timeframe: str, evaluation_config: EvaluationConfig) -> None:
    evaluation_timeframe = _normalize_timeframe(evaluation_config.timeframe)
    if timeframe != evaluation_timeframe:
        raise PortfolioWalkForwardError(
            "Portfolio timeframe and evaluation timeframe must match. "
            f"Received portfolio timeframe {timeframe!r} and evaluation timeframe {evaluation_config.timeframe!r}."
        )


__all__ = ["PORTFOLIO_WALK_FORWARD_METRIC_KEYS", "PortfolioWalkForwardError", "run_portfolio_walk_forward"]
