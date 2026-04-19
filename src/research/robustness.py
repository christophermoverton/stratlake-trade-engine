from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.evaluation import EVALUATION_CONFIG
from src.config.execution import ExecutionConfig
from src.config.robustness import (
    MetricThreshold,
    RobustnessConfig,
)
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.config.settings import load_yaml_config
from src.data.load_features import load_features
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import (
    save_robustness_experiment,
)
from src.research.metrics import (
    compute_benchmark_relative_metrics,
    compute_performance_metrics,
)
from src.research.registry import serialize_canonical_json
from src.research.sanity import SanityCheckError, validate_strategy_backtest_sanity
from src.research.signal_engine import generate_signals
from src.research.splits import generate_evaluation_splits
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.strategies import build_strategy
from src.research.walk_forward import (
    WalkForwardExecutionError,
    build_aggregate_summary,
    execute_split,
    load_walk_forward_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"
_LOWER_IS_BETTER_METRICS = {
    "max_drawdown",
    "relative_drawdown",
    "volatility",
    "annualized_volatility",
    "total_transaction_cost",
    "total_slippage_cost",
    "total_execution_friction",
}


class RobustnessAnalysisError(ValueError):
    """Raised when deterministic robustness analysis cannot be completed safely."""


@dataclass(frozen=True)
class StrategyVariant:
    """Deterministic parameter variant produced by sweep expansion."""

    variant_id: str
    variant_label: str
    parameter_values: dict[str, Any]
    strategy_config: dict[str, Any]
    order_index: int

    def to_record(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_label": self.variant_label,
            "variant_order": self.order_index,
            **self.parameter_values,
        }


@dataclass(frozen=True)
class RobustnessRunResult:
    """Structured result returned by the centralized robustness runner."""

    strategy_name: str
    run_id: str
    experiment_dir: Path
    summary: dict[str, Any]
    variants: list[StrategyVariant]
    variant_metrics: pd.DataFrame
    stability_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    neighbor_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _PeriodDefinition:
    period_id: str
    period_mode: str
    period_start: str
    period_end: str
    mask: pd.Series

    def to_record(self) -> dict[str, Any]:
        return {
            "period_id": self.period_id,
            "period_mode": self.period_mode,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }


def load_strategy_config(strategy_name: str, path: Path = STRATEGIES_CONFIG) -> dict[str, Any]:
    """Load one strategy config from the shared registry YAML."""

    payload = load_yaml_config(path)
    if not isinstance(payload, dict):
        raise RobustnessAnalysisError("Strategy configuration must be a mapping of strategy names to config dictionaries.")
    config = payload.get(strategy_name)
    if not isinstance(config, dict):
        available = ", ".join(sorted(payload)) or "<none>"
        raise RobustnessAnalysisError(
            f"Unknown strategy '{strategy_name}'. Available strategies: {available}."
        )
    return config


def expand_parameter_sweeps(
    strategy_name: str,
    strategy_config: dict[str, Any],
    robustness_config: RobustnessConfig,
) -> list[StrategyVariant]:
    """Expand ordered parameter sweeps into deterministic strategy variants."""

    parameters = strategy_config.get("parameters")
    if not isinstance(parameters, dict):
        raise RobustnessAnalysisError(f"Strategy '{strategy_name}' parameters must be a dictionary.")

    if not robustness_config.sweep:
        raise RobustnessAnalysisError("Robustness configuration must define at least one sweep entry.")

    normalized_sweeps: list[tuple[str, tuple[Any, ...]]] = []
    for definition in robustness_config.sweep:
        parameter = definition.parameter
        if parameter not in parameters:
            raise RobustnessAnalysisError(
                f"Robustness sweep parameter '{parameter}' is not defined for strategy '{strategy_name}'."
            )
        base_value = parameters[parameter]
        normalized_values = tuple(_normalize_sweep_value(parameter, value, base_value) for value in definition.values)
        if len(set(_canonical_variant_value(value) for value in normalized_values)) != len(normalized_values):
            raise RobustnessAnalysisError(f"Robustness sweep parameter '{parameter}' contains duplicate values.")
        normalized_sweeps.append((parameter, normalized_values))

    variants: list[StrategyVariant] = []
    seen_signatures: set[str] = set()
    sweep_parameters = [parameter for parameter, _ in normalized_sweeps]
    sweep_values = [values for _, values in normalized_sweeps]
    for index, value_tuple in enumerate(product(*sweep_values)):
        parameter_values = dict(zip(sweep_parameters, value_tuple, strict=True))
        variant_parameters = dict(parameters)
        variant_parameters.update(parameter_values)
        signature = serialize_canonical_json(variant_parameters)
        if signature in seen_signatures:
            raise RobustnessAnalysisError("Robustness sweep expansion produced duplicate strategy variants.")
        seen_signatures.add(signature)
        variants.append(
            StrategyVariant(
                variant_id=f"variant_{index:04d}_{_variant_slug(parameter_values)}",
                variant_label=_variant_label(parameter_values),
                parameter_values=parameter_values,
                strategy_config={**strategy_config, "parameters": variant_parameters},
                order_index=index,
            )
        )

    if not variants:
        raise RobustnessAnalysisError("Robustness sweep expansion did not produce any strategy variants.")
    return variants


def run_robustness_experiment(
    strategy_name: str | None,
    *,
    robustness_config: RobustnessConfig,
    start: str | None = None,
    end: str | None = None,
    evaluation_path: Path | None = None,
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
    strategy_config_path: Path = STRATEGIES_CONFIG,
) -> RobustnessRunResult:
    """Run deterministic robustness analysis over a strategy parameter sweep."""

    if robustness_config.is_extended_sweep:
        from src.research.extended_robustness import run_extended_robustness_experiment

        return run_extended_robustness_experiment(
            strategy_name,
            robustness_config=robustness_config,
            start=start,
            end=end,
            evaluation_path=evaluation_path,
            runtime_config=runtime_config,
            execution_config=execution_config,
            strict=strict,
            strategy_config_path=strategy_config_path,
        )

    resolved_strategy_name = robustness_config.resolve_strategy_name(strategy_name)
    strategy_config = load_strategy_config(resolved_strategy_name, path=strategy_config_path)
    variants = expand_parameter_sweeps(resolved_strategy_name, strategy_config, robustness_config)
    resolved_runtime = runtime_config or resolve_runtime_config(
        strategy_config,
        cli_overrides=None if execution_config is None else {"execution": execution_config.to_dict()},
        cli_strict=strict,
    )
    higher_is_better = _resolve_metric_direction(
        robustness_config.ranking_metric,
        robustness_config.higher_is_better,
    )

    base_strategy = build_strategy(resolved_strategy_name, strategy_config)
    if robustness_config.stability.mode == "walk_forward":
        variant_metrics, stability_metrics = _run_walk_forward_robustness(
            resolved_strategy_name,
            base_strategy.dataset,
            variants,
            robustness_config,
            resolved_runtime,
            evaluation_path=evaluation_path,
        )
    else:
        variant_metrics, stability_metrics = _run_single_window_robustness(
            resolved_strategy_name,
            base_strategy.dataset,
            variants,
            robustness_config,
            resolved_runtime,
            start=start,
            end=end,
        )

    variant_metrics = _rank_variant_metrics(variant_metrics, robustness_config.ranking_metric, higher_is_better)
    stability_metrics = _rank_stability_metrics(stability_metrics, robustness_config.ranking_metric, higher_is_better)
    neighbor_metrics = _build_neighbor_metrics(variants, variant_metrics, robustness_config.ranking_metric)
    summary = _build_robustness_summary(
        strategy_name=resolved_strategy_name,
        robustness_config=robustness_config,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        neighbor_metrics=neighbor_metrics,
        higher_is_better=higher_is_better,
    )
    _validate_robustness_outputs(variant_metrics, stability_metrics, neighbor_metrics, summary, robustness_config)

    artifact_config = resolved_runtime.apply_to_payload(
        {
            "strategy_name": resolved_strategy_name,
            "dataset": base_strategy.dataset,
            "parameters": dict(strategy_config.get("parameters", {})),
            "start": start,
            "end": end,
            "robustness": robustness_config.to_dict(),
        },
        include_validation_section=False,
    )
    experiment_dir = save_robustness_experiment(
        resolved_strategy_name,
        config=artifact_config,
        variants=variants,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        neighbor_metrics=neighbor_metrics,
        summary=summary,
    )
    return RobustnessRunResult(
        strategy_name=resolved_strategy_name,
        run_id=experiment_dir.name,
        experiment_dir=experiment_dir,
        summary=summary,
        variants=variants,
        variant_metrics=variant_metrics,
        stability_metrics=stability_metrics,
        neighbor_metrics=neighbor_metrics,
    )


def _run_single_window_robustness(
    strategy_name: str,
    dataset_name: str,
    variants: list[StrategyVariant],
    robustness_config: RobustnessConfig,
    runtime_config: RuntimeConfig,
    *,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_features(dataset_name, start=start, end=end)
    if dataset.empty:
        raise RobustnessAnalysisError(
            f"Feature dataset '{dataset_name}' returned no rows for robustness window [{start}, {end})."
        )

    benchmark_results = _build_benchmark_results(dataset, dataset_name, runtime_config.execution)
    periods = _generate_subperiods(dataset, robustness_config)
    variant_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for variant in variants:
        try:
            strategy = build_strategy(strategy_name, variant.strategy_config)
            signal_frame = generate_signals(dataset, strategy)
            results_df = run_backtest(
                signal_frame,
                runtime_config.execution,
                require_managed_signals=True,
            )
            metrics = compute_performance_metrics(results_df)
            metrics.update(compute_benchmark_relative_metrics(results_df, benchmark_results))
            metrics = _apply_sanity(
                metrics,
                results_df,
                runtime_config,
                scope=f"robustness:{variant.variant_id}",
            )
        except (SanityCheckError, ResearchStrictModeError, ValueError, WalkForwardExecutionError) as exc:
            raise RobustnessAnalysisError(
                f"Robustness variant '{variant.variant_id}' failed: {exc}"
            ) from exc

        variant_rows.append({**variant.to_record(), **metrics})
        for period in periods:
            period_df = results_df.loc[period.mask].reset_index(drop=True)
            if period_df.empty:
                raise RobustnessAnalysisError(
                    f"Robustness period '{period.period_id}' produced no rows for variant '{variant.variant_id}'."
                )
            period_metrics = compute_performance_metrics(period_df)
            stability_rows.append({**variant.to_record(), **period.to_record(), **period_metrics})

    return pd.DataFrame(variant_rows), pd.DataFrame(stability_rows)


def _run_walk_forward_robustness(
    strategy_name: str,
    dataset_name: str,
    variants: list[StrategyVariant],
    robustness_config: RobustnessConfig,
    runtime_config: RuntimeConfig,
    *,
    evaluation_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_evaluation_path = evaluation_path or (
        None if robustness_config.stability.evaluation_path is None else Path(robustness_config.stability.evaluation_path)
    ) or EVALUATION_CONFIG
    evaluation_config = load_walk_forward_config(resolved_evaluation_path)
    splits = generate_evaluation_splits(evaluation_config)
    if not splits:
        raise RobustnessAnalysisError("Robustness walk-forward stability did not produce any splits.")

    start = min(split.train_start for split in splits)
    end = max(split.test_end for split in splits)
    dataset = load_features(dataset_name, start=start, end=end)
    if dataset.empty:
        raise RobustnessAnalysisError(
            f"Feature dataset '{dataset_name}' returned no rows for robustness walk-forward window [{start}, {end})."
        )

    variant_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for variant in variants:
        try:
            strategy = build_strategy(strategy_name, variant.strategy_config)
            split_results = [
                execute_split(
                    strategy,
                    dataset,
                    split,
                    execution_config=runtime_config.execution,
                    sanity_config=runtime_config.sanity,
                    strict_mode=runtime_config.strict_mode.enabled,
                )
                for split in splits
            ]
            aggregate_summary = build_aggregate_summary(split_results)
        except (ValueError, WalkForwardExecutionError) as exc:
            raise RobustnessAnalysisError(
                f"Robustness variant '{variant.variant_id}' failed: {exc}"
            ) from exc

        variant_rows.append({**variant.to_record(), **aggregate_summary})
        for split_result in split_results:
            stability_rows.append(
                {
                    **variant.to_record(),
                    "period_id": split_result.split_id,
                    "period_mode": split_result.mode,
                    "period_start": split_result.test_start,
                    "period_end": split_result.test_end,
                    "train_start": split_result.train_start,
                    "train_end": split_result.train_end,
                    **split_result.metrics,
                }
            )
    return pd.DataFrame(variant_rows), pd.DataFrame(stability_rows)


def _apply_sanity(
    metrics: dict[str, Any],
    results_df: pd.DataFrame,
    runtime_config: RuntimeConfig,
    *,
    scope: str,
) -> dict[str, Any]:
    try:
        sanity_report = validate_strategy_backtest_sanity(results_df, metrics, runtime_config.sanity, scope=scope)
    except SanityCheckError as exc:
        try:
            raise_research_validation_error(
                validator="sanity",
                scope=scope,
                exc=exc,
                strict_mode=runtime_config.strict_mode.enabled,
            )
        except ResearchStrictModeError as strict_exc:
            raise strict_exc from strict_exc
        raise
    return sanity_report.apply_to_metrics(metrics)


def _generate_subperiods(dataset: pd.DataFrame, robustness_config: RobustnessConfig) -> list[_PeriodDefinition]:
    if robustness_config.stability.mode != "subperiods":
        return []

    key_column = "date" if "date" in dataset.columns else "ts_utc" if "ts_utc" in dataset.columns else None
    if key_column is None:
        raise RobustnessAnalysisError("Robustness subperiod stability requires a 'date' or 'ts_utc' column.")

    time_values = dataset[key_column].astype("string")
    unique_values = list(dict.fromkeys(time_values.tolist()))
    period_count = int(robustness_config.stability.periods or 0)
    if len(unique_values) < period_count:
        raise RobustnessAnalysisError(
            "Robustness subperiod stability requires at least as many distinct timestamps as configured periods."
        )

    periods: list[_PeriodDefinition] = []
    base_width, remainder = divmod(len(unique_values), period_count)
    start_index = 0
    for index in range(period_count):
        width = base_width + (1 if index < remainder else 0)
        end_index = start_index + width
        period_values = unique_values[start_index:end_index]
        if not period_values:
            raise RobustnessAnalysisError("Robustness subperiod generation produced an empty period.")
        periods.append(
            _PeriodDefinition(
                period_id=f"period_{index:04d}",
                period_mode="subperiods",
                period_start=str(period_values[0]),
                period_end=str(period_values[-1]),
                mask=time_values.isin(period_values),
            )
        )
        start_index = end_index
    return periods


def _rank_variant_metrics(frame: pd.DataFrame, ranking_metric: str, higher_is_better: bool) -> pd.DataFrame:
    if frame.empty:
        raise RobustnessAnalysisError("Robustness analysis produced no variant metrics.")
    if ranking_metric not in frame.columns:
        raise RobustnessAnalysisError(f"Robustness ranking metric '{ranking_metric}' is missing from variant metrics.")

    ordered = frame.copy()
    ordered["_metric_order"] = pd.to_numeric(ordered[ranking_metric], errors="coerce")
    if ordered["_metric_order"].isna().any():
        raise RobustnessAnalysisError(f"Robustness ranking metric '{ranking_metric}' must be finite for all variants.")
    ordered = ordered.sort_values(
        by=["_metric_order", "variant_order", "variant_id"],
        ascending=[not higher_is_better, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["rank"] = pd.Series(range(1, len(ordered) + 1), dtype="int64")
    ordered["is_best_variant"] = ordered["rank"] == 1
    return ordered.drop(columns="_metric_order")


def _rank_stability_metrics(frame: pd.DataFrame, ranking_metric: str, higher_is_better: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if ranking_metric not in frame.columns:
        raise RobustnessAnalysisError(
            f"Robustness ranking metric '{ranking_metric}' is missing from stability metrics."
        )
    ordered = frame.copy()
    ordered[ranking_metric] = pd.to_numeric(ordered[ranking_metric], errors="coerce")
    if ordered[ranking_metric].isna().any():
        raise RobustnessAnalysisError(
            f"Robustness stability metrics must contain finite '{ranking_metric}' values."
        )
    ordered["_row_order"] = pd.Series(range(len(ordered)), dtype="int64")
    ordered = ordered.sort_values(
        by=["period_id", ranking_metric, "_row_order"],
        ascending=[True, not higher_is_better, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["period_rank"] = ordered.groupby("period_id", sort=False).cumcount() + 1
    return ordered.drop(columns="_row_order")


def _build_neighbor_metrics(
    variants: list[StrategyVariant],
    variant_metrics: pd.DataFrame,
    ranking_metric: str,
) -> pd.DataFrame:
    metric_by_variant = {
        str(record["variant_id"]): float(record[ranking_metric])
        for record in variant_metrics.to_dict(orient="records")
    }
    rows: list[dict[str, Any]] = []
    grouped_by_parameter: dict[str, list[Any]] = {}
    for variant in variants:
        for parameter, value in variant.parameter_values.items():
            grouped_by_parameter.setdefault(parameter, [])
            if _canonical_variant_value(value) not in {
                _canonical_variant_value(existing) for existing in grouped_by_parameter[parameter]
            }:
                grouped_by_parameter[parameter].append(value)

    for parameter, ordered_values in grouped_by_parameter.items():
        if len(ordered_values) < 2:
            continue
        sibling_groups: dict[str, dict[str, StrategyVariant]] = {}
        for variant in variants:
            key_payload = {
                key: value
                for key, value in variant.parameter_values.items()
                if key != parameter
            }
            sibling_groups.setdefault(serialize_canonical_json(key_payload), {})[
                _canonical_variant_value(variant.parameter_values[parameter])
            ] = variant

        for siblings in sibling_groups.values():
            for left_value, right_value in zip(ordered_values, ordered_values[1:], strict=False):
                left_variant = siblings.get(_canonical_variant_value(left_value))
                right_variant = siblings.get(_canonical_variant_value(right_value))
                if left_variant is None or right_variant is None:
                    continue
                rows.append(
                    {
                        "parameter": parameter,
                        "left_variant_id": left_variant.variant_id,
                        "right_variant_id": right_variant.variant_id,
                        "left_value": left_value,
                        "right_value": right_value,
                        "metric_gap": abs(
                            metric_by_variant[left_variant.variant_id] - metric_by_variant[right_variant.variant_id]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _build_robustness_summary(
    *,
    strategy_name: str,
    robustness_config: RobustnessConfig,
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    neighbor_metrics: pd.DataFrame,
    higher_is_better: bool,
) -> dict[str, Any]:
    best_row = variant_metrics.iloc[0]
    ranking_metric = robustness_config.ranking_metric
    metric_values = pd.to_numeric(variant_metrics[ranking_metric], errors="raise").astype("float64")
    threshold_flags = _threshold_flags(variant_metrics, robustness_config.thresholds)
    worst_metric_value = float(metric_values.iloc[-1])
    return {
        "strategy_name": strategy_name,
        "ranking_metric": ranking_metric,
        "higher_is_better": higher_is_better,
        "variant_count": int(len(variant_metrics)),
        "best_variant_id": str(best_row["variant_id"]),
        "best_variant_label": str(best_row["variant_label"]),
        "best_metric_value": float(best_row[ranking_metric]),
        "worst_metric_value": worst_metric_value,
        "metric_spread": float(metric_values.max() - metric_values.min()),
        "metric_median": float(metric_values.median()),
        "neighbor_pair_count": int(len(neighbor_metrics)),
        "neighbor_gap_mean": None if neighbor_metrics.empty else float(pd.to_numeric(neighbor_metrics["metric_gap"]).mean()),
        "neighbor_gap_max": None if neighbor_metrics.empty else float(pd.to_numeric(neighbor_metrics["metric_gap"]).max()),
        "best_variant_neighbor_gap": _best_variant_neighbor_gap(str(best_row["variant_id"]), neighbor_metrics),
        "split_count": 0 if stability_metrics.empty else int(stability_metrics["period_id"].nunique()),
        "stability_mode": robustness_config.stability.mode,
        "threshold_pass_count": None if threshold_flags is None else int(threshold_flags.sum()),
        "threshold_pass_rate": None if threshold_flags is None else float(threshold_flags.mean()),
        "best_variant_split_win_rate": _best_variant_split_win_rate(str(best_row["variant_id"]), stability_metrics),
        "best_variant_top_half_rate": _best_variant_top_half_rate(str(best_row["variant_id"]), stability_metrics),
        "rank_consistency_mean": _rank_consistency_mean(variant_metrics, stability_metrics),
    }


def _threshold_flags(frame: pd.DataFrame, thresholds: tuple[MetricThreshold, ...]) -> pd.Series | None:
    if not thresholds:
        return None
    flags = pd.Series(True, index=frame.index, dtype="bool")
    for threshold in thresholds:
        if threshold.metric not in frame.columns:
            raise RobustnessAnalysisError(
                f"Robustness threshold metric '{threshold.metric}' is missing from variant metrics."
            )
        values = pd.to_numeric(frame[threshold.metric], errors="coerce")
        if values.isna().any():
            raise RobustnessAnalysisError(
                f"Robustness threshold metric '{threshold.metric}' must be finite for all variants."
            )
        if threshold.min_value is not None:
            flags &= values >= threshold.min_value
        if threshold.max_value is not None:
            flags &= values <= threshold.max_value
    return flags


def _best_variant_neighbor_gap(variant_id: str, neighbor_metrics: pd.DataFrame) -> float | None:
    if neighbor_metrics.empty:
        return None
    subset = neighbor_metrics.loc[
        (neighbor_metrics["left_variant_id"].astype("string") == variant_id)
        | (neighbor_metrics["right_variant_id"].astype("string") == variant_id)
    ]
    if subset.empty:
        return None
    return float(pd.to_numeric(subset["metric_gap"]).max())


def _best_variant_split_win_rate(variant_id: str, stability_metrics: pd.DataFrame) -> float | None:
    if stability_metrics.empty:
        return None
    subset = stability_metrics.loc[stability_metrics["variant_id"].astype("string") == variant_id]
    if subset.empty:
        return None
    return float((pd.to_numeric(subset["period_rank"]) == 1).mean())


def _best_variant_top_half_rate(variant_id: str, stability_metrics: pd.DataFrame) -> float | None:
    if stability_metrics.empty:
        return None
    counts = stability_metrics.groupby("period_id", sort=False)["variant_id"].size()
    if counts.empty:
        return None
    threshold_by_period = counts.apply(lambda count: max(1, math.ceil(int(count) / 2)))
    subset = stability_metrics.loc[stability_metrics["variant_id"].astype("string") == variant_id].copy()
    if subset.empty:
        return None
    subset["top_half_threshold"] = subset["period_id"].map(threshold_by_period)
    return float((pd.to_numeric(subset["period_rank"]) <= subset["top_half_threshold"]).mean())


def _rank_consistency_mean(variant_metrics: pd.DataFrame, stability_metrics: pd.DataFrame) -> float | None:
    if stability_metrics.empty or len(variant_metrics) < 2:
        return None
    overall_ranks = variant_metrics.set_index("variant_id")["rank"].astype("float64")
    correlations: list[float] = []
    for _, period_frame in stability_metrics.groupby("period_id", sort=False):
        period_ranks = period_frame.set_index("variant_id")["period_rank"].astype("float64")
        aligned = overall_ranks.to_frame("overall").join(period_ranks.rename("period"), how="inner")
        if len(aligned) < 2:
            continue
        correlation = aligned["overall"].corr(aligned["period"], method="pearson")
        if correlation is None or pd.isna(correlation):
            continue
        correlations.append(float(correlation))
    if not correlations:
        return None
    return float(sum(correlations) / len(correlations))


def _resolve_metric_direction(metric_name: str, configured: bool | None) -> bool:
    if configured is not None:
        return configured
    return metric_name not in _LOWER_IS_BETTER_METRICS


def _build_benchmark_results(
    dataset: pd.DataFrame,
    dataset_name: str,
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    benchmark_strategy = build_strategy("buy_and_hold_v1", {"dataset": dataset_name, "parameters": {}})
    benchmark_signal_frame = generate_signals(dataset, benchmark_strategy)
    return run_backtest(
        benchmark_signal_frame,
        execution_config,
        require_managed_signals=True,
    )


def _normalize_sweep_value(parameter: str, value: Any, base_value: Any) -> Any:
    if isinstance(base_value, bool):
        if not isinstance(value, bool):
            raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be boolean.")
        return value
    if isinstance(base_value, int) and not isinstance(base_value, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be an integer.")
        return value
    if isinstance(base_value, float):
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be numeric.")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be finite.")
        return normalized
    if isinstance(base_value, str):
        if not isinstance(value, str) or not value.strip():
            raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be a non-empty string.")
        return value
    if not isinstance(value, int | float | str | bool):
        raise RobustnessAnalysisError(
            f"Robustness sweep value for '{parameter}' must be a JSON-serializable scalar."
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise RobustnessAnalysisError(f"Robustness sweep value for '{parameter}' must be finite.")
    return value


def _variant_slug(parameter_values: dict[str, Any]) -> str:
    parts = []
    for key, value in parameter_values.items():
        key_slug = "".join(char if char.isalnum() else "_" for char in key).strip("_") or "param"
        value_slug = "".join(char if char.isalnum() else "_" for char in str(value)).strip("_") or "value"
        parts.append(f"{key_slug}_{value_slug}")
    return "__".join(parts)


def _variant_label(parameter_values: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in parameter_values.items())


def _canonical_variant_value(value: Any) -> str:
    return serialize_canonical_json(value)


def _validate_robustness_outputs(
    variant_metrics: pd.DataFrame,
    stability_metrics: pd.DataFrame,
    neighbor_metrics: pd.DataFrame,
    summary: dict[str, Any],
    robustness_config: RobustnessConfig,
) -> None:
    if variant_metrics["variant_id"].astype("string").duplicated().any():
        raise RobustnessAnalysisError("Robustness variant metrics must contain unique variant_id values.")
    if robustness_config.ranking_metric not in variant_metrics.columns:
        raise RobustnessAnalysisError(
            f"Robustness variant metrics are missing ranking metric '{robustness_config.ranking_metric}'."
        )
    if not pd.to_numeric(variant_metrics[robustness_config.ranking_metric], errors="coerce").notna().all():
        raise RobustnessAnalysisError(
            f"Robustness variant metrics contain non-finite '{robustness_config.ranking_metric}' values."
        )
    if not stability_metrics.empty and stability_metrics["period_id"].astype("string").eq("").any():
        raise RobustnessAnalysisError("Robustness stability metrics must contain non-empty period_id values.")
    if not neighbor_metrics.empty and pd.to_numeric(neighbor_metrics["metric_gap"], errors="coerce").isna().any():
        raise RobustnessAnalysisError("Robustness neighbor metrics must contain finite metric_gap values.")
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RobustnessAnalysisError(f"Robustness summary field '{key}' must be finite.")
