from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.execution import ExecutionConfig
from src.config.evaluation import EVALUATION_CONFIG, EvaluationConfig, load_evaluation_config
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.config.sanity import SanityCheckConfig
from src.data.load_features import load_features
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.signal_engine import generate_signals
from src.research.sanity import SanityCheckError, validate_strategy_backtest_sanity
from src.research.splits import EvaluationSplit, generate_evaluation_splits
from src.research.strategy_qa import generate_strategy_qa_summary
from src.research.strategy_base import BaseStrategy


class WalkForwardExecutionError(ValueError):
    """Raised when a walk-forward run cannot be executed safely."""


@dataclass(frozen=True)
class SplitExecutionResult:
    """Structured execution result for a single walk-forward split."""

    split_id: str
    mode: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    split_rows: int
    train_rows: int
    test_rows: int
    metrics: dict[str, float | None]
    results_df: pd.DataFrame
    sanity: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return serializable split metadata and metrics for tabular logging."""

        return {
            "split_id": self.split_id,
            "mode": self.mode,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "split_rows": self.split_rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            **self.metrics,
        }


@dataclass(frozen=True)
class WalkForwardRunResult:
    """Structured result returned from a full walk-forward strategy run."""

    strategy_name: str
    run_id: str
    experiment_dir: Path
    metrics: dict[str, float | None]
    aggregate_summary: dict[str, Any]
    splits: list[SplitExecutionResult]
    signal_diagnostics: dict[str, Any] = field(default_factory=dict)
    qa_summary: dict[str, Any] = field(default_factory=dict)
    sanity_summary: dict[str, Any] = field(default_factory=dict)


def compute_metrics(results_df: pd.DataFrame) -> dict[str, float | None]:
    """Compute the standard research metrics for a backtest results frame."""

    return compute_performance_metrics(results_df)


def load_walk_forward_config(path: Path | None = None) -> EvaluationConfig:
    """Load the evaluation configuration used for walk-forward execution."""

    return load_evaluation_config(path or EVALUATION_CONFIG)


def run_walk_forward_experiment(
    strategy_name: str,
    strategy: BaseStrategy,
    *,
    evaluation_path: Path | None = None,
    strategy_config: dict[str, Any] | None = None,
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
) -> WalkForwardRunResult:
    """Execute one strategy across deterministic evaluation splits and persist artifacts."""

    evaluation_config = load_walk_forward_config(evaluation_path)
    resolved_runtime = runtime_config or resolve_runtime_config(
        strategy_config or {},
        cli_overrides=None if execution_config is None else {"execution": execution_config.to_dict()},
        cli_strict=strict,
    )
    splits = generate_evaluation_splits(evaluation_config)
    if not splits:
        raise WalkForwardExecutionError("Walk-forward evaluation did not produce any splits.")

    dataset = _load_dataset_for_splits(strategy, splits)
    split_results = [
        execute_split(
            strategy,
            dataset,
            split,
            execution_config=resolved_runtime.execution,
            sanity_config=resolved_runtime.sanity,
            strict_mode=resolved_runtime.strict_mode.enabled,
        )
        for split in splits
    ]
    aggregate_summary = build_aggregate_summary(split_results)
    aggregate_results = pd.concat([result.results_df for result in split_results], ignore_index=True)
    aggregate_results.attrs["dataset"] = strategy.dataset
    aggregate_results.attrs["timeframe"] = evaluation_config.timeframe
    try:
        aggregate_report = validate_strategy_backtest_sanity(
            aggregate_results,
            aggregate_summary,
            resolved_runtime.sanity,
            scope="strategy_walk_forward_aggregate",
        )
    except SanityCheckError as exc:
        try:
            raise_research_validation_error(
                validator="sanity",
                scope=f"strategy_walk_forward:{strategy_name}:aggregate",
                exc=exc,
                strict_mode=resolved_runtime.strict_mode.enabled,
            )
        except ResearchStrictModeError as strict_exc:
            raise WalkForwardExecutionError(str(strict_exc)) from strict_exc
    aggregate_summary = aggregate_report.apply_to_metrics(aggregate_summary)
    aggregate_summary["sanity"] = {
        "aggregate": aggregate_report.to_dict(),
        "flagged_split_count": int(sum(result.sanity.get("issue_count", 0) > 0 for result in split_results)),
        "failed_split_count": int(sum(result.sanity.get("status") == "fail" for result in split_results)),
        "flagged_splits": [result.split_id for result in split_results if result.sanity.get("issue_count", 0) > 0],
    }
    aggregate_summary["flagged_split_count"] = float(aggregate_summary["sanity"]["flagged_split_count"])
    aggregate_results.attrs["sanity_check"] = aggregate_report.to_dict()
    signal_diagnostics = compute_signal_diagnostics(aggregate_results["signal"], aggregate_results)

    run_config = resolved_runtime.apply_to_payload(
        {
        "strategy_name": strategy_name,
        "dataset": strategy.dataset,
        "parameters": dict((strategy_config or {}).get("parameters", {})),
        "evaluation_config_path": str(evaluation_path or EVALUATION_CONFIG),
        "evaluation": {
            "mode": evaluation_config.mode,
            "timeframe": evaluation_config.timeframe,
            "start": evaluation_config.start,
            "end": evaluation_config.end,
            "train_window": evaluation_config.train_window,
            "test_window": evaluation_config.test_window,
            "step": evaluation_config.step,
            "train_start": evaluation_config.train_start,
            "train_end": evaluation_config.train_end,
            "test_start": evaluation_config.test_start,
            "test_end": evaluation_config.test_end,
        },
        },
        include_validation_section=False,
    )
    experiment_dir = save_walk_forward_experiment(
        strategy_name,
        [
            {
                "split_id": split_result.split_id,
                "split_metadata": {
                    "split_id": split_result.split_id,
                    "mode": split_result.mode,
                    "train_start": split_result.train_start,
                    "train_end": split_result.train_end,
                    "test_start": split_result.test_start,
                    "test_end": split_result.test_end,
                },
                "split_rows": split_result.split_rows,
                "train_rows": split_result.train_rows,
                "test_rows": split_result.test_rows,
                "metrics": split_result.metrics,
                "results_df": split_result.results_df,
            }
            for split_result in split_results
        ],
        aggregate_summary,
        run_config,
    )
    qa_summary = generate_strategy_qa_summary(
        aggregate_results,
        aggregate_results["signal"],
        signal_diagnostics,
        aggregate_summary,
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
    )

    return WalkForwardRunResult(
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
        experiment_dir=experiment_dir,
        metrics=_coerce_metric_map(aggregate_summary),
        aggregate_summary=aggregate_summary,
        splits=split_results,
        signal_diagnostics=signal_diagnostics,
        qa_summary=qa_summary,
        sanity_summary=dict(aggregate_summary.get("sanity", {})),
    )


def execute_split(
    strategy: BaseStrategy,
    dataset: pd.DataFrame,
    split: EvaluationSplit,
    *,
    execution_config: ExecutionConfig | None = None,
    sanity_config: SanityCheckConfig | dict[str, Any] | None = None,
    strict_mode: bool = False,
) -> SplitExecutionResult:
    """Run the research pipeline for one split and score only the test window."""

    split_frame = slice_dataset_by_date(dataset, start=split.train_start, end=split.test_end)
    if split_frame.empty:
        raise WalkForwardExecutionError(
            f"Split '{split.split_id}' produced no rows for the combined train/test window."
        )

    train_frame = slice_dataset_by_date(split_frame, start=split.train_start, end=split.train_end)
    if train_frame.empty:
        raise WalkForwardExecutionError(f"Split '{split.split_id}' produced no training rows.")

    signal_frame = generate_signals(split_frame, strategy)
    backtest_frame = run_backtest(signal_frame, execution_config)
    test_frame = slice_dataset_by_date(backtest_frame, start=split.test_start, end=split.test_end)
    if test_frame.empty:
        raise WalkForwardExecutionError(f"Split '{split.split_id}' produced no test rows.")

    split_results = _attach_split_metadata(test_frame, split)
    split_results.attrs = {}
    metrics = compute_metrics(split_results)
    try:
        sanity_report = validate_strategy_backtest_sanity(
            split_results,
            metrics,
            sanity_config,
            scope=f"strategy_walk_forward_split:{split.split_id}",
        )
    except SanityCheckError as exc:
        try:
            raise_research_validation_error(
                validator="sanity",
                scope=f"strategy_walk_forward_split:{split.split_id}",
                exc=exc,
                strict_mode=strict_mode,
            )
        except ResearchStrictModeError as strict_exc:
            raise WalkForwardExecutionError(str(strict_exc)) from strict_exc
    split_results.attrs["sanity_check"] = sanity_report.to_dict()
    metrics = sanity_report.apply_to_metrics(metrics)
    return SplitExecutionResult(
        split_id=split.split_id,
        mode=split.mode,
        train_start=split.train_start,
        train_end=split.train_end,
        test_start=split.test_start,
        test_end=split.test_end,
        split_rows=len(split_frame),
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        metrics=metrics,
        results_df=split_results,
        sanity=sanity_report.to_dict(),
    )


def build_aggregate_summary(split_results: list[SplitExecutionResult]) -> dict[str, Any]:
    """Build a deterministic aggregate summary from ordered split test results."""

    if not split_results:
        raise WalkForwardExecutionError("Walk-forward evaluation requires at least one executed split.")

    concatenated = pd.concat([result.results_df for result in split_results], ignore_index=True)
    summary: dict[str, Any] = {
        "aggregation_method": "metrics computed on concatenated split test windows in split order",
        "mode": split_results[0].mode,
        "split_count": len(split_results),
        "train_rows": sum(result.train_rows for result in split_results),
        "test_rows": sum(result.test_rows for result in split_results),
        "split_rows": sum(result.split_rows for result in split_results),
        "first_train_start": split_results[0].train_start,
        "last_test_end": split_results[-1].test_end,
    }
    summary.update(compute_metrics(concatenated))
    return summary


def slice_dataset_by_date(
    dataset: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return a half-open `[start, end)` date slice from a feature or backtest frame."""

    date_series = _resolve_date_series(dataset)
    mask = (date_series >= start) & (date_series < end)
    return dataset.loc[mask].reset_index(drop=True)


def _load_dataset_for_splits(strategy: BaseStrategy, splits: list[EvaluationSplit]) -> pd.DataFrame:
    start = min(split.train_start for split in splits)
    end = max(split.test_end for split in splits)
    dataset = load_features(strategy.dataset, start=start, end=end)
    if dataset.empty:
        raise WalkForwardExecutionError(
            f"Feature dataset '{strategy.dataset}' returned no rows for evaluation window [{start}, {end})."
        )
    return dataset


def _resolve_date_series(dataset: pd.DataFrame) -> pd.Series:
    if "date" in dataset.columns:
        return dataset["date"].astype("string")
    if "ts_utc" in dataset.columns:
        return pd.to_datetime(dataset["ts_utc"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    raise WalkForwardExecutionError("Walk-forward evaluation requires a 'date' or 'ts_utc' column.")


def _attach_split_metadata(df: pd.DataFrame, split: EvaluationSplit) -> pd.DataFrame:
    result = df.copy()
    result["split_id"] = split.split_id
    result["mode"] = split.mode
    result["train_start"] = split.train_start
    result["train_end"] = split.train_end
    result["test_start"] = split.test_start
    result["test_end"] = split.test_end
    return result


def _coerce_metric_map(summary: dict[str, Any]) -> dict[str, float | None]:
    metric_keys = (
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
    return {key: (None if summary.get(key) is None else float(summary[key])) for key in metric_keys}
