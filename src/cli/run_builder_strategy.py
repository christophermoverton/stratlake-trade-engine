from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config.execution import ExecutionConfig
from src.config.runtime import RuntimeConfig, resolve_runtime_config
from src.data.load_features import load_features
from src.pipeline.builder import (
    load_builder_config,
    transform_signal,
    validate_declarative_strategy_mapping,
)
from src.pipeline.cli_adapter import build_pipeline_cli_result
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.metrics import compute_benchmark_relative_metrics
from src.research.promotion import load_promotion_gate_config
from src.research.sanity import SanityCheckError, validate_strategy_backtest_sanity
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.signal_engine import generate_signals
from src.research.signal_semantics import Signal, attach_signal_metadata, extract_signal_metadata
from src.research.strict_mode import ResearchStrictModeError, raise_research_validation_error
from src.research.strategies import build_strategy
from src.research.strategy_qa import generate_strategy_qa_summary
from src.research.walk_forward import compute_metrics


@dataclass(frozen=True)
class DeclarativeStrategyRunResult:
    """Structured result returned from one declarative strategy run."""

    strategy_name: str
    run_id: str
    metrics: dict[str, Any]
    experiment_dir: Path
    results_df: pd.DataFrame
    signal_diagnostics: dict[str, Any] = field(default_factory=dict)
    qa_summary: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one builder-rendered strategy run."""

    parser = argparse.ArgumentParser(
        description="Run one declarative builder strategy spec through the existing artifact system."
    )
    parser.add_argument("--config", required=True, help="Declarative strategy config path.")
    parser.add_argument("--start", dest="start", help="Inclusive experiment start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="end", help="Exclusive experiment end date (YYYY-MM-DD).")
    parser.add_argument("--execution-delay", type=int, help="Override execution delay in bars.")
    parser.add_argument("--transaction-cost-bps", type=float, help="Override transaction cost in basis points.")
    parser.add_argument("--slippage-bps", type=float, help="Override slippage in basis points.")
    parser.add_argument("--execution-enabled", action="store_true", help="Enable execution frictions.")
    parser.add_argument("--disable-execution-model", action="store_true", help="Disable execution frictions.")
    parser.add_argument("--strict", action="store_true", help="Enable strict research-validity enforcement.")
    parser.add_argument("--promotion-gates", help="Optional YAML/JSON promotion gate config override.")
    return parser.parse_args(argv)


def run_declarative_strategy_experiment(
    config: Mapping[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    runtime_config: RuntimeConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
) -> DeclarativeStrategyRunResult:
    """Execute one normalized declarative strategy spec and persist standard artifacts."""

    normalized = validate_declarative_strategy_mapping(config)
    strategy_section = normalized["strategy"]
    signal_section = normalized["signal"]
    constructor_section = normalized["constructor"]
    asymmetry_section = normalized["asymmetry"]["params"]

    experiment_config = {
        "dataset": strategy_section["dataset"],
        "parameters": dict(strategy_section["params"]),
        "strategy_id": strategy_section["strategy_id"],
        "strategy_version": strategy_section["version"],
        "signal": dict(signal_section),
        "constructor": {
            "name": constructor_section["name"],
            "version": constructor_section["version"],
            "params": {
                **dict(constructor_section["params"]),
                **dict(asymmetry_section),
            },
        },
        "start": start,
        "end": end,
    }
    resolved_runtime = runtime_config or resolve_runtime_config(
        experiment_config,
        cli_overrides=None if execution_config is None else {"execution": execution_config.to_dict()},
        cli_strict=strict,
    )
    experiment_config = resolved_runtime.apply_to_payload(experiment_config, include_validation_section=False)

    strategy = build_strategy(
        strategy_section["strategy_id"],
        {
            "dataset": strategy_section["dataset"],
            "parameters": dict(strategy_section["params"]),
            "signal_type": strategy_section["source_signal_type"],
            "signal_params": {},
            "position_constructor": None,
        },
    )
    strategy.position_constructor_name = None
    strategy.position_constructor_params = {}

    dataset = load_features(strategy_section["dataset"], start=start, end=end)
    base_frame = generate_signals(dataset, strategy)
    base_signal = Signal(
        strategy_section["source_signal_type"],
        "1.0.0",
        base_frame,
        "signal",
        dict(extract_signal_metadata(base_frame) or {}),
    )
    transformed_signal = transform_signal(
        base_signal,
        signal_section["type"],
        signal_params=dict(signal_section["params"]),
    )
    managed_frame = transformed_signal.data.copy(deep=True)
    signal_metadata = dict(extract_signal_metadata(managed_frame) or {})
    signal_metadata["constructor_id"] = constructor_section["name"]
    signal_metadata["constructor_params"] = {
        **dict(constructor_section["params"]),
        **dict(asymmetry_section),
    }
    attach_signal_metadata(managed_frame, signal_metadata)

    results_df = run_backtest(
        managed_frame,
        resolved_runtime.execution,
        require_managed_signals=True,
    )
    results_df.attrs["dataset"] = strategy_section["dataset"]
    results_df.attrs["runtime_config"] = resolved_runtime.to_dict()
    metrics = compute_metrics(results_df)
    metrics.update(_compute_benchmark_metrics(results_df, dataset, strategy_section["dataset"], resolved_runtime.execution))
    try:
        sanity_report = validate_strategy_backtest_sanity(results_df, metrics, resolved_runtime.sanity)
    except SanityCheckError as exc:
        raise_research_validation_error(
            validator="sanity",
            scope=f"strategy:{strategy_section['name']}",
            exc=exc,
            strict_mode=resolved_runtime.strict_mode.enabled,
        )
    metrics = sanity_report.apply_to_metrics(metrics)
    results_df.attrs["sanity_check"] = sanity_report.to_dict()

    experiment_dir = save_experiment(strategy_section["name"], results_df, metrics, experiment_config)
    signal_diagnostics = compute_signal_diagnostics(results_df["signal"], results_df)
    qa_summary = generate_strategy_qa_summary(
        results_df,
        results_df["signal"],
        signal_diagnostics,
        metrics,
        strategy_name=strategy_section["name"],
        run_id=experiment_dir.name,
    )
    return DeclarativeStrategyRunResult(
        strategy_name=strategy_section["name"],
        run_id=experiment_dir.name,
        metrics=metrics,
        experiment_dir=experiment_dir,
        results_df=results_df,
        signal_diagnostics=signal_diagnostics,
        qa_summary=qa_summary,
    )


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    state: Mapping[str, Any] | None = None,
    pipeline_context: Mapping[str, Any] | None = None,
) -> DeclarativeStrategyRunResult | dict[str, Any]:
    """Execute one declarative strategy from CLI arguments."""

    del state
    args = parse_args(argv)
    config = load_builder_config(args.config)
    if args.promotion_gates is not None and isinstance(config.get("strategy"), dict):
        config = dict(config)
        strategy_config = dict(config["strategy"])
        strategy_config["promotion_gates"] = load_promotion_gate_config(args.promotion_gates)
        config["strategy"] = strategy_config

    execution_override = _execution_override_from_args(args)
    runtime_config = resolve_runtime_config(
        config,
        cli_overrides=None if execution_override is None else {"execution": execution_override},
        cli_strict=args.strict,
    )
    result = run_declarative_strategy_experiment(
        config,
        start=args.start,
        end=args.end,
        execution_config=runtime_config.execution,
        strict=args.strict,
    )
    print_summary(result)
    if pipeline_context is not None:
        artifact_dir = result.experiment_dir
        return build_pipeline_cli_result(
            identifier=result.run_id,
            name=result.strategy_name,
            artifact_dir=artifact_dir,
            manifest_path=artifact_dir / "manifest.json",
            output_paths={
                "manifest_json": artifact_dir / "manifest.json",
                "metrics_json": artifact_dir / "metrics.json",
                "qa_summary_json": artifact_dir / "qa_summary.json",
                "equity_curve_csv": artifact_dir / "equity_curve.csv",
            },
            metrics=dict(result.metrics),
            extra={
                "strategy_name": result.strategy_name,
                "result_type": result.__class__.__name__,
            },
        )
    return result


def print_summary(result: DeclarativeStrategyRunResult) -> None:
    """Print a concise declarative strategy run summary."""

    print(f"strategy: {result.strategy_name}")
    print(f"run_id: {result.run_id}")
    print(f"cumulative_return: {result.metrics['cumulative_return']:.6f}")
    print(f"sharpe_ratio: {result.metrics['sharpe_ratio']:.6f}")


def _execution_override_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    override: dict[str, Any] = {}
    if args.execution_delay is not None:
        override["execution_delay"] = args.execution_delay
    if args.transaction_cost_bps is not None:
        override["transaction_cost_bps"] = args.transaction_cost_bps
    if args.slippage_bps is not None:
        override["slippage_bps"] = args.slippage_bps
    if args.execution_enabled:
        override["enabled"] = True
    if args.disable_execution_model:
        override["enabled"] = False
    return override or None


def _compute_benchmark_metrics(
    results_df: pd.DataFrame,
    dataset: pd.DataFrame,
    strategy_dataset: str,
    execution_config: ExecutionConfig,
) -> dict[str, float | dict[str, bool]]:
    benchmark_strategy = build_strategy("buy_and_hold_v1", {"dataset": strategy_dataset, "parameters": {}})
    benchmark_signal_frame = generate_signals(dataset, benchmark_strategy)
    benchmark_results = run_backtest(
        benchmark_signal_frame,
        execution_config,
        require_managed_signals=True,
    )
    return compute_benchmark_relative_metrics(results_df, benchmark_results)


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (ResearchStrictModeError, ValueError) as exc:
        print(f"Declarative strategy run failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
