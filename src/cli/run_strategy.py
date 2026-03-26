from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from src.config.execution import ExecutionConfig, resolve_execution_config
from src.config.evaluation import EVALUATION_CONFIG
from src.data.load_features import load_features
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.input_validation import StrategyInputError
from src.research.metrics import compute_benchmark_relative_metrics
from src.research.strict_mode import (
    ResearchStrictModeError,
    apply_strict_mode_to_sanity_config,
    raise_research_validation_error,
    resolve_strict_mode_policy,
)
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.signal_engine import generate_signals
from src.research.sanity import SanityCheckError, validate_strategy_backtest_sanity
from src.research.strategies import build_strategy
from src.research.strategy_qa import generate_strategy_qa_summary
from src.research.walk_forward import WalkForwardRunResult, compute_metrics, run_walk_forward_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"


@dataclass(frozen=True)
class StrategyRunResult:
    """Structured result returned from a strategy experiment run."""

    strategy_name: str
    run_id: str
    metrics: dict[str, Any]
    experiment_dir: Path
    results_df: pd.DataFrame
    signal_diagnostics: dict[str, Any] = field(default_factory=dict)
    qa_summary: dict[str, Any] = field(default_factory=dict)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a strategy experiment run."""

    parser = argparse.ArgumentParser(
        description="Run a full strategy experiment using the StratLake research pipeline."
    )
    parser.add_argument("--strategy", required=True, help="Strategy name defined in configs/strategies.yml.")
    parser.add_argument("--start", dest="start", help="Inclusive experiment start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="end", help="Exclusive experiment end date (YYYY-MM-DD).")
    parser.add_argument(
        "--evaluation",
        nargs="?",
        const=str(EVALUATION_CONFIG),
        help="Enable walk-forward evaluation using configs/evaluation.yml or a provided path.",
    )
    parser.add_argument(
        "--execution-delay",
        type=int,
        help="Override execution delay in bars.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        help="Override deterministic transaction cost in basis points.",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        help="Override deterministic slippage in basis points.",
    )
    parser.add_argument(
        "--execution-enabled",
        action="store_true",
        help="Enable execution frictions even when config defaults are disabled.",
    )
    parser.add_argument(
        "--disable-execution-model",
        action="store_true",
        help="Disable transaction-cost and slippage frictions for this run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict research-validity enforcement and block artifact or registry writes on flagged runs.",
    )
    return parser.parse_args(argv)


def load_strategies_config(path: Path = STRATEGIES_CONFIG) -> dict[str, dict[str, Any]]:
    """Load the strategy registry YAML file."""

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError("Strategy configuration must be a mapping of strategy names to config dictionaries.")

    return payload


def get_strategy_config(
    strategy_name: str, strategies: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return the requested strategy configuration or raise a clear validation error."""

    registry = strategies or load_strategies_config()
    if strategy_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available strategies: {available}."
        )

    config = registry[strategy_name]
    if not isinstance(config, dict):
        raise ValueError(f"Strategy '{strategy_name}' configuration must be a dictionary.")

    return config


def run_strategy_experiment(
    strategy_name: str,
    *,
    start: str | None = None,
    end: str | None = None,
    execution_config: ExecutionConfig | None = None,
    strict: bool = False,
) -> StrategyRunResult:
    """Run the full strategy research pipeline and persist experiment artifacts."""

    config = get_strategy_config(strategy_name)
    strategy = build_strategy(strategy_name, config)
    dataset = load_features(strategy.dataset, start=start, end=end)
    signal_frame = generate_signals(dataset, strategy)
    resolved_execution = execution_config or resolve_execution_config(config.get("execution"))
    results_df = run_backtest(signal_frame, resolved_execution)
    results_df.attrs["dataset"] = strategy.dataset
    metrics = compute_metrics(results_df)
    metrics.update(_compute_benchmark_metrics(results_df, dataset, strategy.dataset, resolved_execution))
    strict_policy = resolve_strict_mode_policy(cli_strict=strict, sanity_config=config.get("sanity"))
    sanity_config = apply_strict_mode_to_sanity_config(config.get("sanity"), strict_policy)
    try:
        sanity_report = validate_strategy_backtest_sanity(results_df, metrics, sanity_config)
    except SanityCheckError as exc:
        raise_research_validation_error(
            validator="sanity",
            scope=f"strategy:{strategy_name}",
            exc=exc,
            strict_mode=strict_policy.enabled,
        )
    metrics = sanity_report.apply_to_metrics(metrics)
    results_df.attrs["sanity_check"] = sanity_report.to_dict()
    signal_diagnostics = compute_signal_diagnostics(results_df["signal"], results_df)

    experiment_config = {
        "strategy_name": strategy_name,
        "dataset": strategy.dataset,
        "parameters": dict(config.get("parameters", {})),
        "start": start,
        "end": end,
        "execution": resolved_execution.to_dict(),
        "sanity": sanity_config.to_dict(),
        "strict_mode": strict_policy.to_dict(),
    }
    experiment_dir = save_experiment(strategy_name, results_df, metrics, experiment_config)
    qa_summary = generate_strategy_qa_summary(
        results_df,
        results_df["signal"],
        signal_diagnostics,
        metrics,
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
    )

    return StrategyRunResult(
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
        metrics=metrics,
        experiment_dir=experiment_dir,
        results_df=results_df,
        signal_diagnostics=signal_diagnostics,
        qa_summary=qa_summary,
    )


def print_summary(result: StrategyRunResult | WalkForwardRunResult) -> None:
    """Print a concise experiment summary for CLI users."""

    print(f"strategy: {result.strategy_name}")
    print(f"run_id: {result.run_id}")
    if isinstance(result, WalkForwardRunResult):
        print(f"split_count: {result.aggregate_summary['split_count']}")
    print(f"cumulative_return: {result.metrics['cumulative_return']:.6f}")
    print(f"sharpe_ratio: {result.metrics['sharpe_ratio']:.6f}")
    diagnostics = getattr(result, "signal_diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        print("Signal diagnostics:")
        print(
            f"- long: {diagnostics['pct_long']:.0%} | "
            f"short: {diagnostics['pct_short']:.0%} | "
            f"flat: {diagnostics['pct_flat']:.0%}"
        )
        print(
            f"- trades: {diagnostics['total_trades']} | "
            f"turnover: {diagnostics['turnover']:.2f}"
        )
        print(f"- avg holding: {diagnostics['avg_holding_period']:.1f} bars")
    qa_summary = getattr(result, "qa_summary", None)
    if isinstance(qa_summary, dict) and qa_summary:
        print("QA Summary:")
        print(f"- status: {str(qa_summary['overall_status']).upper()}")
        print(
            f"- rows: {qa_summary['row_count']:,} | "
            f"symbols: {qa_summary['symbols_present']}"
        )
        print(
            f"- trades: {qa_summary['signal']['total_trades']} | "
            f"turnover: {qa_summary['signal']['turnover']:.2f}"
        )
        _print_benchmark_summary(result.metrics)
        warnings_list = summarize_qa_warnings(qa_summary)
        if warnings_list:
            print("Warnings:")
            for warning in warnings_list:
                print(f"- {warning}")


def summarize_qa_warnings(qa_summary: dict[str, Any]) -> list[str]:
    flags = qa_summary.get("flags")
    if not isinstance(flags, dict):
        return []

    warnings_list: list[str] = []
    if bool(flags.get("low_data")):
        warnings_list.append("insufficient data for a high-confidence analysis")
    if bool(flags.get("no_trades")):
        warnings_list.append("no trades were generated")
    if bool(flags.get("high_benchmark_correlation")):
        correlation = qa_summary.get("relative", {}).get("correlation")
        if correlation is None:
            warnings_list.append("strategy is highly correlated with the benchmark")
        else:
            warnings_list.append(f"strategy is highly correlated with the benchmark ({float(correlation):.2f})")
    if bool(flags.get("low_excess_return")):
        warnings_list.append("strategy delivered little excess return versus buy and hold")
    if bool(flags.get("high_turnover_low_edge")):
        warnings_list.append("strategy turnover is high relative to its excess return")
    if bool(flags.get("beta_dominated_strategy")):
        warnings_list.append("strategy returns appear largely benchmark-driven")
    sanity = qa_summary.get("sanity")
    if isinstance(sanity, dict):
        for issue in sanity.get("issues", []):
            if isinstance(issue, dict) and issue.get("message"):
                warnings_list.append(str(issue["message"]))
    return warnings_list


def _compute_benchmark_metrics(
    results_df: pd.DataFrame,
    dataset: pd.DataFrame,
    strategy_dataset: str,
    execution_config: ExecutionConfig,
) -> dict[str, float | dict[str, bool]]:
    benchmark_strategy = build_strategy("buy_and_hold_v1", {"dataset": strategy_dataset, "parameters": {}})
    benchmark_signal_frame = generate_signals(dataset, benchmark_strategy)
    benchmark_results = run_backtest(benchmark_signal_frame, execution_config)
    return compute_benchmark_relative_metrics(results_df, benchmark_results)


def _print_benchmark_summary(metrics: dict[str, Any]) -> None:
    benchmark_return = metrics.get("benchmark_total_return")
    excess_return = metrics.get("excess_return")
    correlation = metrics.get("benchmark_correlation")
    if benchmark_return is None and excess_return is None and correlation is None:
        return

    print("Benchmark comparison:")
    print(f"- benchmark return: {_format_pct(benchmark_return)}")
    print(f"- excess return: {_format_signed_pct(excess_return)}")
    print(f"- correlation: {_format_decimal(correlation)}")


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0%}"


def _format_signed_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.0%}"


def _format_decimal(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def run_cli(argv: Sequence[str] | None = None) -> StrategyRunResult | WalkForwardRunResult:
    """Execute the strategy runner CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    execution_override = _execution_override_from_args(args)
    if args.evaluation:
        if args.start or args.end:
            raise ValueError("The --start and --end arguments cannot be combined with --evaluation.")

        config = get_strategy_config(args.strategy)
        strategy = build_strategy(args.strategy, config)
        execution_config = resolve_execution_config(config.get("execution"), execution_override)
        result = run_walk_forward_experiment(
            args.strategy,
            strategy,
            evaluation_path=Path(args.evaluation),
            strategy_config=config,
            execution_config=execution_config,
            strict=args.strict,
        )
    else:
        config = get_strategy_config(args.strategy)
        execution_config = resolve_execution_config(config.get("execution"), execution_override)
        result = run_strategy_experiment(
            args.strategy,
            start=args.start,
            end=args.end,
            execution_config=execution_config,
            strict=args.strict,
        )
    print_summary(result)
    return result


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


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    try:
        run_cli()
    except (ResearchStrictModeError, StrategyInputError, ValueError) as exc:
        print(_format_run_failure(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def _format_run_failure(exc: Exception) -> str:
    message = str(exc).strip()
    if message.startswith("Run failed:"):
        return message
    return f"Run failed: {message}"


if __name__ == "__main__":
    main()
