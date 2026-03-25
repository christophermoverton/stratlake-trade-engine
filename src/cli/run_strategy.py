from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from src.config.evaluation import EVALUATION_CONFIG
from src.data.load_features import load_features
from src.research.backtest_runner import run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.signal_engine import generate_signals
from src.research.strategies import build_strategy
from src.research.walk_forward import WalkForwardRunResult, compute_metrics, run_walk_forward_experiment

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"


@dataclass(frozen=True)
class StrategyRunResult:
    """Structured result returned from a strategy experiment run."""

    strategy_name: str
    run_id: str
    metrics: dict[str, float | None]
    experiment_dir: Path
    results_df: pd.DataFrame
    signal_diagnostics: dict[str, Any] = field(default_factory=dict)


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
) -> StrategyRunResult:
    """Run the full strategy research pipeline and persist experiment artifacts."""

    config = get_strategy_config(strategy_name)
    strategy = build_strategy(strategy_name, config)
    dataset = load_features(strategy.dataset, start=start, end=end)
    signal_frame = generate_signals(dataset, strategy)
    results_df = run_backtest(signal_frame)
    metrics = compute_metrics(results_df)
    signal_diagnostics = compute_signal_diagnostics(results_df["signal"], results_df)

    experiment_config = {
        "strategy_name": strategy_name,
        "dataset": strategy.dataset,
        "parameters": dict(config.get("parameters", {})),
        "start": start,
        "end": end,
    }
    experiment_dir = save_experiment(strategy_name, results_df, metrics, experiment_config)

    return StrategyRunResult(
        strategy_name=strategy_name,
        run_id=experiment_dir.name,
        metrics=metrics,
        experiment_dir=experiment_dir,
        results_df=results_df,
        signal_diagnostics=signal_diagnostics,
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


def run_cli(argv: Sequence[str] | None = None) -> StrategyRunResult | WalkForwardRunResult:
    """Execute the strategy runner CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    if args.evaluation:
        if args.start or args.end:
            raise ValueError("The --start and --end arguments cannot be combined with --evaluation.")

        config = get_strategy_config(args.strategy)
        strategy = build_strategy(args.strategy, config)
        result = run_walk_forward_experiment(
            args.strategy,
            strategy,
            evaluation_path=Path(args.evaluation),
            strategy_config=config,
        )
    else:
        result = run_strategy_experiment(args.strategy, start=args.start, end=args.end)
    print_summary(result)
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
