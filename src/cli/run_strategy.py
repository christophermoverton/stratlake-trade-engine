from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from src.data.load_features import load_features
from src.research.backtest_runner import RETURN_COLUMN_CANDIDATES, run_backtest
from src.research.experiment_tracker import save_experiment
from src.research.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    volatility,
    win_rate,
)
from src.research.signal_engine import generate_signals
from src.research.strategy_base import BaseStrategy

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGIES_CONFIG = REPO_ROOT / "configs" / "strategies.yml"


@dataclass(frozen=True)
class StrategyRunResult:
    """Structured result returned from a strategy experiment run."""

    strategy_name: str
    run_id: str
    metrics: dict[str, float]
    experiment_dir: Path
    results_df: pd.DataFrame


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy driven by rolling average returns."""

    name = "momentum_v1"
    dataset = "features_daily"

    def __init__(self, *, lookback_short: int, lookback_long: int) -> None:
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return_column = _resolve_return_column(df)
        short_trend = df[return_column].rolling(window=self.lookback_short, min_periods=1).mean()
        long_trend = df[return_column].rolling(window=self.lookback_long, min_periods=1).mean()
        return ((short_trend > long_trend).astype("int64") - (short_trend < long_trend).astype("int64")).rename(
            "signal"
        )


class MeanReversionStrategy(BaseStrategy):
    """Simple mean-reversion strategy based on a rolling return z-score."""

    name = "mean_reversion_v1"
    dataset = "features_daily"

    def __init__(self, *, zscore_window: int, entry_threshold: float) -> None:
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return_column = _resolve_return_column(df)
        rolling_mean = df[return_column].rolling(window=self.zscore_window, min_periods=1).mean()
        rolling_std = (
            df[return_column]
            .rolling(window=self.zscore_window, min_periods=1)
            .std()
            .replace(0.0, pd.NA)
        )
        zscore = ((df[return_column] - rolling_mean) / rolling_std).fillna(0.0)
        threshold = abs(float(self.entry_threshold))

        signals = pd.Series(0, index=df.index, dtype="int64")
        signals.loc[zscore <= -threshold] = 1
        signals.loc[zscore >= threshold] = -1
        return signals.rename("signal")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a strategy experiment run."""

    parser = argparse.ArgumentParser(
        description="Run a full strategy experiment using the StratLake research pipeline."
    )
    parser.add_argument("--strategy", required=True, help="Strategy name defined in configs/strategies.yml.")
    parser.add_argument("--start", dest="start", help="Inclusive experiment start date (YYYY-MM-DD).")
    parser.add_argument("--end", dest="end", help="Exclusive experiment end date (YYYY-MM-DD).")
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


def _resolve_return_column(df: pd.DataFrame) -> str:
    """Return the first supported return column present in the feature dataset."""

    for column in RETURN_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    expected = ", ".join(RETURN_COLUMN_CANDIDATES)
    raise ValueError(f"Feature dataset must include one of the supported return columns: {expected}.")


def build_strategy(strategy_name: str, config: dict[str, Any]) -> BaseStrategy:
    """Instantiate a concrete strategy implementation from registry config."""

    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"Strategy '{strategy_name}' parameters must be a dictionary.")

    if strategy_name == "momentum_v1":
        strategy = MomentumStrategy(
            lookback_short=int(parameters["lookback_short"]),
            lookback_long=int(parameters["lookback_long"]),
        )
    elif strategy_name == "mean_reversion_v1":
        strategy = MeanReversionStrategy(
            zscore_window=int(parameters["zscore_window"]),
            entry_threshold=float(parameters["entry_threshold"]),
        )
    else:
        raise ValueError(f"No strategy implementation is registered for '{strategy_name}'.")

    dataset = config.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"Strategy '{strategy_name}' must define a non-empty dataset name.")

    strategy.dataset = dataset
    return strategy


def compute_metrics(results_df: pd.DataFrame) -> dict[str, float]:
    """Compute the standard strategy performance summary for an experiment run."""

    strategy_return = results_df["strategy_return"]
    return {
        "cumulative_return": cumulative_return(strategy_return),
        "sharpe_ratio": sharpe_ratio(strategy_return),
        "volatility": volatility(strategy_return),
        "max_drawdown": max_drawdown(strategy_return),
        "win_rate": win_rate(strategy_return),
    }


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
    )


def print_summary(result: StrategyRunResult) -> None:
    """Print a concise experiment summary for CLI users."""

    print(f"strategy: {result.strategy_name}")
    print(f"run_id: {result.run_id}")
    print(f"cumulative_return: {result.metrics['cumulative_return']:.6f}")
    print(f"sharpe_ratio: {result.metrics['sharpe_ratio']:.6f}")


def run_cli(argv: Sequence[str] | None = None) -> StrategyRunResult:
    """Execute the strategy runner CLI flow from parsed command-line arguments."""

    args = parse_args(argv)
    result = run_strategy_experiment(args.strategy, start=args.start, end=args.end)
    print_summary(result)
    return result


def main() -> None:
    """CLI entrypoint used by direct module execution."""

    run_cli()


if __name__ == "__main__":
    main()
