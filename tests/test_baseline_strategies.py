from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.backtest_runner import run_backtest
from src.research.strategies import BuyAndHoldStrategy, SMACrossoverStrategy, SeededRandomStrategy
from src.research.walk_forward import compute_metrics, run_walk_forward_experiment


def _daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "ts_utc": pd.to_datetime(dates, utc=True),
            "timeframe": ["1d"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "feature_ret_1d": [None, 0.01, 0.02, -0.01, 0.03, -0.02],
        }
    )


def _minute_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2022-01-03 14:30:00", periods=6, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(timestamps),
            "ts_utc": timestamps,
            "timeframe": ["1m"] * len(timestamps),
            "date": timestamps.strftime("%Y-%m-%d"),
            "feature_ret_1m": [None, 0.001, 0.002, -0.001, 0.0005, -0.002],
        }
    )


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _expected_metric_keys() -> set[str]:
    return {
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
    }


def _expected_walk_forward_metric_keys() -> set[str]:
    return {
        *_expected_metric_keys(),
        "benchmark_total_return",
        "excess_return",
        "benchmark_correlation",
        "relative_drawdown",
    }


def test_buy_and_hold_enters_once_after_first_valid_bar() -> None:
    signals = BuyAndHoldStrategy().generate_signals(_daily_frame())

    assert signals.tolist() == [0, 1, 1, 1, 1, 1]


def test_buy_and_hold_supports_minute_datasets() -> None:
    signals = BuyAndHoldStrategy().generate_signals(_minute_frame())

    assert signals.tolist() == [0, 1, 1, 1, 1, 1]


def test_sma_crossover_produces_expected_signals_for_known_inputs() -> None:
    df = pd.DataFrame(
        {
            "feature_ret_1d": [0.0, 0.1, 0.1, -0.1, -0.1],
        }
    )

    signals = SMACrossoverStrategy(fast_window=2, slow_window=3).generate_signals(df)

    assert signals.tolist() == [0, 0, 1, 1, 0]


def test_seeded_random_strategy_is_deterministic_for_same_seed() -> None:
    df = _daily_frame()

    first = SeededRandomStrategy(seed=13).generate_signals(df)
    second = SeededRandomStrategy(seed=13).generate_signals(df)
    third = SeededRandomStrategy(seed=21).generate_signals(df)

    assert first.tolist() == second.tolist()
    assert first.tolist() != third.tolist()


@pytest.mark.parametrize(
    ("strategy", "dataset"),
    [
        (BuyAndHoldStrategy(), _daily_frame()),
        (SMACrossoverStrategy(fast_window=2, slow_window=3), _daily_frame().fillna({"feature_ret_1d": 0.0})),
        (SeededRandomStrategy(seed=5), _daily_frame()),
    ],
)
def test_metrics_compute_without_errors_for_all_baselines(strategy, dataset: pd.DataFrame) -> None:
    signal_frame = dataset.copy()
    signal_frame["signal"] = strategy.generate_signals(dataset)
    results = run_backtest(signal_frame)

    metrics = compute_metrics(results)

    assert set(metrics) == _expected_metric_keys()


@pytest.mark.parametrize(
    "strategy",
    [
        BuyAndHoldStrategy(),
        SMACrossoverStrategy(fast_window=2, slow_window=3),
        SeededRandomStrategy(seed=11),
    ],
)
def test_baselines_run_successfully_through_walk_forward_execution(
    strategy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2022-01-01",
            "end": "2022-01-07",
            "train_window": "2D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    monkeypatch.setattr("src.research.walk_forward.load_features", lambda dataset, start=None, end=None: _daily_frame())

    result = run_walk_forward_experiment(
        strategy.name,
        strategy,
        evaluation_path=config_path,
        strategy_config={"parameters": {}},
    )

    assert result.aggregate_summary["split_count"] == 4
    assert len(result.splits) == 4
    assert set(result.metrics) == _expected_walk_forward_metric_keys()


def test_sma_crossover_returns_flat_when_dataset_is_shorter_than_slow_window() -> None:
    df = pd.DataFrame({"feature_ret_1d": [0.01, -0.02]})

    signals = SMACrossoverStrategy(fast_window=2, slow_window=5).generate_signals(df)

    assert signals.tolist() == [0, 0]


@pytest.mark.parametrize(
    "strategy",
    [
        BuyAndHoldStrategy(),
        SMACrossoverStrategy(fast_window=2, slow_window=3),
        SeededRandomStrategy(seed=19),
    ],
)
def test_baselines_handle_empty_datasets(strategy) -> None:
    df = pd.DataFrame(
        {
            "symbol": pd.Series(dtype="string"),
            "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "timeframe": pd.Series(dtype="string"),
            "feature_ret_1d": pd.Series(dtype="float64"),
        }
    )

    signals = strategy.generate_signals(df)

    assert signals.empty
    with pytest.raises(ValueError, match="contains no usable values"):
        run_backtest(df.assign(signal=signals))


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (BuyAndHoldStrategy(), [1]),
        (SMACrossoverStrategy(fast_window=1, slow_window=2), [0]),
        (SeededRandomStrategy(seed=3), [1]),
    ],
)
def test_baselines_handle_single_row_datasets(strategy, expected: list[int]) -> None:
    df = pd.DataFrame({"feature_ret_1d": [0.01]})

    signals = strategy.generate_signals(df)

    assert signals.tolist() == expected
