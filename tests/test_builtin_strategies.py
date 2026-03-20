from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_engine import generate_signals
from src.research.strategies import MeanReversionStrategy, MomentumStrategy, build_strategy
from src.research.walk_forward import compute_metrics, run_walk_forward_experiment


def _write_evaluation_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"evaluation": payload}, sort_keys=False), encoding="utf-8")


def _momentum_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_ret_1d": [0.0, 0.1, 0.1, -0.1, -0.1],
        },
        index=pd.Index([10, 11, 12, 13, 14], name="row_id"),
    )


def _multi_symbol_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "date": pd.date_range("2022-01-01", periods=6, freq="D").strftime("%Y-%m-%d"),
            "feature_ret_1d": [0.0, 0.1, 0.1, 0.0, -0.1, -0.1],
        }
    )


def _walk_forward_frame() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "ts_utc": pd.to_datetime(dates, utc=True),
            "timeframe": ["1d"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "feature_ret_1d": [0.0, 0.03, 0.02, -0.01, -0.02, 0.01, 0.02, -0.01],
        }
    )


def _mean_reversion_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0, 103.0, 97.0, 100.0],
        },
        index=pd.Index([20, 21, 22, 23, 24, 25, 26], name="row_id"),
    )


def _mean_reversion_multi_symbol_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
            "close": [100.0, 100.0, 100.0, 103.0, 50.0, 50.0, 50.0, 47.0],
        },
        index=pd.Index(range(100, 108), name="row_id"),
    )


def test_momentum_produces_expected_signals_for_known_inputs() -> None:
    strategy = MomentumStrategy(lookback_short=2, lookback_long=3)

    signals = strategy.generate_signals(_momentum_frame())

    assert signals.tolist() == [0, 0, 1, -1, -1]
    assert signals.index.tolist() == [10, 11, 12, 13, 14]
    assert set(signals.unique()) <= {-1, 0, 1}


def test_momentum_handles_multi_symbol_frames_without_cross_symbol_bleed() -> None:
    strategy = MomentumStrategy(lookback_short=2, lookback_long=3)

    signals = strategy.generate_signals(_multi_symbol_frame())

    assert signals.tolist() == [0, 0, 1, 0, 0, -1]


def test_momentum_is_deterministic_across_repeated_runs() -> None:
    strategy = MomentumStrategy(lookback_short=2, lookback_long=3)
    df = _multi_symbol_frame()

    first = strategy.generate_signals(df)
    second = strategy.generate_signals(df.copy())

    pd.testing.assert_series_equal(first, second)


def test_momentum_rejects_invalid_window_configuration() -> None:
    with pytest.raises(ValueError, match="lookback_short must be smaller than lookback_long"):
        MomentumStrategy(lookback_short=5, lookback_long=5)


def test_mean_reversion_produces_expected_zscore_signals_for_known_inputs() -> None:
    strategy = MeanReversionStrategy(lookback=4, threshold=1.0)

    signals = strategy.generate_signals(_mean_reversion_frame())

    assert signals.tolist() == [0, 0, 0, 0, -1, 1, 0]
    assert signals.index.tolist() == [20, 21, 22, 23, 24, 25, 26]
    assert set(signals.unique()) <= {-1, 0, 1}


def test_mean_reversion_handles_multi_symbol_frames_without_cross_symbol_bleed() -> None:
    strategy = MeanReversionStrategy(lookback=3, threshold=1.0)

    signals = strategy.generate_signals(_mean_reversion_multi_symbol_frame())

    assert signals.tolist() == [0, 0, 0, -1, 0, 0, 0, 1]


def test_mean_reversion_is_deterministic_across_repeated_runs() -> None:
    strategy = MeanReversionStrategy(lookback=3, threshold=1.0)
    df = _mean_reversion_multi_symbol_frame()

    first = strategy.generate_signals(df)
    second = strategy.generate_signals(df.copy())

    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lookback": 1, "threshold": 1.0}, "lookback must be greater than 1"),
        ({"lookback": 5, "threshold": 0.0}, "threshold must be greater than 0"),
    ],
)
def test_mean_reversion_rejects_invalid_configuration(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MeanReversionStrategy(**kwargs)


def test_mean_reversion_integrates_with_strategy_build_and_signal_engine() -> None:
    config = {"dataset": "features_daily", "parameters": {"lookback": 4, "threshold": 1.0}}
    strategy = build_strategy("mean_reversion_v1", config)

    signal_frame = generate_signals(_mean_reversion_frame(), strategy)

    assert signal_frame["signal"].tolist() == [0, 0, 0, 0, -1, 1, 0]
    assert signal_frame.index.tolist() == [20, 21, 22, 23, 24, 25, 26]


def test_momentum_runs_successfully_through_walk_forward_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "evaluation.yml"
    _write_evaluation_config(
        config_path,
        {
            "mode": "rolling",
            "timeframe": "1d",
            "start": "2022-01-01",
            "end": "2022-01-09",
            "train_window": "3D",
            "test_window": "1D",
            "step": "1D",
        },
    )

    strategy = MomentumStrategy(lookback_short=2, lookback_long=3)
    monkeypatch.setattr(
        "src.research.walk_forward.load_features",
        lambda dataset, start=None, end=None: _walk_forward_frame(),
    )

    first = run_walk_forward_experiment(
        strategy.name,
        strategy,
        evaluation_path=config_path,
        strategy_config={"parameters": {"lookback_short": 2, "lookback_long": 3}},
    )
    second = run_walk_forward_experiment(
        strategy.name,
        strategy,
        evaluation_path=config_path,
        strategy_config={"parameters": {"lookback_short": 2, "lookback_long": 3}},
    )

    assert first.aggregate_summary["split_count"] == 5
    assert len(first.splits) == 5
    assert set(compute_metrics(first.splits[0].results_df)) == set(first.metrics)
    assert first.metrics == second.metrics
    assert first.aggregate_summary == second.aggregate_summary
