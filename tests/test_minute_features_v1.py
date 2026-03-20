import math

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.minute_features import MinuteFeatureConfig, compute_minute_features_v1


def _make_minute_bars(
    symbol: str = "AAPL",
    minute_offsets: list[int] | None = None,
    close_start: float = 100.0,
    volume_start: float = 1_000.0,
) -> pd.DataFrame:
    if minute_offsets is None:
        minute_offsets = list(range(12))
    base = pd.Timestamp("2025-01-01T09:30:00Z")
    timestamps = [base + pd.Timedelta(minutes=offset) for offset in minute_offsets]
    closes = [close_start + offset for offset in minute_offsets]
    volumes = [volume_start + (10.0 * offset) for offset in minute_offsets]

    return pd.DataFrame(
        {
            "symbol": [symbol] * len(timestamps),
            "ts_utc": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "source": ["test"] * len(timestamps),
            "timeframe": ["1Min"] * len(timestamps),
            "date": [t.strftime("%Y-%m-%d") for t in timestamps],
        }
    )


def test_minute_returns_respect_exact_lags_and_missing_minutes() -> None:
    offsets = [0, 1, 2, 4, 5, 6, 8, 9, 10]
    features = compute_minute_features_v1(
        _make_minute_bars(minute_offsets=offsets),
        cfg=MinuteFeatureConfig(min_periods_roll=1),
    )
    by_ts = features.set_index("ts_utc")

    ts_931 = pd.Timestamp("2025-01-01T09:31:00Z")
    ts_934 = pd.Timestamp("2025-01-01T09:34:00Z")
    ts_935 = pd.Timestamp("2025-01-01T09:35:00Z")
    ts_938 = pd.Timestamp("2025-01-01T09:38:00Z")

    assert np.isclose(by_ts.loc[ts_931, "feature_ret_1m"], (101.0 / 100.0) - 1.0)
    assert math.isnan(by_ts.loc[ts_934, "feature_ret_1m"])
    assert np.isclose(by_ts.loc[ts_935, "feature_ret_1m"], (105.0 / 104.0) - 1.0)
    assert np.isclose(by_ts.loc[ts_935, "feature_ret_5m"], (105.0 / 100.0) - 1.0)
    assert math.isnan(by_ts.loc[ts_938, "feature_ret_5m"])


def test_minute_rolling_features_match_expected_values_on_irregular_timestamps() -> None:
    offsets = [0, 1, 2, 4, 5, 6]
    bars = _make_minute_bars(minute_offsets=offsets)
    features = compute_minute_features_v1(
        bars,
        cfg=MinuteFeatureConfig(min_periods_roll=3),
    )
    by_ts = features.set_index("ts_utc")

    ts_931 = pd.Timestamp("2025-01-01T09:31:00Z")
    ts_934 = pd.Timestamp("2025-01-01T09:34:00Z")
    ts_935 = pd.Timestamp("2025-01-01T09:35:00Z")
    ts_936 = pd.Timestamp("2025-01-01T09:36:00Z")

    assert math.isnan(by_ts.loc[ts_934, "feature_vol_30m"])
    assert math.isnan(by_ts.loc[ts_934, "feature_rv_30m"])
    assert math.isnan(by_ts.loc[ts_931, "feature_vol_ratio"])

    ret_1m_values = np.array(
        [
            (101.0 / 100.0) - 1.0,
            (102.0 / 101.0) - 1.0,
            (105.0 / 104.0) - 1.0,
            (106.0 / 105.0) - 1.0,
        ],
        dtype=float,
    )
    expected_vol = np.std(ret_1m_values, ddof=0)
    expected_rv = math.sqrt(np.sum(ret_1m_values**2))
    expected_volume_mean = np.mean([1000.0, 1010.0, 1020.0, 1040.0, 1050.0, 1060.0])
    expected_volume_ratio = 1060.0 / expected_volume_mean

    assert np.isclose(by_ts.loc[ts_936, "feature_vol_30m"], expected_vol)
    assert np.isclose(by_ts.loc[ts_936, "feature_rv_30m"], expected_rv)
    assert np.isclose(by_ts.loc[ts_936, "feature_vol_ratio"], expected_volume_ratio)
    assert np.isclose(by_ts.loc[ts_935, "feature_vol_ratio"], 1050.0 / np.mean([1000.0, 1010.0, 1020.0, 1040.0, 1050.0]))


def test_minute_features_sort_deterministically_on_irregular_input() -> None:
    aapl = _make_minute_bars("AAPL", [0, 1, 2, 4, 5, 6], close_start=100.0)
    msft = _make_minute_bars("MSFT", [0, 2, 3, 5, 6, 8], close_start=200.0)
    shuffled = pd.concat([aapl, msft], ignore_index=True).iloc[[5, 0, 8, 2, 10, 1, 7, 3, 11, 4, 9, 6]].reset_index(
        drop=True
    )

    features_first = compute_minute_features_v1(shuffled, cfg=MinuteFeatureConfig(min_periods_roll=1))
    features_second = compute_minute_features_v1(shuffled, cfg=MinuteFeatureConfig(min_periods_roll=1))
    expected_order = shuffled.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(drop=True)

    assert list(features_first["symbol"]) == list(expected_order["symbol"])
    assert list(features_first["ts_utc"]) == list(expected_order["ts_utc"])
    pdt.assert_frame_equal(features_first, features_second)


def test_minute_features_empty_input_returns_expected_schema() -> None:
    empty = _make_minute_bars(minute_offsets=[]).iloc[0:0]

    features = compute_minute_features_v1(empty)

    assert features.empty
    assert list(features.columns) == [
        "symbol",
        "ts_utc",
        "timeframe",
        "date",
        "feature_ret_1m",
        "feature_ret_5m",
        "feature_vol_30m",
        "feature_rv_30m",
        "feature_vol_ratio",
    ]
