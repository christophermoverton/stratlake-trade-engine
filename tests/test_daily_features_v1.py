import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features.daily_features import compute_daily_features_v1


def _make_daily_bars(symbol: str = "AAPL", closes: list[float] | None = None) -> pd.DataFrame:
    if closes is None:
        closes = [100.0, 101.0, 103.0, 106.0, 110.0]
    ts = pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "ts_utc": ts,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000] * len(closes),
            "source": ["test"] * len(closes),
            "timeframe": ["1D"] * len(closes),
            "date": [t.strftime("%Y-%m-%d") for t in ts],
        }
    )


def test_daily_returns_match_expected_windows() -> None:
    closes = [100.0 + i for i in range(25)]
    features = compute_daily_features_v1(_make_daily_bars(closes=closes))

    expected_ret_1d = [np.nan] + [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]
    expected_ret_5d = [np.nan] * 5 + [(closes[i] / closes[i - 5]) - 1.0 for i in range(5, len(closes))]
    expected_ret_20d = [np.nan] * 20 + [(closes[i] / closes[i - 20]) - 1.0 for i in range(20, len(closes))]

    np.testing.assert_allclose(features["feature_ret_1d"], expected_ret_1d, equal_nan=True)
    np.testing.assert_allclose(features["feature_ret_5d"], expected_ret_5d, equal_nan=True)
    np.testing.assert_allclose(features["feature_ret_20d"], expected_ret_20d, equal_nan=True)


def test_daily_volatility_and_moving_averages_have_expected_warmup_and_values() -> None:
    closes = [100.0 + i for i in range(55)]
    features = compute_daily_features_v1(_make_daily_bars(closes=closes))
    ret_1d = np.array([(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))], dtype=float)

    assert np.isnan(features.loc[19, "feature_vol_20d"])
    assert np.isnan(features.loc[18, "feature_sma20"])
    assert np.isnan(features.loc[48, "feature_sma50"])
    assert np.isnan(features.loc[18, "feature_sma_20"])
    assert np.isnan(features.loc[18, "feature_close_to_sma20"])

    expected_vol_idx_20 = np.std(ret_1d[:20], ddof=0)
    expected_sma20_idx_19 = np.mean(closes[:20])
    expected_sma20_idx_20 = np.mean(closes[1:21])
    expected_sma50_idx_49 = np.mean(closes[:50])
    expected_close_to_sma20_idx_19 = (closes[19] / expected_sma20_idx_19) - 1.0

    assert np.isclose(features.loc[20, "feature_vol_20d"], expected_vol_idx_20)
    assert np.isclose(features.loc[19, "feature_sma20"], expected_sma20_idx_19)
    assert np.isclose(features.loc[20, "feature_sma20"], expected_sma20_idx_20)
    assert np.isclose(features.loc[19, "feature_sma_20"], expected_sma20_idx_19)
    assert np.isclose(features.loc[49, "feature_sma50"], expected_sma50_idx_49)
    assert np.isclose(features.loc[19, "feature_close_to_sma20"], expected_close_to_sma20_idx_19)


def test_daily_features_sort_deterministically() -> None:
    aapl = _make_daily_bars("AAPL", [100.0, 101.0, 102.0, 103.0])
    msft = _make_daily_bars("MSFT", [200.0, 202.0, 204.0, 206.0])
    shuffled = pd.concat([aapl, msft], ignore_index=True).iloc[[5, 0, 7, 2, 6, 1, 4, 3]].reset_index(drop=True)

    features_first = compute_daily_features_v1(shuffled)
    features_second = compute_daily_features_v1(shuffled)
    expected_order = shuffled.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(drop=True)

    assert list(features_first["symbol"]) == list(expected_order["symbol"])
    assert list(features_first["ts_utc"]) == list(expected_order["ts_utc"])
    pdt.assert_frame_equal(features_first, features_second)


def test_daily_features_empty_input_returns_expected_schema() -> None:
    empty = _make_daily_bars(closes=[]).iloc[0:0]

    features = compute_daily_features_v1(empty)

    assert features.empty
    assert list(features.columns) == [
        "symbol",
        "ts_utc",
        "timeframe",
        "date",
        "feature_ret_1d",
        "feature_ret_5d",
        "feature_ret_20d",
        "feature_vol_20d",
        "feature_sma_20",
        "feature_sma_50",
        "feature_close_to_sma20",
    ]
