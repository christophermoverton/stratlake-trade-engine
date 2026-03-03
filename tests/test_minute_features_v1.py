import numpy as np
import pandas as pd

from src.features.minute_features import compute_minute_features_v1, MinuteFeatureConfig


def _make_1m(symbol: str, start: str, minutes: int, missing: set[int] | None = None):
    missing = missing or set()
    ts = []
    close = []
    vol = []
    base = pd.Timestamp(start, tz="UTC")
    c = 100.0

    for i in range(minutes):
        if i in missing:
            c += 1.0
            continue
        ts.append(base + pd.Timedelta(minutes=i))
        close.append(c)
        vol.append(100 + i)
        c += 1.0

    df = pd.DataFrame(
        {
            "symbol": [symbol] * len(ts),
            "ts_utc": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": vol,
            "source": ["test"] * len(ts),
            "timeframe": ["1Min"] * len(ts),
            "date": [t.strftime("%Y-%m-%d") for t in ts],
        }
    )
    return df


def test_returns_exact_lag_nan_on_missing_minutes():
    # Missing minute at i=1 means:
    # at t=00:02, t-1m would be 00:01 which doesn't exist => ret_1m should be NaN
    df = _make_1m("AAPL", "2025-01-01T00:00:00Z", minutes=6, missing={1})
    feats = compute_minute_features_v1(df, cfg=MinuteFeatureConfig(min_periods_roll=1))

    # Find row with ts=00:02
    t = pd.Timestamp("2025-01-01T00:02:00Z")
    row = feats[feats["ts_utc"] == t].iloc[0]
    assert np.isnan(row["feature_ret_1m"])

    # ret_5m at t=00:05 needs 00:00 exact present -> should exist
    t5 = pd.Timestamp("2025-01-01T00:05:00Z")
    row5 = feats[feats["ts_utc"] == t5].iloc[0]
    # close at 00:05 is 105, close at 00:00 is 100 => 0.05
    assert row5["feature_ret_5m"] == (105.0 / 100.0 - 1.0)


def test_multi_symbol_small_window_runs_and_outputs_contract():
    df1 = _make_1m("AAPL", "2025-01-01T00:00:00Z", minutes=120, missing={10, 11, 50})
    df2 = _make_1m("MSFT", "2025-01-01T00:00:00Z", minutes=120, missing={5, 90})
    df = pd.concat([df1, df2], ignore_index=True)

    feats = compute_minute_features_v1(df)

    # Contract columns
    assert list(feats.columns[:4]) == ["symbol", "ts_utc", "timeframe", "date"]
    assert any(c.startswith("feature_") for c in feats.columns)

    # Should not explode and should have both symbols
    assert set(feats["symbol"].unique().tolist()) == {"AAPL", "MSFT"}