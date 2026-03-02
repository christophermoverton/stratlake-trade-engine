import pandas as pd
import numpy as np

from src.features.daily_features import compute_daily_features_v1


def _make_daily(symbol="AAPL", n=25, close_start=100.0):
    ts = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = [close_start + i for i in range(n)] #simple monotonic series
    df = pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "ts_utc": ts,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1000] * n,
            "source": ["test"] * n,
            "timeframe": ["1D"] * n,
            "date": [d.strftime("%Y-%m-%d") for d in ts],
        }
    )
    return df

def test_ret_1d_is_close_to_close():
    df = _make_daily(n=3, close_start=100.0) #closes: 100,101,102
    feats = compute_daily_features_v1(df)
    
    #Row 0 ret_1d is NaN
    assert np.isnan(feats.loc[0, "feature_ret_1d"])
    
    #Row 1: 101/100 - 1 = 0.01
    assert feats.loc[1, "feature_ret_1d"] == (101.0 / 100.0 - 1.0)
    
    #Row 2: 102/101 - 1
    assert feats.loc[2, "feature_ret_1d"] == (102.0 /101.0  -1.0)
    

def test_sma_20_deterministic():
    df = _make_daily(n=25, close_start=100.0)  #close=100..124
    feats = compute_daily_features_v1(df)
    
    #sma_20 becomes available at index 19 (20th observation)
    assert np.isnan(feats.loc[18, "feature_sma_20"])
    
    #At index 19, SMA is mean of 100..119 = (100+119)/2 = 109.5
    assert feats.loc[19, "feature_sma_20"] == 109.5
    
    #At index 20, SMA is mean of 101..120 = (101+120)/2 = 110.5
    assert feats.loc[20, "feature_sma_20"] == 110.5