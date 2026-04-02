from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class DailyFeatureConfig:
    ret_windows: tuple[int, ...] = (1,5,20)
    vol_window: int = 20
    sma_windows: tuple[int, ...] = (20, 50)
    std_ddof: int = 0 #ddof=0 > determministic popultation std
    
def compute_daily_features_v1(
    bars_daily: pd.DataFrame,
    *,
    cfg: DailyFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Compute daily features v1 from curated daily bars.

    Expects input to already be canonical/validated:
      - symbol, ts_utc, close, timeframe, date present
      - ts_utc sorted per symbol (we sort defensively)
      - timeframe should be "1D" for daily bars

    Output columns:
      symbol, ts_utc, timeframe, date, close, feature_*
    """
    cfg = cfg or DailyFeatureConfig()
    
    required = {"symbol", "ts_utc", "close", "timeframe", "date"}
    missing = sorted(list(required - set(bars_daily.columns)))
    if missing:
        raise ValueError(f"Missing required columns for feature computation: {missing}")
    
    if bars_daily.empty:
        #Return empty with the right schema
        base_cols = ["symbol", "ts_utc", "timeframe", "date", "close"]
        feat_cols = (
            [f"feature_ret_{w}d" for w in cfg.ret_windows]
            + [f"feature_vol_{cfg.vol_window}d"]
            + [f"feature_sma_{w}" for w in cfg.sma_windows]
            + ["feature_close_to_sma20"]
        )
        return pd.DataFrame(columns=base_cols + feat_cols)
    
    df = bars_daily.copy()
    
    #Defensive sort for determinist rolling behavior
    df = df.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(drop=True)
    
    g = df.groupby("symbol", sort=False)
    
    #Returns
    for w in cfg.ret_windows:
        df[f"feature_ret_{w}d"] = g["close"].transform(lambda s: s.astype("float64") / s.shift(w) - 1.0)
        
    
    #vol_20d (std of 1d returns)
    r1 = df["feature_ret_1d"].astype("float64")
    df[f"feature_vol_{cfg.vol_window}d"] = g[r1.name].transform(
        lambda s: s.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=cfg.std_ddof)
    )
    
    #SMAs
    for w in cfg.sma_windows:
        df[f"feature_sma_{w}"] = g["close"].transform(
            lambda s: s.astype("float64").rolling(w, min_periods=w).mean()
        )
        
    df["feature_close_to_sma20"] = df["close"].astype("float64") / df["feature_sma_20"] - 1.0
    
    #Output contract
    out_cols = ["symbol", "ts_utc", "timeframe", "date", "close"] + [
        c for c in df.columns if c.startswith("feature_")
    ]
    out = df.loc[:, out_cols].copy()
    
    #Keep types clean
    out["symbol"] = out["symbol"].astype("string")
    out["timeframe"] = out["timeframe"].astype("string")
    out["date"] = out["date"].astype("string")
    
    return out
    
    
    
    
    
    
    
    
    
    
    
    
        
