from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MinuteFeatureConfig:
    """
    v1 1-minute feature set configuration.
    """
    ret_lags_min: tuple[int, ...] = (1, 5)   # ret_1m, ret_5m
    roll_window: str = "30min"              # vol_30m, rv_30m, vol_mean_30m
    std_ddof: int = 0                       # deterministic population std
    min_periods_roll: int = 10              # require at least N obs in 30min window


def compute_minute_features_v1(
    bars_1m: pd.DataFrame,
    *,
    cfg: MinuteFeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Compute 1-minute features v1 for short-horizon analytics.

    Features:
      - feature_ret_1m: exact 1-minute close-to-close return
      - feature_ret_5m: exact 5-minute close-to-close return
      - feature_vol_30m: rolling std of feature_ret_1m over last 30 minutes (time-based)
      - feature_rv_30m: realized vol proxy sqrt(sum(ret^2)) over last 30 minutes (time-based)
      - feature_vol_ratio: volume / rolling mean(volume) over last 30 minutes (time-based)

    Missing minutes handling:
      - ret_1m and ret_5m are NaN unless an EXACT bar exists at t-1m / t-5m
      - rolling features operate on whatever data exists in the 30-minute lookback,
        but will be NaN until min_periods_roll is met.

    Output contract:
      symbol, ts_utc, timeframe, date, feature_*
    """
    cfg = cfg or MinuteFeatureConfig()

    required = {"symbol", "ts_utc", "close", "volume", "timeframe", "date"}
    missing = sorted(list(required - set(bars_1m.columns)))
    if missing:
        raise ValueError(f"Missing required columns for 1m feature computation: {missing}")

    # Empty -> return empty with expected schema
    if bars_1m.empty:
        base_cols = ["symbol", "ts_utc", "timeframe", "date"]
        feat_cols = [
            "feature_ret_1m",
            "feature_ret_5m",
            "feature_vol_30m",
            "feature_rv_30m",
            "feature_vol_ratio",
        ]
        return pd.DataFrame(columns=base_cols + feat_cols)

    df = bars_1m.copy()

    # Defensive sort for determinism
    df = df.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(drop=True)

    # Ensure ts_utc is datetime (tz-aware preferred; contract validator should enforce)
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    if df["ts_utc"].isna().any():
        raise ValueError("ts_utc contains unparseable values after to_datetime().")

    # Make sure close/volume are numeric for math
    df["close"] = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("float64")

    # ---- Exact-lag returns via key lookup (symbol, ts_utc) ----
    # Build a keyed frame for exact timestamp lookups
    key = df[["symbol", "ts_utc", "close"]].rename(columns={"ts_utc": "ts_key", "close": "close_key"})

    def _exact_return(lag_min: int) -> pd.Series:
        target_ts = df["ts_utc"] - pd.Timedelta(minutes=lag_min)
        lookup = df[["symbol"]].copy()
        lookup["ts_key"] = target_ts

        joined = lookup.merge(key, on=["symbol", "ts_key"], how="left")
        prev_close = joined["close_key"].astype("float64")

        return df["close"] / prev_close - 1.0

    df["feature_ret_1m"] = _exact_return(1)
    df["feature_ret_5m"] = _exact_return(5)

    # ---- Time-based rolling features (per symbol) ----
    # Use groupby apply with ts_utc index; works on irregular timestamps
    # ---- Time-based rolling features (per symbol) via groupby.rolling ----
    # Use ts_utc as index for time-based windows, then merge results back by (symbol, ts_utc)
    # ---- Time-based rolling features (per symbol) via groupby(...).rolling(on="ts_utc") ----
    tmp = df[["symbol", "ts_utc", "feature_ret_1m", "volume"]].copy()

    # Must be sorted for time-based rolling windows
    tmp = tmp.sort_values(["symbol", "ts_utc"], kind="mergesort").reset_index(drop=True)

    grp = tmp.groupby("symbol", sort=False)

    # Rolling std of 1m returns
    vol_30m = (
        grp.rolling(cfg.roll_window, on="ts_utc", min_periods=cfg.min_periods_roll)["feature_ret_1m"]
        .std(ddof=cfg.std_ddof)
        .reset_index()
        .rename(columns={"feature_ret_1m": "feature_vol_30m"})
    )

    # Realized vol proxy: sqrt(sum(ret^2))
    tmp["ret1_sq"] = tmp["feature_ret_1m"] ** 2
    rv_sum = (
        tmp.groupby("symbol", sort=False)
        .rolling(cfg.roll_window, on="ts_utc", min_periods=cfg.min_periods_roll)["ret1_sq"]
        .sum()
        .reset_index()
        .rename(columns={"ret1_sq": "rv_sum_30m"})
    )
    rv_sum["feature_rv_30m"] = np.sqrt(rv_sum["rv_sum_30m"])
    rv_30m = rv_sum.drop(columns=["rv_sum_30m"])

    # Rolling mean volume and volume ratio
    vol_mean = (
        grp.rolling(cfg.roll_window, on="ts_utc", min_periods=cfg.min_periods_roll)["volume"]
        .mean()
        .reset_index()
        .rename(columns={"volume": "vol_mean_30m"})
    )

    # Merge rolling features back into df
    df = df.merge(vol_30m[["symbol", "ts_utc", "feature_vol_30m"]], on=["symbol", "ts_utc"], how="left")
    df = df.merge(rv_30m[["symbol", "ts_utc", "feature_rv_30m"]], on=["symbol", "ts_utc"], how="left")
    df = df.merge(vol_mean[["symbol", "ts_utc", "vol_mean_30m"]], on=["symbol", "ts_utc"], how="left")

    df["feature_vol_ratio"] = df["volume"] / df["vol_mean_30m"]
    df = df.drop(columns=["vol_mean_30m"])
    
    # Robustly ensure symbol is a column (groupby/apply sometimes leaves it in the index)
    if "symbol" not in df.columns:
        df = df.reset_index()

    # Output contract
    out_cols = ["symbol", "ts_utc", "timeframe", "date"] + [c for c in df.columns if c.startswith("feature_")]
    out = df.loc[:, out_cols].copy()

    # Keep stringy columns consistent
    out["symbol"] = out["symbol"].astype("string")
    out["timeframe"] = out["timeframe"].astype("string")
    out["date"] = out["date"].astype("string")

    return out