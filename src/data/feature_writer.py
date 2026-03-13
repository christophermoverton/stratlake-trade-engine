from __future__ import annotations

from pathlib import Path
import pandas as pd

FEATURE_PATHS = {
    "1D": Path("data/curated/features_daily"),
    "1Min": Path("data/curated/features_1m"),
}

PRIMARY_KEY = ["symbol", "ts_utc", "timeframe"]

def write_features(
    df: pd.DataFrame,
    timeframe: str,
) -> None:
    """
    Write feature dataframe to partitioned parquet.

    Partitioning:
        Daily  -> symbol/date
        1Min   -> symbol/date

    Idempotent via partition overwrite.

    Required columns:
        symbol, ts_utc, timeframe    
    """
    if not set(PRIMARY_KEY).issubset(df.columns):
        raise ValueError("Features DF is missing required key columns")
    
    df = df.drop_duplicates(PRIMARY_KEY)
    
    base_path = FEATURE_PATHS[timeframe]
    
    if timeframe not in FEATURE_PATHS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    #derive partition column
    df["date"] = pd.to_datetime(df["ts_utc"]).dt.date.astype(str)
    
    partition_cols = ["symbol", "date"]
    
    for keys, subdf in df.groupby(partition_cols):
        symbol, date = keys
    
        partition_path = (
            base_path 
            / f"symbol={symbol}"
            / f"date={date}"
        )
        
        partition_path.mkdir(parents=True, exist_ok=True)
        
        file_path = partition_path / "part-0.parquet"
        
        subdf.to_parquet(file_path, index=False)