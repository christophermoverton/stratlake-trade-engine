from __future__ import annotations

from pathlib import Path
import pandas as pd

FEATURE_PATHS = {
    "1D": Path("data/curated/features_daily"),
    "1Min": Path("data/curated/features_1m"),
}

PRIMARY_KEY = ["symbol", "ts_utc", "timeframe"]

PARTITION_CONFIG = {
    "1D": ("year", lambda frame: pd.to_datetime(frame["ts_utc"], utc=True).dt.strftime("%Y")),
    "1Min": ("date", lambda frame: pd.to_datetime(frame["ts_utc"], utc=True).dt.strftime("%Y-%m-%d")),
}


def _load_existing_partition(partition_path: Path) -> pd.DataFrame:
    parquet_files = sorted(partition_path.glob("part-*.parquet"))
    if not parquet_files:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in parquet_files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _overwrite_partition(partition_path: Path, df: pd.DataFrame) -> None:
    partition_path.mkdir(parents=True, exist_ok=True)
    for existing_file in partition_path.glob("part-*.parquet"):
        existing_file.unlink()

    df.to_parquet(partition_path / "part-0.parquet", index=False)


def write_features(
    df: pd.DataFrame,
    timeframe: str,
) -> None:
    """
    Write feature dataframe to partitioned parquet.

    Partitioning:
        Daily  -> symbol/year
        1Min   -> symbol/date

    Idempotent via touched-partition overwrite after dedupe.

    Required columns:
        symbol, ts_utc, timeframe    
    """
    if timeframe not in FEATURE_PATHS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if not set(PRIMARY_KEY).issubset(df.columns):
        raise ValueError("Features DF is missing required key columns")

    if df.empty:
        return

    df = df.copy()
    df = df.drop_duplicates(PRIMARY_KEY, keep="last")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    base_path = FEATURE_PATHS[timeframe]
    partition_col, partition_value_factory = PARTITION_CONFIG[timeframe]
    df[partition_col] = partition_value_factory(df)

    partition_cols = ["symbol", partition_col]

    for keys, subdf in df.groupby(partition_cols):
        symbol, partition_value = keys

        partition_path = (
            base_path
            / f"symbol={symbol}"
            / f"{partition_col}={partition_value}"
        )

        existing_df = _load_existing_partition(partition_path)
        if not existing_df.empty:
            existing_df = existing_df.copy()
            existing_df["ts_utc"] = pd.to_datetime(existing_df["ts_utc"], utc=True)

        combined = pd.concat([existing_df, subdf], ignore_index=True)
        combined = combined.drop_duplicates(PRIMARY_KEY, keep="last")
        combined = combined.sort_values(PRIMARY_KEY, kind="mergesort").reset_index(drop=True)

        _overwrite_partition(partition_path, combined)
