from __future__ import annotations

from typing import Optional, Sequence

import duckdb
import pandas as pd

from src.data.catalog import CuratedPaths
from src.data.feature_writer import write_features
from src.data.loaders import LoadConfig, load_bars_1m, load_bars_daily
from src.features.daily_features import DailyFeatureConfig, compute_daily_features_v1
from src.features.minute_features import MinuteFeatureConfig, compute_minute_features_v1


def run_daily_feature_pipeline(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    load_cfg: Optional[LoadConfig] = None,
    feature_cfg: Optional[DailyFeatureConfig] = None,
    validate_contract: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    bars_daily = load_bars_daily(
        symbols,
        start_date=start_date,
        end_date=end_date,
        con=con,
        paths=paths,
        cfg=load_cfg,
        validate_contract=validate_contract,
        strict=strict,
    )
    features = compute_daily_features_v1(bars_daily, cfg=feature_cfg)
    write_features(features, "1D")
    return features


def run_minute_feature_pipeline(
    symbols: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    paths: Optional[CuratedPaths] = None,
    load_cfg: Optional[LoadConfig] = None,
    feature_cfg: Optional[MinuteFeatureConfig] = None,
    validate_contract: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    bars_1m = load_bars_1m(
        symbols,
        start_date=start_date,
        end_date=end_date,
        con=con,
        paths=paths,
        cfg=load_cfg,
        validate_contract=validate_contract,
        strict=strict,
    )
    features = compute_minute_features_v1(bars_1m, cfg=feature_cfg)
    write_features(features, "1Min")
    return features
