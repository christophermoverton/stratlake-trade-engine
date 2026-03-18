from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import duckdb
import pandas as pd

from src.data.catalog import CuratedPaths
from src.data.feature_metadata import export_feature_metadata
from src.data.feature_qa import write_feature_qa_artifacts
from src.data.feature_writer import write_features
from src.data.loaders import LoadConfig, _default_curated_paths, load_bars_1m, load_bars_daily
from src.features.daily_features import DailyFeatureConfig, compute_daily_features_v1
from src.features.minute_features import MinuteFeatureConfig, compute_minute_features_v1

LOGGER = logging.getLogger(__name__)


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
    qa_artifacts_root: Optional[Path] = None,
) -> pd.DataFrame:
    resolved_paths = paths or _default_curated_paths()
    bars_daily = load_bars_daily(
        symbols,
        start_date=start_date,
        end_date=end_date,
        con=con,
        paths=resolved_paths,
        cfg=load_cfg,
        validate_contract=validate_contract,
        strict=strict,
    )
    if bars_daily.empty:
        LOGGER.warning(
            "No bars loaded timeframe=%s start=%s end=%s marketlake_root=%s parquet_scan_path=%s",
            "1D",
            start_date,
            end_date,
            resolved_paths.root,
            resolved_paths.dataset_glob("bars_daily"),
        )
    features = compute_daily_features_v1(bars_daily, cfg=feature_cfg)
    write_features(features, "1D")
    write_feature_qa_artifacts(
        features,
        timeframe="1D",
        expected_symbols=symbols,
        qa_root=qa_artifacts_root,
    )
    export_feature_metadata()
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
    qa_artifacts_root: Optional[Path] = None,
) -> pd.DataFrame:
    resolved_paths = paths or _default_curated_paths()
    bars_1m = load_bars_1m(
        symbols,
        start_date=start_date,
        end_date=end_date,
        con=con,
        paths=resolved_paths,
        cfg=load_cfg,
        validate_contract=validate_contract,
        strict=strict,
    )
    if bars_1m.empty:
        LOGGER.warning(
            "No bars loaded timeframe=%s start=%s end=%s marketlake_root=%s parquet_scan_path=%s",
            "1Min",
            start_date,
            end_date,
            resolved_paths.root,
            resolved_paths.dataset_glob("bars_1m"),
        )
    features = compute_minute_features_v1(bars_1m, cfg=feature_cfg)
    write_features(features, "1Min")
    write_feature_qa_artifacts(
        features,
        timeframe="1Min",
        expected_symbols=symbols,
        qa_root=qa_artifacts_root,
    )
    export_feature_metadata()
    return features
