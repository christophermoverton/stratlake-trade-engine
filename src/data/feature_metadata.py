from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.load_features import FeaturePaths, load_features

DEFAULT_FEATURE_METADATA_PATH = Path("artifacts/features/feature_metadata.json")
DEFAULT_FEATURE_REGISTRY_PATH = Path("configs/features.yml")
DEFAULT_FEATURE_DATA_ROOT = Path("data/curated")
TARGET_DATASETS = ("features_1m", "features_daily")
CORE_COLUMNS = {"symbol", "ts_utc", "timeframe", "date"}


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Feature registry must be a mapping: {path}")
    return payload


def _registry_entry_for_dataset(
    registry: dict[str, dict[str, Any]],
    dataset_name: str,
) -> dict[str, Any]:
    prefix = f"{dataset_name}_"
    matches = [key for key in sorted(registry) if key == dataset_name or key.startswith(prefix)]
    if not matches:
        raise ValueError(f"Missing feature registry entry for dataset: {dataset_name}")
    entry = registry[matches[0]]
    if not isinstance(entry, dict):
        raise ValueError(f"Feature registry entry must be a mapping: {matches[0]}")
    return entry


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column for column in df.columns if column.startswith("feature_"))


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if df.empty:
        return {"start": None, "end": None}

    if "date" in df.columns:
        non_null_dates = df["date"].dropna().astype(str)
        if not non_null_dates.empty:
            ordered = sorted(non_null_dates.tolist())
            return {"start": ordered[0], "end": ordered[-1]}

    if "ts_utc" in df.columns:
        ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            return {
                "start": ts.min().strftime("%Y-%m-%d"),
                "end": ts.max().strftime("%Y-%m-%d"),
            }

    return {"start": None, "end": None}


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _feature_statistics(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float | None]]:
    row_count = len(df.index)
    stats: dict[str, dict[str, float | None]] = {}

    for column in feature_columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        finite_values = numeric[finite_mask]

        stats[column] = {
            "null_pct": round(float(series.isna().mean()) if row_count else 0.0, 6),
            "min": _finite_float(finite_values.min()) if not finite_values.empty else None,
            "max": _finite_float(finite_values.max()) if not finite_values.empty else None,
        }

    return stats


def _data_types(df: pd.DataFrame) -> dict[str, str]:
    return {column: str(df[column].dtype) for column in df.columns}


def build_feature_metadata_summary(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    source_dataset: str,
    feature_list: list[str],
) -> dict[str, Any]:
    feature_columns = _feature_columns(df)
    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in df.columns else []

    return {
        "dataset_name": dataset_name,
        "source_dataset": source_dataset,
        "feature_list": list(feature_list),
        "feature_count": int(len(feature_list)),
        "schema": {
            "column_names": list(df.columns),
            "feature_columns": feature_columns,
            "data_types": _data_types(df),
        },
        "metrics": {
            "row_count": int(len(df.index)),
            "symbol_coverage": int(len(symbols)),
            "date_range": _date_range(df),
        },
        "feature_statistics": _feature_statistics(df, feature_columns),
    }


def export_feature_metadata(
    *,
    artifact_path: Path | None = None,
    registry_path: Path | None = None,
    feature_data_root: Path | None = None,
) -> Path:
    resolved_artifact_path = artifact_path or DEFAULT_FEATURE_METADATA_PATH
    resolved_registry_path = registry_path or DEFAULT_FEATURE_REGISTRY_PATH
    resolved_feature_data_root = feature_data_root or DEFAULT_FEATURE_DATA_ROOT

    registry = _load_registry(resolved_registry_path)
    feature_paths = FeaturePaths(root=resolved_feature_data_root)
    datasets: list[dict[str, Any]] = []

    for dataset_name in TARGET_DATASETS:
        entry = _registry_entry_for_dataset(registry, dataset_name)
        feature_list = sorted(str(item) for item in entry.get("features", []))
        # Metadata export summarizes stored datasets as-is, even when they contain
        # warm-up nulls or outlier values that strategy execution would later validate.
        df = load_features(dataset_name, paths=feature_paths, validate_integrity=False)
        datasets.append(
            build_feature_metadata_summary(
                dataset_name,
                df,
                source_dataset=str(entry.get("source_dataset", "")),
                feature_list=feature_list,
            )
        )

    payload = {"datasets": datasets}
    resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_artifact_path
