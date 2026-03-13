from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PRIMARY_KEY = ["symbol", "ts_utc", "timeframe"]
DEFAULT_QA_ROOT = Path("artifacts/qa/features")


def _dataset_name_for_timeframe(timeframe: str) -> str:
    if timeframe == "1D":
        return "features_daily"
    if timeframe == "1Min":
        return "features_1m"
    raise ValueError(f"Unsupported timeframe for QA: {timeframe}")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(column for column in df.columns if column.startswith("feature_"))


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_metrics(df: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    row_count = len(df.index)

    for column in feature_columns:
        series = df[column]
        numeric = _safe_numeric(series)
        inf_mask = pd.Series(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)), index=series.index)
        finite_values = numeric[~series.isna() & ~inf_mask]

        null_pct = float(series.isna().mean()) if row_count else 0.0
        metrics[f"{column}_null_pct"] = round(null_pct, 6)
        metrics[f"{column}_inf_count"] = int(inf_mask.sum())
        metrics[f"{column}_min"] = None if finite_values.empty else float(finite_values.min())
        metrics[f"{column}_max"] = None if finite_values.empty else float(finite_values.max())

    return metrics


def _status_for_summary(
    *,
    total_rows: int,
    duplicate_row_count: int,
    duplicate_key_count: int,
    inf_count: int,
    has_nulls: bool,
) -> str:
    if total_rows == 0 or duplicate_row_count > 0 or duplicate_key_count > 0 or inf_count > 0:
        return "FAIL"
    if has_nulls:
        return "WARN"
    return "PASS"


def _base_summary(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    timeframe: str,
    expected_symbols: Sequence[str] | None,
) -> dict[str, object]:
    total_rows = int(len(df.index))
    duplicate_row_count = int(df.duplicated().sum())
    duplicate_key_count = int(df.duplicated(PRIMARY_KEY).sum()) if set(PRIMARY_KEY).issubset(df.columns) else total_rows
    observed_symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in df.columns else []
    normalized_expected = sorted({symbol.strip().upper() for symbol in (expected_symbols or []) if symbol and symbol.strip()})
    observed_upper = {symbol.upper() for symbol in observed_symbols}
    missing_symbols = [symbol for symbol in normalized_expected if symbol not in observed_upper]
    coverage_pct = (
        round((len(normalized_expected) - len(missing_symbols)) / len(normalized_expected), 6)
        if normalized_expected
        else 1.0
    )
    feature_columns = _feature_columns(df)
    feature_metrics = _feature_metrics(df, feature_columns)
    inf_count = sum(int(feature_metrics[f"{column}_inf_count"]) for column in feature_columns)
    has_nulls = any(float(feature_metrics[f"{column}_null_pct"]) > 0.0 for column in feature_columns)

    return {
        "dataset_name": dataset_name,
        "timeframe": timeframe,
        "total_rows": total_rows,
        "duplicate_row_count": duplicate_row_count,
        "duplicate_key_count": duplicate_key_count,
        "expected_symbol_count": int(len(normalized_expected)),
        "observed_symbol_count": int(len(observed_symbols)),
        "missing_symbol_count": int(len(missing_symbols)),
        "symbol_coverage_pct": round(coverage_pct, 6),
        "missing_symbols": "|".join(missing_symbols),
        "dataset_status": _status_for_summary(
            total_rows=total_rows,
            duplicate_row_count=duplicate_row_count,
            duplicate_key_count=duplicate_key_count,
            inf_count=inf_count,
            has_nulls=has_nulls,
        ),
        **feature_metrics,
    }


def build_feature_qa_summaries(
    df: pd.DataFrame,
    *,
    timeframe: str,
    expected_symbols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_name = _dataset_name_for_timeframe(timeframe)
    working = df.copy()

    if "ts_utc" in working.columns:
        working["ts_utc"] = pd.to_datetime(working["ts_utc"], utc=True)

    by_symbol_rows: list[dict[str, object]] = []
    if "symbol" in working.columns and not working.empty:
        for symbol in sorted(working["symbol"].dropna().astype(str).unique().tolist()):
            symbol_df = working[working["symbol"].astype(str) == symbol].reset_index(drop=True)
            row = _base_summary(
                symbol_df,
                dataset_name=dataset_name,
                timeframe=timeframe,
                expected_symbols=[symbol],
            )
            row["symbol"] = symbol
            by_symbol_rows.append(row)
    elif expected_symbols:
        for symbol in sorted({item.strip().upper() for item in expected_symbols if item and item.strip()}):
            row = _base_summary(
                working.iloc[0:0],
                dataset_name=dataset_name,
                timeframe=timeframe,
                expected_symbols=[symbol],
            )
            row["symbol"] = symbol
            by_symbol_rows.append(row)

    by_symbol = pd.DataFrame(by_symbol_rows)
    global_summary = pd.DataFrame(
        [_base_summary(working, dataset_name=dataset_name, timeframe=timeframe, expected_symbols=expected_symbols)]
    )
    return by_symbol, global_summary


def _merge_with_existing(path: Path, df: pd.DataFrame, key_columns: Sequence[str]) -> pd.DataFrame:
    working = df.copy()
    for column in key_columns:
        if column not in working.columns:
            working[column] = pd.Series(dtype="object")

    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            incoming_keys = working[list(key_columns)].drop_duplicates()
            existing = existing.merge(incoming_keys, on=list(key_columns), how="left", indicator=True)
            existing = existing[existing["_merge"] == "left_only"].drop(columns=["_merge"])
        combined = pd.concat([existing, working], ignore_index=True, sort=False)
    else:
        combined = working

    base_columns = [column for column in key_columns if column in combined.columns]
    remaining_columns = sorted(column for column in combined.columns if column not in base_columns)
    combined = combined.reindex(columns=base_columns + remaining_columns)
    combined = combined.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    return combined


def write_feature_qa_artifacts(
    df: pd.DataFrame,
    *,
    timeframe: str,
    expected_symbols: Sequence[str] | None = None,
    qa_root: Path | None = None,
) -> tuple[Path, Path]:
    root = qa_root or DEFAULT_QA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    by_symbol, global_summary = build_feature_qa_summaries(
        df,
        timeframe=timeframe,
        expected_symbols=expected_symbols,
    )

    by_symbol_path = root / "qa_features_summary_by_symbol.csv"
    global_path = root / "qa_features_summary_global.csv"

    merged_by_symbol = _merge_with_existing(by_symbol_path, by_symbol, ["dataset_name", "timeframe", "symbol"])
    merged_global = _merge_with_existing(global_path, global_summary, ["dataset_name", "timeframe"])

    merged_by_symbol.to_csv(by_symbol_path, index=False)
    merged_global.to_csv(global_path, index=False)
    return by_symbol_path, global_path
