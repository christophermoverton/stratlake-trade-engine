from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc")


class AlphaCrossSectionError(ValueError):
    """Raised when deterministic cross-sectional access is invalid or ambiguous."""


def get_cross_section(
    df: pd.DataFrame,
    ts_utc: str | pd.Timestamp,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return one deterministic same-timestamp asset slice."""

    validated = validate_cross_section_input(df)
    requested_timestamp = _coerce_timestamp(ts_utc, field_name="ts_utc")
    selected_columns = _resolve_selected_columns(validated, columns=columns)

    cross_section = validated.loc[validated["ts_utc"].eq(requested_timestamp), selected_columns].copy(deep=True)
    if cross_section.empty:
        raise AlphaCrossSectionError(
            "No rows found for requested cross-section timestamp "
            f"{requested_timestamp.isoformat().replace('+00:00', 'Z')}."
        )

    cross_section.attrs = {}
    return cross_section


def list_cross_section_timestamps(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return distinct available cross-section timestamps in deterministic UTC order."""

    validated = validate_cross_section_input(df)
    unique_timestamps = validated["ts_utc"].drop_duplicates().tolist()
    return [timestamp for timestamp in unique_timestamps]


def iter_cross_sections(
    df: pd.DataFrame,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield deterministic `(timestamp, cross_section)` pairs in ascending UTC order."""

    validated = validate_cross_section_input(df)
    selected_columns = _resolve_selected_columns(validated, columns=columns)

    for timestamp in validated["ts_utc"].drop_duplicates().tolist():
        cross_section = validated.loc[validated["ts_utc"].eq(timestamp), selected_columns].copy(deep=True)
        cross_section.attrs = {}
        yield timestamp, cross_section


def validate_cross_section_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate deterministic multi-symbol input for cross-sectional utilities."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Cross-sectional input must be a pandas DataFrame.")
    if df.empty:
        raise AlphaCrossSectionError("Cross-sectional input must not be empty.")

    missing = [column for column in STRUCTURAL_COLUMNS if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise AlphaCrossSectionError(
            f"Cross-sectional input must include required columns: {formatted}."
        )

    if df["symbol"].isna().any():
        raise AlphaCrossSectionError("Cross-sectional input contains null values in 'symbol'.")

    normalized = df.copy(deep=True)
    normalized.attrs = {}
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["ts_utc"] = _coerce_timestamp_series(normalized["ts_utc"])

    keys = normalized.loc[:, ["symbol", "ts_utc"]]
    sorted_keys = keys.sort_values(["symbol", "ts_utc"], kind="stable")
    if not keys.index.equals(sorted_keys.index):
        first_mismatch = next(
            position
            for position, (actual_index, expected_index) in enumerate(
                zip(keys.index, sorted_keys.index, strict=False)
            )
            if actual_index != expected_index
        )
        bad_row = keys.iloc[first_mismatch]
        raise AlphaCrossSectionError(
            "Cross-sectional input must be sorted by (symbol, ts_utc). "
            f"First out-of-order row: symbol={bad_row['symbol']}, ts_utc={bad_row['ts_utc']}."
        )

    duplicate_mask = keys.duplicated(subset=["symbol", "ts_utc"], keep=False)
    if duplicate_mask.any():
        first_duplicate = keys.loc[duplicate_mask, ["symbol", "ts_utc"]].iloc[0]
        raise AlphaCrossSectionError(
            "Cross-sectional input must not contain duplicate (symbol, ts_utc) rows. "
            f"First duplicate key: symbol={first_duplicate['symbol']}, ts_utc={first_duplicate['ts_utc']}."
        )

    return normalized


def _resolve_selected_columns(
    df: pd.DataFrame,
    *,
    columns: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if columns is None:
        return list(df.columns)

    if not isinstance(columns, (list, tuple)):
        raise TypeError("columns must be a list, tuple, or None.")
    if not columns:
        raise AlphaCrossSectionError("columns must contain at least one column when provided explicitly.")

    normalized_columns: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise AlphaCrossSectionError("columns entries must be non-empty strings.")
        if column not in df.columns:
            raise AlphaCrossSectionError(f"Cross-sectional input must include requested column '{column}'.")
        if column in STRUCTURAL_COLUMNS:
            continue
        if column in seen:
            raise AlphaCrossSectionError(f"columns must not contain duplicates. Found duplicate: '{column}'.")
        normalized_columns.append(column)
        seen.add(column)

    return [*STRUCTURAL_COLUMNS, *normalized_columns]


def _coerce_timestamp(value: str | pd.Timestamp, *, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AlphaCrossSectionError(
            f"{field_name} must be a valid timestamp or timestamp-like string."
        ) from exc

    if pd.isna(timestamp):
        raise AlphaCrossSectionError(f"{field_name} must be a valid timestamp or timestamp-like string.")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _coerce_timestamp_series(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise AlphaCrossSectionError("Cross-sectional input contains unparsable 'ts_utc' values.")
    return timestamps
