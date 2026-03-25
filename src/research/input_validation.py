from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STRATEGY_INPUT_MIN_ROWS = 100
_COMMON_REQUIRED_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "timeframe")
_PRIMARY_KEY_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "timeframe")


class StrategyInputError(ValueError):
    """Raised when a strategy input frame violates the research contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StrategyInputAssessment:
    row_count: int
    valid_row_count: int
    low_data: bool


def assess_strategy_input(
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    timeframe: str,
) -> StrategyInputAssessment:
    """Assess a strategy input frame and return deterministic validation metadata."""

    _validate_required_columns(df, required_columns)
    if df.empty:
        raise StrategyInputError(
            "no_data",
            "no data available for selected timeframe and symbols",
        )

    _validate_structure(df, timeframe=timeframe)
    valid_rows = _valid_rows_mask(df, required_columns)
    valid_row_count = int(valid_rows.sum())
    if valid_row_count == 0:
        raise StrategyInputError(
            "no_usable_rows",
            "rows were loaded, but no valid rows remained after applying required-column and warm-up checks",
        )

    row_count = int(len(df))
    return StrategyInputAssessment(
        row_count=row_count,
        valid_row_count=valid_row_count,
        low_data=row_count < STRATEGY_INPUT_MIN_ROWS,
    )


def validate_strategy_input(
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    timeframe: str,
) -> None:
    """Validate the research contract for a strategy input frame."""

    assessment = assess_strategy_input(
        df,
        required_columns=required_columns,
        timeframe=timeframe,
    )
    df.attrs["input_validation"] = {
        "row_count": assessment.row_count,
        "valid_row_count": assessment.valid_row_count,
        "low_data": assessment.low_data,
        "required_columns": list(dict.fromkeys([*_COMMON_REQUIRED_COLUMNS, *required_columns])),
        "timeframe": timeframe,
    }


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    all_required = list(dict.fromkeys([*_COMMON_REQUIRED_COLUMNS, *required_columns]))
    missing = [column for column in all_required if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise StrategyInputError(
            "missing_columns",
            f"missing required strategy input columns: {formatted}",
        )


def _validate_structure(df: pd.DataFrame, *, timeframe: str) -> pd.Series:
    null_columns = [column for column in _PRIMARY_KEY_COLUMNS if df[column].isna().any()]
    if null_columns:
        formatted = ", ".join(repr(column) for column in null_columns)
        raise StrategyInputError(
            "invalid_structure",
            f"strategy input contains null values in key columns: {formatted}",
        )

    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise StrategyInputError("invalid_structure", "strategy input contains unparsable 'ts_utc' values")

    duplicate_mask = df.duplicated(subset=list(_PRIMARY_KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        duplicate_row = df.loc[duplicate_mask, list(_PRIMARY_KEY_COLUMNS)].iloc[0]
        raise StrategyInputError(
            "duplicate_keys",
            "strategy input contains duplicate (symbol, ts_utc, timeframe) rows. "
            f"First duplicate key: ({duplicate_row['symbol']}, {duplicate_row['ts_utc']}, {duplicate_row['timeframe']})",
        )

    actual_timeframes = df["timeframe"].astype("string").str.strip()
    mismatch_mask = actual_timeframes.ne(timeframe)
    if mismatch_mask.any():
        actual = actual_timeframes.loc[mismatch_mask].iloc[0]
        raise StrategyInputError(
            "invalid_structure",
            f"strategy input timeframe mismatch: expected {timeframe!r}, found {actual!r}",
        )

    keys = pd.DataFrame(
        {
            "symbol": df["symbol"].astype("string"),
            "ts_utc": ts_utc,
            "timeframe": actual_timeframes,
        },
        index=df.index,
    )
    sorted_keys = keys.sort_values(list(_PRIMARY_KEY_COLUMNS), kind="stable")
    if not keys.index.equals(sorted_keys.index):
        first_mismatch = next(
            position
            for position, (actual_index, expected_index) in enumerate(zip(keys.index, sorted_keys.index, strict=False))
            if actual_index != expected_index
        )
        actual_row = keys.iloc[first_mismatch]
        raise StrategyInputError(
            "invalid_structure",
            "strategy input must be sorted by (symbol, ts_utc, timeframe) before strategy execution. "
            f"First out-of-order row: symbol={actual_row['symbol']}, ts_utc={actual_row['ts_utc']}, timeframe={actual_row['timeframe']}",
        )
    return ts_utc


def _valid_rows_mask(df: pd.DataFrame, required_columns: list[str]) -> pd.Series:
    feature_columns = list(dict.fromkeys(required_columns))
    if not feature_columns:
        return pd.Series([True] * len(df), index=df.index, dtype="bool")
    return df[feature_columns].notna().all(axis=1)
