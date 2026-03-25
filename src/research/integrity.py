from __future__ import annotations

import warnings

import pandas as pd

_REQUIRED_COLUMNS: tuple[str, str] = ("symbol", "ts_utc")
_VALID_SIGNAL_VALUES = (-1, 0, 1)
_WARMUP_NAME_HINTS: tuple[str, ...] = (
    "sma",
    "ema",
    "zscore",
    "rolling",
    "std",
    "mean",
    "rsi",
    "atr",
    "lookback",
    "momentum",
    "vol",
)
_NON_FEATURE_COLUMNS = {
    "symbol",
    "ts_utc",
    "timeframe",
    "date",
    "signal",
    "strategy_return",
    "equity_curve",
    "asset_return",
    "return",
    "returns",
    "ret_1",
    "ret_1m",
    "ret_1d",
    "feature_ret_1m",
    "feature_ret_1d",
}


def validate_research_integrity(
    df: pd.DataFrame,
    signals: pd.Series | None = None,
    *,
    positions: pd.Series | None = None,
    warmup_rows: int = 3,
) -> None:
    """
    Validate deterministic research integrity constraints for strategy execution.

    Args:
        df: Canonical research frame expected to contain ``symbol`` and ``ts_utc``.
        signals: Optional signal series that must align exactly to ``df.index`` and
            only contain ``-1``, ``0``, or ``1``.
        positions: Optional executed position series. When provided, it must equal
            ``signals.shift(1).fillna(0.0)`` to enforce lagged execution.
        warmup_rows: Number of earliest rows per symbol inspected by the warm-up
            leakage heuristic.

    Raises:
        ValueError: If structural integrity, signal integrity, or lagged execution
            rules are violated.
    """

    _validate_required_columns(df)
    ts_utc = _validate_required_values(df)
    _validate_duplicate_keys(df)
    _validate_sorted_keys(df, ts_utc)
    _validate_monotonic_time(df, ts_utc)

    if signals is not None:
        _validate_signal_alignment(df, signals)
        _validate_signal_values(signals)

    if positions is not None:
        if signals is None:
            raise ValueError("Lagged execution validation requires a signal series.")
        _validate_signal_alignment(df, positions.rename("position"))
        _validate_lagged_execution(signals, positions)

    _warn_on_missing_warmup_nans(df, ts_utc, warmup_rows=warmup_rows)


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise ValueError(
            f"Research input must include required columns: {formatted}. "
            "Load canonical feature data before generating signals or running a backtest."
        )


def _validate_required_values(df: pd.DataFrame) -> pd.Series:
    null_columns = [column for column in _REQUIRED_COLUMNS if df[column].isna().any()]
    if null_columns:
        formatted = ", ".join(repr(column) for column in null_columns)
        raise ValueError(f"Research input contains null values in required columns: {formatted}.")

    ts_utc = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise ValueError("Research input contains unparsable 'ts_utc' values.")
    return ts_utc


def _validate_duplicate_keys(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=["symbol", "ts_utc"], keep=False)
    if not duplicate_mask.any():
        return

    duplicate_row = df.loc[duplicate_mask, ["symbol", "ts_utc"]].iloc[0]
    raise ValueError(
        "Research input contains duplicate (symbol, ts_utc) rows. "
        f"First duplicate key: ({duplicate_row['symbol']}, {duplicate_row['ts_utc']})."
    )


def _validate_sorted_keys(df: pd.DataFrame, ts_utc: pd.Series) -> None:
    keys = pd.DataFrame(
        {
            "symbol": df["symbol"].astype("string"),
            "ts_utc": ts_utc,
        },
        index=df.index,
    )
    sorted_keys = keys.sort_values(["symbol", "ts_utc"], kind="stable")
    if keys.index.equals(sorted_keys.index):
        return

    first_mismatch = next(
        position
        for position, (actual_index, expected_index) in enumerate(zip(keys.index, sorted_keys.index, strict=False))
        if actual_index != expected_index
    )
    actual_row = keys.iloc[first_mismatch]
    raise ValueError(
        "Research input must be sorted by (symbol, ts_utc) before strategy execution. "
        f"First out-of-order row: symbol={actual_row['symbol']}, ts_utc={actual_row['ts_utc']}."
    )


def _validate_monotonic_time(df: pd.DataFrame, ts_utc: pd.Series) -> None:
    working = pd.DataFrame({"symbol": df["symbol"].astype("string"), "ts_utc": ts_utc}, index=df.index)
    deltas = working.groupby("symbol", sort=False)["ts_utc"].diff()
    backward_mask = deltas < pd.Timedelta(0)
    if not backward_mask.any():
        return

    row = working.loc[backward_mask].iloc[0]
    prior_ts = working.groupby("symbol", sort=False)["ts_utc"].shift(1).loc[backward_mask].iloc[0]
    raise ValueError(
        "Research input moves backward in time within a symbol. "
        f"Symbol={row['symbol']} regressed from {prior_ts} to {row['ts_utc']}."
    )


def _validate_signal_alignment(df: pd.DataFrame, signals: pd.Series) -> None:
    if not isinstance(signals, pd.Series):
        raise TypeError("Research integrity validation requires signals and positions to be pandas Series.")
    if not signals.index.equals(df.index):
        raise ValueError(
            "Strategy signals must be aligned exactly with the input DataFrame index. "
            "Found missing, extra, or reordered rows."
        )


def _validate_signal_values(signals: pd.Series) -> None:
    invalid = signals[~signals.isin(_VALID_SIGNAL_VALUES)]
    if invalid.empty:
        return

    invalid_values = ", ".join(str(value) for value in invalid.drop_duplicates().head(5).tolist())
    raise ValueError(
        "Strategy signals must only contain the values -1, 0, and 1. "
        f"Unexpected values: {invalid_values}."
    )


def _validate_lagged_execution(signals: pd.Series, positions: pd.Series) -> None:
    normalized_signals = signals.astype("float64")
    normalized_positions = positions.astype("float64")
    expected_positions = normalized_signals.shift(1).fillna(0.0)

    if normalized_positions.equals(expected_positions):
        return

    same_bar_positions = normalized_signals.fillna(0.0)
    if normalized_positions.equals(same_bar_positions):
        raise ValueError(
            "Same-bar execution detected. Backtests must use lagged positions computed as "
            "signals.shift(1).fillna(0.0)."
        )

    mismatched = normalized_positions.ne(expected_positions)
    first_bad_label = mismatched[mismatched].index[0]
    raise ValueError(
        "Backtest positions are not properly lagged relative to the signal series. "
        f"First mismatch at index {first_bad_label!r}: expected {expected_positions.loc[first_bad_label]}, "
        f"got {normalized_positions.loc[first_bad_label]}."
    )


def _warn_on_missing_warmup_nans(df: pd.DataFrame, ts_utc: pd.Series, *, warmup_rows: int) -> None:
    if warmup_rows <= 0:
        return

    candidate_columns = _resolve_warmup_columns(df)
    if not candidate_columns:
        return

    ordered = df.assign(_ts_utc_integrity=ts_utc)
    early_rows = ordered.groupby("symbol", sort=False, group_keys=False).head(warmup_rows)
    if early_rows.empty or early_rows[candidate_columns].isna().any().any():
        return

    warnings.warn(
        "Warm-up leakage heuristic: early rows are fully populated for rolling-style feature columns "
        f"({', '.join(candidate_columns)}). Verify those features do not leak future information.",
        stacklevel=2,
    )


def _resolve_warmup_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        lower = column.lower()
        if column in _NON_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if any(hint in lower for hint in _WARMUP_NAME_HINTS):
            columns.append(column)
    return columns
