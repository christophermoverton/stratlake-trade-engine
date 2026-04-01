from __future__ import annotations

import pandas as pd

STRUCTURAL_COLUMNS: tuple[str, ...] = ("symbol", "ts_utc", "timeframe")


class ForwardReturnAlignmentError(ValueError):
    """Raised when prediction-to-forward-return alignment is invalid or ambiguous."""


def align_forward_returns(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    price_column: str | None = None,
    realized_return_column: str | None = None,
    horizon: int = 1,
    drop_incomplete: bool = True,
    forward_return_column: str = "forward_return",
) -> pd.DataFrame:
    """Align predictions at time ``t`` with realized returns over a future horizon."""

    validated = validate_forward_return_alignment_input(
        df,
        prediction_column=prediction_column,
        price_column=price_column,
        realized_return_column=realized_return_column,
        horizon=horizon,
    )
    aligned = validated.copy(deep=True)

    if price_column is not None:
        aligned[forward_return_column] = _compute_price_forward_returns(
            aligned,
            price_column=price_column,
            horizon=horizon,
        )
    else:
        assert realized_return_column is not None
        aligned[forward_return_column] = _compute_compounded_forward_returns(
            aligned,
            realized_return_column=realized_return_column,
            horizon=horizon,
        )

    if drop_incomplete:
        aligned = aligned.loc[aligned[forward_return_column].notna()].copy(deep=True)

    aligned.attrs = {}
    return aligned


def validate_forward_return_alignment_input(
    df: pd.DataFrame,
    *,
    prediction_column: str = "prediction_score",
    price_column: str | None = None,
    realized_return_column: str | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Validate and normalize canonical alpha-evaluation alignment input."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Forward return alignment input must be a pandas DataFrame.")
    if df.empty:
        raise ForwardReturnAlignmentError("Forward return alignment input must not be empty.")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ForwardReturnAlignmentError("horizon must be a positive integer.")
    if (price_column is None) == (realized_return_column is None):
        raise ForwardReturnAlignmentError(
            "Specify exactly one of price_column or realized_return_column."
        )

    missing = [column for column in (*STRUCTURAL_COLUMNS, prediction_column) if column not in df.columns]
    if price_column is not None and price_column not in df.columns:
        missing.append(price_column)
    if realized_return_column is not None and realized_return_column not in df.columns:
        missing.append(realized_return_column)
    if missing:
        formatted = ", ".join(repr(column) for column in missing)
        raise ForwardReturnAlignmentError(
            f"Forward return alignment input must include required columns: {formatted}."
        )

    normalized = df.copy(deep=True)
    normalized.attrs = {}
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["timeframe"] = normalized["timeframe"].astype("string")

    if normalized["symbol"].isna().any():
        raise ForwardReturnAlignmentError("Forward return alignment input contains null values in 'symbol'.")
    if normalized["timeframe"].isna().any():
        raise ForwardReturnAlignmentError(
            "Forward return alignment input contains null values in 'timeframe'."
        )

    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="coerce")
    if normalized["ts_utc"].isna().any():
        raise ForwardReturnAlignmentError(
            "Forward return alignment input contains unparsable 'ts_utc' values."
        )

    keys = normalized.loc[:, list(STRUCTURAL_COLUMNS)]
    duplicate_mask = keys.duplicated(subset=list(STRUCTURAL_COLUMNS), keep=False)
    if duplicate_mask.any():
        first_duplicate = keys.loc[duplicate_mask, list(STRUCTURAL_COLUMNS)].iloc[0]
        raise ForwardReturnAlignmentError(
            "Forward return alignment input must not contain duplicate "
            "(symbol, ts_utc, timeframe) rows. "
            f"First duplicate key: symbol={first_duplicate['symbol']}, "
            f"ts_utc={first_duplicate['ts_utc']}, timeframe={first_duplicate['timeframe']}."
        )

    normalized[prediction_column] = _coerce_numeric_column(
        normalized[prediction_column],
        column_name=prediction_column,
    )

    source_column = price_column if price_column is not None else realized_return_column
    assert source_column is not None
    normalized[source_column] = _coerce_numeric_column(
        normalized[source_column],
        column_name=source_column,
    )

    normalized = normalized.sort_values(list(STRUCTURAL_COLUMNS), kind="stable").copy(deep=True)
    normalized.attrs = {}
    return normalized


def _compute_price_forward_returns(
    df: pd.DataFrame,
    *,
    price_column: str,
    horizon: int,
) -> pd.Series:
    future_price = df.groupby(["symbol", "timeframe"], sort=False)[price_column].shift(-horizon)
    return future_price.div(df[price_column]).sub(1.0).astype("float64")


def _compute_compounded_forward_returns(
    df: pd.DataFrame,
    *,
    realized_return_column: str,
    horizon: int,
) -> pd.Series:
    grouped_returns = df.groupby(["symbol", "timeframe"], sort=False)[realized_return_column]
    future_anchor = df.groupby(["symbol", "timeframe"], sort=False)["ts_utc"].shift(-horizon)

    compounded = pd.Series(1.0, index=df.index, dtype="float64")
    for step in range(horizon):
        compounded = compounded.mul(grouped_returns.shift(-step).add(1.0))
    return compounded.sub(1.0).where(future_anchor.notna()).astype("float64")


def _coerce_numeric_column(values: pd.Series, *, column_name: str) -> pd.Series:
    try:
        normalized = pd.to_numeric(values, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ForwardReturnAlignmentError(
            f"Forward return alignment column '{column_name}' must be numeric."
        ) from exc

    if normalized.isna().any():
        raise ForwardReturnAlignmentError(
            f"Forward return alignment column '{column_name}' must not contain NaN values."
        )
    return normalized
