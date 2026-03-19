from __future__ import annotations

import pandas as pd

RETURN_COLUMN_CANDIDATES: tuple[str, ...] = (
    "ret_1",
    "ret_1m",
    "ret_1d",
    "feature_ret_1m",
    "feature_ret_1d",
    "asset_return",
    "return",
    "returns",
)


def _resolve_return_column(df: pd.DataFrame) -> str:
    """Return the preferred asset return column name from a backtest input frame."""

    for column in RETURN_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Backtest input must include a 'signal' column and an asset return column. "
        f"Expected one of: {', '.join(RETURN_COLUMN_CANDIDATES)}."
    )


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deterministic strategy returns and an equity curve from strategy signals.

    The strategy return uses the previous period's signal so the backtest does not
    look ahead at the same row's realized return.

    Args:
        df: Feature dataset containing a standardized ``signal`` column and a
            supported asset return column such as ``ret_1`` or ``feature_ret_1d``.

    Returns:
        A copy of ``df`` with added ``strategy_return`` and ``equity_curve`` columns.

    Raises:
        ValueError: If the required ``signal`` column or a supported asset return
            column is missing.
    """

    if "signal" not in df.columns:
        raise ValueError("Backtest input must include a 'signal' column.")

    return_column = _resolve_return_column(df)

    result = df.copy()
    shifted_signal = result["signal"].shift(1).fillna(0.0)
    result["strategy_return"] = shifted_signal * result[return_column]
    result["equity_curve"] = (1.0 + result["strategy_return"]).cumprod()
    return result
