from __future__ import annotations

import pandas as pd

from src.research.backtest_runner import RETURN_COLUMN_CANDIDATES


def resolve_return_column(df: pd.DataFrame) -> str:
    """Return the first supported asset return column present in a feature frame."""

    for column in RETURN_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    expected = ", ".join(RETURN_COLUMN_CANDIDATES)
    raise ValueError(f"Feature dataset must include one of the supported return columns: {expected}.")


def valid_return_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask identifying rows with a usable asset return value."""

    return df[resolve_return_column(df)].notna()
