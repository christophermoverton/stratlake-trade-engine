from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.input_validation import (
    STRATEGY_INPUT_MIN_ROWS,
    StrategyInputError,
    assess_strategy_input,
    validate_strategy_input,
)


def _frame(*, rows: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"] * rows, dtype="string"),
            "ts_utc": ts,
            "timeframe": pd.Series(["1D"] * rows, dtype="string"),
            "feature_alpha": pd.Series(range(rows), dtype="float64"),
            "feature_ret_1d": pd.Series([0.01] * rows, dtype="float64"),
        }
    )


def test_validate_strategy_input_fails_for_empty_dataframe() -> None:
    df = _frame(rows=0)

    with pytest.raises(StrategyInputError, match="no data available"):
        validate_strategy_input(
            df,
            required_columns=["feature_alpha"],
            timeframe="1D",
        )


def test_validate_strategy_input_fails_for_missing_required_columns() -> None:
    df = _frame().drop(columns=["feature_alpha"])

    with pytest.raises(StrategyInputError, match="missing required strategy input columns"):
        validate_strategy_input(
            df,
            required_columns=["feature_alpha"],
            timeframe="1D",
        )


def test_validate_strategy_input_fails_for_duplicate_keys() -> None:
    df = pd.concat([_frame(rows=3), _frame(rows=1).iloc[[0]]], ignore_index=True)

    with pytest.raises(StrategyInputError, match="duplicate \\(symbol, ts_utc, timeframe\\) rows"):
        validate_strategy_input(
            df,
            required_columns=["feature_alpha"],
            timeframe="1D",
        )


def test_validate_strategy_input_fails_when_no_usable_rows_remain() -> None:
    df = _frame(rows=5)
    df["feature_alpha"] = pd.Series([pd.NA] * len(df), dtype="Float64")

    with pytest.raises(StrategyInputError, match="no valid rows remained"):
        validate_strategy_input(
            df,
            required_columns=["feature_alpha"],
            timeframe="1D",
        )


def test_assess_strategy_input_marks_low_data_without_failing() -> None:
    df = _frame(rows=20)

    assessment = assess_strategy_input(
        df,
        required_columns=["feature_alpha"],
        timeframe="1D",
    )

    assert assessment.row_count == 20
    assert assessment.valid_row_count == 20
    assert assessment.low_data is True


def test_validate_strategy_input_accepts_valid_frame() -> None:
    df = _frame(rows=STRATEGY_INPUT_MIN_ROWS)

    validate_strategy_input(
        df,
        required_columns=["feature_alpha", "feature_ret_1d"],
        timeframe="1D",
    )

    assert df.attrs["input_validation"] == {
        "row_count": STRATEGY_INPUT_MIN_ROWS,
        "valid_row_count": STRATEGY_INPUT_MIN_ROWS,
        "low_data": False,
        "required_columns": ["symbol", "ts_utc", "timeframe", "feature_alpha", "feature_ret_1d"],
        "timeframe": "1D",
    }
