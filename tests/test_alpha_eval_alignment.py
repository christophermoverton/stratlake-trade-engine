from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha_eval import (
    ForwardReturnAlignmentError,
    align_forward_returns,
    validate_forward_return_alignment_input,
)


@pytest.fixture
def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "MSFT", "AAPL", "AAPL", "MSFT"],
            "ts_utc": [
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "prediction_score": [0.9, 0.1, 0.2, 0.7, 0.4, 0.3],
            "close": [220.0, 100.0, 242.0, 121.0, 110.0, 200.0],
            "realized_ret_1d": [0.10, 0.10, 0.05, 0.0, 0.10, 0.10],
        },
        index=pd.Index(["m1", "a0", "m2", "a2", "a1", "m0"], name="row_id"),
    )


def test_align_forward_returns_computes_one_period_forward_return_from_price(
    prediction_frame: pd.DataFrame,
) -> None:
    result = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
    )

    expected = pd.DataFrame(
        {
            "symbol": pd.array(["AAPL", "AAPL", "MSFT", "MSFT"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": pd.array(["1d", "1d", "1d", "1d"], dtype="string"),
            "prediction_score": [0.1, 0.4, 0.3, 0.9],
            "close": [100.0, 110.0, 200.0, 220.0],
            "realized_ret_1d": [0.10, 0.10, 0.10, 0.10],
            "forward_return": [0.10, 0.10, 0.10, 0.10],
        },
        index=pd.Index(["a0", "a1", "m0", "m1"], name="row_id"),
    )

    pd.testing.assert_frame_equal(
        result.drop(columns=["forward_return"]),
        expected.drop(columns=["forward_return"]),
        check_dtype=True,
        check_exact=True,
    )
    assert result["forward_return"].tolist() == pytest.approx([0.10, 0.10, 0.10, 0.10])


def test_align_forward_returns_supports_multi_period_horizon_from_price(
    prediction_frame: pd.DataFrame,
) -> None:
    result = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=2,
    )

    expected = pd.DataFrame(
        {
            "symbol": pd.array(["AAPL", "MSFT"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
            "timeframe": pd.array(["1d", "1d"], dtype="string"),
            "prediction_score": [0.1, 0.3],
            "close": [100.0, 200.0],
            "realized_ret_1d": [0.10, 0.10],
            "forward_return": [0.21, 0.21],
        },
        index=pd.Index(["a0", "m0"], name="row_id"),
    )

    pd.testing.assert_frame_equal(
        result.drop(columns=["forward_return"]),
        expected.drop(columns=["forward_return"]),
        check_dtype=True,
        check_exact=True,
    )
    assert result["forward_return"].tolist() == pytest.approx([0.21, 0.21])


def test_align_forward_returns_can_compound_one_step_realized_returns_over_horizon(
    prediction_frame: pd.DataFrame,
) -> None:
    result = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        realized_return_column="realized_ret_1d",
        horizon=2,
    )

    assert result.index.tolist() == ["a0", "m0"]
    assert result["forward_return"].tolist() == pytest.approx([0.21, 0.21])


def test_align_forward_returns_can_keep_incomplete_terminal_rows_with_nan_forward_returns(
    prediction_frame: pd.DataFrame,
) -> None:
    result = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
        drop_incomplete=False,
    )

    assert result.index.tolist() == ["a0", "a1", "a2", "m0", "m1", "m2"]
    assert pd.isna(result.loc["a2", "forward_return"])
    assert pd.isna(result.loc["m2", "forward_return"])
    assert result.loc["a1", "forward_return"] == pytest.approx(0.10)


def test_validate_forward_return_alignment_input_rejects_duplicate_logical_keys(
    prediction_frame: pd.DataFrame,
) -> None:
    duplicate = pd.concat([prediction_frame, prediction_frame.iloc[[0]]], axis=0)

    with pytest.raises(ForwardReturnAlignmentError, match="duplicate \\(symbol, ts_utc, timeframe\\)"):
        validate_forward_return_alignment_input(
            duplicate,
            prediction_column="prediction_score",
            price_column="close",
        )


def test_validate_forward_return_alignment_input_rejects_missing_required_columns(
    prediction_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ForwardReturnAlignmentError, match="required columns"):
        validate_forward_return_alignment_input(
            prediction_frame.drop(columns=["timeframe"]),
            prediction_column="prediction_score",
            price_column="close",
        )


def test_validate_forward_return_alignment_input_rejects_invalid_timestamp_values(
    prediction_frame: pd.DataFrame,
) -> None:
    malformed = prediction_frame.copy(deep=True)
    malformed.loc["a1", "ts_utc"] = "not-a-timestamp"

    with pytest.raises(ForwardReturnAlignmentError, match="unparsable 'ts_utc'"):
        validate_forward_return_alignment_input(
            malformed,
            prediction_column="prediction_score",
            price_column="close",
        )


def test_validate_forward_return_alignment_input_rejects_invalid_horizon(
    prediction_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ForwardReturnAlignmentError, match="positive integer"):
        validate_forward_return_alignment_input(
            prediction_frame,
            prediction_column="prediction_score",
            price_column="close",
            horizon=0,
        )


def test_validate_forward_return_alignment_input_requires_exactly_one_return_source(
    prediction_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ForwardReturnAlignmentError, match="exactly one of price_column or realized_return_column"):
        validate_forward_return_alignment_input(
            prediction_frame,
            prediction_column="prediction_score",
            price_column="close",
            realized_return_column="realized_ret_1d",
        )


def test_validate_forward_return_alignment_input_rejects_non_numeric_prediction_column(
    prediction_frame: pd.DataFrame,
) -> None:
    malformed = prediction_frame.copy(deep=True)
    malformed["prediction_score"] = malformed["prediction_score"].astype("object")
    malformed.loc["a0", "prediction_score"] = "bad"

    with pytest.raises(ForwardReturnAlignmentError, match="'prediction_score' must be numeric"):
        validate_forward_return_alignment_input(
            malformed,
            prediction_column="prediction_score",
            price_column="close",
        )


def test_align_forward_returns_is_deterministic_and_does_not_mutate_input(
    prediction_frame: pd.DataFrame,
) -> None:
    baseline = prediction_frame.copy(deep=True)
    baseline.attrs = dict(prediction_frame.attrs)

    first = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
    )
    second = align_forward_returns(
        prediction_frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
    )

    pd.testing.assert_frame_equal(first, second, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(prediction_frame, baseline, check_dtype=True, check_exact=True)
    assert prediction_frame.attrs == baseline.attrs


def test_align_forward_returns_keeps_prediction_timestamp_and_prevents_current_row_leakage() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": ["1d", "1d", "1d"],
            "prediction_score": [9.0, 8.0, 7.0],
            "close": [100.0, 50.0, 200.0],
        },
        index=pd.Index(["t0", "t1", "t2"], name="row_id"),
    )

    result = align_forward_returns(
        frame,
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
        drop_incomplete=False,
    )

    assert result.loc["t0", "ts_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert result.loc["t0", "forward_return"] == pytest.approx(-0.50)
    assert result.loc["t1", "forward_return"] == pytest.approx(3.0)
