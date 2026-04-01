from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha_eval import (
    AlphaEvaluationError,
    evaluate_alpha_predictions,
    validate_alpha_evaluation_input,
)


@pytest.fixture
def valid_alpha_eval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "prediction_score": [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
            "forward_return": [0.2, 0.3, 0.1, 0.4, 0.0, 0.2],
        },
        index=pd.Index(["a0", "a1", "b0", "b1", "c0", "c1"], name="row_id"),
    )


def test_validate_alpha_evaluation_input_accepts_valid_contract(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    validated = validate_alpha_evaluation_input(valid_alpha_eval_frame)

    assert validated.index.tolist() == valid_alpha_eval_frame.index.tolist()
    assert validated["symbol"].dtype == "string"
    assert validated["timeframe"].dtype == "string"
    assert str(validated["prediction_score"].dtype) == "float64"
    assert str(validated["forward_return"].dtype) == "float64"
    assert validated["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]


def test_validate_alpha_evaluation_input_rejects_unsorted_input(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    unsorted = valid_alpha_eval_frame.iloc[[4, 0, 2, 1, 3, 5]].copy(deep=True)

    with pytest.raises(AlphaEvaluationError, match="sorted by \\(symbol, ts_utc, timeframe\\)"):
        validate_alpha_evaluation_input(unsorted)


def test_validate_alpha_evaluation_input_rejects_duplicate_rows(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    duplicate = pd.concat([valid_alpha_eval_frame, valid_alpha_eval_frame.iloc[[0]]], axis=0)

    with pytest.raises(AlphaEvaluationError, match="duplicate rows"):
        validate_alpha_evaluation_input(duplicate)


def test_validate_alpha_evaluation_input_rejects_missing_columns(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    with pytest.raises(AlphaEvaluationError, match="required columns"):
        validate_alpha_evaluation_input(valid_alpha_eval_frame.drop(columns=["forward_return"]))


def test_validate_alpha_evaluation_input_rejects_invalid_prediction_type(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = valid_alpha_eval_frame.copy(deep=True)
    malformed["prediction_score"] = malformed["prediction_score"].astype("object")
    malformed.loc["a0", "prediction_score"] = "bad"

    with pytest.raises(AlphaEvaluationError, match="'prediction_score' must be numeric"):
        validate_alpha_evaluation_input(malformed)


def test_validate_alpha_evaluation_input_rejects_insufficient_cross_section(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    sparse = valid_alpha_eval_frame.copy(deep=True)
    sparse.loc[["b1", "c1"], "forward_return"] = float("nan")

    with pytest.raises(AlphaEvaluationError, match="insufficient cross-section"):
        validate_alpha_evaluation_input(sparse)


def test_validate_alpha_evaluation_input_rejects_constant_prediction_cross_section(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = valid_alpha_eval_frame.copy(deep=True)
    malformed.loc[["a1", "b1", "c1"], "prediction_score"] = 1.0

    with pytest.raises(AlphaEvaluationError, match="variation in the prediction column"):
        validate_alpha_evaluation_input(malformed)


def test_validate_alpha_evaluation_input_rejects_constant_forward_return_cross_section(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = valid_alpha_eval_frame.copy(deep=True)
    malformed.loc[["a1", "b1", "c1"], "forward_return"] = 0.5

    with pytest.raises(AlphaEvaluationError, match="variation in the forward return column"):
        validate_alpha_evaluation_input(malformed)


def test_validate_alpha_evaluation_input_does_not_mutate_input(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    baseline = valid_alpha_eval_frame.copy(deep=True)
    baseline.attrs = dict(valid_alpha_eval_frame.attrs)

    _ = validate_alpha_evaluation_input(valid_alpha_eval_frame)

    pd.testing.assert_frame_equal(valid_alpha_eval_frame, baseline, check_dtype=True, check_exact=True)
    assert valid_alpha_eval_frame.attrs == baseline.attrs


def test_evaluate_alpha_predictions_fails_fast_before_ic_on_invalid_group(
    valid_alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = valid_alpha_eval_frame.copy(deep=True)
    malformed.loc[["a1", "b1", "c1"], "forward_return"] = 0.5

    with pytest.raises(AlphaEvaluationError, match="variation in the forward return column"):
        evaluate_alpha_predictions(malformed)
