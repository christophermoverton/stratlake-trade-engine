from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.research.alpha import predict_alpha_model, register_alpha_model, train_alpha_model
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha_eval import (
    AlphaEvaluationError,
    evaluate_alpha_predictions,
    evaluate_information_coefficient,
    align_forward_returns,
    validate_alpha_evaluation_input,
)

TARGET_COLUMN = "target_ret_1d"


class WeightedFeatureAlphaModel(BaseAlphaModel):
    name = "weighted_feature_alpha_model_for_eval"

    def __init__(self) -> None:
        self.feature_columns: list[str] = []
        self.feature_means: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        self.feature_columns = [
            column
            for column in df.columns
            if column not in {"symbol", "ts_utc", TARGET_COLUMN}
        ]
        self.feature_means = {
            column: float(pd.to_numeric(df[column], errors="coerce").mean())
            for column in self.feature_columns
        }

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index, dtype="float64")
        for idx, column in enumerate(self.feature_columns, start=1):
            centered = pd.to_numeric(df[column], errors="coerce").astype("float64") - self.feature_means[column]
            score = score + (centered * float(idx))
        return score.rename("prediction")


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


@pytest.fixture
def alpha_eval_frame() -> pd.DataFrame:
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
            "y_pred": [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
            "forward_return": [0.2, 0.3, 0.1, 0.4, 0.0, 0.2],
        },
        index=pd.Index(["t0_a", "t1_a", "t0_b", "t1_b", "t0_c", "t1_c"], name="row_id"),
    )


@pytest.fixture
def alpha_research_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            TARGET_COLUMN: [0.01, 0.02, 0.03, 0.05, 0.04, 0.06],
            "feature_alpha": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "feature_beta": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            "close": [100.0, 110.0, 121.0, 200.0, 180.0, 216.0],
        },
        index=pd.Index([f"row_{idx}" for idx in range(6)], name="row_id"),
    )


def test_evaluate_information_coefficient_computes_deterministic_ic_and_rank_ic(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    ic_frame, summary = evaluate_information_coefficient(
        alpha_eval_frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )

    assert ic_frame["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
    assert ic_frame["ic"].tolist() == pytest.approx([-1.0, 0.5])
    assert ic_frame["rank_ic"].tolist() == pytest.approx([-1.0, 0.5])
    assert ic_frame["sample_size"].tolist() == [3, 3]
    assert summary["mean_ic"] == pytest.approx(-0.25)
    assert summary["std_ic"] == pytest.approx(1.0606601717798212)
    assert summary["ic_ir"] == pytest.approx(-0.23570226039551587)
    assert summary["mean_rank_ic"] == pytest.approx(-0.25)
    assert summary["std_rank_ic"] == pytest.approx(1.0606601717798212)
    assert summary["rank_ic_ir"] == pytest.approx(-0.23570226039551587)
    assert summary["n_periods"] == 2
    assert summary["ic_positive_rate"] == pytest.approx(0.5)
    assert summary["valid_timestamps"] == 2.0


def test_evaluate_information_coefficient_drops_only_rows_needed_for_correlation() -> None:
    frame = pd.DataFrame(
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
            "y_pred": [0.3, 0.1, 0.4, 0.2, 0.5, float("nan")],
            "forward_return": [0.1, 0.2, float("nan"), 0.1, 0.5, 0.3],
        }
    )

    ic_frame, summary = evaluate_information_coefficient(
        frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )

    assert ic_frame["sample_size"].tolist() == [2, 2]
    assert ic_frame["ic"].tolist() == pytest.approx([1.0, -1.0])
    assert ic_frame["rank_ic"].tolist() == pytest.approx([1.0, -1.0])
    assert summary["mean_ic"] == pytest.approx(0.0)
    assert summary["std_ic"] == pytest.approx(1.4142135623730951)
    assert summary["ic_ir"] == pytest.approx(0.0)
    assert summary["mean_rank_ic"] == pytest.approx(0.0)
    assert summary["std_rank_ic"] == pytest.approx(1.4142135623730951)
    assert summary["rank_ic_ir"] == pytest.approx(0.0)
    assert summary["n_periods"] == 2
    assert summary["ic_positive_rate"] == pytest.approx(0.5)
    assert summary["valid_timestamps"] == 2.0


def test_evaluate_alpha_predictions_integrates_with_predict_and_alignment_outputs(
    alpha_research_frame: pd.DataFrame,
) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(
        alpha_research_frame,
        model_name=WeightedFeatureAlphaModel.name,
        target_column=TARGET_COLUMN,
    )
    prediction_result = predict_alpha_model(trained, alpha_research_frame)
    aligned = align_forward_returns(
        prediction_result.predictions.merge(
            alpha_research_frame.loc[:, ["symbol", "ts_utc", "timeframe", "close"]],
            on=["symbol", "ts_utc", "timeframe"],
            how="inner",
            sort=False,
        ),
        prediction_column="prediction_score",
        price_column="close",
        horizon=1,
    )

    result = evaluate_alpha_predictions(aligned)

    assert result.prediction_column == "prediction_score"
    assert result.forward_return_column == "forward_return"
    assert result.row_count == 4
    assert result.timestamp_count == 2
    assert result.symbol_count == 2
    assert result.metadata["timeframe"] == "1d"
    assert result.metadata["artifact_scaffold"] == {
        "coefficients": "coefficients.json",
        "ic_timeseries": "ic_timeseries.csv",
        "alpha_metrics": "alpha_metrics.json",
        "cross_section_diagnostics": "cross_section_diagnostics.json",
        "predictions": "predictions.parquet",
        "qa_summary": "qa_summary.json",
        "signal_mapping": "signal_mapping.json",
        "signals": "signals.parquet",
        "training_summary": "training_summary.json",
    }
    assert result.metadata["ic_timeseries_columns"] == ["ts_utc", "ic", "rank_ic", "sample_size"]
    assert result.ic_timeseries["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
    assert result.ic_timeseries["ic"].iloc[0] == pytest.approx(-1.0)
    assert result.ic_timeseries["rank_ic"].iloc[0] == pytest.approx(-1.0)
    assert result.ic_timeseries["ic"].iloc[1] == pytest.approx(1.0)
    assert result.ic_timeseries["rank_ic"].iloc[1] == pytest.approx(1.0)
    assert result.ic_timeseries["sample_size"].tolist() == [2, 2]
    assert result.summary["mean_ic"] == pytest.approx(0.0)
    assert result.summary["std_ic"] == pytest.approx(1.4142135623730951)
    assert result.summary["ic_ir"] == pytest.approx(0.0)
    assert result.summary["mean_rank_ic"] == pytest.approx(0.0)
    assert result.summary["std_rank_ic"] == pytest.approx(1.4142135623730951)
    assert result.summary["rank_ic_ir"] == pytest.approx(0.0)
    assert result.summary["n_periods"] == 2
    assert result.summary["ic_positive_rate"] == pytest.approx(0.5)
    assert result.summary["valid_timestamps"] == 2.0


def test_validate_alpha_evaluation_input_rejects_duplicate_logical_keys(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    duplicate = pd.concat([alpha_eval_frame, alpha_eval_frame.iloc[[0]]], axis=0)
    duplicate.iloc[-1, duplicate.columns.get_loc("y_pred")] = 9.9

    with pytest.raises(AlphaEvaluationError, match="duplicate \\(symbol, ts_utc, timeframe\\)"):
        validate_alpha_evaluation_input(
            duplicate,
            prediction_column="y_pred",
            forward_return_column="forward_return",
        )


def test_validate_alpha_evaluation_input_rejects_missing_forward_return_column(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    with pytest.raises(AlphaEvaluationError, match="required columns"):
        validate_alpha_evaluation_input(
            alpha_eval_frame.drop(columns=["forward_return"]),
            prediction_column="y_pred",
            forward_return_column="forward_return",
        )


def test_validate_alpha_evaluation_input_rejects_non_numeric_forward_returns(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = alpha_eval_frame.copy(deep=True)
    malformed["forward_return"] = malformed["forward_return"].astype("object")
    malformed.loc["t0_a", "forward_return"] = "bad"

    with pytest.raises(AlphaEvaluationError, match="'forward_return' must be numeric"):
        validate_alpha_evaluation_input(
            malformed,
            prediction_column="y_pred",
            forward_return_column="forward_return",
        )


def test_validate_alpha_evaluation_input_rejects_invalid_minimum_cross_section_size(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    with pytest.raises(AlphaEvaluationError, match="greater than or equal to 2"):
        validate_alpha_evaluation_input(
            alpha_eval_frame,
            prediction_column="y_pred",
            forward_return_column="forward_return",
            min_cross_section_size=1,
        )


def test_validate_alpha_evaluation_input_rejects_multiple_timeframes(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    malformed = alpha_eval_frame.copy(deep=True)
    malformed.loc["t1_c", "timeframe"] = "1h"

    with pytest.raises(AlphaEvaluationError, match="exactly one timeframe"):
        validate_alpha_evaluation_input(
            malformed,
            prediction_column="y_pred",
            forward_return_column="forward_return",
        )


def test_evaluate_information_coefficient_is_deterministic_and_does_not_mutate_input(
    alpha_eval_frame: pd.DataFrame,
) -> None:
    baseline = alpha_eval_frame.copy(deep=True)
    baseline.attrs = dict(alpha_eval_frame.attrs)

    first_frame, first_summary = evaluate_information_coefficient(
        alpha_eval_frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )
    second_frame, second_summary = evaluate_information_coefficient(
        alpha_eval_frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )

    pd.testing.assert_frame_equal(first_frame, second_frame, check_dtype=True, check_exact=True)
    assert first_summary == second_summary
    pd.testing.assert_frame_equal(alpha_eval_frame, baseline, check_dtype=True, check_exact=True)
    assert alpha_eval_frame.attrs == baseline.attrs


def test_evaluate_information_coefficient_keeps_summary_key_order() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-03T00:00:00Z",
            ],
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "y_pred": [0.1, 0.3, 0.6, 0.2, 0.4, 0.5],
            "forward_return": [0.1, 0.5, 0.4, 0.2, 0.8, 0.3],
        }
    )

    _, summary = evaluate_information_coefficient(
        frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )

    assert list(summary) == [
        "mean_ic",
        "std_ic",
        "ic_ir",
        "mean_rank_ic",
        "std_rank_ic",
        "rank_ic_ir",
        "n_periods",
        "ic_positive_rate",
        "valid_timestamps",
    ]
    assert summary["n_periods"] == 3
    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["std_ic"] >= 0.0
    assert summary["ic_ir"] >= 0.0
    assert summary["mean_rank_ic"] == pytest.approx(1.0)
    assert summary["std_rank_ic"] >= 0.0
    assert summary["rank_ic_ir"] >= 0.0
    assert summary["ic_positive_rate"] == pytest.approx(1.0)
    assert summary["valid_timestamps"] == 3.0


def test_evaluate_information_coefficient_returns_zero_ir_for_zero_volatility() -> None:
    frame = pd.DataFrame(
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
            "y_pred": [0.1, 1.0, 0.2, 2.0, 0.3, 3.0],
            "forward_return": [0.1, 1.0, 0.2, 2.0, 0.3, 3.0],
        }
    )

    _, summary = evaluate_information_coefficient(
        frame,
        prediction_column="y_pred",
        forward_return_column="forward_return",
    )

    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["std_ic"] == pytest.approx(0.0)
    assert summary["ic_ir"] == pytest.approx(0.0)
    assert summary["mean_rank_ic"] == pytest.approx(1.0)
    assert summary["std_rank_ic"] == pytest.approx(0.0)
    assert summary["rank_ic_ir"] == pytest.approx(0.0)
    assert summary["n_periods"] == 2
    assert summary["ic_positive_rate"] == pytest.approx(1.0)
