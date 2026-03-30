from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.research.alpha import predict_alpha_model, register_alpha_model, train_alpha_model
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha.predictor import AlphaPredictionError


TARGET_COLUMN = "target_ret_1d"


class WeightedFeatureAlphaModel(BaseAlphaModel):
    name = "weighted_feature_alpha_model"

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
        if not self.feature_columns:
            raise RuntimeError("Model must be fit before predict.")

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
def research_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
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
            "noise_column": [100, 101, 102, 103, 104, 105],
        },
        index=pd.Index([f"row_{idx}" for idx in range(6)], name="row_id"),
    )
    frame.attrs["source"] = "predictor-tests"
    return frame


def test_predict_alpha_model_returns_aligned_structured_output(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(
        research_frame,
        model_name=WeightedFeatureAlphaModel.name,
        target_column=TARGET_COLUMN,
    )

    result = predict_alpha_model(
        trained,
        research_frame,
        predict_start="2025-01-02T00:00:00Z",
        predict_end="2025-01-03T00:00:00Z",
    )

    expected = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "ts_utc": pd.to_datetime(["2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "timeframe": ["1d", "1d"],
            "prediction_score": [-5.25, 5.25],
        },
        index=pd.Index(["row_1", "row_4"], name="row_id"),
    )

    assert result.model_name == WeightedFeatureAlphaModel.name
    assert result.target_column == TARGET_COLUMN
    assert result.feature_columns == ["feature_alpha", "feature_beta"]
    assert result.predict_start == pd.Timestamp("2025-01-02T00:00:00Z")
    assert result.predict_end == pd.Timestamp("2025-01-03T00:00:00Z")
    assert result.row_count == 2
    assert result.symbol_count == 2
    assert result.metadata["window_semantics"] == "[predict_start, predict_end)"
    assert result.metadata["predict_columns"] == ["symbol", "ts_utc", "feature_alpha", "feature_beta", "timeframe"]
    pd.testing.assert_frame_equal(result.predictions, expected, check_dtype=True, check_exact=True)


def test_predict_alpha_model_rejects_missing_required_structural_columns(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    with pytest.raises(AlphaPredictionError, match="required columns"):
        predict_alpha_model(trained, research_frame.drop(columns=["symbol"]))


def test_predict_alpha_model_rejects_missing_trained_feature_column(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    with pytest.raises(AlphaPredictionError, match="trained feature columns"):
        predict_alpha_model(trained, research_frame.drop(columns=["feature_beta"]))


def test_predict_alpha_model_rejects_empty_prediction_input() -> None:
    trained = type("FakeTrained", (), {})()

    with pytest.raises(TypeError, match="TrainedAlphaModel"):
        predict_alpha_model(trained, pd.DataFrame())


def test_predict_alpha_model_rejects_empty_post_filter_window(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    with pytest.raises(AlphaPredictionError, match="produced no rows"):
        predict_alpha_model(
            trained,
            research_frame,
            predict_start="2025-02-01T00:00:00Z",
            predict_end="2025-02-02T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("predict_start", "predict_end", "expected_message"),
    [
        ("2025-01-03T00:00:00Z", "2025-01-02T00:00:00Z", "predict_start must be earlier than predict_end"),
        ("not-a-timestamp", None, "predict_start must be a valid timestamp"),
    ],
)
def test_predict_alpha_model_rejects_invalid_prediction_boundaries(
    research_frame: pd.DataFrame,
    predict_start: str,
    predict_end: str | None,
    expected_message: str,
) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    with pytest.raises(AlphaPredictionError, match=expected_message):
        predict_alpha_model(
            trained,
            research_frame,
            predict_start=predict_start,
            predict_end=predict_end,
        )


def test_predict_alpha_model_rejects_empty_prediction_input_frame(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    with pytest.raises(AlphaPredictionError, match="must not be empty"):
        predict_alpha_model(trained, research_frame.iloc[0:0].copy(deep=True))


def test_predict_alpha_model_is_deterministic_across_repeated_calls(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    first = predict_alpha_model(trained, research_frame, predict_start="2025-01-02T00:00:00Z")
    second = predict_alpha_model(trained, research_frame, predict_start="2025-01-02T00:00:00Z")

    pd.testing.assert_frame_equal(first.predictions, second.predictions, check_dtype=True, check_exact=True)


def test_predict_alpha_model_does_not_mutate_caller_input(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)
    baseline = research_frame.copy(deep=True)
    baseline.attrs = dict(research_frame.attrs)

    _ = predict_alpha_model(trained, research_frame, predict_start="2025-01-01T00:00:00Z")

    pd.testing.assert_frame_equal(research_frame, baseline, check_dtype=True, check_exact=True)
    assert research_frame.attrs == baseline.attrs


def test_predict_alpha_model_preserves_deterministic_row_ordering(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    result = predict_alpha_model(trained, research_frame)

    assert result.predictions.index.tolist() == research_frame.index.tolist()
    assert result.predictions["symbol"].tolist() == ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"]


def test_predict_alpha_model_preserves_optional_timeframe_column(research_frame: pd.DataFrame) -> None:
    register_alpha_model(WeightedFeatureAlphaModel.name, WeightedFeatureAlphaModel)
    trained = train_alpha_model(research_frame, model_name=WeightedFeatureAlphaModel.name, target_column=TARGET_COLUMN)

    result = predict_alpha_model(trained, research_frame)

    assert list(result.predictions.columns) == ["symbol", "ts_utc", "timeframe", "prediction_score"]
    assert result.predictions["timeframe"].tolist() == ["1d", "1d", "1d", "1d", "1d", "1d"]
