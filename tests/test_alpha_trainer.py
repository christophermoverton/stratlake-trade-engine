from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.research.alpha import register_alpha_model, train_alpha_model
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha.trainer import AlphaTrainingError


TARGET_COLUMN = "target_ret_1d"


class MeanTargetAlphaModel(BaseAlphaModel):
    name = "mean_target_alpha_model"

    def __init__(self) -> None:
        self.fit_called = False
        self.learned_target_mean: float | None = None
        self.training_columns: list[str] = []
        self.training_index: list[str] = []

    def _fit(self, df: pd.DataFrame) -> None:
        self.fit_called = True
        self.training_columns = list(df.columns)
        self.training_index = list(df.index)
        self.learned_target_mean = float(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().mean())

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        if self.learned_target_mean is None:
            raise RuntimeError("Model must be fit before predict.")
        return pd.Series(self.learned_target_mean, index=df.index, dtype="float64", name="prediction")


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


@pytest.fixture
def training_frame() -> pd.DataFrame:
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
            TARGET_COLUMN: [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "feature_zeta": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            "feature_alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "context_regime": ["bull", "bull", "bear", "bull", "bear", "bear"],
        },
        index=pd.Index([f"row_{idx}" for idx in range(6)], name="row_id"),
    )
    frame.attrs["source"] = "synthetic"
    return frame


def test_train_alpha_model_fits_registered_model_on_explicit_time_window(
    training_frame: pd.DataFrame,
) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    trained = train_alpha_model(
        training_frame,
        model_name=MeanTargetAlphaModel.name,
        target_column=TARGET_COLUMN,
        train_start="2025-01-02T00:00:00Z",
        train_end="2025-01-03T00:00:00Z",
    )

    assert trained.model_name == MeanTargetAlphaModel.name
    assert trained.feature_columns == ["feature_alpha", "feature_zeta"]
    assert trained.row_count == 2
    assert trained.symbol_count == 2
    assert trained.train_start == pd.Timestamp("2025-01-02T00:00:00Z")
    assert trained.train_end == pd.Timestamp("2025-01-03T00:00:00Z")
    assert trained.metadata["window_semantics"] == "[train_start, train_end)"
    assert trained.metadata["fit_columns"] == ["symbol", "ts_utc", TARGET_COLUMN, "feature_alpha", "feature_zeta"]
    assert isinstance(trained.model, MeanTargetAlphaModel)
    assert trained.model.fit_called is True
    assert trained.model.learned_target_mean == pytest.approx(11.0)
    assert trained.model.training_columns == ["symbol", "ts_utc", TARGET_COLUMN, "feature_alpha", "feature_zeta"]


@pytest.mark.parametrize(
    ("drop_column", "expected_message"),
    [
        ("symbol", "required columns"),
        (TARGET_COLUMN, "target column"),
    ],
)
def test_train_alpha_model_rejects_missing_required_columns(
    training_frame: pd.DataFrame,
    drop_column: str,
    expected_message: str,
) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    with pytest.raises(AlphaTrainingError, match=expected_message):
        train_alpha_model(
            training_frame.drop(columns=[drop_column]),
            model_name=MeanTargetAlphaModel.name,
            target_column=TARGET_COLUMN,
        )


def test_train_alpha_model_rejects_unknown_model(training_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaTrainingError, match="No alpha model implementation is registered"):
        train_alpha_model(training_frame, model_name="missing_model", target_column=TARGET_COLUMN)


def test_train_alpha_model_rejects_empty_training_slice(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    with pytest.raises(AlphaTrainingError, match="produced no rows"):
        train_alpha_model(
            training_frame,
            model_name=MeanTargetAlphaModel.name,
            target_column=TARGET_COLUMN,
            train_start="2025-02-01T00:00:00Z",
            train_end="2025-02-02T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("train_start", "train_end", "expected_message"),
    [
        ("2025-01-03T00:00:00Z", "2025-01-02T00:00:00Z", "train_start must be earlier than train_end"),
        ("not-a-timestamp", None, "train_start must be a valid timestamp"),
    ],
)
def test_train_alpha_model_rejects_invalid_time_boundaries(
    training_frame: pd.DataFrame,
    train_start: str,
    train_end: str | None,
    expected_message: str,
) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    with pytest.raises(AlphaTrainingError, match=expected_message):
        train_alpha_model(
            training_frame,
            model_name=MeanTargetAlphaModel.name,
            target_column=TARGET_COLUMN,
            train_start=train_start,
            train_end=train_end,
        )


def test_train_alpha_model_derives_feature_columns_deterministically(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    trained = train_alpha_model(
        training_frame,
        model_name=MeanTargetAlphaModel.name,
        target_column=TARGET_COLUMN,
    )

    assert trained.feature_columns == ["feature_alpha", "feature_zeta"]


def test_train_alpha_model_supports_explicit_feature_subset(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    trained = train_alpha_model(
        training_frame,
        model_name=MeanTargetAlphaModel.name,
        target_column=TARGET_COLUMN,
        feature_columns=["feature_zeta"],
    )

    assert trained.feature_columns == ["feature_zeta"]
    assert trained.model.training_columns == ["symbol", "ts_utc", TARGET_COLUMN, "feature_zeta"]


def test_train_alpha_model_rejects_unsorted_input(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)
    unsorted = training_frame.iloc[[1, 0, 2, 3, 4, 5]].copy(deep=True)

    with pytest.raises(AlphaTrainingError, match="sorted by \\(symbol, ts_utc\\)"):
        train_alpha_model(
            unsorted,
            model_name=MeanTargetAlphaModel.name,
            target_column=TARGET_COLUMN,
        )


def test_train_alpha_model_does_not_mutate_caller_input(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)
    baseline = training_frame.copy(deep=True)
    baseline.attrs = dict(training_frame.attrs)

    _ = train_alpha_model(
        training_frame,
        model_name=MeanTargetAlphaModel.name,
        target_column=TARGET_COLUMN,
        train_start="2025-01-01T00:00:00Z",
        train_end="2025-01-03T00:00:00Z",
    )

    pd.testing.assert_frame_equal(training_frame, baseline, check_dtype=True, check_exact=True)
    assert training_frame.attrs == baseline.attrs


def test_train_alpha_model_returns_metadata_for_filtered_training_slice(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    trained = train_alpha_model(
        training_frame,
        model_name=MeanTargetAlphaModel.name,
        target_column=TARGET_COLUMN,
        train_end="2025-01-03T00:00:00Z",
    )

    assert trained.row_count == 4
    assert trained.symbol_count == 2
    assert trained.train_start is None
    assert trained.train_end == pd.Timestamp("2025-01-03T00:00:00Z")


def test_repeated_training_produces_equivalent_trained_behavior(training_frame: pd.DataFrame) -> None:
    register_alpha_model(MeanTargetAlphaModel.name, MeanTargetAlphaModel)

    first = train_alpha_model(training_frame, model_name=MeanTargetAlphaModel.name, target_column=TARGET_COLUMN)
    second = train_alpha_model(training_frame, model_name=MeanTargetAlphaModel.name, target_column=TARGET_COLUMN)

    assert first.feature_columns == second.feature_columns
    assert first.row_count == second.row_count
    assert first.model.learned_target_mean == second.model.learned_target_mean
    pd.testing.assert_series_equal(
        first.model.predict(training_frame[["symbol", "ts_utc", "feature_alpha", "feature_zeta"]]),
        second.model.predict(training_frame[["symbol", "ts_utc", "feature_alpha", "feature_zeta"]]),
        check_dtype=True,
        check_exact=True,
    )
