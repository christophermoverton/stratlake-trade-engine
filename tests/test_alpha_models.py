from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.research.alpha.registry as alpha_registry
from src.research.alpha.base import AlphaModelValidationError, BaseAlphaModel, DummyAlphaModel
from src.research.alpha.registry import get_alpha_model, register_alpha_model


class ConstantAlphaModel(BaseAlphaModel):
    name = "constant_alpha_model"

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.25, index=df.index, dtype="float64", name="prediction")


class WrongIndexAlphaModel(BaseAlphaModel):
    name = "wrong_index_alpha_model"

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=range(len(df)), dtype="float64")


class NaNWarmupLeakAlphaModel(BaseAlphaModel):
    name = "nan_warmup_leak_alpha_model"
    warmup_rows = 1

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0, float("nan"), 0.0], index=df.index, dtype="float64")


class MutatingAlphaModel(BaseAlphaModel):
    name = "mutating_alpha_model"

    def _fit(self, df: pd.DataFrame) -> None:
        df["mutated"] = 1.0

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        df.iloc[0, df.columns.get_loc("feature_alpha")] = 999.0
        return pd.Series(0.0, index=df.index, dtype="float64")


class NonDeterministicAlphaModel(BaseAlphaModel):
    name = "non_deterministic_alpha_model"

    def __init__(self) -> None:
        self._prediction_calls = 0

    def _fit(self, df: pd.DataFrame) -> None:
        return None

    def _predict(self, df: pd.DataFrame) -> pd.Series:
        self._prediction_calls += 1
        return pd.Series(float(self._prediction_calls), index=df.index, dtype="float64")


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


@pytest.fixture
def alpha_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "ts_utc": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z", "2025-01-02T00:00:00Z"],
                utc=True,
            ),
            "feature_alpha": [0.1, -0.2, 0.3],
        },
        index=pd.Index(["row_a", "row_b", "row_c"], name="row_id"),
    )


def test_base_alpha_model_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseAlphaModel()


def test_dummy_alpha_model_predicts_zero_series_aligned_to_input(alpha_frame: pd.DataFrame) -> None:
    model = DummyAlphaModel()

    predictions = model.predict(alpha_frame)

    assert predictions.index.equals(alpha_frame.index)
    assert predictions.dtype == "float64"
    assert predictions.tolist() == [0.0, 0.0, 0.0]


def test_registry_registration_and_retrieval_instantiates_model() -> None:
    register_alpha_model(ConstantAlphaModel.name, ConstantAlphaModel)

    model = get_alpha_model(ConstantAlphaModel.name)

    assert isinstance(model, ConstantAlphaModel)


def test_registry_rejects_duplicate_registration() -> None:
    register_alpha_model(ConstantAlphaModel.name, ConstantAlphaModel)

    with pytest.raises(ValueError, match="already registered"):
        register_alpha_model(ConstantAlphaModel.name, ConstantAlphaModel)


def test_registry_rejects_unknown_model_lookup() -> None:
    with pytest.raises(ValueError, match="No alpha model implementation is registered"):
        get_alpha_model("missing_alpha_model")


def test_predict_is_deterministic_for_same_input(alpha_frame: pd.DataFrame) -> None:
    model = ConstantAlphaModel()

    assert model.predict(alpha_frame).equals(model.predict(alpha_frame))


def test_predict_rejects_index_misalignment(alpha_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaModelValidationError, match="align exactly"):
        WrongIndexAlphaModel().predict(alpha_frame)


def test_predict_rejects_nans_after_warmup(alpha_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaModelValidationError, match="must not contain NaN values after warmup"):
        NaNWarmupLeakAlphaModel().predict(alpha_frame)


def test_predict_rejects_non_deterministic_outputs(alpha_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaModelValidationError, match="must be deterministic"):
        NonDeterministicAlphaModel().predict(alpha_frame)


def test_fit_and_predict_do_not_mutate_input(alpha_frame: pd.DataFrame) -> None:
    fit_frame = alpha_frame.copy(deep=True)
    predict_frame = alpha_frame.copy(deep=True)

    with pytest.raises(AlphaModelValidationError, match="fit\\(\\) must not mutate"):
        MutatingAlphaModel().fit(fit_frame)
    with pytest.raises(AlphaModelValidationError, match="predict\\(\\) must not mutate"):
        MutatingAlphaModel().predict(predict_frame)


def test_predict_rejects_unsorted_input() -> None:
    unsorted = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ts_utc": pd.to_datetime(["2025-01-03T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "feature_alpha": [0.1, 0.2],
        }
    )

    with pytest.raises(AlphaModelValidationError, match="sorted by \\(symbol, ts_utc\\)"):
        DummyAlphaModel().predict(unsorted)


def test_predict_does_not_mutate_input_frame(alpha_frame: pd.DataFrame) -> None:
    model = ConstantAlphaModel()
    baseline = alpha_frame.copy(deep=True)

    _ = model.predict(alpha_frame)

    pd.testing.assert_frame_equal(alpha_frame, baseline, check_dtype=True, check_exact=True)
