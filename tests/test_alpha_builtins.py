from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha.builtins import (
    CrossSectionalElasticNetAlphaModel,
    CrossSectionalLinearAlphaModel,
    CrossSectionalLightGBMAlphaModel,
    CrossSectionalXGBoostAlphaModel,
    ElasticNetModelSpec,
    LightGBMModelSpec,
    LinearModelSpec,
    RankCompositeAlphaModel,
    RidgeLinearAlphaModel,
    XGBoostModelSpec,
)


@pytest.fixture
def daily_alpha_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "target_ret_5d": [0.09, 0.03, -0.03, 0.08, 0.02, -0.04, 0.07, 0.01, -0.05],
            "feature_ret_5d": [1.0, 0.2, -0.6, 0.9, 0.1, -0.7, 0.8, 0.0, -0.8],
            "feature_ret_20d": [0.7, 0.1, -0.4, 0.6, 0.0, -0.5, 0.5, -0.1, -0.6],
            "feature_close_to_sma20": [0.25, 0.04, -0.12, 0.2, 0.03, -0.15, 0.18, 0.01, -0.18],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)


def test_cross_sectional_linear_alpha_persists_coefficients_and_is_deterministic(
    daily_alpha_frame: pd.DataFrame,
) -> None:
    model = CrossSectionalLinearAlphaModel(spec=LinearModelSpec())

    model.fit(daily_alpha_frame)
    prediction_frame = daily_alpha_frame.drop(columns=["target_ret_5d"])
    first = model.predict(prediction_frame)
    second = model.predict(prediction_frame)

    assert model.target_column == "target_ret_5d"
    assert model.feature_columns == [
        "feature_ret_5d",
        "feature_ret_20d",
        "feature_close_to_sma20",
    ]
    assert sorted(model.coefficient_by_feature) == sorted(model.feature_columns)
    assert model.training_metadata["n_training_samples"] == len(daily_alpha_frame)
    assert first.equals(second)
    assert first.iloc[0] > first.iloc[3] > first.iloc[6]
    assert first.iloc[1] > first.iloc[4] > first.iloc[7]
    assert first.iloc[2] > first.iloc[5] > first.iloc[8]


def test_ridge_linear_alpha_shrinks_coefficients_relative_to_unpenalized_fit(
    daily_alpha_frame: pd.DataFrame,
) -> None:
    linear = CrossSectionalLinearAlphaModel(spec=LinearModelSpec(ridge_penalty=0.0))
    ridge = RidgeLinearAlphaModel(spec=LinearModelSpec(ridge_penalty=5.0))

    linear.fit(daily_alpha_frame)
    ridge.fit(daily_alpha_frame)
    prediction_frame = daily_alpha_frame.drop(columns=["target_ret_5d"])

    linear_norm = sum(abs(value) for value in linear.coefficient_by_feature.values())
    ridge_norm = sum(abs(value) for value in ridge.coefficient_by_feature.values())

    assert ridge_norm < linear_norm
    pd.testing.assert_series_equal(
        ridge.predict(prediction_frame),
        ridge.predict(prediction_frame),
        check_dtype=True,
        check_exact=True,
    )


def test_rank_composite_alpha_learns_inspectable_ic_weights_and_predicts_ordering(
    daily_alpha_frame: pd.DataFrame,
) -> None:
    model = RankCompositeAlphaModel()

    model.fit(daily_alpha_frame)
    prediction_frame = daily_alpha_frame.drop(columns=["target_ret_5d"])
    predictions = model.predict(prediction_frame)

    assert model.target_column == "target_ret_5d"
    assert sorted(model.feature_ic_by_name) == sorted(model.feature_columns)
    assert sorted(model.feature_weight_by_name) == sorted(model.feature_columns)
    assert sum(abs(weight) for weight in model.feature_weight_by_name.values()) == pytest.approx(1.0)
    assert predictions.iloc[0] > predictions.iloc[3] > predictions.iloc[6]
    assert predictions.iloc[1] > predictions.iloc[4] > predictions.iloc[7]
    assert predictions.iloc[2] > predictions.iloc[5] > predictions.iloc[8]


def test_cross_sectional_xgboost_alpha_is_deterministic_with_fixed_seed_and_no_sampling(
    daily_alpha_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeXGBRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)
            self.feature_importances_ = None
            self._weights = None

        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            filled = X.fillna(0.0).to_numpy(dtype="float64")
            target = y.to_numpy(dtype="float64")
            self._weights = filled.T @ target
            total = abs(self._weights).sum()
            if total > 0.0:
                self.feature_importances_ = abs(self._weights) / total
            else:
                self.feature_importances_ = abs(self._weights)

        def predict(self, X: pd.DataFrame):
            filled = X.fillna(0.0).to_numpy(dtype="float64")
            return filled @ self._weights

    fake_module = types.SimpleNamespace(XGBRegressor=FakeXGBRegressor)
    monkeypatch.setitem(sys.modules, "xgboost", fake_module)

    frame = daily_alpha_frame.copy(deep=True)
    frame["feature_sma_20"] = [1.0, 0.3, -0.5, 0.9, 0.2, -0.6, 0.8, 0.1, -0.7]
    frame["feature_sma_50"] = [0.8, 0.2, -0.4, 0.7, 0.1, -0.5, 0.6, 0.0, -0.6]
    frame.loc[frame.index[0], "feature_sma_20"] = None
    frame.loc[frame.index[1], "feature_sma_50"] = None

    model = CrossSectionalXGBoostAlphaModel(
        spec=XGBoostModelSpec(
            random_state=7,
            n_estimators=25,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            colsample_bylevel=1.0,
            colsample_bynode=1.0,
            n_jobs=1,
        )
    )

    model.fit(frame)
    prediction_frame = frame.drop(columns=["target_ret_5d"])
    first = model.predict(prediction_frame)
    second = model.predict(prediction_frame)

    assert first.equals(second)
    assert first.dtype == "float64"
    assert first.index.equals(prediction_frame.index)
    assert model.model_params["random_state"] == 7
    assert model.model_params["subsample"] == pytest.approx(1.0)
    assert model.model_params["colsample_bytree"] == pytest.approx(1.0)
    assert model.model_params["colsample_bylevel"] == pytest.approx(1.0)
    assert model.model_params["colsample_bynode"] == pytest.approx(1.0)
    assert model.model_params["n_jobs"] == 1
    assert sorted(model.feature_importance_by_name) == sorted(model.feature_columns)


def test_cross_sectional_xgboost_alpha_raises_clear_error_when_dependency_missing(
    daily_alpha_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "xgboost", raising=False)

    import importlib

    real_import_module = importlib.import_module

    def _raise_for_xgboost(name: str, package: str | None = None):
        if name == "xgboost":
            raise ModuleNotFoundError("No module named 'xgboost'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _raise_for_xgboost)

    model = CrossSectionalXGBoostAlphaModel()

    with pytest.raises(RuntimeError, match="xgboost is required"):
        model.fit(daily_alpha_frame)


def test_cross_sectional_lightgbm_alpha_is_deterministic_with_fixed_seed_and_no_sampling(
    daily_alpha_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLGBMRegressor:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)
            self.feature_importances_ = None
            self._weights = None

        def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
            filled = X.fillna(0.0).to_numpy(dtype="float64")
            target = y.to_numpy(dtype="float64")
            self._weights = filled.T @ target
            self.feature_importances_ = abs(self._weights)

        def predict(self, X: pd.DataFrame):
            filled = X.fillna(0.0).to_numpy(dtype="float64")
            return filled @ self._weights

    fake_module = types.SimpleNamespace(LGBMRegressor=FakeLGBMRegressor)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_module)

    frame = daily_alpha_frame.copy(deep=True)
    frame["feature_sma_20"] = [1.0, 0.3, -0.5, 0.9, 0.2, -0.6, 0.8, 0.1, -0.7]
    frame["feature_sma_50"] = [0.8, 0.2, -0.4, 0.7, 0.1, -0.5, 0.6, 0.0, -0.6]
    frame.loc[frame.index[0], "feature_sma_20"] = None
    frame.loc[frame.index[1], "feature_sma_50"] = None

    model = CrossSectionalLightGBMAlphaModel(
        spec=LightGBMModelSpec(
            random_state=7,
            n_estimators=25,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
        )
    )

    model.fit(frame)
    prediction_frame = frame.drop(columns=["target_ret_5d"])
    first = model.predict(prediction_frame)
    second = model.predict(prediction_frame)

    assert first.equals(second)
    assert model.model_params["random_state"] == 7
    assert model.model_params["subsample"] == pytest.approx(1.0)
    assert model.model_params["colsample_bytree"] == pytest.approx(1.0)
    assert model.model_params["n_jobs"] == 1
    assert sorted(model.feature_importance_by_name) == sorted(model.feature_columns)


def test_cross_sectional_lightgbm_alpha_raises_clear_error_when_dependency_missing(
    daily_alpha_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "lightgbm", raising=False)

    import importlib

    real_import_module = importlib.import_module

    def _raise_for_lightgbm(name: str, package: str | None = None):
        if name == "lightgbm":
            raise ModuleNotFoundError("No module named 'lightgbm'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _raise_for_lightgbm)

    model = CrossSectionalLightGBMAlphaModel()

    with pytest.raises(RuntimeError, match="lightgbm is required"):
        model.fit(daily_alpha_frame)


def test_cross_sectional_elastic_net_alpha_is_deterministic_and_tracks_coefficients(
    daily_alpha_frame: pd.DataFrame,
) -> None:
    frame = daily_alpha_frame.copy(deep=True)
    frame["feature_ret_1d"] = [0.9, 0.15, -0.5, 0.8, 0.1, -0.6, 0.7, 0.05, -0.7]
    frame["feature_vol_20d"] = [0.2, 0.18, 0.22, 0.19, 0.17, 0.23, 0.21, 0.16, 0.24]

    model = CrossSectionalElasticNetAlphaModel(
        spec=ElasticNetModelSpec(
            alpha=0.05,
            l1_ratio=0.35,
            fit_intercept=True,
            max_iter=4000,
            tol=1e-5,
            selection="cyclic",
            min_cross_section_size=2,
        )
    )

    model.fit(frame)
    prediction_frame = frame.drop(columns=["target_ret_5d"])
    first = model.predict(prediction_frame)
    second = model.predict(prediction_frame)

    assert first.equals(second)
    assert first.dtype == "float64"
    assert first.index.equals(prediction_frame.index)
    assert model.model_params["alpha"] == pytest.approx(0.05)
    assert model.model_params["l1_ratio"] == pytest.approx(0.35)
    assert model.model_params["selection"] == "cyclic"
    assert sorted(model.coefficient_by_feature) == sorted(model.feature_columns)
    assert model.training_metadata["n_training_samples"] == len(frame)
