from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import src.research.alpha.registry as alpha_registry
from src.research.alpha.base import BaseAlphaModel
from src.research.alpha.builtins import (
    CrossSectionalLinearAlphaModel,
    CrossSectionalXGBoostAlphaModel,
    RankCompositeAlphaModel,
    RidgeLinearAlphaModel,
)
from src.research.alpha.catalog import (
    ALPHAS_CONFIG,
    get_alpha_config,
    load_alphas_config,
    register_builtin_alpha_catalog,
    resolve_alpha_config,
)


@pytest.fixture(autouse=True)
def reset_alpha_registry() -> None:
    original_registry = dict(alpha_registry._ALPHA_MODEL_REGISTRY)
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)
    yield
    alpha_registry._ALPHA_MODEL_REGISTRY.clear()
    alpha_registry._ALPHA_MODEL_REGISTRY.update(original_registry)


def test_alphas_config_file_exists() -> None:
    assert ALPHAS_CONFIG.exists()


def test_alphas_config_parses_successfully() -> None:
    alphas = load_alphas_config()

    assert isinstance(alphas, dict)
    assert "cs_linear_ret_1d" in alphas
    assert "cs_linear_ret_5d" in alphas
    assert "ridge_ret_5d" in alphas
    assert "rank_composite_momentum" in alphas


def test_each_alpha_entry_has_required_fields() -> None:
    alphas = load_alphas_config()

    for alpha_name, alpha_config in alphas.items():
        assert isinstance(alpha_config, dict), f"{alpha_name} must map to a dictionary"
        assert alpha_config["alpha_name"] == alpha_name
        assert isinstance(alpha_config["dataset"], str)
        assert isinstance(alpha_config["target_column"], str)
        assert isinstance(alpha_config["feature_columns"], list)
        assert alpha_config["feature_columns"]
        assert isinstance(alpha_config["model_type"], str)
        assert isinstance(alpha_config["model_params"], dict)
        assert isinstance(alpha_config["alpha_horizon"], int)
        assert alpha_config["alpha_horizon"] > 0


def test_get_alpha_config_returns_registry_entry_with_defaults_applied() -> None:
    config = get_alpha_config("cs_linear_ret_1d")

    assert config["price_column"] == "close"
    assert config["min_cross_section_size"] == 2
    assert config["alpha_model"] == "cs_linear_ret_1d"


def test_resolve_alpha_config_merges_registry_defaults_and_runtime_overrides() -> None:
    resolved = resolve_alpha_config(
        {
            "alpha_name": "ridge_ret_5d",
            "alpha_horizon": 10,
            "feature_columns": ["feature_ret_5d"],
        }
    )

    assert resolved["dataset"] == "features_daily"
    assert resolved["target_column"] == "target_ret_5d"
    assert resolved["alpha_model"] == "ridge_ret_5d"
    assert resolved["alpha_horizon"] == 10
    assert resolved["feature_columns"] == ["feature_ret_5d"]
    assert resolved["price_column"] == "close"


def test_register_builtin_alpha_catalog_registers_seeded_alphas() -> None:
    register_builtin_alpha_catalog()

    model = alpha_registry.get_alpha_model("cs_linear_ret_1d")

    assert isinstance(model, BaseAlphaModel)
    assert isinstance(model, CrossSectionalLinearAlphaModel)


def test_registered_builtin_linear_alpha_can_fit_and_predict() -> None:
    register_builtin_alpha_catalog()
    model = alpha_registry.get_alpha_model("cs_linear_ret_1d")
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "AAA", "BBB", "BBB"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "target_ret_1d": [0.1, 0.2, -0.1, -0.2],
            "feature_ret_1d": [1.0, 2.0, -1.0, -2.0],
            "feature_ret_5d": [0.5, 1.0, -0.5, -1.0],
            "feature_ret_20d": [0.25, 0.5, -0.25, -0.5],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)

    model.fit(frame)
    predictions = model.predict(frame.drop(columns=["target_ret_1d"]))

    assert predictions.dtype == "float64"
    assert len(predictions) == len(frame)
    assert predictions.iloc[0] > predictions.iloc[2]
    assert predictions.iloc[1] > predictions.iloc[3]


def test_register_builtin_alpha_catalog_supports_native_ridge_baseline() -> None:
    register_builtin_alpha_catalog()

    model = alpha_registry.get_alpha_model("ridge_ret_5d")
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "AAA", "BBB", "BBB"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "target_ret_5d": [0.1, 0.2, -0.1, -0.2],
            "feature_ret_1d": [1.0, 2.0, -1.0, -2.0],
            "feature_ret_5d": [0.5, 1.0, -0.5, -1.0],
            "feature_ret_20d": [0.25, 0.5, -0.25, -0.5],
            "feature_vol_20d": [0.2, 0.3, 0.2, 0.3],
            "feature_close_to_sma20": [0.01, 0.02, -0.01, -0.02],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)

    model.fit(frame)
    predictions = model.predict(frame.drop(columns=["target_ret_5d"]))

    assert predictions.dtype == "float64"
    assert len(predictions) == len(frame)
    assert isinstance(model, RidgeLinearAlphaModel)
    assert sorted(model.coefficient_by_feature) == [
        "feature_close_to_sma20",
        "feature_ret_1d",
        "feature_ret_20d",
        "feature_ret_5d",
        "feature_vol_20d",
    ]

def test_load_alphas_config_uses_native_cross_sectional_linear_model_type() -> None:
    config = get_alpha_config("cs_linear_ret_1d")

    assert config["model_type"] == "cross_sectional_linear"
    assert config["model_params"]["fit_intercept"] is True


def test_custom_catalog_supports_cross_sectional_xgboost_model_type(tmp_path: Path) -> None:
    config_path = tmp_path / "alphas.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ml_cross_sectional_xgb_2026_q1": {
                    "alpha_name": "ml_cross_sectional_xgb_2026_q1",
                    "dataset": "features_daily",
                    "target_column": "target_ret_5d",
                    "feature_columns": [
                        "feature_ret_1d",
                        "feature_ret_5d",
                        "feature_ret_20d",
                    ],
                    "model_type": "cross_sectional_xgboost",
                    "model_params": {
                        "random_state": 20260302,
                        "n_estimators": 32,
                        "max_depth": 3,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "colsample_bylevel": 1.0,
                        "colsample_bynode": 1.0,
                    },
                    "alpha_horizon": 5,
                    "defaults": {
                        "price_column": "close",
                        "min_cross_section_size": 25,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    alphas = load_alphas_config(config_path)
    assert alphas["ml_cross_sectional_xgb_2026_q1"]["model_type"] == "cross_sectional_xgboost"

    register_builtin_alpha_catalog(config_path)
    model = alpha_registry.get_alpha_model("ml_cross_sectional_xgb_2026_q1")

    assert isinstance(model, CrossSectionalXGBoostAlphaModel)
    assert model.model_params["random_state"] == 20260302
    assert model.model_params["subsample"] == pytest.approx(1.0)
    assert model.model_params["colsample_bytree"] == pytest.approx(1.0)


def test_register_builtin_alpha_catalog_supports_rank_composite_baseline() -> None:
    register_builtin_alpha_catalog()
    model = alpha_registry.get_alpha_model("rank_composite_momentum")
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "target_ret_5d": [0.06, 0.02, -0.01, 0.05, 0.01, -0.02],
            "feature_ret_5d": [0.9, 0.3, -0.2, 0.8, 0.1, -0.4],
            "feature_ret_20d": [0.6, 0.1, -0.3, 0.5, 0.0, -0.2],
            "feature_close_to_sma20": [0.2, 0.05, -0.1, 0.15, 0.02, -0.08],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)

    model.fit(frame)
    predictions = model.predict(frame.drop(columns=["target_ret_5d"]))

    assert isinstance(model, RankCompositeAlphaModel)
    assert predictions.dtype == "float64"
    assert len(predictions) == len(frame)
    assert sum(abs(weight) for weight in model.feature_weight_by_name.values()) == pytest.approx(1.0)
    assert predictions.iloc[0] > predictions.iloc[2]
    assert predictions.iloc[1] > predictions.iloc[3]


def test_load_alphas_config_rejects_invalid_defaults_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "alphas.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "broken_alpha": {
                    "alpha_name": "broken_alpha",
                    "dataset": "features_daily",
                    "target_column": "target_ret_1d",
                    "feature_columns": ["feature_ret_1d"],
                    "model_type": "linear",
                    "model_params": {},
                    "alpha_horizon": 1,
                    "defaults": ["bad"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="defaults must be a dictionary"):
        load_alphas_config(config_path)
