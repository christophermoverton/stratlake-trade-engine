from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha.builtins import (
    CrossSectionalLinearAlphaModel,
    LinearModelSpec,
    RankCompositeAlphaModel,
    RidgeLinearAlphaModel,
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
