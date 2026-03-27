from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import (
    OptimizerAllocator,
    construct_portfolio,
    optimize_portfolio,
    resolve_portfolio_optimizer_config,
)


def _returns() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00:00+00:00"),
            pd.Timestamp("2025-01-02 00:00:00+00:00"),
            pd.Timestamp("2025-01-03 00:00:00+00:00"),
            pd.Timestamp("2025-01-04 00:00:00+00:00"),
        ],
        name="ts_utc",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "alpha": [0.020, 0.018, 0.021, 0.019],
            "beta": [0.010, 0.011, 0.009, 0.010],
            "gamma": [0.015, -0.005, 0.020, -0.010],
        },
        index=index,
        dtype="float64",
    )


def test_optimize_portfolio_equal_weight_returns_deterministic_weights() -> None:
    result = optimize_portfolio(_returns(), {"method": "equal_weight"})

    assert result.weights.index.tolist() == ["alpha", "beta", "gamma"]
    assert result.weights.tolist() == pytest.approx([1.0 / 3.0] * 3)
    assert result.diagnostics["method"] == "equal_weight"


def test_optimize_portfolio_max_sharpe_prefers_high_mean_low_variance_strategy() -> None:
    result = optimize_portfolio(
        _returns(),
        {
            "method": "max_sharpe",
            "max_weight": 0.8,
            "max_iterations": 300,
            "tolerance": 1e-10,
        },
    )

    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.index.tolist() == ["alpha", "beta", "gamma"]
    assert result.weights["alpha"] > result.weights["beta"]
    assert result.weights["alpha"] > result.weights["gamma"]
    assert result.weights.max() <= 0.8 + 1e-10


def test_optimize_portfolio_risk_parity_returns_balanced_positive_weights() -> None:
    result = optimize_portfolio(
        _returns(),
        {
            "method": "risk_parity",
            "max_weight": 0.7,
            "max_iterations": 400,
        },
    )

    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights >= 0.0).all()
    assert result.weights.max() <= 0.7 + 1e-10
    assert result.diagnostics["gross_exposure"] == pytest.approx(1.0)


def test_optimize_portfolio_rejects_invalid_constraint_combination() -> None:
    with pytest.raises(ValueError, match="leverage_ceiling"):
        resolve_portfolio_optimizer_config(
            {
                "method": "max_sharpe",
                "target_weight_sum": 1.0,
                "leverage_ceiling": 0.5,
            }
        )


def test_optimize_portfolio_rejects_non_finite_returns() -> None:
    returns = _returns()
    returns.loc[returns.index[0], "alpha"] = float("nan")

    with pytest.raises(ValueError, match="must not contain NaN values"):
        optimize_portfolio(returns, {"method": "max_sharpe"})


def test_optimize_portfolio_applies_max_turnover_against_previous_weights() -> None:
    result = optimize_portfolio(
        _returns(),
        {
            "method": "max_sharpe",
            "max_weight": 0.9,
            "max_turnover": 0.2,
        },
        previous_weights=pd.Series({"alpha": 0.4, "beta": 0.3, "gamma": 0.3}, dtype="float64"),
    )

    turnover = float((result.weights - pd.Series({"alpha": 0.4, "beta": 0.3, "gamma": 0.3})).abs().sum())
    assert turnover <= 0.2 + 1e-10


def test_optimizer_allocator_uses_training_window_for_application_weights() -> None:
    train_returns = _returns().iloc[:3]
    test_returns = _returns().iloc[3:].copy()
    allocator = OptimizerAllocator(
        {
            "method": "max_sharpe",
            "max_weight": 0.85,
        }
    )
    expected = optimize_portfolio(
        train_returns,
        {
            "method": "max_sharpe",
            "max_weight": 0.85,
        },
    )

    portfolio_output = construct_portfolio(
        test_returns,
        allocator=allocator,
        optimization_returns=train_returns,
    )

    assert portfolio_output.attrs["portfolio_constructor"]["optimizer"]["config"]["method"] == "max_sharpe"
    for strategy_name, weight_value in expected.weights.items():
        assert portfolio_output[f"weight__{strategy_name}"].iloc[0] == pytest.approx(weight_value)
