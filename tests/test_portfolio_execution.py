from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig
from src.portfolio.execution import PortfolioExecutionError, apply_portfolio_execution_model


def _returns() -> pd.DataFrame:
    index = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "alpha": [0.01, 0.02, -0.03],
            "beta": [0.03, -0.01, 0.02],
        },
        index=index,
        dtype="float64",
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "alpha": [1.0, 0.0, 1.0],
            "beta": [0.0, 1.0, 0.0],
        },
        index=_returns().index,
        dtype="float64",
    )


def test_apply_portfolio_execution_model_is_deterministic() -> None:
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        fixed_fee=0.001,
    )

    first = apply_portfolio_execution_model(_returns(), _weights(), config)
    second = apply_portfolio_execution_model(_returns(), _weights(), config)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.summary == second.summary


def test_apply_portfolio_execution_model_turnover_scaled_slippage() -> None:
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=0.0,
        slippage_bps=10.0,
        slippage_model="turnover_scaled",
        slippage_turnover_scale=1.5,
    )

    result = apply_portfolio_execution_model(_returns(), _weights(), config)

    assert result.frame["portfolio_turnover"].tolist() == pytest.approx([1.0, 2.0, 2.0])
    assert result.frame["portfolio_slippage_cost"].tolist() == pytest.approx([0.0015, 0.006, 0.006])


def test_apply_portfolio_execution_model_rejects_non_finite_inputs() -> None:
    returns = _returns()
    returns.loc[returns.index[0], "alpha"] = float("nan")

    with pytest.raises(PortfolioExecutionError, match="finite aligned returns"):
        apply_portfolio_execution_model(returns, _weights(), ExecutionConfig.default())


def test_apply_portfolio_execution_model_tracks_directional_costs() -> None:
    weights = pd.DataFrame(
        {
            "alpha": [0.5, -0.5, 0.0],
            "beta": [0.5, 1.5, 1.0],
        },
        index=_returns().index,
        dtype="float64",
    )
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        long_transaction_cost_bps=5.0,
        short_transaction_cost_bps=20.0,
        short_slippage_multiplier=2.0,
        short_borrow_cost_bps=10.0,
    )

    result = apply_portfolio_execution_model(_returns(), weights, config)

    assert result.frame["portfolio_long_turnover"].tolist() == pytest.approx([1.0, 1.0, 0.5])
    assert result.frame["portfolio_short_turnover"].tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert result.frame["portfolio_transaction_cost"].tolist() == pytest.approx([0.0005, 0.0025, 0.00125])
    assert result.frame["portfolio_short_borrow_cost"].tolist() == pytest.approx([0.0, 0.0005, 0.0])
    assert result.frame["portfolio_execution_friction"].tolist() == pytest.approx([0.001, 0.0045, 0.002])
    assert result.summary["directional_asymmetry"]["enabled"] is True
