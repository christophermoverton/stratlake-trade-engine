from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig
from src.portfolio import (
    EqualWeightAllocator,
    compute_portfolio_equity_curve,
    compute_portfolio_returns,
    construct_portfolio,
)


def _returns_two_strategies() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-01 00:00:00+00:00"),
            pd.Timestamp("2025-01-02 00:00:00+00:00"),
            pd.Timestamp("2025-01-03 00:00:00+00:00"),
        ],
        name="ts_utc",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "beta": [0.03, -0.01, 0.02],
            "alpha": [0.01, 0.02, -0.03],
        },
        index=index,
        dtype="float64",
    )


def _returns_three_strategies() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-02-01 00:00:00+00:00"),
            pd.Timestamp("2025-02-02 00:00:00+00:00"),
        ],
        name="ts_utc",
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "gamma": [0.03, 0.00],
            "alpha": [0.01, 0.02],
            "beta": [-0.02, 0.01],
        },
        index=index,
        dtype="float64",
    )


def test_equal_weight_allocator_allocates_half_per_strategy_for_two_strategies() -> None:
    allocator = EqualWeightAllocator()

    weights = allocator.allocate(_returns_two_strategies())

    assert weights.columns.tolist() == ["alpha", "beta"]
    assert weights.index.tolist() == list(
        pd.to_datetime(
            ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
            utc=True,
        )
    )
    assert weights["alpha"].tolist() == [0.5, 0.5, 0.5]
    assert weights["beta"].tolist() == [0.5, 0.5, 0.5]


def test_equal_weight_allocator_allocates_equal_weights_for_multiple_strategies() -> None:
    allocator = EqualWeightAllocator()

    weights = allocator.allocate(_returns_three_strategies())

    assert weights.columns.tolist() == ["alpha", "beta", "gamma"]
    for column in weights.columns:
        assert weights[column].tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0])


def test_equal_weight_allocator_weights_sum_to_one_on_every_row() -> None:
    allocator = EqualWeightAllocator()

    weights = allocator.allocate(_returns_three_strategies())

    assert weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_compute_portfolio_returns_aggregates_weighted_component_returns() -> None:
    returns_wide = _returns_two_strategies()
    weights_wide = EqualWeightAllocator().allocate(returns_wide)

    portfolio_returns = compute_portfolio_returns(returns_wide, weights_wide)

    assert portfolio_returns.columns.tolist() == [
        "ts_utc",
        "strategy_return__alpha",
        "strategy_return__beta",
        "weight__alpha",
        "weight__beta",
        "gross_portfolio_return",
        "portfolio_weight_change",
        "portfolio_abs_weight_change",
        "portfolio_turnover",
        "portfolio_rebalance_event",
        "portfolio_transaction_cost",
        "portfolio_slippage_cost",
        "portfolio_execution_friction",
        "net_portfolio_return",
        "portfolio_return",
    ]
    assert portfolio_returns["portfolio_return"].tolist() == pytest.approx([0.02, 0.005, -0.005])


def test_compute_portfolio_equity_curve_compounds_deterministic_returns() -> None:
    portfolio_returns = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                ["2025-03-01T00:00:00Z", "2025-03-02T00:00:00Z", "2025-03-03T00:00:00Z"],
                utc=True,
            ),
            "strategy_return__alpha": [0.01, 0.02, 0.00],
            "weight__alpha": [1.0, 1.0, 1.0],
            "portfolio_return": [0.01, 0.02, 0.00],
        }
    )

    output = compute_portfolio_equity_curve(portfolio_returns, initial_capital=100.0)

    assert output["portfolio_equity_curve"].tolist() == pytest.approx([101.0, 103.02, 103.02])


def test_construct_portfolio_builds_end_to_end_validated_output() -> None:
    output = construct_portfolio(
        _returns_two_strategies(),
        allocator=EqualWeightAllocator(),
        initial_capital=10.0,
    )

    assert output.columns.tolist() == [
        "ts_utc",
        "strategy_return__alpha",
        "strategy_return__beta",
        "weight__alpha",
        "weight__beta",
        "gross_portfolio_return",
        "portfolio_weight_change",
        "portfolio_abs_weight_change",
        "portfolio_turnover",
        "portfolio_rebalance_event",
        "portfolio_transaction_cost",
        "portfolio_slippage_cost",
        "portfolio_execution_friction",
        "net_portfolio_return",
        "portfolio_return",
        "portfolio_equity_curve",
    ]
    assert output["portfolio_return"].tolist() == pytest.approx([0.02, 0.005, -0.005])
    assert output["portfolio_equity_curve"].tolist() == pytest.approx([10.2, 10.251, 10.199745])


def test_compute_portfolio_returns_applies_turnover_aware_execution_costs() -> None:
    returns_wide = _returns_two_strategies()
    weights_wide = pd.DataFrame(
        {
            "alpha": [1.0, 0.0, 1.0],
            "beta": [0.0, 1.0, 0.0],
        },
        index=returns_wide.index,
        dtype="float64",
    )
    config = ExecutionConfig(
        enabled=True,
        execution_delay=1,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
    )

    portfolio_returns = compute_portfolio_returns(
        returns_wide,
        weights_wide,
        execution_config=config,
    )

    assert portfolio_returns["gross_portfolio_return"].tolist() == pytest.approx([0.01, -0.01, -0.03])
    assert portfolio_returns["portfolio_turnover"].tolist() == pytest.approx([1.0, 2.0, 2.0])
    assert portfolio_returns["portfolio_rebalance_event"].tolist() == [1, 1, 1]
    assert portfolio_returns["portfolio_transaction_cost"].tolist() == pytest.approx([0.001, 0.002, 0.002])
    assert portfolio_returns["portfolio_slippage_cost"].tolist() == pytest.approx([0.0005, 0.001, 0.001])
    assert portfolio_returns["portfolio_execution_friction"].tolist() == pytest.approx([0.0015, 0.003, 0.003])
    assert portfolio_returns["portfolio_return"].tolist() == pytest.approx([0.0085, -0.013, -0.033])


def test_compute_portfolio_returns_rejects_mismatched_index() -> None:
    returns_wide = _returns_two_strategies()
    weights_wide = EqualWeightAllocator().allocate(returns_wide)
    shifted_index = weights_wide.index + pd.Timedelta(days=1)
    weights_wide = weights_wide.copy()
    weights_wide.index = shifted_index

    with pytest.raises(ValueError, match="exactly matching indices"):
        compute_portfolio_returns(returns_wide, weights_wide)


def test_compute_portfolio_returns_rejects_mismatched_columns() -> None:
    returns_wide = _returns_two_strategies()
    weights_wide = EqualWeightAllocator().allocate(returns_wide).rename(columns={"alpha": "alpha_alt"})

    with pytest.raises(ValueError, match="exactly matching columns"):
        compute_portfolio_returns(returns_wide, weights_wide)


def test_equal_weight_allocator_rejects_empty_input() -> None:
    empty_returns = _returns_two_strategies().iloc[0:0]

    with pytest.raises(ValueError, match="Aligned return matrix is empty"):
        EqualWeightAllocator().allocate(empty_returns)


def test_compute_portfolio_equity_curve_rejects_non_positive_initial_capital() -> None:
    portfolio_returns = compute_portfolio_returns(
        _returns_two_strategies(),
        EqualWeightAllocator().allocate(_returns_two_strategies()),
    )

    with pytest.raises(ValueError, match="initial_capital must be positive"):
        compute_portfolio_equity_curve(portfolio_returns, initial_capital=0.0)
