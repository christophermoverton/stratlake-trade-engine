from __future__ import annotations

import pandas as pd
import pytest

from src.research.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    volatility,
    win_rate,
)


def _strategy_returns() -> pd.Series:
    return pd.Series(
        [0.10, -0.05, 0.02, -0.03, 0.04],
        index=pd.Index(["row_a", "row_b", "row_c", "row_d", "row_e"], name="row_id"),
        name="strategy_return",
        dtype="float64",
    )


def test_cumulative_return_computes_total_compounded_return() -> None:
    strategy_return = _strategy_returns()

    result = cumulative_return(strategy_return)

    assert result == pytest.approx(0.07527992)


def test_volatility_returns_sample_standard_deviation() -> None:
    strategy_return = _strategy_returns()

    result = volatility(strategy_return)

    assert result == pytest.approx(0.0594138031100518)


def test_sharpe_ratio_uses_zero_risk_free_rate() -> None:
    strategy_return = _strategy_returns()

    result = sharpe_ratio(strategy_return)

    assert result == pytest.approx(0.269297691150376)


def test_max_drawdown_computes_largest_peak_to_trough_decline() -> None:
    strategy_return = _strategy_returns()

    result = max_drawdown(strategy_return)

    assert result == pytest.approx(0.06007)


def test_win_rate_counts_positive_return_periods() -> None:
    strategy_return = _strategy_returns()

    result = win_rate(strategy_return)

    assert result == pytest.approx(0.6)
