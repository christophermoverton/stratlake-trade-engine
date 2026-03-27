from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import compute_portfolio_metrics
from src.research.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


def _portfolio_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                [
                    "2025-04-01T00:00:00Z",
                    "2025-04-02T00:00:00Z",
                    "2025-04-03T00:00:00Z",
                    "2025-04-04T00:00:00Z",
                ],
                utc=True,
            ),
            "strategy_return__alpha": [0.03, -0.01, 0.02, -0.03],
            "strategy_return__beta": [0.005, 0.00, 0.04, -0.01],
            "weight__alpha": [0.60, 0.50, 0.50, 0.20],
            "weight__beta": [0.40, 0.50, 0.50, 0.80],
            "portfolio_return": [0.02, -0.005, 0.03, -0.014],
            "portfolio_equity_curve": [102.0, 101.49, 104.5347, 103.0712142],
        }
    )


def test_compute_portfolio_metrics_reuses_return_metrics_deterministically() -> None:
    portfolio_output = _portfolio_output()
    portfolio_returns = portfolio_output["portfolio_return"]

    metrics = compute_portfolio_metrics(portfolio_output, timeframe="1d")

    expected_total = ((1.02 * 0.995 * 1.03 * 0.986) - 1.0)
    expected_period_volatility = portfolio_returns.std()
    expected_annual_volatility = expected_period_volatility * math.sqrt(TRADING_DAYS_PER_YEAR)
    expected_sharpe = (
        portfolio_returns.mean() * TRADING_DAYS_PER_YEAR / expected_annual_volatility
    )
    expected_drawdown = 1.0 - ((1.02 * 0.995 * 1.03 * 0.986) / (1.02 * 0.995 * 1.03))

    assert list(metrics.keys()) == [
        "cumulative_return",
        "total_return",
        "volatility",
        "annualized_return",
        "annualized_volatility",
        "rolling_volatility_window",
        "rolling_volatility_latest",
        "rolling_volatility_mean",
        "rolling_volatility_max",
        "target_volatility",
        "realized_volatility",
        "latest_rolling_volatility",
        "volatility_target_scale",
        "volatility_target_scale_capped",
        "sharpe_ratio",
        "max_drawdown",
        "current_drawdown",
        "max_drawdown_duration",
        "current_drawdown_duration",
        "value_at_risk",
        "value_at_risk_confidence_level",
        "conditional_value_at_risk",
        "conditional_value_at_risk_confidence_level",
        "win_rate",
        "hit_rate",
        "profit_factor",
        "turnover",
        "total_turnover",
        "average_turnover",
        "trade_count",
        "rebalance_count",
        "percent_periods_traded",
        "average_trade_size",
        "total_transaction_cost",
        "total_slippage_cost",
        "total_execution_friction",
        "average_execution_friction_per_trade",
        "exposure_pct",
        "average_gross_exposure",
        "max_gross_exposure",
        "average_net_exposure",
        "min_net_exposure",
        "max_net_exposure",
        "average_leverage",
        "max_leverage",
        "max_single_weight",
        "max_weight_sum_deviation",
        "validation_issue_count",
        "sanity_issue_count",
        "sanity_warning_count",
        "sanity_status",
        "sanity_strict_mode",
    ]
    assert metrics["cumulative_return"] == pytest.approx(metrics["total_return"])
    assert metrics["total_return"] == pytest.approx(expected_total)
    assert metrics["volatility"] == pytest.approx(expected_period_volatility)
    assert metrics["annualized_volatility"] == pytest.approx(expected_annual_volatility)
    assert metrics["rolling_volatility_window"] == pytest.approx(20.0)
    assert metrics["rolling_volatility_latest"] is None
    assert metrics["rolling_volatility_mean"] is None
    assert metrics["rolling_volatility_max"] is None
    assert metrics["target_volatility"] is None
    assert metrics["realized_volatility"] == pytest.approx(expected_annual_volatility)
    assert metrics["latest_rolling_volatility"] is None
    assert metrics["volatility_target_scale"] is None
    assert metrics["volatility_target_scale_capped"] == pytest.approx(0.0)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert metrics["max_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["current_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["max_drawdown_duration"] == pytest.approx(1.0)
    assert metrics["current_drawdown_duration"] == pytest.approx(1.0)
    assert metrics["value_at_risk"] == pytest.approx(0.014)
    assert metrics["value_at_risk_confidence_level"] == pytest.approx(0.95)
    assert metrics["conditional_value_at_risk"] == pytest.approx(0.014)
    assert metrics["conditional_value_at_risk_confidence_level"] == pytest.approx(0.95)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert metrics["profit_factor"] == pytest.approx((0.02 + 0.03) / (0.005 + 0.014))
    assert metrics["turnover"] == pytest.approx(0.45)
    assert metrics["total_turnover"] == pytest.approx(1.8)
    assert metrics["average_turnover"] == pytest.approx(0.45)
    assert metrics["trade_count"] == pytest.approx(3.0)
    assert metrics["rebalance_count"] == pytest.approx(3.0)
    assert metrics["percent_periods_traded"] == pytest.approx(75.0)
    assert metrics["average_trade_size"] == pytest.approx(0.6)
    assert metrics["total_execution_friction"] == pytest.approx(0.0)
    assert metrics["exposure_pct"] == pytest.approx(100.0)
    assert metrics["average_gross_exposure"] == pytest.approx(1.0)
    assert metrics["max_gross_exposure"] == pytest.approx(1.0)
    assert metrics["average_net_exposure"] == pytest.approx(1.0)
    assert metrics["max_leverage"] == pytest.approx(1.0)
    assert metrics["max_single_weight"] == pytest.approx(0.8)
    assert metrics["max_weight_sum_deviation"] == pytest.approx(0.0)
    assert metrics["validation_issue_count"] == pytest.approx(0.0)
    assert metrics["sanity_issue_count"] == pytest.approx(0.0)
    assert metrics["sanity_warning_count"] == pytest.approx(0.0)
    assert metrics["sanity_status"] == "pass"
    assert metrics["sanity_strict_mode"] is False


def test_compute_portfolio_metrics_matches_research_metric_primitives() -> None:
    portfolio_output = _portfolio_output()
    portfolio_returns = portfolio_output["portfolio_return"]

    metrics = compute_portfolio_metrics(portfolio_output, timeframe="1d")

    assert metrics["total_return"] == pytest.approx(total_return(portfolio_returns))
    assert metrics["annualized_volatility"] == pytest.approx(
        annualized_volatility(portfolio_returns, periods_per_year=TRADING_DAYS_PER_YEAR)
    )
    assert metrics["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(portfolio_returns, periods_per_year=TRADING_DAYS_PER_YEAR)
    )
    assert metrics["max_drawdown"] == pytest.approx(max_drawdown(portfolio_returns))
    assert metrics["value_at_risk"] == pytest.approx(0.014)
    assert metrics["conditional_value_at_risk"] == pytest.approx(0.014)


def test_compute_portfolio_metrics_omits_weight_based_metrics_without_traceability() -> None:
    portfolio_output = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                [
                    "2025-05-01T00:00:00Z",
                    "2025-05-02T00:00:00Z",
                    "2025-05-03T00:00:00Z",
                ],
                utc=True,
            ),
            "portfolio_return": [0.01, -0.02, 0.03],
            "portfolio_equity_curve": [1.01, 0.9898, 1.019494],
        }
    )

    metrics = compute_portfolio_metrics(portfolio_output, timeframe="1d")

    assert metrics["turnover"] is None
    assert metrics["trade_count"] is None
    assert metrics["exposure_pct"] is None
    assert metrics["average_gross_exposure"] == pytest.approx(0.0)
    assert metrics["value_at_risk"] == pytest.approx(0.02)
    assert metrics["conditional_value_at_risk"] == pytest.approx(0.02)
    assert metrics["validation_issue_count"] == pytest.approx(0.0)


def test_compute_portfolio_metrics_handles_all_zero_return_streams() -> None:
    portfolio_output = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                ["2025-06-01T00:00:00Z", "2025-06-02T00:00:00Z", "2025-06-03T00:00:00Z"],
                utc=True,
            ),
            "strategy_return__alpha": [0.0, 0.0, 0.0],
            "weight__alpha": [1.0, 1.0, 1.0],
            "portfolio_return": [0.0, 0.0, 0.0],
            "portfolio_equity_curve": [1.0, 1.0, 1.0],
        }
    )

    metrics = compute_portfolio_metrics(portfolio_output, timeframe="1d")

    assert metrics["total_return"] == 0.0
    assert metrics["annualized_return"] == 0.0
    assert metrics["annualized_volatility"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["current_drawdown"] == 0.0
    assert metrics["value_at_risk"] == 0.0
    assert metrics["conditional_value_at_risk"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["turnover"] == pytest.approx(1.0 / 3.0)
    assert metrics["exposure_pct"] == pytest.approx(100.0)


def test_compute_portfolio_metrics_handles_zero_volatility_non_zero_returns() -> None:
    portfolio_output = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(
                ["2025-07-01T00:00:00Z", "2025-07-02T00:00:00Z", "2025-07-03T00:00:00Z"],
                utc=True,
            ),
            "strategy_return__alpha": [0.01, 0.01, 0.01],
            "weight__alpha": [1.0, 1.0, 1.0],
            "portfolio_return": [0.01, 0.01, 0.01],
            "portfolio_equity_curve": [1.01, 1.0201, 1.030301],
        }
    )

    metrics = compute_portfolio_metrics(portfolio_output, timeframe="1d")

    assert metrics["annualized_volatility"] == 0.0
    assert metrics["realized_volatility"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["total_return"] == pytest.approx((1.01**3) - 1.0)


def test_compute_portfolio_metrics_rejects_empty_input() -> None:
    empty_output = pd.DataFrame(
        {
            "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "portfolio_return": pd.Series(dtype="float64"),
        }
    )

    with pytest.raises(ValueError, match="portfolio_output is empty"):
        compute_portfolio_metrics(empty_output, timeframe="1d")


def test_compute_portfolio_metrics_rejects_invalid_timeframe() -> None:
    with pytest.raises(ValueError, match="Unsupported portfolio metrics timeframe"):
        compute_portfolio_metrics(_portfolio_output(), timeframe="4h")


def test_compute_portfolio_metrics_counts_non_strict_sanity_issues() -> None:
    portfolio_output = _portfolio_output()
    portfolio_output.loc[2, "portfolio_return"] = 0.2
    running_equity = 100.0
    recomputed_equity: list[float] = []
    for portfolio_return in portfolio_output["portfolio_return"].tolist():
        running_equity *= 1.0 + float(portfolio_return)
        recomputed_equity.append(running_equity)
    portfolio_output["portfolio_equity_curve"] = recomputed_equity

    metrics = compute_portfolio_metrics(
        portfolio_output,
        timeframe="1d",
        validation_config={"max_abs_period_return": 0.1, "strict_sanity_checks": False},
    )

    assert metrics["validation_issue_count"] == pytest.approx(1.0)
