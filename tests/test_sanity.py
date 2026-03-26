from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.sanity import resolve_sanity_check_config
from src.research.sanity import (
    SanityCheckError,
    validate_portfolio_output_sanity,
    validate_strategy_backtest_sanity,
)


def test_resolve_sanity_check_config_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="smoothness_min_positive_return_fraction"):
        resolve_sanity_check_config({"smoothness_min_positive_return_fraction": 1.5})


def test_validate_strategy_backtest_sanity_fails_on_non_finite_returns() -> None:
    results_df = pd.DataFrame(
        {
            "strategy_return": [0.01, float("nan")],
            "net_strategy_return": [0.01, float("nan")],
            "equity_curve": [1.01, 1.01],
        }
    )
    metrics = {
        "annualized_return": 0.1,
        "annualized_volatility": 0.2,
        "sharpe_ratio": 0.5,
        "max_drawdown": 0.01,
    }

    with pytest.raises(SanityCheckError, match="non-finite"):
        validate_strategy_backtest_sanity(results_df, metrics)


def test_validate_strategy_backtest_sanity_warns_in_non_strict_mode() -> None:
    results_df = pd.DataFrame(
        {
            "strategy_return": [0.0, 0.75],
            "net_strategy_return": [0.0, 0.75],
            "equity_curve": [1.0, 1.75],
        }
    )
    metrics = {
        "annualized_return": 12.0,
        "annualized_volatility": 0.1,
        "sharpe_ratio": 5.0,
        "max_drawdown": 0.0,
    }

    report = validate_strategy_backtest_sanity(
        results_df,
        metrics,
        {
            "strict_sanity_checks": False,
            "max_abs_period_return": 0.5,
            "max_annualized_return": 2.0,
            "max_sharpe_ratio": 3.0,
        },
    )

    assert report.status == "warn"
    assert report.issue_count >= 3


def test_validate_strategy_backtest_sanity_fails_in_strict_mode() -> None:
    results_df = pd.DataFrame(
        {
            "strategy_return": [0.0, 0.75],
            "net_strategy_return": [0.0, 0.75],
            "equity_curve": [1.0, 1.75],
        }
    )
    metrics = {
        "annualized_return": 12.0,
        "annualized_volatility": 0.1,
        "sharpe_ratio": 5.0,
        "max_drawdown": 0.0,
    }

    with pytest.raises(SanityCheckError, match="flagged"):
        validate_strategy_backtest_sanity(
            results_df,
            metrics,
            {
                "strict_sanity_checks": True,
                "max_abs_period_return": 0.5,
            },
        )


def test_validate_portfolio_output_sanity_flags_extreme_equity_multiple() -> None:
    portfolio_output = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "portfolio_return": [0.1, 0.2],
            "portfolio_equity_curve": [110.0, 132.0],
        }
    )
    metrics = {
        "annualized_return": 5.0,
        "annualized_volatility": 0.5,
        "sharpe_ratio": 3.5,
        "max_drawdown": 0.0,
    }

    report = validate_portfolio_output_sanity(
        portfolio_output,
        metrics,
        {"strict_sanity_checks": False, "max_equity_multiple": 1.2},
        initial_capital=100.0,
    )

    assert report.status == "warn"
    assert any(issue.code == "equity_multiple_exceeds_threshold" for issue in report.issues)
