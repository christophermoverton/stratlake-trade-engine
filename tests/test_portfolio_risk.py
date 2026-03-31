from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio.risk import (
    apply_volatility_targeting,
    PortfolioRiskError,
    historical_cvar,
    historical_var,
    resolve_portfolio_risk_config,
    resolve_portfolio_volatility_targeting_config,
    rolling_volatility,
    summarize_drawdown,
    summarize_portfolio_risk,
    volatility_target_diagnostics,
)


def _returns() -> pd.Series:
    return pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.02],
        index=pd.date_range("2025-01-01", periods=5, tz="UTC"),
        dtype="float64",
    )


def test_rolling_volatility_is_deterministic_and_warmup_is_nan() -> None:
    result = rolling_volatility(_returns(), window=3, periods_per_year=252)

    assert result.iloc[0] != result.iloc[0]
    assert result.iloc[1] != result.iloc[1]
    assert result.iloc[2] == pytest.approx(_returns().iloc[:3].std(ddof=1) * (252**0.5))


def test_rolling_volatility_rejects_invalid_window() -> None:
    with pytest.raises(PortfolioRiskError, match="greater than 1"):
        rolling_volatility(_returns(), window=1)


def test_historical_var_and_cvar_use_positive_loss_sign_convention() -> None:
    returns = pd.Series([-0.10, -0.04, -0.03, 0.01, 0.02], dtype="float64")

    assert historical_var(returns, confidence_level=0.95) == pytest.approx(0.10)
    assert historical_cvar(returns, confidence_level=0.95) == pytest.approx(0.10)


def test_drawdown_summary_uses_equity_curve_deterministically() -> None:
    equity_curve = pd.Series([100.0, 110.0, 99.0, 105.0, 95.0], dtype="float64")

    summary = summarize_drawdown(equity_curve=equity_curve)

    assert summary["max_drawdown"] == pytest.approx(1.0 - (95.0 / 110.0))
    assert summary["current_drawdown"] == pytest.approx(1.0 - (95.0 / 110.0))
    assert summary["max_drawdown_duration"] == 3
    assert summary["current_drawdown_duration"] == 3


def test_volatility_targeting_handles_zero_volatility_safely() -> None:
    diagnostics = volatility_target_diagnostics(
        pd.Series([0.0, 0.0, 0.0], dtype="float64"),
        config={"target_volatility": 0.15},
        periods_per_year=252,
    )

    assert diagnostics["recommended_scale"] == pytest.approx(0.0)
    assert diagnostics["scaling_limited_by_zero_volatility"] is True


def test_volatility_targeting_respects_scale_cap_without_hidden_leverage() -> None:
    diagnostics = volatility_target_diagnostics(
        _returns(),
        config={
            "volatility_window": 3,
            "target_volatility": 0.50,
            "allow_scale_up": False,
        },
        periods_per_year=252,
    )

    assert diagnostics["recommended_scale"] <= 1.0
    assert diagnostics["scale_cap"] == pytest.approx(1.0)


def test_summarize_portfolio_risk_returns_auditable_payload() -> None:
    returns = _returns()
    equity_curve = (1.0 + returns).cumprod()

    summary = summarize_portfolio_risk(
        returns,
        equity_curve=equity_curve,
        config={"volatility_window": 3, "target_volatility": 0.12},
        periods_per_year=252,
        leverage_ceiling=1.0,
    )

    assert summary["config"]["volatility_window"] == 3
    assert summary["rolling_volatility"]["latest"] is not None
    assert summary["tail_risk"]["var"] >= 0.0
    assert summary["tail_risk"]["cvar"] >= summary["tail_risk"]["var"]
    assert summary["drawdown"]["max_drawdown"] >= 0.0


def test_risk_config_rejects_invalid_scale_settings() -> None:
    with pytest.raises(PortfolioRiskError, match="allow_scale_up=False"):
        resolve_portfolio_risk_config({"max_volatility_scale": 1.5, "allow_scale_up": False})


def test_risk_functions_reject_non_finite_returns() -> None:
    with pytest.raises(PortfolioRiskError, match="finite numeric values"):
        summarize_portfolio_risk(
            pd.Series([0.01, float("nan")], dtype="float64"),
            periods_per_year=252,
        )


def test_resolve_volatility_targeting_config_requires_target_when_enabled() -> None:
    with pytest.raises(PortfolioRiskError, match="target_volatility is required"):
        resolve_portfolio_volatility_targeting_config({"enabled": True})


def test_apply_volatility_targeting_scales_weights_and_surfaces_metadata() -> None:
    index = pd.date_range("2025-02-01", periods=4, tz="UTC", name="ts_utc")
    returns = pd.DataFrame(
        {
            "alpha": [0.02, 0.01, -0.01, 0.02],
            "beta": [0.01, 0.00, 0.01, -0.01],
        },
        index=index,
        dtype="float64",
    )
    weights = pd.DataFrame(
        {
            "alpha": [0.5, 0.5, 0.5, 0.5],
            "beta": [0.5, 0.5, 0.5, 0.5],
        },
        index=index,
        dtype="float64",
    )

    scaled, metadata = apply_volatility_targeting(
        weights,
        returns,
        config={"enabled": True, "target_volatility": 0.10, "lookback_periods": 3},
        periods_per_year=252,
    )

    assert metadata["enabled"] is True
    assert metadata["estimated_pre_target_volatility"] is not None
    assert metadata["volatility_scaling_factor"] is not None
    assert metadata["estimated_post_target_volatility"] == pytest.approx(0.10)
    assert scaled.equals(weights * float(metadata["volatility_scaling_factor"]))


def test_apply_volatility_targeting_disabled_preserves_weights() -> None:
    index = pd.date_range("2025-03-01", periods=3, tz="UTC", name="ts_utc")
    returns = pd.DataFrame({"alpha": [0.01, 0.02, 0.00]}, index=index, dtype="float64")
    weights = pd.DataFrame({"alpha": [1.0, 1.0, 1.0]}, index=index, dtype="float64")

    scaled, metadata = apply_volatility_targeting(
        weights,
        returns,
        config={"enabled": False, "target_volatility": 0.10, "lookback_periods": 2},
        periods_per_year=252,
    )

    assert scaled.equals(weights)
    assert metadata["enabled"] is False
    assert metadata["volatility_scaling_factor"] is None
