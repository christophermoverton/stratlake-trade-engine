from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


_DEFAULT_VOLATILITY_WINDOW = 20
_DEFAULT_VAR_CONFIDENCE_LEVEL = 0.95
_DEFAULT_CVAR_CONFIDENCE_LEVEL = 0.95
_DEFAULT_MIN_VOLATILITY_SCALE = 0.0
_DEFAULT_MAX_VOLATILITY_SCALE = 1.0
_DEFAULT_VOLATILITY_EPSILON = 1e-12


class PortfolioRiskError(ValueError):
    """Raised when deterministic portfolio risk inputs or outputs are invalid."""


@dataclass(frozen=True)
class PortfolioRiskConfig:
    """Deterministic centralized risk-model configuration."""

    volatility_window: int = _DEFAULT_VOLATILITY_WINDOW
    target_volatility: float | None = None
    min_volatility_scale: float = _DEFAULT_MIN_VOLATILITY_SCALE
    max_volatility_scale: float = _DEFAULT_MAX_VOLATILITY_SCALE
    allow_scale_up: bool = False
    var_confidence_level: float = _DEFAULT_VAR_CONFIDENCE_LEVEL
    cvar_confidence_level: float = _DEFAULT_CVAR_CONFIDENCE_LEVEL
    volatility_epsilon: float = _DEFAULT_VOLATILITY_EPSILON
    periods_per_year_override: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "volatility_window": self.volatility_window,
            "target_volatility": self.target_volatility,
            "min_volatility_scale": self.min_volatility_scale,
            "max_volatility_scale": self.max_volatility_scale,
            "allow_scale_up": self.allow_scale_up,
            "var_confidence_level": self.var_confidence_level,
            "cvar_confidence_level": self.cvar_confidence_level,
            "volatility_epsilon": self.volatility_epsilon,
            "periods_per_year_override": self.periods_per_year_override,
        }


@dataclass(frozen=True)
class PortfolioVolatilityTargetingConfig:
    """Deterministic post-optimizer volatility-targeting controls."""

    enabled: bool = False
    target_volatility: float | None = None
    lookback_periods: int = _DEFAULT_VOLATILITY_WINDOW
    volatility_epsilon: float = _DEFAULT_VOLATILITY_EPSILON

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "target_volatility": self.target_volatility,
            "lookback_periods": self.lookback_periods,
            "volatility_epsilon": self.volatility_epsilon,
        }


def resolve_portfolio_risk_config(
    payload: PortfolioRiskConfig | dict[str, Any] | None,
) -> PortfolioRiskConfig:
    if payload is None:
        return PortfolioRiskConfig()
    if isinstance(payload, PortfolioRiskConfig):
        return payload
    if not isinstance(payload, dict):
        raise PortfolioRiskError("portfolio risk config must be a dictionary when provided.")

    volatility_window = _coerce_positive_int(
        payload.get("volatility_window", _DEFAULT_VOLATILITY_WINDOW),
        field_name="risk.volatility_window",
    )
    target_volatility = _coerce_optional_positive_float(
        payload.get("target_volatility"),
        field_name="risk.target_volatility",
    )
    min_volatility_scale = _coerce_non_negative_float(
        payload.get("min_volatility_scale", _DEFAULT_MIN_VOLATILITY_SCALE),
        field_name="risk.min_volatility_scale",
    )
    max_volatility_scale = _coerce_non_negative_float(
        payload.get("max_volatility_scale", _DEFAULT_MAX_VOLATILITY_SCALE),
        field_name="risk.max_volatility_scale",
    )
    allow_scale_up = payload.get("allow_scale_up", False)
    if not isinstance(allow_scale_up, bool):
        raise PortfolioRiskError("portfolio risk field 'allow_scale_up' must be a boolean.")
    var_confidence_level = _coerce_confidence_level(
        payload.get("var_confidence_level", _DEFAULT_VAR_CONFIDENCE_LEVEL),
        field_name="risk.var_confidence_level",
    )
    cvar_confidence_level = _coerce_confidence_level(
        payload.get("cvar_confidence_level", _DEFAULT_CVAR_CONFIDENCE_LEVEL),
        field_name="risk.cvar_confidence_level",
    )
    volatility_epsilon = _coerce_positive_float(
        payload.get("volatility_epsilon", _DEFAULT_VOLATILITY_EPSILON),
        field_name="risk.volatility_epsilon",
    )
    periods_per_year_override = _coerce_optional_positive_int(
        payload.get("periods_per_year_override"),
        field_name="risk.periods_per_year_override",
    )

    if min_volatility_scale > max_volatility_scale:
        raise PortfolioRiskError(
            "portfolio risk requires min_volatility_scale <= max_volatility_scale."
        )
    if not allow_scale_up and max_volatility_scale > 1.0:
        raise PortfolioRiskError(
            "portfolio risk cannot set max_volatility_scale > 1.0 when allow_scale_up=False."
        )

    return PortfolioRiskConfig(
        volatility_window=volatility_window,
        target_volatility=target_volatility,
        min_volatility_scale=min_volatility_scale,
        max_volatility_scale=max_volatility_scale,
        allow_scale_up=allow_scale_up,
        var_confidence_level=var_confidence_level,
        cvar_confidence_level=cvar_confidence_level,
        volatility_epsilon=volatility_epsilon,
        periods_per_year_override=periods_per_year_override,
    )


def resolve_portfolio_volatility_targeting_config(
    payload: PortfolioVolatilityTargetingConfig | dict[str, Any] | None,
) -> PortfolioVolatilityTargetingConfig:
    if payload is None:
        return PortfolioVolatilityTargetingConfig()
    if isinstance(payload, PortfolioVolatilityTargetingConfig):
        return payload
    if not isinstance(payload, dict):
        raise PortfolioRiskError(
            "portfolio volatility targeting config must be a dictionary when provided."
        )

    enabled = payload.get("enabled", False)
    if not isinstance(enabled, bool):
        raise PortfolioRiskError(
            "portfolio volatility targeting field 'enabled' must be a boolean."
        )
    lookback_periods = _coerce_positive_int(
        payload.get("lookback_periods", _DEFAULT_VOLATILITY_WINDOW),
        field_name="volatility_targeting.lookback_periods",
    )
    if lookback_periods <= 1:
        raise PortfolioRiskError(
            "volatility_targeting.lookback_periods must be an integer greater than 1."
        )
    volatility_epsilon = _coerce_positive_float(
        payload.get("volatility_epsilon", _DEFAULT_VOLATILITY_EPSILON),
        field_name="volatility_targeting.volatility_epsilon",
    )
    target_volatility_value = payload.get("target_volatility")
    if enabled:
        if target_volatility_value is None:
            raise PortfolioRiskError(
                "volatility_targeting.target_volatility is required when volatility targeting is enabled."
            )
        target_volatility = _coerce_positive_float(
            target_volatility_value,
            field_name="volatility_targeting.target_volatility",
        )
    else:
        target_volatility = (
            None
            if target_volatility_value is None
            else _coerce_positive_float(
                target_volatility_value,
                field_name="volatility_targeting.target_volatility",
            )
        )

    return PortfolioVolatilityTargetingConfig(
        enabled=enabled,
        target_volatility=target_volatility,
        lookback_periods=lookback_periods,
        volatility_epsilon=volatility_epsilon,
    )


def validate_return_series(
    returns: pd.Series,
    *,
    owner: str = "returns",
    min_samples: int = 1,
) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise PortfolioRiskError(f"{owner} must be provided as a pandas Series.")
    normalized = pd.to_numeric(returns, errors="coerce").astype("float64")
    if normalized.isna().any():
        failing_index = normalized.index[normalized.isna()][0]
        raise PortfolioRiskError(
            f"{owner} must contain only finite numeric values. First invalid index={failing_index!r}."
        )
    if not normalized.map(math.isfinite).all():
        failing_index = normalized.index[~normalized.map(math.isfinite)][0]
        raise PortfolioRiskError(
            f"{owner} must contain only finite numeric values. First invalid index={failing_index!r}."
        )
    if len(normalized) < min_samples:
        raise PortfolioRiskError(
            f"{owner} must contain at least {min_samples} observations."
        )
    return normalized


def validate_equity_curve_series(
    equity_curve: pd.Series,
    *,
    owner: str = "equity_curve",
) -> pd.Series:
    if not isinstance(equity_curve, pd.Series):
        raise PortfolioRiskError(f"{owner} must be provided as a pandas Series.")
    normalized = pd.to_numeric(equity_curve, errors="coerce").astype("float64")
    if normalized.isna().any():
        failing_index = normalized.index[normalized.isna()][0]
        raise PortfolioRiskError(
            f"{owner} must contain only finite numeric values. First invalid index={failing_index!r}."
        )
    if not normalized.map(math.isfinite).all():
        failing_index = normalized.index[~normalized.map(math.isfinite)][0]
        raise PortfolioRiskError(
            f"{owner} must contain only finite numeric values. First invalid index={failing_index!r}."
        )
    if (normalized <= 0.0).any():
        failing_index = normalized.index[normalized.le(0.0)][0]
        raise PortfolioRiskError(
            f"{owner} must remain strictly positive. First non-positive index={failing_index!r}."
        )
    return normalized


def rolling_volatility(
    returns: pd.Series,
    *,
    window: int,
    periods_per_year: int | None = None,
) -> pd.Series:
    normalized = validate_return_series(returns, owner="returns")
    if not isinstance(window, int) or window <= 1:
        raise PortfolioRiskError("rolling volatility window must be an integer greater than 1.")
    series = normalized.rolling(window=window, min_periods=window).std(ddof=1)
    if periods_per_year is not None:
        series = series * math.sqrt(_validate_periods_per_year(periods_per_year))
    return series.astype("float64")


def drawdown_series_from_equity(equity_curve: pd.Series) -> pd.Series:
    equity = validate_equity_curve_series(equity_curve)
    drawdown = 1.0 - (equity / equity.cummax())
    _validate_drawdown_bounds(drawdown)
    return drawdown.astype("float64")


def drawdown_series_from_returns(returns: pd.Series) -> pd.Series:
    normalized = validate_return_series(returns, owner="returns")
    return drawdown_series_from_equity((1.0 + normalized).cumprod())


def summarize_drawdown(
    *,
    equity_curve: pd.Series | None = None,
    returns: pd.Series | None = None,
) -> dict[str, float | int]:
    if equity_curve is None and returns is None:
        raise PortfolioRiskError("drawdown summary requires either equity_curve or returns.")
    drawdown = (
        drawdown_series_from_equity(equity_curve)
        if equity_curve is not None
        else drawdown_series_from_returns(returns if returns is not None else pd.Series(dtype="float64"))
    )
    return {
        "max_drawdown": float(drawdown.max()) if not drawdown.empty else 0.0,
        "current_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else 0.0,
        "max_drawdown_duration": int(_max_drawdown_duration(drawdown)),
        "current_drawdown_duration": int(_current_drawdown_duration(drawdown)),
    }


def historical_var(
    returns: pd.Series,
    *,
    confidence_level: float,
) -> float:
    normalized = validate_return_series(returns, owner="returns", min_samples=1)
    alpha = 1.0 - _coerce_confidence_level(
        confidence_level,
        field_name="confidence_level",
    )
    quantile = float(normalized.quantile(alpha, interpolation="lower"))
    return float(max(0.0, -quantile))


def historical_cvar(
    returns: pd.Series,
    *,
    confidence_level: float,
) -> float:
    normalized = validate_return_series(returns, owner="returns", min_samples=1)
    alpha = 1.0 - _coerce_confidence_level(
        confidence_level,
        field_name="confidence_level",
    )
    quantile = float(normalized.quantile(alpha, interpolation="lower"))
    tail = normalized.loc[normalized <= quantile]
    if tail.empty:
        tail = normalized.nsmallest(1)
    return float(max(0.0, -tail.mean()))


def volatility_target_diagnostics(
    returns: pd.Series,
    *,
    config: PortfolioRiskConfig | dict[str, Any] | None = None,
    periods_per_year: int,
    leverage_ceiling: float | None = None,
) -> dict[str, float | bool | None]:
    resolved = resolve_portfolio_risk_config(config)
    normalized = validate_return_series(returns, owner="returns", min_samples=1)
    annualization = _validate_periods_per_year(
        resolved.periods_per_year_override
        if resolved.periods_per_year_override is not None
        else periods_per_year
    )
    realized_volatility = (
        float(normalized.std(ddof=1) * math.sqrt(annualization))
        if len(normalized) >= 2
        else 0.0
    )
    rolling = rolling_volatility(
        normalized,
        window=resolved.volatility_window,
        periods_per_year=annualization,
    )
    latest_rolling = None if rolling.dropna().empty else float(rolling.dropna().iloc[-1])
    if resolved.target_volatility is None:
        return {
            "target_volatility": None,
            "realized_volatility": realized_volatility,
            "latest_rolling_volatility": latest_rolling,
            "recommended_scale": None,
            "recommended_scale_unclipped": None,
            "scale_floor": float(resolved.min_volatility_scale),
            "scale_cap": float(_effective_scale_cap(resolved, leverage_ceiling=leverage_ceiling)),
            "scale_was_capped": False,
            "scaling_limited_by_zero_volatility": False,
            "allow_scale_up": resolved.allow_scale_up,
            "volatility_window": int(resolved.volatility_window),
        }

    scale_cap = _effective_scale_cap(resolved, leverage_ceiling=leverage_ceiling)
    reference_volatility = latest_rolling if latest_rolling is not None else realized_volatility
    if reference_volatility <= resolved.volatility_epsilon:
        return {
            "target_volatility": float(resolved.target_volatility),
            "realized_volatility": realized_volatility,
            "latest_rolling_volatility": latest_rolling,
            "recommended_scale": float(resolved.min_volatility_scale),
            "recommended_scale_unclipped": None,
            "scale_floor": float(resolved.min_volatility_scale),
            "scale_cap": float(scale_cap),
            "scale_was_capped": False,
            "scaling_limited_by_zero_volatility": True,
            "allow_scale_up": resolved.allow_scale_up,
            "volatility_window": int(resolved.volatility_window),
        }

    recommended_unclipped = float(resolved.target_volatility / reference_volatility)
    recommended = min(max(recommended_unclipped, resolved.min_volatility_scale), scale_cap)
    return {
        "target_volatility": float(resolved.target_volatility),
        "realized_volatility": realized_volatility,
        "latest_rolling_volatility": latest_rolling,
        "recommended_scale": float(recommended),
        "recommended_scale_unclipped": recommended_unclipped,
        "scale_floor": float(resolved.min_volatility_scale),
        "scale_cap": float(scale_cap),
        "scale_was_capped": bool(abs(recommended - recommended_unclipped) > 1e-12),
        "scaling_limited_by_zero_volatility": False,
        "allow_scale_up": resolved.allow_scale_up,
        "volatility_window": int(resolved.volatility_window),
    }


def summarize_portfolio_risk(
    returns: pd.Series,
    *,
    equity_curve: pd.Series | None = None,
    config: PortfolioRiskConfig | dict[str, Any] | None = None,
    periods_per_year: int,
    leverage_ceiling: float | None = None,
) -> dict[str, Any]:
    resolved = resolve_portfolio_risk_config(config)
    annualization = _validate_periods_per_year(
        resolved.periods_per_year_override
        if resolved.periods_per_year_override is not None
        else periods_per_year
    )
    normalized = validate_return_series(returns, owner="returns")
    rolling = rolling_volatility(
        normalized,
        window=resolved.volatility_window,
        periods_per_year=annualization,
    )
    rolling_values = rolling.dropna()
    drawdown = summarize_drawdown(equity_curve=equity_curve, returns=normalized)
    vol_target = volatility_target_diagnostics(
        normalized,
        config=resolved,
        periods_per_year=annualization,
        leverage_ceiling=leverage_ceiling,
    )
    var_value = historical_var(normalized, confidence_level=resolved.var_confidence_level)
    cvar_value = historical_cvar(normalized, confidence_level=resolved.cvar_confidence_level)
    if cvar_value + 1e-12 < var_value:
        raise PortfolioRiskError("portfolio CVaR must be greater than or equal to portfolio VaR.")

    summary = {
        "config": resolved.to_dict(),
        "sample_count": int(len(normalized)),
        "periods_per_year": int(annualization),
        "rolling_volatility": {
            "window": int(resolved.volatility_window),
            "warmup_periods": int(max(resolved.volatility_window - 1, 0)),
            "latest": None if rolling_values.empty else float(rolling_values.iloc[-1]),
            "mean": None if rolling_values.empty else float(rolling_values.mean()),
            "min": None if rolling_values.empty else float(rolling_values.min()),
            "max": None if rolling_values.empty else float(rolling_values.max()),
        },
        "volatility_targeting": vol_target,
        "drawdown": drawdown,
        "tail_risk": {
            "var": var_value,
            "var_confidence_level": float(resolved.var_confidence_level),
            "cvar": cvar_value,
            "cvar_confidence_level": float(resolved.cvar_confidence_level),
            "quantile_method": "historical_empirical_lower_tail",
            "loss_sign_convention": "positive_loss_fraction",
        },
    }
    _validate_risk_summary(summary)
    return summary


def apply_volatility_targeting(
    weights_wide: pd.DataFrame,
    returns_wide: pd.DataFrame,
    *,
    config: PortfolioVolatilityTargetingConfig | dict[str, Any] | None = None,
    periods_per_year: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = resolve_portfolio_volatility_targeting_config(config)
    normalized_weights = _validate_targeting_matrix(weights_wide, owner="weights_wide")
    normalized_returns = _validate_targeting_matrix(returns_wide, owner="returns_wide")

    if not normalized_weights.index.equals(normalized_returns.index):
        raise PortfolioRiskError(
            "weights_wide and returns_wide must have exactly matching indices for volatility targeting."
        )
    if not normalized_weights.columns.equals(normalized_returns.columns):
        raise PortfolioRiskError(
            "weights_wide and returns_wide must have exactly matching columns for volatility targeting."
        )

    base_weights = normalized_weights.copy()
    base_portfolio_returns = (normalized_returns * base_weights).sum(axis=1).astype("float64")
    diagnostics = volatility_target_diagnostics(
        base_portfolio_returns,
        config={
            "volatility_window": resolved.lookback_periods,
            "target_volatility": resolved.target_volatility,
            "volatility_epsilon": resolved.volatility_epsilon,
            "min_volatility_scale": 0.0,
            "max_volatility_scale": 1_000_000_000_000.0,
            "allow_scale_up": True,
        },
        periods_per_year=periods_per_year,
        leverage_ceiling=None,
    )

    metadata: dict[str, Any] = {
        "enabled": bool(resolved.enabled),
        "config": resolved.to_dict(),
        "target_volatility": resolved.target_volatility,
        "lookback_periods": int(resolved.lookback_periods),
        "base_weights": base_weights.copy(),
        "estimated_pre_target_volatility": diagnostics["latest_rolling_volatility"]
        if diagnostics["latest_rolling_volatility"] is not None
        else diagnostics["realized_volatility"],
        "estimated_post_target_volatility": None,
        "volatility_scaling_factor": None,
        "scaling_limited_by_zero_volatility": False,
    }

    if not resolved.enabled:
        metadata["targeted_weights"] = base_weights.copy()
        return base_weights, metadata

    estimated_pre_target_volatility = metadata["estimated_pre_target_volatility"]
    if estimated_pre_target_volatility is None or float(estimated_pre_target_volatility) <= resolved.volatility_epsilon:
        raise PortfolioRiskError(
            "cannot apply volatility targeting when estimated portfolio volatility is effectively zero."
        )

    scaling_factor = float(resolved.target_volatility / float(estimated_pre_target_volatility))
    if not math.isfinite(scaling_factor) or scaling_factor < 0.0:
        raise PortfolioRiskError("volatility targeting scaling factor must be finite and non-negative.")

    targeted_weights = (base_weights * scaling_factor).astype("float64")
    metadata["volatility_scaling_factor"] = scaling_factor
    metadata["estimated_post_target_volatility"] = float(
        float(estimated_pre_target_volatility) * scaling_factor
    )
    metadata["targeted_weights"] = targeted_weights.copy()
    return targeted_weights, metadata


def _validate_risk_summary(summary: dict[str, Any]) -> None:
    rolling = summary.get("rolling_volatility", {})
    tail_risk = summary.get("tail_risk", {})
    drawdown = summary.get("drawdown", {})
    vol_target = summary.get("volatility_targeting", {})

    for key in ("latest", "mean", "min", "max"):
        value = rolling.get(key)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise PortfolioRiskError(f"rolling volatility summary field {key!r} must be finite and non-negative.")

    for key in ("max_drawdown", "current_drawdown"):
        value = drawdown.get(key)
        if value is None or not math.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
            raise PortfolioRiskError(f"drawdown summary field {key!r} must be within [0, 1].")

    if int(drawdown.get("max_drawdown_duration", 0)) < 0 or int(drawdown.get("current_drawdown_duration", 0)) < 0:
        raise PortfolioRiskError("drawdown duration fields must be non-negative integers.")

    for key in ("var", "cvar"):
        value = tail_risk.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            raise PortfolioRiskError(f"tail risk field {key!r} must be finite and non-negative.")

    recommended_scale = vol_target.get("recommended_scale")
    if recommended_scale is not None and (not math.isfinite(float(recommended_scale)) or float(recommended_scale) < 0.0):
        raise PortfolioRiskError("volatility targeting recommended_scale must be finite and non-negative.")


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    duration = 0
    max_duration = 0
    for value in drawdown.tolist():
        if float(value) > 0.0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_duration


def _current_drawdown_duration(drawdown: pd.Series) -> int:
    duration = 0
    for value in reversed(drawdown.tolist()):
        if float(value) > 0.0:
            duration += 1
            continue
        break
    return duration


def _validate_drawdown_bounds(drawdown: pd.Series) -> None:
    if drawdown.isna().any():
        raise PortfolioRiskError("drawdown series must not contain NaN values.")
    out_of_bounds = (drawdown < -1e-12) | (drawdown > 1.0 + 1e-12)
    if out_of_bounds.any():
        failing_index = drawdown.index[out_of_bounds][0]
        raise PortfolioRiskError(
            f"drawdown series must remain within [0, 1]. First invalid index={failing_index!r}."
        )


def _effective_scale_cap(config: PortfolioRiskConfig, *, leverage_ceiling: float | None) -> float:
    scale_cap = float(config.max_volatility_scale)
    if leverage_ceiling is not None:
        scale_cap = min(scale_cap, float(leverage_ceiling))
    if not config.allow_scale_up:
        scale_cap = min(scale_cap, 1.0)
    return float(scale_cap)


def _validate_periods_per_year(periods_per_year: int) -> int:
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise PortfolioRiskError("periods_per_year must be a positive integer.")
    return periods_per_year


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise PortfolioRiskError(f"{field_name} must be a positive integer.")
    return value


def _coerce_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _coerce_positive_int(value, field_name=field_name)


def _coerce_positive_float(value: Any, *, field_name: str) -> float:
    numeric = _coerce_float(value, field_name=field_name)
    if numeric <= 0.0:
        raise PortfolioRiskError(f"{field_name} must be positive.")
    return numeric


def _coerce_optional_positive_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _coerce_positive_float(value, field_name=field_name)


def _coerce_non_negative_float(value: Any, *, field_name: str) -> float:
    numeric = _coerce_float(value, field_name=field_name)
    if numeric < 0.0:
        raise PortfolioRiskError(f"{field_name} must be non-negative.")
    return numeric


def _coerce_confidence_level(value: Any, *, field_name: str) -> float:
    numeric = _coerce_float(value, field_name=field_name)
    if not (0.0 < numeric < 1.0):
        raise PortfolioRiskError(f"{field_name} must be strictly between 0 and 1.")
    return numeric


def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PortfolioRiskError(f"{field_name} must be a finite float.") from exc
    if not math.isfinite(numeric):
        raise PortfolioRiskError(f"{field_name} must be a finite float.")
    return numeric


__all__ = [
    "PortfolioRiskConfig",
    "PortfolioRiskError",
    "PortfolioVolatilityTargetingConfig",
    "apply_volatility_targeting",
    "drawdown_series_from_equity",
    "drawdown_series_from_returns",
    "historical_cvar",
    "historical_var",
    "resolve_portfolio_risk_config",
    "resolve_portfolio_volatility_targeting_config",
    "rolling_volatility",
    "summarize_drawdown",
    "summarize_portfolio_risk",
    "validate_equity_curve_series",
    "validate_return_series",
    "volatility_target_diagnostics",
]


def _validate_targeting_matrix(df: pd.DataFrame, *, owner: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise PortfolioRiskError(f"{owner} must be provided as a pandas DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise PortfolioRiskError(f"{owner} must use a DatetimeIndex.")
    if df.index.name != "ts_utc":
        raise PortfolioRiskError(f"{owner} index must be named 'ts_utc'.")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise PortfolioRiskError(f"{owner} index must be timezone-aware UTC.")
    if df.columns.empty:
        raise PortfolioRiskError(f"{owner} must contain at least one strategy column.")
    if not df.columns.is_unique:
        raise PortfolioRiskError(f"{owner} must contain unique strategy columns.")

    normalized = df.copy()
    normalized = normalized.sort_index(kind="stable")
    normalized = normalized.loc[:, sorted(str(column) for column in normalized.columns)]
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    if normalized.isna().any().any():
        bad_index, bad_column = normalized.isna().stack()[lambda values: values].index[0]
        raise PortfolioRiskError(
            f"{owner} must contain only finite numeric values. First invalid location=({bad_index!r}, {bad_column!r})."
        )
    if not normalized.map(lambda value: math.isfinite(float(value))).all().all():
        raise PortfolioRiskError(f"{owner} must contain only finite numeric values.")
    return normalized
