from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import (
    PortfolioContractError,
    PortfolioValidationConfig,
    resolve_portfolio_validation_config,
    validate_portfolio_output,
    validate_weights,
)


class PortfolioValidationError(ValueError):
    """Raised when portfolio-level invariants or sanity checks fail."""


def validate_portfolio_weights(
    weights_wide: pd.DataFrame,
    validation_config: PortfolioValidationConfig | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate deterministic allocation, exposure, and sleeve-bound invariants."""

    config = resolve_portfolio_validation_config(validation_config)
    try:
        normalized = validate_weights(weights_wide, validation_config=config)
    except PortfolioContractError as exc:
        raise PortfolioValidationError(f"Portfolio weight validation failed: {exc}") from exc

    diagnostics = summarize_weight_diagnostics(normalized)
    _raise_on_weight_constraint_breach(normalized, diagnostics=diagnostics, config=config)
    normalized.attrs["portfolio_validation"] = {
        "weight_diagnostics": diagnostics,
        "sanity_issues": [],
        "strict_sanity_checks": config.strict_sanity_checks,
    }
    return normalized


def validate_portfolio_output_constraints(
    portfolio_output: pd.DataFrame,
    *,
    validation_config: PortfolioValidationConfig | dict[str, Any] | None = None,
    initial_capital: float | None = None,
    require_traceability: bool = False,
    strict_sanity_checks: bool | None = None,
) -> pd.DataFrame:
    """Validate portfolio exposures plus return-stream and compounding sanity."""

    config = resolve_portfolio_validation_config(validation_config)
    strict = config.strict_sanity_checks if strict_sanity_checks is None else strict_sanity_checks
    try:
        normalized = validate_portfolio_output(portfolio_output)
    except PortfolioContractError as exc:
        raise PortfolioValidationError(f"Portfolio output validation failed: {exc}") from exc

    weight_frame = _extract_weight_frame(normalized)
    if require_traceability and weight_frame is None:
        raise PortfolioValidationError(
            "Portfolio output validation failed: weight__<strategy> traceability columns are required."
        )

    diagnostics = (
        summarize_weight_diagnostics(weight_frame)
        if weight_frame is not None
        else _empty_weight_diagnostics(row_count=len(normalized))
    )
    if weight_frame is not None:
        _raise_on_weight_constraint_breach(weight_frame, diagnostics=diagnostics, config=config)

    sanity_issues = _collect_sanity_issues(
        normalized,
        config=config,
        initial_capital=initial_capital,
        weight_diagnostics=diagnostics,
        has_weight_traceability=weight_frame is not None,
    )
    if sanity_issues and strict:
        raise PortfolioValidationError("; ".join(sanity_issues))

    normalized.attrs["portfolio_validation"] = {
        "weight_diagnostics": diagnostics,
        "sanity_issues": sanity_issues,
        "strict_sanity_checks": strict,
    }
    return normalized


def summarize_weight_diagnostics(weights_wide: pd.DataFrame) -> dict[str, float | int]:
    """Return deterministic exposure diagnostics for one weight matrix."""

    if weights_wide.empty:
        return _empty_weight_diagnostics(row_count=0)

    row_sums = weights_wide.sum(axis=1).astype("float64")
    gross_exposure = weights_wide.abs().sum(axis=1).astype("float64")
    net_exposure = row_sums.astype("float64")
    leverage = gross_exposure.astype("float64")
    max_single_sleeve_weight = weights_wide.abs().max(axis=1).astype("float64")

    return {
        "row_count": int(len(weights_wide)),
        "average_gross_exposure": float(gross_exposure.mean()),
        "max_gross_exposure": float(gross_exposure.max()),
        "average_net_exposure": float(net_exposure.mean()),
        "min_net_exposure": float(net_exposure.min()),
        "max_net_exposure": float(net_exposure.max()),
        "average_leverage": float(leverage.mean()),
        "max_leverage": float(leverage.max()),
        "max_single_weight": float(max_single_sleeve_weight.max()),
        "max_weight_sum_deviation": float((row_sums - 1.0).abs().max()),
    }


def _raise_on_weight_constraint_breach(
    weights_wide: pd.DataFrame,
    *,
    diagnostics: dict[str, float | int],
    config: PortfolioValidationConfig,
) -> None:
    row_sums = weights_wide.sum(axis=1).astype("float64")
    gross_exposure = weights_wide.abs().sum(axis=1).astype("float64")
    net_exposure = row_sums.astype("float64")
    leverage = gross_exposure.astype("float64")

    invalid_net = (net_exposure - config.target_net_exposure).abs() > config.net_exposure_tolerance
    if invalid_net.any():
        bad_ts = net_exposure.index[invalid_net][0]
        raise PortfolioValidationError(
            "Portfolio weight validation failed: net exposure must equal "
            f"{config.target_net_exposure} within tolerance {config.net_exposure_tolerance}. "
            f"First failing ts_utc={bad_ts}, net_exposure={net_exposure.loc[bad_ts]}."
        )

    gross_violation = gross_exposure > config.max_gross_exposure
    if gross_violation.any():
        bad_ts = gross_exposure.index[gross_violation][0]
        raise PortfolioValidationError(
            "Portfolio weight validation failed: gross exposure exceeds configured maximum "
            f"{config.max_gross_exposure}. First failing ts_utc={bad_ts}, "
            f"gross_exposure={gross_exposure.loc[bad_ts]}."
        )

    leverage_violation = leverage > config.max_leverage
    if leverage_violation.any():
        bad_ts = leverage.index[leverage_violation][0]
        raise PortfolioValidationError(
            "Portfolio weight validation failed: leverage exceeds configured maximum "
            f"{config.max_leverage}. First failing ts_utc={bad_ts}, leverage={leverage.loc[bad_ts]}."
        )

    if config.long_only:
        negative_mask = weights_wide < -config.weight_sum_tolerance
        if negative_mask.any().any():
            bad_ts, bad_strategy = negative_mask.stack()[lambda values: values].index[0]
            raise PortfolioValidationError(
                "Portfolio weight validation failed: long-only portfolios cannot hold negative sleeve weights. "
                f"First failing ts_utc={bad_ts}, strategy={bad_strategy!r}, "
                f"weight={weights_wide.loc[bad_ts, bad_strategy]}."
            )

    if config.max_single_sleeve_weight is not None:
        overweight_mask = weights_wide.abs() > config.max_single_sleeve_weight
        if overweight_mask.any().any():
            bad_ts, bad_strategy = overweight_mask.stack()[lambda values: values].index[0]
            raise PortfolioValidationError(
                "Portfolio weight validation failed: sleeve weight exceeds configured maximum "
                f"{config.max_single_sleeve_weight}. First failing ts_utc={bad_ts}, "
                f"strategy={bad_strategy!r}, weight={weights_wide.loc[bad_ts, bad_strategy]}."
            )

    if config.min_single_sleeve_weight is not None:
        underweight_mask = weights_wide < config.min_single_sleeve_weight
        if underweight_mask.any().any():
            bad_ts, bad_strategy = underweight_mask.stack()[lambda values: values].index[0]
            raise PortfolioValidationError(
                "Portfolio weight validation failed: sleeve weight is below configured minimum "
                f"{config.min_single_sleeve_weight}. First failing ts_utc={bad_ts}, "
                f"strategy={bad_strategy!r}, weight={weights_wide.loc[bad_ts, bad_strategy]}."
            )

    diagnostics["max_weight_sum_deviation"] = float(
        (row_sums - config.target_weight_sum).abs().max() if not row_sums.empty else 0.0
    )


def _collect_sanity_issues(
    portfolio_output: pd.DataFrame,
    *,
    config: PortfolioValidationConfig,
    initial_capital: float | None,
    weight_diagnostics: dict[str, float | int],
    has_weight_traceability: bool,
) -> list[str]:
    issues: list[str] = []
    portfolio_return = portfolio_output["portfolio_return"].astype("float64")

    below_negative_one = portfolio_return < -1.0
    if below_negative_one.any():
        bad_index = int(below_negative_one[below_negative_one].index[0])
        raise PortfolioValidationError(
            "Portfolio output validation failed: portfolio_return cannot be less than -1.0. "
            f"First failing row index={bad_index}, value={portfolio_return.iloc[bad_index]}."
        )

    if config.max_abs_period_return is not None:
        extreme_period = portfolio_return.abs() > config.max_abs_period_return
        if extreme_period.any():
            bad_index = int(extreme_period[extreme_period].index[0])
            issues.append(
                "Portfolio output sanity check failed: absolute portfolio_return exceeds configured "
                f"maximum {config.max_abs_period_return} at row index {bad_index}."
            )

    strategy_returns = _extract_strategy_return_frame(portfolio_output)
    weight_frame = _extract_weight_frame(portfolio_output)
    if strategy_returns is not None and weight_frame is not None:
        recomputed_gross = (strategy_returns * weight_frame).sum(axis=1).astype("float64")
        if "gross_portfolio_return" in portfolio_output.columns:
            mismatch = (recomputed_gross - portfolio_output["gross_portfolio_return"]).abs() > 1e-10
            if mismatch.any():
                bad_index = int(mismatch[mismatch].index[0])
                raise PortfolioValidationError(
                    "Portfolio output validation failed: gross_portfolio_return does not match "
                    f"weighted component returns at row index {bad_index}."
                )
        theoretical_bound = (strategy_returns.abs() * weight_frame.abs()).sum(axis=1).astype("float64") + 1e-10
        impossible_magnitude = recomputed_gross.abs() > theoretical_bound
        if impossible_magnitude.any():
            bad_index = int(impossible_magnitude[impossible_magnitude].index[0])
            raise PortfolioValidationError(
                "Portfolio output validation failed: gross portfolio returns exceed deterministic "
                f"exposure-implied bounds at row index {bad_index}."
            )

    if "net_portfolio_return" in portfolio_output.columns:
        expected_net = portfolio_output["gross_portfolio_return"].astype("float64")
        if "portfolio_transaction_cost" in portfolio_output.columns:
            expected_net = expected_net - portfolio_output["portfolio_transaction_cost"].astype("float64")
        if "portfolio_slippage_cost" in portfolio_output.columns:
            expected_net = expected_net - portfolio_output["portfolio_slippage_cost"].astype("float64")
        mismatch = (expected_net - portfolio_output["net_portfolio_return"].astype("float64")).abs() > 1e-10
        if mismatch.any():
            bad_index = int(mismatch[mismatch].index[0])
            raise PortfolioValidationError(
                "Portfolio output validation failed: net_portfolio_return does not match gross "
                f"returns minus execution frictions at row index {bad_index}."
            )

    if "portfolio_equity_curve" in portfolio_output.columns:
        equity = portfolio_output["portfolio_equity_curve"].astype("float64")
        if (equity <= 0.0).any():
            bad_index = int(equity.index[equity.le(0.0)][0])
            raise PortfolioValidationError(
                "Portfolio output validation failed: portfolio_equity_curve must remain positive. "
                f"First non-positive row index={bad_index}."
            )
        inferred_initial_capital = _resolve_initial_capital(
            portfolio_output,
            initial_capital=initial_capital,
        )
        expected_equity = float(inferred_initial_capital) * (1.0 + portfolio_return).cumprod()
        mismatch = (expected_equity - equity).abs() > 1e-10
        if mismatch.any():
            bad_index = int(mismatch[mismatch].index[0])
            raise PortfolioValidationError(
                "Portfolio output validation failed: portfolio_equity_curve does not match "
                f"compounded portfolio_return values at row index {bad_index}."
            )
        if config.max_equity_multiple is not None:
            max_multiple = equity / float(inferred_initial_capital)
            too_large = max_multiple > config.max_equity_multiple
            if too_large.any():
                bad_index = int(too_large[too_large].index[0])
                issues.append(
                    "Portfolio output sanity check failed: portfolio_equity_curve exceeds configured "
                    f"maximum equity multiple {config.max_equity_multiple} at row index {bad_index}."
                )

    if has_weight_traceability and weight_diagnostics["max_gross_exposure"] == 0.0 and portfolio_return.abs().gt(0.0).any():
        bad_index = int(portfolio_return.abs().gt(0.0).idxmax())
        raise PortfolioValidationError(
            "Portfolio output validation failed: portfolio_return is non-zero while gross exposure "
            f"is zero at row index {bad_index}."
        )

    return issues


def _extract_weight_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame | None:
    weight_columns = [column for column in portfolio_output.columns if column.startswith("weight__")]
    if not weight_columns:
        return None
    weights = portfolio_output.loc[:, weight_columns].copy()
    weights.columns = [column.removeprefix("weight__") for column in weight_columns]
    weights.index = pd.DatetimeIndex(pd.to_datetime(portfolio_output["ts_utc"], utc=True), name="ts_utc")
    return weights.astype("float64")


def _extract_strategy_return_frame(portfolio_output: pd.DataFrame) -> pd.DataFrame | None:
    return_columns = [column for column in portfolio_output.columns if column.startswith("strategy_return__")]
    if not return_columns:
        return None
    returns = portfolio_output.loc[:, return_columns].copy()
    returns.columns = [column.removeprefix("strategy_return__") for column in return_columns]
    returns.index = pd.DatetimeIndex(pd.to_datetime(portfolio_output["ts_utc"], utc=True), name="ts_utc")
    return returns.astype("float64")


def _resolve_initial_capital(
    portfolio_output: pd.DataFrame,
    *,
    initial_capital: float | None,
) -> float:
    if initial_capital is not None:
        return float(initial_capital)
    if portfolio_output.empty:
        raise PortfolioValidationError(
            "Portfolio output validation failed: cannot infer initial capital from an empty portfolio."
        )
    if "portfolio_equity_curve" not in portfolio_output.columns:
        raise PortfolioValidationError(
            "Portfolio output validation failed: initial_capital is required when portfolio_equity_curve is absent."
        )

    first_return = float(portfolio_output["portfolio_return"].iloc[0])
    denominator = 1.0 + first_return
    if abs(denominator) <= 1e-10:
        raise PortfolioValidationError(
            "Portfolio output validation failed: cannot infer initial capital when the first "
            "portfolio return equals -100%."
        )
    return float(portfolio_output["portfolio_equity_curve"].iloc[0]) / denominator


def _empty_weight_diagnostics(*, row_count: int) -> dict[str, float | int]:
    return {
        "row_count": int(row_count),
        "average_gross_exposure": 0.0,
        "max_gross_exposure": 0.0,
        "average_net_exposure": 0.0,
        "min_net_exposure": 0.0,
        "max_net_exposure": 0.0,
        "average_leverage": 0.0,
        "max_leverage": 0.0,
        "max_single_weight": 0.0,
        "max_weight_sum_deviation": 0.0,
    }


__all__ = [
    "PortfolioValidationError",
    "summarize_weight_diagnostics",
    "validate_portfolio_output_constraints",
    "validate_portfolio_weights",
]
