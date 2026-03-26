from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import pandas as pd

from src.config.sanity import SanityCheckConfig, resolve_sanity_check_config


class SanityCheckError(ValueError):
    """Raised when deterministic sanity checks fail execution."""


@dataclass(frozen=True)
class SanityCheckIssue:
    code: str
    message: str
    severity: str
    value: float | None = None
    threshold: float | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "value": self.value,
            "threshold": self.threshold,
            "location": self.location,
        }


@dataclass(frozen=True)
class SanityCheckReport:
    scope: str
    strict_sanity_checks: bool
    hard_failures: tuple[SanityCheckIssue, ...]
    warnings: tuple[SanityCheckIssue, ...]

    @property
    def issues(self) -> tuple[SanityCheckIssue, ...]:
        return self.hard_failures + self.warnings

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def status(self) -> str:
        if self.hard_failures:
            return "fail"
        if self.warnings and self.strict_sanity_checks:
            return "fail"
        if self.warnings:
            return "warn"
        return "pass"

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "status": self.status,
            "issue_count": self.issue_count,
            "warning_count": self.warning_count,
            "strict_sanity_checks": self.strict_sanity_checks,
            "hard_failures": [issue.to_dict() for issue in self.hard_failures],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_for_failure(self) -> None:
        if not self.failed:
            return
        messages = "; ".join(issue.message for issue in self.issues)
        raise SanityCheckError(messages)

    def apply_to_metrics(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(metrics)
        payload["sanity_status"] = self.status
        payload["sanity_issue_count"] = float(self.issue_count)
        payload["sanity_warning_count"] = float(self.warning_count)
        payload["sanity_strict_mode"] = self.strict_sanity_checks
        return payload


def validate_strategy_backtest_sanity(
    results_df: pd.DataFrame,
    metrics: Mapping[str, Any],
    sanity_config: SanityCheckConfig | Mapping[str, Any] | None = None,
    *,
    scope: str = "strategy_backtest",
) -> SanityCheckReport:
    return _validate_return_stream_sanity(
        return_streams={
            "strategy_return": _optional_numeric_series(results_df, "strategy_return"),
            "net_strategy_return": _optional_numeric_series(results_df, "net_strategy_return"),
        },
        equity_curve=_optional_numeric_series(results_df, "equity_curve"),
        metrics=metrics,
        sanity_config=sanity_config,
        scope=scope,
        equity_multiple=_strategy_equity_multiple(results_df),
    )


def validate_portfolio_output_sanity(
    portfolio_output: pd.DataFrame,
    metrics: Mapping[str, Any],
    sanity_config: SanityCheckConfig | Mapping[str, Any] | None = None,
    *,
    initial_capital: float | None = None,
    scope: str = "portfolio_backtest",
) -> SanityCheckReport:
    equity_curve = _optional_numeric_series(portfolio_output, "portfolio_equity_curve")
    return _validate_return_stream_sanity(
        return_streams={"portfolio_return": _optional_numeric_series(portfolio_output, "portfolio_return")},
        equity_curve=equity_curve,
        metrics=metrics,
        sanity_config=sanity_config,
        scope=scope,
        equity_multiple=_portfolio_equity_multiple(portfolio_output, initial_capital=initial_capital),
    )


def summarize_walk_forward_sanity(
    split_reports: Iterable[SanityCheckReport],
    aggregate_report: SanityCheckReport,
    *,
    strict_sanity_checks: bool,
    scope: str,
) -> dict[str, Any]:
    reports = list(split_reports)
    flagged_split_ids = [str(index) for index, report in enumerate(reports) if report.issue_count > 0]
    failed_split_ids = [str(index) for index, report in enumerate(reports) if report.failed]
    return {
        "scope": scope,
        "status": "fail"
        if aggregate_report.failed or failed_split_ids or (strict_sanity_checks and flagged_split_ids)
        else "warn"
        if aggregate_report.issue_count or flagged_split_ids
        else "pass",
        "strict_sanity_checks": strict_sanity_checks,
        "aggregate": aggregate_report.to_dict(),
        "flagged_split_count": len(flagged_split_ids),
        "failed_split_count": len(failed_split_ids),
        "flagged_split_indexes": flagged_split_ids,
        "failed_split_indexes": failed_split_ids,
    }


def _validate_return_stream_sanity(
    *,
    return_streams: Mapping[str, pd.Series],
    equity_curve: pd.Series,
    metrics: Mapping[str, Any],
    sanity_config: SanityCheckConfig | Mapping[str, Any] | None,
    scope: str,
    equity_multiple: float | None,
) -> SanityCheckReport:
    config = resolve_sanity_check_config(sanity_config)
    hard_failures: list[SanityCheckIssue] = []
    warnings: list[SanityCheckIssue] = []

    for name, series in return_streams.items():
        if series.empty:
            continue
        invalid_mask = _invalid_numeric_mask(series)
        if invalid_mask.any():
            bad_index = invalid_mask[invalid_mask].index[0]
            hard_failures.append(
                SanityCheckIssue(
                    code=f"{name}_non_finite",
                    message=f"Sanity check failed: {name} contains non-finite values at row index {bad_index}.",
                    severity="hard",
                    location=f"row_index={bad_index}",
                )
            )
            continue

        if config.max_abs_period_return is not None and not series.dropna().empty:
            max_abs_return = float(series.abs().max())
            if max_abs_return > config.max_abs_period_return:
                bad_index = int(series.abs().idxmax())
                warnings.append(
                    SanityCheckIssue(
                        code=f"{name}_max_abs_period_return",
                        message=(
                            f"Sanity check flagged: absolute {name} exceeds configured maximum "
                            f"{config.max_abs_period_return} at row index {bad_index}."
                        ),
                        severity="warning",
                        value=max_abs_return,
                        threshold=config.max_abs_period_return,
                        location=f"row_index={bad_index}",
                    )
                )

    if not equity_curve.empty:
        invalid_equity = _invalid_numeric_mask(equity_curve)
        if invalid_equity.any():
            bad_index = invalid_equity[invalid_equity].index[0]
            hard_failures.append(
                SanityCheckIssue(
                    code="equity_curve_non_finite",
                    message=f"Sanity check failed: equity curve contains non-finite values at row index {bad_index}.",
                    severity="hard",
                    location=f"row_index={bad_index}",
                )
            )

    for metric_name, value in metrics.items():
        if value is None or isinstance(value, bool | str | dict | list | tuple):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            hard_failures.append(
                SanityCheckIssue(
                    code=f"{metric_name}_non_finite",
                    message=f"Sanity check failed: metric '{metric_name}' is non-finite.",
                    severity="hard",
                    value=numeric,
                )
            )

    annualized_return = _metric_value(metrics, "annualized_return")
    if config.max_annualized_return is not None and annualized_return is not None and annualized_return > config.max_annualized_return:
        warnings.append(
            SanityCheckIssue(
                code="annualized_return_exceeds_threshold",
                message=(
                    "Sanity check flagged: annualized_return exceeds configured maximum "
                    f"{config.max_annualized_return}."
                ),
                severity="warning",
                value=annualized_return,
                threshold=config.max_annualized_return,
            )
        )

    sharpe_ratio = _metric_value(metrics, "sharpe_ratio")
    if config.max_sharpe_ratio is not None and sharpe_ratio is not None and sharpe_ratio > config.max_sharpe_ratio:
        warnings.append(
            SanityCheckIssue(
                code="sharpe_ratio_exceeds_threshold",
                message=f"Sanity check flagged: sharpe_ratio exceeds configured maximum {config.max_sharpe_ratio}.",
                severity="warning",
                value=sharpe_ratio,
                threshold=config.max_sharpe_ratio,
            )
        )

    if config.max_equity_multiple is not None and equity_multiple is not None and equity_multiple > config.max_equity_multiple:
        warnings.append(
            SanityCheckIssue(
                code="equity_multiple_exceeds_threshold",
                message=(
                    "Sanity check flagged: equity multiple exceeds configured maximum "
                    f"{config.max_equity_multiple}."
                ),
                severity="warning",
                value=equity_multiple,
                threshold=config.max_equity_multiple,
            )
        )

    annualized_volatility = _metric_value(metrics, "annualized_volatility")
    if (
        config.min_annualized_volatility_floor is not None
        and annualized_volatility is not None
        and annualized_volatility < config.min_annualized_volatility_floor
        and (
            (
                config.min_volatility_trigger_sharpe is not None
                and sharpe_ratio is not None
                and sharpe_ratio >= config.min_volatility_trigger_sharpe
            )
            or (
                config.min_volatility_trigger_annualized_return is not None
                and annualized_return is not None
                and annualized_return >= config.min_volatility_trigger_annualized_return
            )
        )
    ):
        warnings.append(
            SanityCheckIssue(
                code="low_volatility_high_performance_combo",
                message=(
                    "Sanity check flagged: annualized volatility is below the configured floor "
                    "despite unusually strong performance."
                ),
                severity="warning",
                value=annualized_volatility,
                threshold=config.min_annualized_volatility_floor,
            )
        )

    primary_returns = next(iter(return_streams.values()), pd.Series(dtype="float64"))
    max_drawdown = _metric_value(metrics, "max_drawdown")
    positive_fraction = _positive_return_fraction(primary_returns)
    if (
        config.smoothness_min_sharpe is not None
        and config.smoothness_min_annualized_return is not None
        and config.smoothness_max_drawdown is not None
        and config.smoothness_min_positive_return_fraction is not None
        and sharpe_ratio is not None
        and annualized_return is not None
        and max_drawdown is not None
        and sharpe_ratio >= config.smoothness_min_sharpe
        and annualized_return >= config.smoothness_min_annualized_return
        and max_drawdown <= config.smoothness_max_drawdown
        and positive_fraction >= config.smoothness_min_positive_return_fraction
    ):
        warnings.append(
            SanityCheckIssue(
                code="suspicious_smoothness",
                message=(
                    "Sanity check flagged: return path is unusually smooth relative to its "
                    "return and drawdown profile."
                ),
                severity="warning",
                value=positive_fraction,
                threshold=config.smoothness_min_positive_return_fraction,
            )
        )

    report = SanityCheckReport(
        scope=scope,
        strict_sanity_checks=config.strict_sanity_checks,
        hard_failures=tuple(hard_failures),
        warnings=tuple(warnings),
    )
    report.raise_for_failure()
    return report


def _metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return numeric
    return numeric


def _optional_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _positive_return_fraction(series: pd.Series) -> float:
    values = series.dropna().astype("float64")
    if values.empty:
        return 0.0
    return float((values > 0.0).mean())


def _invalid_numeric_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    if values.empty:
        return pd.Series(dtype="bool")
    finite_mask = values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    if finite_mask.any():
        first_valid_position = next(index for index, value in enumerate(finite_mask.tolist()) if value)
        relevant = values.iloc[first_valid_position:]
        invalid_relevant = relevant.isna() | ~relevant.map(lambda value: math.isfinite(float(value)))
        return invalid_relevant.reindex(values.index, fill_value=False)
    return values.isna()


def _strategy_equity_multiple(results_df: pd.DataFrame) -> float | None:
    if "equity_curve" not in results_df.columns:
        return None
    equity = pd.to_numeric(results_df["equity_curve"], errors="coerce").dropna().astype("float64")
    if equity.empty:
        return None
    return float(equity.max())


def _portfolio_equity_multiple(
    portfolio_output: pd.DataFrame,
    *,
    initial_capital: float | None,
) -> float | None:
    if "portfolio_equity_curve" not in portfolio_output.columns:
        return None
    equity = pd.to_numeric(portfolio_output["portfolio_equity_curve"], errors="coerce").dropna().astype("float64")
    if equity.empty:
        return None
    resolved_initial = float(initial_capital) if initial_capital is not None else 1.0
    if resolved_initial == 0.0:
        return None
    return float((equity / resolved_initial).max())
