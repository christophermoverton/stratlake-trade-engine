from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import numpy as np

from .contracts import PortfolioContractError, validate_portfolio_output
from .metrics import compute_portfolio_metrics
from .validation import (
    summarize_weight_diagnostics,
    validate_portfolio_output_constraints,
    validate_portfolio_weights,
)

_DEFAULT_TOLERANCE = 1e-10
_QA_SUMMARY_FILENAME = "qa_summary.json"


class PortfolioQAError(ValueError):
    """Raised when portfolio QA checks detect inconsistent results."""


def validate_portfolio_return_consistency(
    portfolio_df: pd.DataFrame,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate weighted-return traceability for one portfolio output frame."""

    normalized = _normalize_portfolio_output(
        portfolio_df,
        owner="portfolio_df",
        validation_config=validation_config,
    )
    required_columns = _required_traceability_columns(normalized)
    if required_columns["return_columns"] == []:
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio output must include strategy_return__<strategy> columns."
        )
    if required_columns["weight_columns"] == []:
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio output must include weight__<strategy> columns."
        )

    _ensure_sorted_unique_timestamps(normalized)
    _ensure_no_nans(
        normalized,
        columns=[
            "ts_utc",
            "portfolio_return",
            *required_columns["return_columns"],
            *required_columns["weight_columns"],
        ],
    )

    weight_frame = _weight_frame(normalized)
    row_sums = weight_frame.sum(axis=1)
    invalid_weight_sums = (row_sums - 1.0).abs() > tolerance
    if invalid_weight_sums.any():
        failing_index = row_sums.index[invalid_weight_sums][0]
        raise PortfolioQAError(
            "Portfolio QA failed: weights must sum to 1.0 within tolerance "
            f"{tolerance}. First failing row index={failing_index}, row_sum={row_sums.iloc[failing_index]}."
        )

    recomputed_gross_returns = pd.Series(
        (_strategy_return_frame(normalized) * weight_frame).sum(axis=1).to_numpy(dtype="float64", copy=True),
        index=normalized.index,
        dtype="float64",
    )
    if "gross_portfolio_return" in normalized.columns:
        mismatched_gross_returns = (
            recomputed_gross_returns - normalized["gross_portfolio_return"]
        ).abs() > tolerance
        if mismatched_gross_returns.any():
            failing_index = recomputed_gross_returns.index[mismatched_gross_returns][0]
            raise PortfolioQAError(
                "Portfolio QA failed: gross_portfolio_return does not equal the weighted sum of component returns "
                f"at row index {failing_index}."
            )

    recomputed_returns = recomputed_gross_returns.copy()
    if "portfolio_execution_friction" in normalized.columns:
        transaction_cost = pd.Series(
            normalized["portfolio_transaction_cost"].to_numpy(dtype="float64", copy=True),
            index=normalized.index,
            dtype="float64",
        ) if "portfolio_transaction_cost" in normalized.columns else pd.Series(0.0, index=normalized.index, dtype="float64")
        slippage_cost = pd.Series(
            normalized["portfolio_slippage_cost"].to_numpy(dtype="float64", copy=True),
            index=normalized.index,
            dtype="float64",
        ) if "portfolio_slippage_cost" in normalized.columns else pd.Series(0.0, index=normalized.index, dtype="float64")
        friction_match = (
            normalized["portfolio_execution_friction"]
            - transaction_cost
            - slippage_cost
        ).abs() > tolerance
        if friction_match.any():
            failing_index = normalized.index[friction_match][0]
            raise PortfolioQAError(
                "Portfolio QA failed: portfolio_execution_friction must equal "
                "portfolio_transaction_cost + portfolio_slippage_cost "
                f"at row index {failing_index}."
            )
    if "portfolio_transaction_cost" in normalized.columns:
        recomputed_returns = recomputed_returns - pd.Series(
            normalized["portfolio_transaction_cost"].to_numpy(dtype="float64", copy=True),
            index=normalized.index,
            dtype="float64",
        )
    if "portfolio_slippage_cost" in normalized.columns:
        recomputed_returns = recomputed_returns - pd.Series(
            normalized["portfolio_slippage_cost"].to_numpy(dtype="float64", copy=True),
            index=normalized.index,
            dtype="float64",
        )

    mismatched_returns = (
        recomputed_returns
        - pd.Series(normalized["portfolio_return"].to_numpy(dtype="float64", copy=True), index=normalized.index, dtype="float64")
    ).abs() > tolerance
    if mismatched_returns.any():
        failing_index = recomputed_returns.index[mismatched_returns][0]
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio_return does not equal the weighted sum of component returns "
            "after execution frictions "
            f"at row index {failing_index}."
        )

    return normalized


def validate_equity_curve(
    portfolio_df: pd.DataFrame,
    *,
    initial_capital: float | None = None,
    allow_non_positive: bool = False,
    tolerance: float = _DEFAULT_TOLERANCE,
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate that the portfolio equity curve matches compounded returns."""

    normalized = validate_portfolio_return_consistency(
        portfolio_df,
        tolerance=tolerance,
        validation_config=validation_config,
    )
    if "portfolio_equity_curve" not in normalized.columns:
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio output must include 'portfolio_equity_curve'."
        )

    equity = pd.to_numeric(normalized["portfolio_equity_curve"], errors="coerce").astype("float64")
    if equity.isna().any():
        failing_index = int(equity.index[equity.isna()][0])
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio_equity_curve contains NaN values "
            f"at row index {failing_index}."
        )

    if not allow_non_positive and (equity <= 0.0).any():
        failing_index = int(equity.index[equity.le(0.0)][0])
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio_equity_curve must remain positive. "
            f"First non-positive value at row index {failing_index}."
        )

    resolved_initial_capital = _resolve_initial_capital(
        normalized,
        equity,
        initial_capital=initial_capital,
    )
    expected_equity = float(resolved_initial_capital) * (1.0 + normalized["portfolio_return"]).cumprod()
    mismatched_equity = (expected_equity - equity).abs() > tolerance
    if mismatched_equity.any():
        failing_index = expected_equity.index[mismatched_equity][0]
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio_equity_curve does not match compounded portfolio_return values "
            f"at row index {failing_index}."
        )

    return normalized


def validate_weights_behavior(
    portfolio_df: pd.DataFrame,
    *,
    allocator_name: str | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Validate weight presence, finiteness, and allocator-specific behavior."""

    normalized = _normalize_portfolio_output(
        portfolio_df,
        owner="portfolio_df",
        validation_config=validation_config,
    )
    weight_columns = [column for column in normalized.columns if column.startswith("weight__")]
    if not weight_columns:
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio output must include weight__<strategy> columns."
        )

    _ensure_no_nans(normalized, columns=weight_columns)
    weights = _weight_frame(normalized)
    if not np.isfinite(weights.to_numpy(dtype="float64")).all():
        raise PortfolioQAError("Portfolio QA failed: weights contain infinite values.")
    try:
        validate_portfolio_weights(weights, validation_config=validation_config)
    except ValueError as exc:
        raise PortfolioQAError(str(exc)) from exc

    normalized_allocator = None if allocator_name is None else str(allocator_name).strip().lower()
    if normalized_allocator == "equal_weight" and len(weights) > 1:
        deltas = weights.sub(weights.iloc[0], axis=1).abs()
        drift_mask = deltas > tolerance
        if drift_mask.any().any():
            first_position = drift_mask.stack()[lambda values: values].index[0]
            raise PortfolioQAError(
                "Portfolio QA failed: equal_weight allocator should produce constant weights. "
                f"First drift detected at row index {first_position[0]}, strategy {first_position[1]!r}."
            )

    return normalized


def validate_portfolio_artifact_consistency(
    output_dir: str | Path,
    *,
    portfolio_output: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    tolerance: float = _DEFAULT_TOLERANCE,
) -> None:
    """Validate that persisted portfolio artifacts match the in-memory portfolio run."""

    resolved_output_dir = Path(output_dir)
    normalized_output = validate_equity_curve(
        portfolio_output,
        initial_capital=_initial_capital_from_config(config),
        tolerance=tolerance,
        validation_config=config.get("validation") if isinstance(config, dict) else None,
    )
    validate_weights_behavior(
        normalized_output,
        allocator_name=config.get("allocator"),
        tolerance=tolerance,
        validation_config=config.get("validation") if isinstance(config, dict) else None,
    )

    returns_path = resolved_output_dir / "portfolio_returns.csv"
    equity_path = resolved_output_dir / "portfolio_equity_curve.csv"
    weights_path = resolved_output_dir / "weights.csv"
    metrics_path = resolved_output_dir / "metrics.json"

    if not returns_path.exists():
        raise PortfolioQAError("Portfolio QA failed: missing artifact 'portfolio_returns.csv'.")
    if not equity_path.exists():
        raise PortfolioQAError("Portfolio QA failed: missing artifact 'portfolio_equity_curve.csv'.")
    if not weights_path.exists():
        raise PortfolioQAError("Portfolio QA failed: missing artifact 'weights.csv'.")
    if not metrics_path.exists():
        raise PortfolioQAError("Portfolio QA failed: missing artifact 'metrics.json'.")

    expected_returns = _portfolio_returns_artifact_frame(normalized_output)
    expected_equity = _portfolio_equity_artifact_frame(normalized_output)
    expected_weights = _weights_artifact_frame(normalized_output)

    actual_returns = pd.read_csv(returns_path)
    actual_equity = pd.read_csv(equity_path)
    actual_weights = pd.read_csv(weights_path)

    _assert_frames_match(
        "portfolio_returns.csv",
        expected_returns,
        actual_returns,
        tolerance=tolerance,
    )
    _assert_frames_match(
        "portfolio_equity_curve.csv",
        expected_equity,
        actual_equity,
        tolerance=tolerance,
    )
    _assert_frames_match(
        "weights.csv",
        expected_weights,
        actual_weights,
        tolerance=tolerance,
    )

    with metrics_path.open("r", encoding="utf-8") as handle:
        actual_metrics = json.load(handle)
    if not isinstance(actual_metrics, dict):
        raise PortfolioQAError("Portfolio QA failed: metrics.json must contain a JSON object.")

    recomputed_metrics = compute_portfolio_metrics(
        normalized_output,
        timeframe=_timeframe_from_config(config),
        validation_config=config.get("validation") if isinstance(config, dict) else None,
    )
    _assert_metrics_match(
        expected=dict(metrics),
        recomputed=recomputed_metrics,
        actual=actual_metrics,
        tolerance=tolerance,
    )


def generate_portfolio_qa_summary(
    portfolio_df: pd.DataFrame,
    metrics: dict[str, Any],
    *,
    portfolio_name: str | None,
    allocator_name: str | None,
    timeframe: str | None,
    run_id: str | None = None,
    issues: list[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic QA summary for a portfolio output."""

    normalized = _normalize_portfolio_output(portfolio_df, owner="portfolio_df")
    weight_columns = [column for column in normalized.columns if column.startswith("weight__")]
    issue_list = sorted(str(issue) for issue in (issues or []))
    sanity = _portfolio_sanity_payload(normalized, metrics)
    diagnostics = (
        summarize_weight_diagnostics(_weight_frame(normalized))
        if weight_columns
        else summarize_weight_diagnostics(pd.DataFrame(dtype="float64"))
    )
    if "portfolio_equity_curve" in normalized.columns:
        equity_series = pd.to_numeric(normalized["portfolio_equity_curve"], errors="coerce")
        ending_equity = None if equity_series.empty or equity_series.isna().all() else float(equity_series.iloc[-1])
    else:
        ending_equity = None

    status = "fail" if issue_list else ("warn" if sanity["issue_count"] > 0 else "pass")
    return {
        "run_id": run_id,
        "portfolio_name": None if portfolio_name is None else str(portfolio_name),
        "allocator": None if allocator_name is None else str(allocator_name),
        "timeframe": None if timeframe is None else str(timeframe),
        "row_count": int(len(normalized)),
        "strategy_count": int(len(weight_columns)),
        "date_range": _date_range(normalized),
        "return_summary": {
            "min": _coerce_number(normalized["portfolio_return"].min() if not normalized.empty else None),
            "max": _coerce_number(normalized["portfolio_return"].max() if not normalized.empty else None),
            "mean": _coerce_number(normalized["portfolio_return"].mean() if not normalized.empty else None),
        },
        "ending_equity": ending_equity,
        "metrics": {
            "total_return": _coerce_number(metrics.get("total_return")),
            "sharpe_ratio": _coerce_number(metrics.get("sharpe_ratio")),
            "max_drawdown": _coerce_number(metrics.get("max_drawdown")),
            "turnover": _coerce_number(metrics.get("turnover")),
            "trade_count": _coerce_number(metrics.get("trade_count")),
            "total_execution_friction": _coerce_number(metrics.get("total_execution_friction")),
            "exposure_pct": _coerce_number(metrics.get("exposure_pct")),
        },
        "diagnostics": {
            "max_gross_exposure": _coerce_number(diagnostics.get("max_gross_exposure")),
            "min_net_exposure": _coerce_number(diagnostics.get("min_net_exposure")),
            "max_net_exposure": _coerce_number(diagnostics.get("max_net_exposure")),
            "max_leverage": _coerce_number(diagnostics.get("max_leverage")),
            "max_single_weight": _coerce_number(diagnostics.get("max_single_weight")),
            "max_weight_sum_deviation": _coerce_number(diagnostics.get("max_weight_sum_deviation")),
            "validation_issue_count": _coerce_number(metrics.get("validation_issue_count", len(issue_list))),
        },
        "sanity": sanity,
        "issues": issue_list,
        "validation_status": status,
    }


def run_portfolio_qa(
    portfolio_df: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    artifacts_dir: str | Path | None = None,
    run_id: str | None = None,
    strict: bool = True,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Execute deterministic portfolio QA checks and optionally persist a summary."""

    issues: list[str] = []
    normalized: pd.DataFrame | None = None
    try:
        normalized = validate_portfolio_return_consistency(
            portfolio_df,
            tolerance=tolerance,
            validation_config=config.get("validation") if isinstance(config, dict) else None,
        )
        validate_equity_curve(
            normalized,
            initial_capital=_initial_capital_from_config(config),
            tolerance=tolerance,
            validation_config=config.get("validation") if isinstance(config, dict) else None,
        )
        validate_weights_behavior(
            normalized,
            allocator_name=config.get("allocator"),
            tolerance=tolerance,
            validation_config=config.get("validation") if isinstance(config, dict) else None,
        )
    except PortfolioQAError as exc:
        issues.append(str(exc))

    if normalized is None:
        try:
            normalized = _normalize_portfolio_output(portfolio_df, owner="portfolio_df")
        except PortfolioQAError as exc:
            issues.append(str(exc))
            normalized = pd.DataFrame(columns=["ts_utc", "portfolio_return"])

    if artifacts_dir is not None and not issues:
        try:
            validate_portfolio_artifact_consistency(
                artifacts_dir,
                portfolio_output=normalized,
                metrics=metrics,
                config=config,
                tolerance=tolerance,
            )
        except PortfolioQAError as exc:
            issues.append(str(exc))

    summary = generate_portfolio_qa_summary(
        normalized,
        metrics,
        portfolio_name=config.get("portfolio_name"),
        allocator_name=config.get("allocator"),
        timeframe=config.get("timeframe"),
        run_id=run_id,
        issues=issues,
    )

    if artifacts_dir is not None:
        _write_qa_summary(Path(artifacts_dir) / _QA_SUMMARY_FILENAME, summary)

    if issues and strict:
        raise PortfolioQAError("; ".join(issues))
    if issues:
        for issue in issues:
            warnings.warn(issue, stacklevel=2)

    return summary


def _normalize_portfolio_output(
    portfolio_df: pd.DataFrame,
    *,
    owner: str,
    validation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    try:
        return validate_portfolio_output(portfolio_df)
    except PortfolioContractError as exc:
        raise PortfolioQAError(f"Portfolio QA failed: {owner} must be valid portfolio output: {exc}") from exc


def _required_traceability_columns(portfolio_df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "return_columns": [column for column in portfolio_df.columns if column.startswith("strategy_return__")],
        "weight_columns": [column for column in portfolio_df.columns if column.startswith("weight__")],
    }


def _ensure_sorted_unique_timestamps(portfolio_df: pd.DataFrame) -> None:
    timestamps = pd.to_datetime(portfolio_df["ts_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise PortfolioQAError("Portfolio QA failed: ts_utc contains unparsable timestamps.")
    if timestamps.duplicated().any():
        duplicate_ts = timestamps.loc[timestamps.duplicated(keep=False)].iloc[0]
        raise PortfolioQAError(
            "Portfolio QA failed: portfolio output timestamps must be unique. "
            f"First duplicate ts_utc={duplicate_ts}."
        )
    sorted_timestamps = timestamps.sort_values(kind="stable").reset_index(drop=True)
    if timestamps.reset_index(drop=True).tolist() != sorted_timestamps.tolist():
        raise PortfolioQAError("Portfolio QA failed: portfolio output timestamps must be sorted ascending.")


def _ensure_no_nans(portfolio_df: pd.DataFrame, *, columns: list[str]) -> None:
    for column in columns:
        if column == "ts_utc":
            series = pd.to_datetime(portfolio_df[column], utc=True, errors="coerce")
        else:
            series = pd.to_numeric(portfolio_df[column], errors="coerce")
        null_mask = pd.isna(series)
        if null_mask.any():
            failing_index = int(null_mask[null_mask].index[0])
            raise PortfolioQAError(
                f"Portfolio QA failed: required column {column!r} contains NaN values at row index {failing_index}."
            )


def _strategy_return_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    return_columns = [column for column in portfolio_df.columns if column.startswith("strategy_return__")]
    frame = portfolio_df.loc[:, return_columns].copy()
    frame.columns = [column.removeprefix("strategy_return__") for column in return_columns]
    frame.index = pd.DatetimeIndex(pd.to_datetime(portfolio_df["ts_utc"], utc=True), name="ts_utc")
    return frame.astype("float64")


def _weight_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    weight_columns = [column for column in portfolio_df.columns if column.startswith("weight__")]
    frame = portfolio_df.loc[:, weight_columns].copy()
    frame.columns = [column.removeprefix("weight__") for column in weight_columns]
    frame.index = pd.DatetimeIndex(pd.to_datetime(portfolio_df["ts_utc"], utc=True), name="ts_utc")
    return frame.astype("float64")


def _resolve_initial_capital(
    portfolio_df: pd.DataFrame,
    equity: pd.Series,
    *,
    initial_capital: float | None,
) -> float:
    if initial_capital is not None:
        return float(initial_capital)

    if portfolio_df.empty:
        raise PortfolioQAError("Portfolio QA failed: cannot infer initial capital from an empty portfolio output.")

    first_return = float(portfolio_df["portfolio_return"].iloc[0])
    first_equity = float(equity.iloc[0])
    denominator = 1.0 + first_return
    if abs(denominator) <= _DEFAULT_TOLERANCE:
        raise PortfolioQAError(
            "Portfolio QA failed: cannot infer initial capital when the first portfolio return equals -100%."
        )
    return first_equity / denominator


def _portfolio_returns_artifact_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    execution_columns = [
        column
        for column in (
            "gross_portfolio_return",
            "portfolio_weight_change",
            "portfolio_abs_weight_change",
            "portfolio_turnover",
            "portfolio_rebalance_event",
            "portfolio_transaction_cost",
            "portfolio_slippage_cost",
            "portfolio_execution_friction",
            "net_portfolio_return",
        )
        if column in portfolio_df.columns
    ]
    return _csv_ready(
        portfolio_df.loc[
            :,
            [
                "ts_utc",
                *sorted(column for column in portfolio_df.columns if column.startswith("strategy_return__")),
                *sorted(column for column in portfolio_df.columns if column.startswith("weight__")),
                *execution_columns,
                "portfolio_return",
            ],
        ].copy()
    )


def _portfolio_equity_artifact_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    return _csv_ready(portfolio_df.loc[:, ["ts_utc", "portfolio_equity_curve"]].copy())


def _weights_artifact_frame(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    return _csv_ready(
        portfolio_df.loc[
            :,
            [
                "ts_utc",
                *sorted(column for column in portfolio_df.columns if column.startswith("weight__")),
            ],
        ].copy()
    )


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="raise").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return normalized.sort_values("ts_utc", kind="stable").reset_index(drop=True)


def _assert_frames_match(
    artifact_name: str,
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    tolerance: float,
) -> None:
    if expected.columns.tolist() != actual.columns.tolist():
        raise PortfolioQAError(
            f"Portfolio QA failed: {artifact_name} columns do not match expected portfolio output."
        )
    if len(expected) != len(actual):
        raise PortfolioQAError(
            f"Portfolio QA failed: {artifact_name} row count mismatch "
            f"(expected={len(expected)} vs actual={len(actual)})."
        )

    for column in expected.columns:
        if column == "ts_utc":
            if expected[column].tolist() != actual[column].astype("string").tolist():
                raise PortfolioQAError(
                    f"Portfolio QA failed: {artifact_name} timestamp values do not match in-memory output."
                )
            continue

        left = pd.to_numeric(expected[column], errors="coerce").astype("float64")
        right = pd.to_numeric(actual[column], errors="coerce").astype("float64")
        mismatch = (left - right).abs() > tolerance
        if mismatch.any():
            failing_index = mismatch.index[mismatch][0]
            raise PortfolioQAError(
                f"Portfolio QA failed: {artifact_name} column {column!r} differs from in-memory output "
                f"at row index {failing_index}."
            )


def _assert_metrics_match(
    *,
    expected: dict[str, Any],
    recomputed: dict[str, Any],
    actual: dict[str, Any],
    tolerance: float,
) -> None:
    metric_keys = sorted(set(expected) | set(recomputed) | set(actual))
    for key in metric_keys:
        _assert_numeric_match(
            f"metrics.json field {key!r} vs provided metrics",
            expected.get(key),
            actual.get(key),
            tolerance=tolerance,
        )
        _assert_numeric_match(
            f"metrics.json field {key!r} vs recomputed metrics",
            recomputed.get(key),
            actual.get(key),
            tolerance=tolerance,
        )


def _assert_numeric_match(name: str, expected: Any, actual: Any, *, tolerance: float) -> None:
    left = _coerce_number(expected)
    right = _coerce_number(actual)
    if left is None and right is None:
        if expected == actual:
            return
        raise PortfolioQAError(f"Portfolio QA failed: {name} mismatch (expected={expected!r} vs actual={actual!r}).")
    if left is None or right is None:
        if expected == actual:
            return
        raise PortfolioQAError(f"Portfolio QA failed: {name} mismatch (expected={expected!r} vs actual={actual!r}).")
    if abs(left - right) > tolerance:
        raise PortfolioQAError(f"Portfolio QA failed: {name} mismatch (expected={left} vs actual={right}).")


def _initial_capital_from_config(config: dict[str, Any]) -> float:
    value = config.get("initial_capital", 1.0)
    return float(value)


def _timeframe_from_config(config: dict[str, Any]) -> str:
    value = config.get("timeframe")
    if not isinstance(value, str) or not value.strip():
        raise PortfolioQAError("Portfolio QA failed: config must include a non-empty timeframe for metric validation.")
    return value


def _date_range(portfolio_df: pd.DataFrame) -> list[str | None]:
    if portfolio_df.empty:
        return [None, None]
    timestamps = pd.to_datetime(portfolio_df["ts_utc"], utc=True, errors="coerce")
    if timestamps.isna().all():
        return [None, None]
    return [
        timestamps.min().strftime("%Y-%m-%dT%H:%M:%SZ"),
        timestamps.max().strftime("%Y-%m-%dT%H:%M:%SZ"),
    ]


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_qa_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _portfolio_sanity_payload(portfolio_df: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = portfolio_df.attrs.get("sanity_check")
    if not isinstance(payload, dict):
        payload = metrics.get("sanity")
    if not isinstance(payload, dict):
        return {
            "status": "pass",
            "issue_count": 0,
            "warning_count": 0,
            "strict_sanity_checks": False,
            "issues": [],
        }

    issues = payload.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    return {
        "status": str(payload.get("status", "pass")),
        "issue_count": int(payload.get("issue_count", len(issues))),
        "warning_count": int(payload.get("warning_count", 0)),
        "strict_sanity_checks": bool(payload.get("strict_sanity_checks", False)),
        "issues": [dict(issue) for issue in issues if isinstance(issue, dict)],
    }


__all__ = [
    "PortfolioQAError",
    "generate_portfolio_qa_summary",
    "run_portfolio_qa",
    "validate_equity_curve",
    "validate_portfolio_artifact_consistency",
    "validate_portfolio_return_consistency",
    "validate_weights_behavior",
]
