from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.signal_diagnostics import compute_signal_diagnostics
from src.research.input_validation import STRATEGY_INPUT_MIN_ROWS

LOW_DATA_THRESHOLD = STRATEGY_INPUT_MIN_ROWS


def generate_strategy_qa_summary(
    df: pd.DataFrame,
    signals: pd.Series,
    diagnostics: dict,
    metrics: dict,
    *,
    strategy_name: str,
    run_id: str,
) -> dict[str, Any]:
    """Build a deterministic QA summary for one completed strategy run."""

    normalized_df = df.copy()
    signal_diagnostics = _resolved_diagnostics(normalized_df, signals, diagnostics)
    date_range = _resolve_date_range(normalized_df)
    metrics_payload = {
        "total_return": _coerce_number(metrics.get("total_return")),
        "sharpe": _coerce_number(metrics.get("sharpe_ratio")),
        "max_drawdown": _coerce_number(metrics.get("max_drawdown")),
    }
    execution = _execution_payload(normalized_df)

    row_count = int(len(normalized_df))
    input_validation = _input_validation_payload(normalized_df)
    flags = {
        "no_data": row_count == 0,
        "degenerate_signal": bool(
            signal_diagnostics["flags"]["always_flat"]
            or signal_diagnostics["flags"]["always_long"]
            or signal_diagnostics["flags"]["always_short"]
        ),
        "no_trades": int(signal_diagnostics["total_trades"]) == 0,
        "high_turnover": bool(signal_diagnostics["flags"]["high_turnover"]),
        "low_data": bool(input_validation.get("low_data", row_count < LOW_DATA_THRESHOLD)),
    }

    integrity_failure = _integrity_failure(normalized_df, signal_diagnostics)
    if flags["no_data"] or integrity_failure or not execution["valid_returns"] or not execution["equity_curve_present"]:
        overall_status = "fail"
    elif any(flags.values()):
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "run_id": run_id,
        "strategy_name": strategy_name,
        "dataset": _resolve_dataset(normalized_df),
        "timeframe": _resolve_timeframe(normalized_df),
        "row_count": row_count,
        "symbols_present": _symbols_present(normalized_df),
        "date_range": date_range,
        "input_validation": input_validation,
        "signal": {
            "pct_long": _coerce_number(signal_diagnostics.get("pct_long")),
            "pct_short": _coerce_number(signal_diagnostics.get("pct_short")),
            "pct_flat": _coerce_number(signal_diagnostics.get("pct_flat")),
            "turnover": _coerce_number(signal_diagnostics.get("turnover")),
            "total_trades": int(signal_diagnostics.get("total_trades", 0)),
        },
        "execution": execution,
        "metrics": metrics_payload,
        "flags": flags,
        "overall_status": overall_status,
    }


def _resolved_diagnostics(df: pd.DataFrame, signals: pd.Series, diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(diagnostics, dict) and diagnostics:
        return diagnostics
    return compute_signal_diagnostics(signals, df)


def _resolve_dataset(df: pd.DataFrame) -> str | None:
    dataset = df.attrs.get("dataset")
    if dataset is None and "dataset" in df.columns:
        dataset = _first_non_empty(df["dataset"])
    return None if dataset is None else str(dataset)


def _resolve_timeframe(df: pd.DataFrame) -> str | None:
    timeframe = df.attrs.get("timeframe")
    if timeframe is None and "timeframe" in df.columns:
        timeframe = _first_non_empty(df["timeframe"])
    if timeframe is None:
        return None
    return str(timeframe)


def _symbols_present(df: pd.DataFrame) -> int:
    if "symbol" not in df.columns:
        return 0
    return int(df["symbol"].dropna().astype("string").nunique())


def _resolve_date_range(df: pd.DataFrame) -> list[str | None]:
    if "ts_utc" in df.columns:
        timestamps = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dropna()
        if not timestamps.empty:
            return [_format_datetime(timestamps.min()), _format_datetime(timestamps.max())]

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
        if not dates.empty:
            return [_format_date(dates.min()), _format_date(dates.max())]

    return [None, None]


def _execution_payload(df: pd.DataFrame) -> dict[str, bool]:
    valid_returns = False
    if "strategy_return" in df.columns:
        valid_returns = bool(pd.to_numeric(df["strategy_return"], errors="coerce").notna().any())

    equity_curve_present = False
    if "equity_curve" in df.columns:
        equity_curve_present = bool(pd.to_numeric(df["equity_curve"], errors="coerce").notna().any())

    return {
        "valid_returns": valid_returns,
        "equity_curve_present": equity_curve_present,
    }


def _input_validation_payload(df: pd.DataFrame) -> dict[str, Any]:
    payload = df.attrs.get("input_validation")
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


def _integrity_failure(df: pd.DataFrame, diagnostics: dict[str, Any]) -> bool:
    if bool(df.attrs.get("integrity_failure", False)):
        return True
    return bool(diagnostics.get("integrity_failure", False))


def _first_non_empty(series: pd.Series) -> str | None:
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return None


def _format_datetime(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_date(value: pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d")


def _coerce_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
