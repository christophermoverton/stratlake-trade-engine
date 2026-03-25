from __future__ import annotations

import math
from typing import Any

import pandas as pd

_EPSILON = 1e-9
_HIGH_TURNOVER_THRESHOLD = 0.5


def compute_signal_diagnostics(signals: pd.Series, df: pd.DataFrame) -> dict[str, Any]:
    """Compute deterministic, symbol-aware diagnostics for a standardized signal series."""

    normalized_signals = _normalize_signals(signals, df.index)
    ordered_frame = _ordered_signal_frame(df, normalized_signals)
    signal_series = ordered_frame["signal"].astype("float64")

    total_rows = int(len(signal_series))
    pct_long = _fraction(signal_series.eq(1.0))
    pct_short = _fraction(signal_series.eq(-1.0))
    pct_flat = _fraction(signal_series.eq(0.0))
    total_trades = int(_count_signal_changes(ordered_frame))
    turnover = float(total_trades / total_rows) if total_rows else 0.0
    holding_periods = _holding_periods(ordered_frame)
    avg_holding_period = float(sum(holding_periods) / len(holding_periods)) if holding_periods else 0.0
    exposure_pct = _fraction(signal_series.ne(0.0))

    flags = {
        "always_flat": math.isclose(pct_flat, 1.0, rel_tol=0.0, abs_tol=_EPSILON),
        "always_long": math.isclose(pct_long, 1.0, rel_tol=0.0, abs_tol=_EPSILON),
        "always_short": math.isclose(pct_short, 1.0, rel_tol=0.0, abs_tol=_EPSILON),
        "no_trades": total_trades == 0,
        "high_turnover": turnover > _HIGH_TURNOVER_THRESHOLD,
    }

    return {
        "total_rows": total_rows,
        "pct_long": pct_long,
        "pct_short": pct_short,
        "pct_flat": pct_flat,
        "total_trades": total_trades,
        "turnover": turnover,
        "avg_holding_period": avg_holding_period,
        "exposure_pct": exposure_pct,
        "flags": flags,
    }


def _normalize_signals(signals: pd.Series, index: pd.Index) -> pd.Series:
    aligned = signals.reindex(index)
    normalized = pd.to_numeric(aligned, errors="coerce").fillna(0.0).astype("float64")
    normalized.name = "signal"
    return normalized


def _ordered_signal_frame(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    frame = df.copy()
    frame["signal"] = signals
    frame["_row_order"] = range(len(frame))

    sort_columns = ["_row_order"]
    if "symbol" in frame.columns:
        sort_columns.insert(0, "symbol")

    if "ts_utc" in frame.columns:
        frame["_sort_ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
        sort_columns.insert(-1, "_sort_ts_utc")
    elif "date" in frame.columns:
        frame["_sort_date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        sort_columns.insert(-1, "_sort_date")

    ordered = frame.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)
    return ordered


def _fraction(mask: pd.Series) -> float:
    if mask.empty:
        return 0.0
    return float(mask.mean())


def _count_signal_changes(frame: pd.DataFrame) -> int:
    total_changes = 0
    for _, group in _iter_symbol_groups(frame):
        signals = group["signal"].astype("float64")
        previous = signals.shift(1)
        changes = signals.ne(previous) & previous.notna()
        total_changes += int(changes.sum())
    return total_changes


def _holding_periods(frame: pd.DataFrame) -> list[int]:
    runs: list[int] = []
    for _, group in _iter_symbol_groups(frame):
        current_signal = 0.0
        current_length = 0
        for value in group["signal"].astype("float64").tolist():
            if value == 0.0:
                if current_length:
                    runs.append(current_length)
                    current_length = 0
                current_signal = 0.0
                continue

            if value == current_signal:
                current_length += 1
                continue

            if current_length:
                runs.append(current_length)
            current_signal = value
            current_length = 1

        if current_length:
            runs.append(current_length)
    return runs


def _iter_symbol_groups(frame: pd.DataFrame):
    if "symbol" not in frame.columns:
        yield None, frame
        return

    grouped = frame.groupby("symbol", sort=False, dropna=False)
    yield from grouped
