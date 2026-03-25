from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_diagnostics import compute_signal_diagnostics


def _frame(signals: list[int], *, symbol: str = "AAPL") -> tuple[pd.Series, pd.DataFrame]:
    ts = pd.date_range("2025-01-01", periods=len(signals), freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": pd.Series([symbol] * len(signals), dtype="string"),
            "ts_utc": ts,
        }
    )
    return pd.Series(signals, index=df.index, dtype="int64", name="signal"), df


def test_compute_signal_diagnostics_for_mixed_signals() -> None:
    signals, df = _frame([0, 1, 1, 0, -1, -1, 0, 1])

    result = compute_signal_diagnostics(signals, df)

    assert result == {
        "total_rows": 8,
        "pct_long": 0.375,
        "pct_short": 0.25,
        "pct_flat": 0.375,
        "total_trades": 5,
        "turnover": 0.625,
        "avg_holding_period": 5 / 3,
        "exposure_pct": 0.625,
        "flags": {
            "always_flat": False,
            "always_long": False,
            "always_short": False,
            "no_trades": False,
            "high_turnover": True,
        },
    }


def test_compute_signal_diagnostics_flags_always_flat() -> None:
    signals, df = _frame([0, 0, 0, 0])

    result = compute_signal_diagnostics(signals, df)

    assert result["pct_flat"] == 1.0
    assert result["total_trades"] == 0
    assert result["avg_holding_period"] == 0.0
    assert result["flags"]["always_flat"] is True
    assert result["flags"]["no_trades"] is True


def test_compute_signal_diagnostics_flags_always_long() -> None:
    signals, df = _frame([1, 1, 1, 1])

    result = compute_signal_diagnostics(signals, df)

    assert result["pct_long"] == 1.0
    assert result["total_trades"] == 0
    assert result["flags"]["always_long"] is True
    assert result["flags"]["no_trades"] is True


def test_compute_signal_diagnostics_treats_no_change_as_no_trade() -> None:
    signals, df = _frame([-1, -1, -1])

    result = compute_signal_diagnostics(signals, df)

    assert result["total_trades"] == 0
    assert result["turnover"] == 0.0
    assert result["flags"]["no_trades"] is True


def test_compute_signal_diagnostics_flags_high_turnover() -> None:
    signals, df = _frame([1, -1, 1, -1, 1, -1])

    result = compute_signal_diagnostics(signals, df)

    assert result["total_trades"] == 5
    assert result["turnover"] == pytest.approx(5 / 6)
    assert result["avg_holding_period"] == 1.0
    assert result["flags"]["high_turnover"] is True


def test_compute_signal_diagnostics_computes_holding_periods_by_symbol_boundary() -> None:
    df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "MSFT", "MSFT", "AAPL", "AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-04T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    signals = pd.Series([1, 1, -1, 0, 0, 0], index=df.index, dtype="int64")

    result = compute_signal_diagnostics(signals, df)

    assert result["total_trades"] == 2
    assert result["avg_holding_period"] == pytest.approx(1.5)
