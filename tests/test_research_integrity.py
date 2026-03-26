from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.integrity import validate_research_integrity


def _research_frame(*, include_warmup_features: bool = False, warmup_nan: bool = True) -> pd.DataFrame:
    ts_utc = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "AAPL", "AAPL"], dtype="string"),
            "ts_utc": ts_utc,
            "feature_ret_1d": [0.01, -0.02, 0.03, -0.01],
            "feature_alpha": [0.1, -0.2, 0.3, 0.4],
        }
    )
    if include_warmup_features:
        frame["fast_sma"] = [101.0, 102.0, 103.0, 104.0]
        frame["slow_sma"] = [100.0, 101.0, 102.0, 103.0]
        if warmup_nan:
            frame.loc[0, ["fast_sma", "slow_sma"]] = pd.NA
    return frame


def test_validate_research_integrity_accepts_valid_dataset() -> None:
    df = _research_frame()
    signals = pd.Series([1, 0, -1, 0], index=df.index, dtype="int64")
    positions = signals.shift(1).fillna(0.0)

    validate_research_integrity(df, signals, positions=positions)


def test_validate_research_integrity_rejects_unsorted_timestamps() -> None:
    df = _research_frame().iloc[[0, 2, 1, 3]].reset_index(drop=True)
    signals = pd.Series([1, 0, -1, 0], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="must be sorted by \\(symbol, ts_utc\\)"):
        validate_research_integrity(df, signals)


def test_validate_research_integrity_rejects_misaligned_signals() -> None:
    df = _research_frame()
    signals = pd.Series([1, 0, -1, 0], index=pd.RangeIndex(start=10, stop=14), dtype="int64")

    with pytest.raises(ValueError, match="aligned exactly with the input DataFrame index"):
        validate_research_integrity(df, signals)


def test_validate_research_integrity_rejects_duplicate_keys() -> None:
    df = pd.concat([_research_frame(), _research_frame().iloc[[1]]], ignore_index=True)
    signals = pd.Series([1, 0, -1, 0, 1], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="duplicate \\(symbol, ts_utc\\)"):
        validate_research_integrity(df, signals)


def test_validate_research_integrity_rejects_invalid_signal_values() -> None:
    df = _research_frame()
    signals = pd.Series([1, 2, -1, 0], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="must only contain the values -1, 0, and 1"):
        validate_research_integrity(df, signals)


def test_validate_research_integrity_rejects_same_bar_execution() -> None:
    df = _research_frame()
    signals = pd.Series([1, 0, -1, 0], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="same_bar_execution"):
        validate_research_integrity(df, signals, positions=signals.astype("float64"))


def test_validate_research_integrity_rejects_forward_filled_warmup_rows() -> None:
    df = _research_frame(include_warmup_features=True, warmup_nan=False)
    signals = pd.Series([1, 0, -1, 0], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="forward_filled_warmup"):
        validate_research_integrity(df, signals)


def test_validate_research_integrity_rejects_future_feature_timestamp() -> None:
    df = _research_frame()
    df["feature_source_ts_utc"] = [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-04T00:00:00Z"),
        pd.Timestamp("2025-01-04T00:00:00Z"),
    ]
    signals = pd.Series([1, 0, -1, 0], index=df.index, dtype="int64")

    with pytest.raises(ValueError, match="future_feature_timestamp"):
        validate_research_integrity(df, signals)
