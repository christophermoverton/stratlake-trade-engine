from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.signal_semantics import (
    SignalSemanticsError,
    create_signal,
    ensure_signal_type_compatible,
    extract_signal_metadata,
    load_signal_type_registry,
    percentile_to_quantile_bucket,
    rank_to_percentile,
    score_to_cross_section_rank,
    score_to_signed_zscore,
    validate_signal_frame,
)


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "prediction_score": [1.5, 0.2, -0.8, 0.9, 0.1, -1.2],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)


def test_load_signal_type_registry_exposes_canonical_definitions() -> None:
    registry = load_signal_type_registry()

    assert {
        "prediction_score",
        "cross_section_rank",
        "cross_section_percentile",
        "signed_zscore",
        "ternary_quantile",
        "binary_signal",
        "spread_zscore",
        "target_weight",
    }.issubset(set(registry))
    assert registry["prediction_score"].executable is False
    assert registry["signed_zscore"].executable is True


def test_validate_signal_frame_rejects_duplicate_keys() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
            "signal": [1.0, 0.0],
        }
    )

    with pytest.raises(SignalSemanticsError, match="duplicate"):
        validate_signal_frame(frame, signal_type="ternary_quantile")


def test_create_signal_attaches_traceable_metadata() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
            "signal": [1.0, 0.0],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable")

    signal = create_signal(
        frame,
        signal_type="binary_signal",
        parameters={"quantile": 0.5},
        source={"layer": "strategy", "component": "unit_test"},
    )

    payload = extract_signal_metadata(signal.data)
    assert payload is not None
    assert payload["signal_type"] == "binary_signal"
    assert payload["parameters"] == {"quantile": 0.5}
    assert payload["source"]["layer"] == "strategy"


def test_transformations_are_deterministic_and_preserve_metadata() -> None:
    prediction_signal = create_signal(
        _prediction_frame(),
        signal_type="prediction_score",
        value_column="prediction_score",
        parameters={"min_cross_section_size": 2},
        source={"layer": "alpha", "component": "unit_test"},
    )

    first_rank = score_to_cross_section_rank(prediction_signal)
    second_rank = score_to_cross_section_rank(prediction_signal)
    pdt.assert_frame_equal(first_rank.data, second_rank.data, check_dtype=True, check_exact=True)

    percentile_signal = rank_to_percentile(first_rank)
    bucket_signal = percentile_to_quantile_bucket(percentile_signal, quantile=0.34)
    zscore_signal = score_to_signed_zscore(prediction_signal, clip=2.0)

    assert first_rank.signal_type == "cross_section_rank"
    assert percentile_signal.signal_type == "cross_section_percentile"
    assert bucket_signal.signal_type == "ternary_quantile"
    assert zscore_signal.signal_type == "signed_zscore"
    assert extract_signal_metadata(bucket_signal.data)["transformation_history"][-1]["operation"] == "percentile_to_quantile_bucket"
    assert extract_signal_metadata(zscore_signal.data)["clipping"] == {"clip": 2.0}


def test_ensure_signal_type_compatible_rejects_non_executable_signal() -> None:
    with pytest.raises(SignalSemanticsError, match="not compatible"):
        ensure_signal_type_compatible(
            "prediction_score",
            position_constructor="backtest_numeric_exposure",
        )
