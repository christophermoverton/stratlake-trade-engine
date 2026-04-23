from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.regimes import (
    REGIME_DIMENSIONS,
    REGIME_TAXONOMY,
    RegimeClassificationConfig,
    RegimeClassificationError,
    RegimeValidationError,
    classify_market_regimes,
    taxonomy_payload,
    validate_regime_labels,
)


def _single_market_return_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC"),
            "market_return": [0.00, 0.01, 0.01, -0.04, -0.03, 0.02, 0.03, 0.00],
        }
    )


def _small_config() -> RegimeClassificationConfig:
    return RegimeClassificationConfig(
        return_column="market_return",
        volatility_window=3,
        trend_window=3,
        trend_return_threshold=0.02,
        drawdown_threshold=0.05,
        near_peak_drawdown_threshold=0.0,
        stress_window=3,
    )


def test_taxonomy_payload_exposes_canonical_dimensions_and_labels() -> None:
    payload = taxonomy_payload()

    assert payload["version"] == "regime_taxonomy_v1"
    assert list(payload["dimensions"]) == list(REGIME_DIMENSIONS)
    assert REGIME_TAXONOMY["volatility"].labels == (
        "undefined",
        "low_volatility",
        "normal_volatility",
        "high_volatility",
    )


def test_classify_market_regimes_returns_stable_schema_and_metadata() -> None:
    result = classify_market_regimes(_single_market_return_frame(), config=_small_config())

    assert list(result.labels.columns) == [
        "ts_utc",
        "volatility_state",
        "trend_state",
        "drawdown_recovery_state",
        "stress_state",
        "regime_label",
        "is_defined",
        "volatility_metric",
        "trend_metric",
        "drawdown_metric",
        "stress_correlation_metric",
        "stress_dispersion_metric",
    ]
    assert result.labels["ts_utc"].is_monotonic_increasing
    assert result.labels.loc[0, "volatility_state"] == "undefined"
    assert result.labels.loc[0, "trend_state"] == "undefined"
    assert result.labels["stress_state"].eq("undefined").all()
    assert result.metadata["taxonomy_version"] == "regime_taxonomy_v1"
    assert result.metadata["input"]["symbol_count"] == 1


def test_classification_is_deterministic_across_reruns() -> None:
    frame = _single_market_return_frame()
    config = _small_config()

    first = classify_market_regimes(frame, config=config)
    second = classify_market_regimes(frame, config=config)

    pd.testing.assert_frame_equal(first.labels, second.labels, check_exact=True)
    assert first.metadata == second.metadata


def test_trend_threshold_boundaries_are_inclusive() -> None:
    frame = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC"),
            "market_return": [0.0, 0.0, 0.01, 0.0, 0.0, -0.01],
        }
    )
    config = RegimeClassificationConfig(
        return_column="market_return",
        volatility_window=2,
        trend_window=2,
        trend_return_threshold=0.01,
        stress_window=2,
    )

    labels = classify_market_regimes(frame, config=config).labels

    assert labels.loc[2, "trend_state"] == "uptrend"
    assert labels.loc[5, "trend_state"] == "downtrend"


def test_multi_symbol_input_flags_correlation_stress() -> None:
    timestamps = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
    rows = []
    for ts_utc, ret in zip(timestamps, [0.01, 0.02, 0.03, 0.04, 0.05], strict=True):
        rows.append({"ts_utc": ts_utc, "symbol": "AAA", "asset_return": ret})
        rows.append({"ts_utc": ts_utc, "symbol": "BBB", "asset_return": ret * 2.0})
    frame = pd.DataFrame(rows)
    config = RegimeClassificationConfig(
        return_column="asset_return",
        volatility_window=3,
        trend_window=3,
        stress_window=3,
        stress_correlation_threshold=0.99,
    )

    labels = classify_market_regimes(frame, config=config).labels

    assert labels.loc[0, "stress_state"] == "undefined"
    assert labels.loc[2:, "stress_state"].eq("correlation_stress").all()


def test_missing_returns_create_partial_undefined_states() -> None:
    frame = _single_market_return_frame()
    frame.loc[3, "market_return"] = float("nan")

    labels = classify_market_regimes(frame, config=_small_config()).labels

    assert labels.loc[3, "drawdown_recovery_state"] == "undefined"
    assert labels["is_defined"].eq(False).all()


def test_flat_low_variance_segments_stay_classifiable_after_warmup() -> None:
    frame = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "market_return": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    config = RegimeClassificationConfig(
        return_column="market_return",
        volatility_window=3,
        trend_window=3,
        stress_window=3,
    )

    labels = classify_market_regimes(frame, config=config).labels

    assert labels.loc[2:, "volatility_state"].eq("low_volatility").all()
    assert labels.loc[2:, "trend_state"].eq("sideways").all()
    assert labels.loc[2:, "drawdown_recovery_state"].eq("near_peak").all()


def test_validate_regime_labels_rejects_unsorted_output() -> None:
    labels = classify_market_regimes(_single_market_return_frame(), config=_small_config()).labels
    unsorted = labels.iloc[[1, 0, *range(2, len(labels))]].reset_index(drop=True)

    with pytest.raises(RegimeValidationError, match="sorted by ts_utc"):
        validate_regime_labels(unsorted)


def test_validate_regime_labels_rejects_bad_composite_label() -> None:
    labels = classify_market_regimes(_single_market_return_frame(), config=_small_config()).labels
    labels.loc[0, "regime_label"] = "bad"

    with pytest.raises(RegimeValidationError, match="composite labels"):
        validate_regime_labels(labels)


def test_classification_rejects_duplicate_symbol_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "ts_utc": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["AAA", "AAA"],
            "asset_return": [0.01, 0.02],
        }
    )

    with pytest.raises(RegimeClassificationError, match="duplicate"):
        classify_market_regimes(frame, config={"return_column": "asset_return"})
