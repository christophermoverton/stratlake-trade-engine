from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha import (
    AlphaSignalMappingConfig,
    AlphaSignalMappingError,
    map_alpha_predictions_to_signals,
)


def _prediction_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "prediction_score": [0.9, 1.0, 0.1, 1.0, -0.2, -1.0],
        }
    )
    return frame.sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)


def test_rank_long_short_maps_cross_sections_deterministically() -> None:
    result = map_alpha_predictions_to_signals(
        _prediction_frame(),
        AlphaSignalMappingConfig(policy="rank_long_short"),
    )

    expected = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "prediction_score": [0.9, 1.0, 0.1, 1.0, -0.2, -1.0],
            "signal": [1.0, 1.0, 0.0, 0.0, -1.0, -1.0],
        }
    )

    expected = expected.astype(
        {
            "symbol": result.signals["symbol"].dtype,
            "timeframe": result.signals["timeframe"].dtype,
            "prediction_score": "float64",
            "signal": "float64",
        }
    )

    pdt.assert_frame_equal(result.signals.reset_index(drop=True), expected, check_dtype=True, check_exact=True)
    assert result.metadata["tie_breaker"] == "prediction_score then symbol ascending"


def test_zscore_continuous_returns_zero_when_cross_section_has_no_dispersion() -> None:
    frame = _prediction_frame()
    frame["prediction_score"] = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

    result = map_alpha_predictions_to_signals(
        frame,
        AlphaSignalMappingConfig(policy="zscore_continuous"),
    )

    assert result.signals["signal"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_top_bottom_quantile_uses_ceiling_selection_and_stable_symbol_ties() -> None:
    result = map_alpha_predictions_to_signals(
        _prediction_frame(),
        AlphaSignalMappingConfig(policy="top_bottom_quantile", quantile=0.34),
    )

    assert result.signals["signal"].tolist() == [1.0, 1.0, 0.0, 0.0, -1.0, -1.0]


def test_long_only_top_quantile_selects_at_least_one_name_per_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3, utc=True),
            "prediction_score": [0.4, 0.2, 0.1],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable")

    result = map_alpha_predictions_to_signals(
        frame,
        AlphaSignalMappingConfig(policy="long_only_top_quantile", quantile=0.2),
    )

    assert result.signals["signal"].tolist() == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (AlphaSignalMappingConfig(policy="top_bottom_quantile"), "requires a quantile"),
        (AlphaSignalMappingConfig(policy="rank_long_short", quantile=0.2), "does not accept a quantile"),
    ],
)
def test_signal_mapping_rejects_invalid_policy_parameters(
    config: AlphaSignalMappingConfig,
    message: str,
) -> None:
    with pytest.raises(AlphaSignalMappingError, match=message):
        map_alpha_predictions_to_signals(_prediction_frame(), config)


def test_signal_mapping_is_deterministic_across_repeated_calls() -> None:
    frame = _prediction_frame()
    config = AlphaSignalMappingConfig(policy="zscore_continuous")

    first = map_alpha_predictions_to_signals(frame, config)
    second = map_alpha_predictions_to_signals(frame, config)

    pdt.assert_frame_equal(first.signals, second.signals, check_dtype=True, check_exact=True)
