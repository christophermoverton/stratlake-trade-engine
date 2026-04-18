from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.execution import ExecutionConfig
from src.research.backtest_runner import run_backtest
from src.research.position_constructors import (
    load_position_constructor_registry,
    resolve_constructor,
)
from src.research.signal_semantics import (
    SignalSemanticsError,
    create_signal,
    ensure_signal_type_compatible,
)


def test_position_constructor_registry_loads_expected_definitions() -> None:
    registry = load_position_constructor_registry()

    assert {
        "identity_weights",
        "top_bottom_equal_weight",
        "rank_dollar_neutral",
        "softmax_long_only",
        "zscore_clip_scale",
    }.issubset(set(registry))
    assert registry["rank_dollar_neutral"].inputs == ("cross_section_rank",)


def test_resolve_constructor_rejects_missing_required_params() -> None:
    with pytest.raises(ValueError, match="requires parameters"):
        resolve_constructor("rank_dollar_neutral", {})


def test_rank_dollar_neutral_constructs_deterministic_net_neutral_positions() -> None:
    signal = create_signal(
        pd.DataFrame(
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
                "signal": [1.0, 0.0, -1.0, 1.0, 0.0, -1.0],
            }
        ).sort_values(["symbol", "ts_utc"], kind="stable"),
        signal_type="cross_section_rank",
        metadata={
            "constructor_id": "rank_dollar_neutral",
            "constructor_params": {"gross_long": 0.5, "gross_short": 0.5},
        },
    )

    constructor = resolve_constructor("rank_dollar_neutral", {"gross_long": 0.5, "gross_short": 0.5})
    positions = constructor.construct(signal)

    assert positions["position"].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0, -0.5, -0.5])
    grouped = positions.groupby("ts_utc", sort=False)["position"].sum()
    assert grouped.tolist() == pytest.approx([0.0, 0.0])


def test_top_bottom_equal_weight_handles_empty_selected_sets() -> None:
    signal = create_signal(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
                "signal": [0.0, 0.0],
            }
        ).sort_values(["symbol", "ts_utc"], kind="stable"),
        signal_type="ternary_quantile",
    )
    constructor = resolve_constructor("top_bottom_equal_weight", {"gross_long": 0.5, "gross_short": 0.5})
    positions = constructor.construct(signal)

    assert positions["position"].tolist() == pytest.approx([0.0, 0.0])


def test_softmax_long_only_equal_scores_split_gross_exposure_evenly() -> None:
    signal = create_signal(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3, utc=True),
                "prediction_score": [1.0, 1.0, 1.0],
            }
        ).sort_values(["symbol", "ts_utc"], kind="stable"),
        signal_type="prediction_score",
        value_column="prediction_score",
        parameters={"min_cross_section_size": 2},
    )
    constructor = resolve_constructor("softmax_long_only", {"gross_exposure": 1.0, "temperature": 1.0})
    positions = constructor.construct(signal)

    assert positions["position"].tolist() == pytest.approx([1.0 / 3.0] * 3)


def test_zscore_clip_scale_clips_and_normalizes_gross_exposure() -> None:
    signal = create_signal(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z"] * 3, utc=True),
                "signal": [4.0, 1.0, -1.0],
            }
        ).sort_values(["symbol", "ts_utc"], kind="stable"),
        signal_type="signed_zscore",
    )
    constructor = resolve_constructor("zscore_clip_scale", {"clip": 2.0, "gross_exposure": 1.0})
    positions = constructor.construct(signal)

    assert positions["position"].tolist() == pytest.approx([0.5, 0.25, -0.25])
    assert float(positions["position"].abs().sum()) == pytest.approx(1.0)


def test_compatibility_enforcement_rejects_invalid_constructor_for_signal_type() -> None:
    with pytest.raises(SignalSemanticsError, match="not compatible"):
        ensure_signal_type_compatible("cross_section_rank", position_constructor="top_bottom_equal_weight")


def test_managed_backtest_runner_constructs_positions_from_metadata() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "signal": [1.0, -1.0, 1.0, -1.0],
            "feature_ret_1d": [0.01, -0.01, 0.02, -0.02],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable")
    managed = create_signal(
        frame,
        signal_type="cross_section_rank",
        metadata={
            "constructor_id": "rank_dollar_neutral",
            "constructor_params": {"gross_long": 0.5, "gross_short": 0.5},
        },
    ).data

    result = run_backtest(
        managed,
        ExecutionConfig(enabled=False, execution_delay=1, transaction_cost_bps=0.0, slippage_bps=0.0),
    )

    assert result["constructed_position"].tolist() == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert result["executed_signal"].tolist() == pytest.approx([0.0, 0.5, 0.0, -0.5])
    assert result.attrs["backtest_signal_semantics"]["constructor_id"] == "rank_dollar_neutral"


def test_managed_backtest_runner_requires_explicit_constructor_metadata() -> None:
    managed = create_signal(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], utc=True),
                "signal": [1.0, -1.0],
                "feature_ret_1d": [0.01, -0.01],
            }
        ).sort_values(["symbol", "ts_utc"], kind="stable"),
        signal_type="cross_section_rank",
    ).data

    with pytest.raises(ValueError, match="position constructor metadata"):
        run_backtest(managed)
