from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha import AlphaSignalMappingConfig, map_alpha_predictions_to_signals
from src.research.alpha_eval import evaluate_alpha_predictions, generate_alpha_qa_summary


def _aligned_frame(*, periods: int = 4) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        pd.date_range("2025-01-01", periods=periods, freq="D", tz="UTC"),
        utc=True,
    )
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for ts_index, ts_utc in enumerate(timestamps):
            rows.append(
                {
                    "symbol": symbol,
                    "ts_utc": ts_utc,
                    "timeframe": "1D",
                    "prediction_score": float(symbol_index) + ts_index * 0.1,
                    "forward_return": (float(symbol_index) - 1.5) * 0.01 + ts_index * 0.001,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)


def test_generate_alpha_qa_summary_passes_for_clean_balanced_signals() -> None:
    aligned = _aligned_frame(periods=12)
    result = evaluate_alpha_predictions(aligned, min_cross_section_size=3)
    signal_mapping = map_alpha_predictions_to_signals(
        aligned.loc[:, ["symbol", "ts_utc", "timeframe", "prediction_score"]],
        AlphaSignalMappingConfig(policy="rank_long_short"),
    )

    summary = generate_alpha_qa_summary(
        aligned,
        result,
        alpha_name="demo_alpha",
        run_id="alpha_run_001",
        signal_mapping_result=signal_mapping,
    )

    assert summary["overall_status"] == "pass"
    assert summary["forecast"]["valid_timestamps"] == 12
    assert summary["nulls"]["prediction_null_rate"] == 0.0
    assert summary["signals"]["enabled"] is True
    assert summary["signals"]["max_single_name_abs_share"] == 0.375
    assert summary["checks"]["sleeve_turnover"]["status"] == "pass"
    assert summary["checks"]["concentration"]["status"] == "pass"
    assert summary["checks"]["net_exposure_sanity"]["status"] == "pass"


def test_generate_alpha_qa_summary_flags_sparse_null_and_concentrated_runs() -> None:
    aligned = _aligned_frame().copy()
    aligned.loc[
        aligned["symbol"].isin(["BBB", "CCC"]) & aligned["ts_utc"].gt(pd.Timestamp("2025-01-01T00:00:00Z")),
        "forward_return",
    ] = None
    result = evaluate_alpha_predictions(aligned, min_cross_section_size=2)
    concentrated_signals = pd.DataFrame(
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
            "timeframe": ["1D", "1D", "1D", "1D"],
            "prediction_score": [5.0, 0.1, 5.0, 0.1],
            "signal": [1.0, 0.0, -1.0, 0.0],
        }
    ).sort_values(["symbol", "ts_utc"], kind="stable").reset_index(drop=True)
    signal_mapping = map_alpha_predictions_to_signals(
        concentrated_signals.loc[:, ["symbol", "ts_utc", "timeframe", "prediction_score"]],
        AlphaSignalMappingConfig(policy="long_only_top_quantile", quantile=0.5),
    )

    summary = generate_alpha_qa_summary(
        aligned,
        result,
        alpha_name="fragile_alpha",
        run_id="alpha_run_002",
        signal_mapping_result=signal_mapping,
    )

    assert summary["overall_status"] == "fail"
    assert summary["checks"]["minimum_valid_timestamps"]["status"] == "warn"
    assert summary["checks"]["valid_timestamp_rate"]["status"] == "pass"
    assert summary["checks"]["post_warmup_null_limit"]["status"] == "fail"
    assert summary["checks"]["concentration"]["status"] == "fail"
    assert summary["checks"]["net_exposure_sanity"]["status"] == "fail"
