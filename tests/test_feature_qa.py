from __future__ import annotations

import pandas as pd

from src.data.feature_qa import build_feature_qa_summaries, write_feature_qa_artifacts


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL", "AAPL", "MSFT", "MSFT"], dtype="string"),
            "ts_utc": pd.to_datetime(
                [
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T14:31:00Z",
                    "2025-01-02T14:32:00Z",
                ],
                utc=True,
            ),
            "timeframe": pd.Series(["1Min", "1Min", "1Min", "1Min"], dtype="string"),
            "date": pd.Series(["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02"], dtype="string"),
            "feature_alpha": [1.0, 1.0, None, float("inf")],
            "feature_beta": [10.0, 10.0, 11.0, 12.0],
        }
    )


def test_build_feature_qa_summaries_reports_duplicates_nulls_and_inf() -> None:
    by_symbol, global_summary = build_feature_qa_summaries(
        _features_df(),
        timeframe="1Min",
        expected_symbols=["AAPL", "MSFT", "NVDA"],
    )

    aapl = by_symbol.loc[by_symbol["symbol"] == "AAPL"].iloc[0]
    msft = by_symbol.loc[by_symbol["symbol"] == "MSFT"].iloc[0]
    overall = global_summary.iloc[0]

    assert aapl["dataset_name"] == "features_1m"
    assert aapl["duplicate_row_count"] == 1
    assert aapl["duplicate_key_count"] == 1
    assert aapl["feature_alpha_null_pct"] == 0.0
    assert aapl["dataset_status"] == "FAIL"

    assert msft["duplicate_row_count"] == 0
    assert msft["duplicate_key_count"] == 0
    assert msft["feature_alpha_null_pct"] == 0.5
    assert msft["feature_alpha_inf_count"] == 1
    assert pd.isna(msft["feature_alpha_min"])
    assert pd.isna(msft["feature_alpha_max"])
    assert msft["dataset_status"] == "FAIL"

    assert overall["total_rows"] == 4
    assert overall["missing_symbol_count"] == 1
    assert overall["symbol_coverage_pct"] == 0.666667
    assert overall["missing_symbols"] == "NVDA"
    assert overall["feature_beta_min"] == 10.0
    assert overall["feature_beta_max"] == 12.0
    assert overall["dataset_status"] == "FAIL"


def test_write_feature_qa_artifacts_merges_datasets_deterministically(tmp_path) -> None:
    qa_root = tmp_path / "artifacts" / "qa" / "features"

    minute_df = _features_df().iloc[[2, 3]].reset_index(drop=True)
    daily_df = pd.DataFrame(
        {
            "symbol": pd.Series(["AAPL"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T21:00:00Z"], utc=True),
            "timeframe": pd.Series(["1D"], dtype="string"),
            "date": pd.Series(["2025-01-02"], dtype="string"),
            "feature_gamma": [0.25],
        }
    )

    write_feature_qa_artifacts(minute_df, timeframe="1Min", expected_symbols=["MSFT"], qa_root=qa_root)
    write_feature_qa_artifacts(daily_df, timeframe="1D", expected_symbols=["AAPL"], qa_root=qa_root)
    write_feature_qa_artifacts(minute_df, timeframe="1Min", expected_symbols=["MSFT"], qa_root=qa_root)

    by_symbol = pd.read_csv(qa_root / "qa_features_summary_by_symbol.csv")
    global_summary = pd.read_csv(qa_root / "qa_features_summary_global.csv")

    assert by_symbol[["dataset_name", "timeframe", "symbol"]].values.tolist() == [
        ["features_1m", "1Min", "MSFT"],
        ["features_daily", "1D", "AAPL"],
    ]
    assert global_summary[["dataset_name", "timeframe"]].values.tolist() == [
        ["features_1m", "1Min"],
        ["features_daily", "1D"],
    ]
    assert len(by_symbol.index) == 2
    assert len(global_summary.index) == 2
