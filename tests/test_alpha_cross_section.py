from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha import (
    AlphaCrossSectionError,
    get_cross_section,
    iter_cross_sections,
    list_cross_section_timestamps,
    validate_cross_section_input,
)


@pytest.fixture
def research_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "GOOG", "GOOG", "MSFT", "MSFT"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "timeframe": ["1d", "1d", "1d", "1d", "1d", "1d"],
            "feature_alpha": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "feature_beta": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        },
        index=pd.Index([f"row_{idx}" for idx in range(6)], name="row_id"),
    )
    frame.attrs["source"] = "cross-section-tests"
    return frame


def test_get_cross_section_returns_deterministic_single_timestamp_slice(research_frame: pd.DataFrame) -> None:
    result = get_cross_section(research_frame, "2025-01-02T00:00:00Z")

    expected = pd.DataFrame(
        {
            "symbol": pd.array(["AAPL", "GOOG", "MSFT"], dtype="string"),
            "ts_utc": pd.to_datetime(["2025-01-02T00:00:00Z"] * 3, utc=True),
            "timeframe": ["1d", "1d", "1d"],
            "feature_alpha": [1.5, 2.5, 3.5],
            "feature_beta": [11.0, 13.0, 15.0],
        },
        index=pd.Index(["row_1", "row_3", "row_5"], name="row_id"),
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=True, check_exact=True)


def test_get_cross_section_supports_requested_column_subset_and_preserves_column_order(
    research_frame: pd.DataFrame,
) -> None:
    result = get_cross_section(
        research_frame,
        pd.Timestamp("2025-01-02 00:00:00"),
        columns=["feature_beta", "timeframe"],
    )

    assert list(result.columns) == ["symbol", "ts_utc", "feature_beta", "timeframe"]
    assert result["timeframe"].tolist() == ["1d", "1d", "1d"]


def test_get_cross_section_rejects_missing_timestamp(research_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaCrossSectionError, match="No rows found"):
        get_cross_section(research_frame, "2025-02-01T00:00:00Z")


def test_get_cross_section_rejects_empty_input() -> None:
    empty = pd.DataFrame(columns=["symbol", "ts_utc"])

    with pytest.raises(AlphaCrossSectionError, match="must not be empty"):
        get_cross_section(empty, "2025-01-01T00:00:00Z")


def test_validate_cross_section_input_rejects_missing_required_column(research_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaCrossSectionError, match="required columns"):
        validate_cross_section_input(research_frame.drop(columns=["symbol"]))


def test_get_cross_section_rejects_missing_requested_column(research_frame: pd.DataFrame) -> None:
    with pytest.raises(AlphaCrossSectionError, match="requested column 'feature_gamma'"):
        get_cross_section(research_frame, "2025-01-01T00:00:00Z", columns=["feature_gamma"])


def test_validate_cross_section_input_rejects_duplicate_symbol_timestamp_rows(
    research_frame: pd.DataFrame,
) -> None:
    duplicate = pd.concat(
        [research_frame.iloc[:1], research_frame.iloc[:1], research_frame.iloc[1:]],
        axis=0,
    )

    with pytest.raises(AlphaCrossSectionError, match="duplicate \\(symbol, ts_utc\\)"):
        validate_cross_section_input(duplicate)


def test_validate_cross_section_input_rejects_unsorted_input(research_frame: pd.DataFrame) -> None:
    unsorted = research_frame.iloc[[0, 2, 1, 3, 4, 5]].copy(deep=True)

    with pytest.raises(AlphaCrossSectionError, match="sorted by \\(symbol, ts_utc\\)"):
        validate_cross_section_input(unsorted)


def test_cross_section_helpers_do_not_mutate_caller_input(research_frame: pd.DataFrame) -> None:
    baseline = research_frame.copy(deep=True)
    baseline.attrs = dict(research_frame.attrs)

    _ = get_cross_section(research_frame, "2025-01-01T00:00:00Z", columns=["feature_alpha"])
    _ = list_cross_section_timestamps(research_frame)
    _ = list(iter_cross_sections(research_frame, columns=["feature_beta"]))

    pd.testing.assert_frame_equal(research_frame, baseline, check_dtype=True, check_exact=True)
    assert research_frame.attrs == baseline.attrs


def test_get_cross_section_is_deterministic_across_repeated_calls(research_frame: pd.DataFrame) -> None:
    first = get_cross_section(research_frame, "2025-01-01T00:00:00Z", columns=["feature_alpha"])
    second = get_cross_section(research_frame, "2025-01-01T00:00:00Z", columns=["feature_alpha"])

    pd.testing.assert_frame_equal(first, second, check_dtype=True, check_exact=True)


def test_list_cross_section_timestamps_returns_unique_sorted_utc_timestamps(
    research_frame: pd.DataFrame,
) -> None:
    timestamps = list_cross_section_timestamps(research_frame)

    assert timestamps == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]


def test_iter_cross_sections_yields_deterministic_timestamp_pairs(research_frame: pd.DataFrame) -> None:
    results = list(iter_cross_sections(research_frame, columns=["feature_beta"]))

    assert [timestamp for timestamp, _ in results] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
    assert [frame["symbol"].tolist() for _, frame in results] == [
        ["AAPL", "GOOG", "MSFT"],
        ["AAPL", "GOOG", "MSFT"],
    ]
    assert all(list(frame.columns) == ["symbol", "ts_utc", "feature_beta"] for _, frame in results)


def test_validate_cross_section_input_normalizes_timestamps_to_utc(research_frame: pd.DataFrame) -> None:
    localized = research_frame.copy(deep=True)
    localized["ts_utc"] = [
        "2024-12-31 18:00:00-06:00",
        "2025-01-01 18:00:00-06:00",
        "2024-12-31 18:00:00-06:00",
        "2025-01-01 18:00:00-06:00",
        "2024-12-31 18:00:00-06:00",
        "2025-01-01 18:00:00-06:00",
    ]

    validated = validate_cross_section_input(localized)

    assert validated["ts_utc"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-02T00:00:00Z"),
    ]
