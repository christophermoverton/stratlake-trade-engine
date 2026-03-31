from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.alpha import (
    AlphaTimeSplit,
    AlphaTimeSplitError,
    generate_alpha_rolling_splits,
    make_alpha_fixed_split,
    validate_alpha_time_split,
)


def test_make_alpha_fixed_split_creates_valid_utc_split() -> None:
    split = make_alpha_fixed_split(
        train_start="2025-01-01",
        train_end="2025-01-10",
        predict_start="2025-01-10",
        predict_end="2025-01-15",
    )

    assert split.mode == "fixed"
    assert split.train_start == pd.Timestamp("2025-01-01T00:00:00Z")
    assert split.train_end == pd.Timestamp("2025-01-10T00:00:00Z")
    assert split.predict_start == pd.Timestamp("2025-01-10T00:00:00Z")
    assert split.predict_end == pd.Timestamp("2025-01-15T00:00:00Z")
    assert split.metadata["window_semantics"] == {
        "train": "[train_start, train_end)",
        "predict": "[predict_start, predict_end)",
    }
    assert split.metadata["train_duration_seconds"] == 9 * 24 * 60 * 60
    assert split.metadata["predict_duration_seconds"] == 5 * 24 * 60 * 60


def test_make_alpha_fixed_split_normalizes_mixed_timestamp_inputs_to_utc() -> None:
    split = make_alpha_fixed_split(
        train_start=pd.Timestamp("2025-01-01 09:30:00"),
        train_end=pd.Timestamp("2025-01-03 15:30:00-05:00"),
        predict_start="2025-01-03T20:30:00Z",
        predict_end="2025-01-04T20:30:00Z",
    )

    assert split.train_start == pd.Timestamp("2025-01-01T09:30:00Z")
    assert split.train_end == pd.Timestamp("2025-01-03T20:30:00Z")
    assert split.predict_start == pd.Timestamp("2025-01-03T20:30:00Z")
    assert split.predict_end == pd.Timestamp("2025-01-04T20:30:00Z")
    assert split.to_dict()["train_end"] == "2025-01-03T20:30:00Z"


def test_make_alpha_fixed_split_supports_open_ended_train_start() -> None:
    split = make_alpha_fixed_split(
        train_start=None,
        train_end="2025-01-10T00:00:00Z",
        predict_start="2025-01-10T00:00:00Z",
        predict_end="2025-01-12T00:00:00Z",
    )

    assert split.train_start is None
    assert split.metadata["train_duration_seconds"] is None


def test_make_alpha_fixed_split_rejects_invalid_timestamp_parsing() -> None:
    with pytest.raises(AlphaTimeSplitError, match="train_end must be a valid timestamp"):
        make_alpha_fixed_split(
            train_start="2025-01-01T00:00:00Z",
            train_end="not-a-timestamp",
            predict_start="2025-01-10T00:00:00Z",
            predict_end="2025-01-11T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {
                "train_start": "2025-01-03T00:00:00Z",
                "train_end": "2025-01-02T00:00:00Z",
                "predict_start": "2025-01-02T00:00:00Z",
                "predict_end": "2025-01-04T00:00:00Z",
            },
            "train_start must be earlier than train_end",
        ),
        (
            {
                "train_start": "2025-01-01T00:00:00Z",
                "train_end": "2025-01-04T00:00:00Z",
                "predict_start": "2025-01-03T00:00:00Z",
                "predict_end": "2025-01-05T00:00:00Z",
            },
            "must not overlap",
        ),
        (
            {
                "train_start": "2025-01-01T00:00:00Z",
                "train_end": "2025-01-03T00:00:00Z",
                "predict_start": "2025-01-05T00:00:00Z",
                "predict_end": "2025-01-05T00:00:00Z",
            },
            "predict_start must be earlier than predict_end",
        ),
    ],
)
def test_make_alpha_fixed_split_rejects_invalid_boundaries(
    kwargs: dict[str, str],
    expected_message: str,
) -> None:
    with pytest.raises(AlphaTimeSplitError, match=expected_message):
        make_alpha_fixed_split(**kwargs)


def test_make_alpha_fixed_split_generates_deterministic_split_id() -> None:
    first = make_alpha_fixed_split(
        train_start="2025-01-01",
        train_end="2025-01-10",
        predict_start="2025-01-10",
        predict_end="2025-01-12",
    )
    second = make_alpha_fixed_split(
        train_start="2025-01-01",
        train_end="2025-01-10",
        predict_start="2025-01-10",
        predict_end="2025-01-12",
    )

    assert first.split_id == second.split_id
    assert first.to_dict() == second.to_dict()
    assert validate_alpha_time_split(first) is first


def test_make_alpha_fixed_split_preserves_custom_split_id_and_metadata_without_mutation() -> None:
    metadata = {"owner": "alpha-research", "sequence_index": 99}

    split = make_alpha_fixed_split(
        train_start="2025-01-01",
        train_end="2025-01-05",
        predict_start="2025-01-05",
        predict_end="2025-01-06",
        split_id="alpha_fixed_custom",
        metadata=metadata,
    )

    metadata["owner"] = "mutated-by-caller"

    assert split.split_id == "alpha_fixed_custom"
    assert split.metadata["owner"] == "alpha-research"
    assert split.metadata["sequence_index"] == 99
    assert split.metadata["mode"] == "fixed"


def test_alpha_time_split_rejects_direct_naive_timestamps() -> None:
    with pytest.raises(AlphaTimeSplitError, match="normalized to UTC"):
        AlphaTimeSplit(
            split_id="bad",
            mode="fixed",
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-01-02"),
            predict_start=pd.Timestamp("2025-01-02"),
            predict_end=pd.Timestamp("2025-01-03"),
        )


def test_generate_alpha_rolling_splits_is_deterministic_and_gap_free() -> None:
    first = generate_alpha_rolling_splits(
        start="2025-01-01T00:00:00Z",
        end="2025-01-10T00:00:00Z",
        train_window="3D",
        predict_window="2D",
        step="2D",
    )
    second = generate_alpha_rolling_splits(
        start="2025-01-01T00:00:00Z",
        end="2025-01-10T00:00:00Z",
        train_window="3D",
        predict_window="2D",
        step="2D",
    )

    assert [split.to_dict() for split in first] == [split.to_dict() for split in second]
    assert [split.split_id for split in first] == ["rolling_0000", "rolling_0001", "rolling_0002"]
    assert [split.train_start for split in first] == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-03T00:00:00Z"),
        pd.Timestamp("2025-01-05T00:00:00Z"),
    ]
    assert [split.train_end for split in first] == [
        pd.Timestamp("2025-01-04T00:00:00Z"),
        pd.Timestamp("2025-01-06T00:00:00Z"),
        pd.Timestamp("2025-01-08T00:00:00Z"),
    ]
    assert [split.predict_start for split in first] == [
        pd.Timestamp("2025-01-04T00:00:00Z"),
        pd.Timestamp("2025-01-06T00:00:00Z"),
        pd.Timestamp("2025-01-08T00:00:00Z"),
    ]
    assert [split.predict_end for split in first] == [
        pd.Timestamp("2025-01-06T00:00:00Z"),
        pd.Timestamp("2025-01-08T00:00:00Z"),
        pd.Timestamp("2025-01-10T00:00:00Z"),
    ]
    assert [split.metadata["sequence_index"] for split in first] == [0, 1, 2]


def test_generate_alpha_rolling_splits_defaults_step_to_predict_window() -> None:
    splits = generate_alpha_rolling_splits(
        start="2025-01-01T00:00:00Z",
        end="2025-01-07T00:00:00Z",
        train_window="2D",
        predict_window="1D",
    )

    assert [split.split_id for split in splits] == ["rolling_0000", "rolling_0001", "rolling_0002", "rolling_0003"]


def test_generate_alpha_rolling_splits_rejects_insufficient_range() -> None:
    with pytest.raises(AlphaTimeSplitError, match="enough data to form at least one rolling alpha split"):
        generate_alpha_rolling_splits(
            start="2025-01-01T00:00:00Z",
            end="2025-01-04T00:00:00Z",
            train_window="3D",
            predict_window="2D",
        )
