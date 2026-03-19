from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import yaml

from src.config.evaluation import EVALUATION_CONFIG, EvaluationConfig, load_evaluation_config
from src.research.splits import EvaluationSplitConfigError, generate_evaluation_splits


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(data, file_obj, sort_keys=False)


def test_evaluation_config_file_exists() -> None:
    assert EVALUATION_CONFIG.exists()


def test_load_evaluation_config_parses_repository_yaml() -> None:
    config = load_evaluation_config()

    assert config == EvaluationConfig(
        mode="rolling",
        timeframe="1d",
        start="2022-01-01",
        end="2024-01-01",
        train_window="12M",
        test_window="3M",
        step="3M",
        train_start=None,
        train_end=None,
        test_start=None,
        test_end=None,
    )


def test_generate_fixed_split_from_explicit_boundaries() -> None:
    splits = generate_evaluation_splits(
        {
            "mode": "fixed",
            "timeframe": "1d",
            "train_start": "2022-01-01",
            "train_end": "2023-01-01",
            "test_start": "2023-01-01",
            "test_end": "2023-04-01",
        }
    )

    assert [split.to_dict() for split in splits] == [
        {
            "split_id": "fixed_0000",
            "mode": "fixed",
            "train_start": "2022-01-01",
            "train_end": "2023-01-01",
            "test_start": "2023-01-01",
            "test_end": "2023-04-01",
        }
    ]


def test_generate_rolling_splits_uses_half_open_boundaries_deterministically() -> None:
    config = {
        "mode": "rolling",
        "start": "2022-01-01",
        "end": "2023-01-01",
        "train_window": "6M",
        "test_window": "3M",
        "step": "3M",
    }

    first = [split.to_dict() for split in generate_evaluation_splits(config)]
    second = [split.to_dict() for split in generate_evaluation_splits(config)]

    assert first == second
    assert first == [
        {
            "split_id": "rolling_0000",
            "mode": "rolling",
            "train_start": "2022-01-01",
            "train_end": "2022-07-01",
            "test_start": "2022-07-01",
            "test_end": "2022-10-01",
        },
        {
            "split_id": "rolling_0001",
            "mode": "rolling",
            "train_start": "2022-04-01",
            "train_end": "2022-10-01",
            "test_start": "2022-10-01",
            "test_end": "2023-01-01",
        },
    ]


def test_generate_expanding_splits_keeps_initial_train_start_fixed() -> None:
    splits = [
        split.to_dict()
        for split in generate_evaluation_splits(
            {
                "mode": "expanding",
                "start": "2022-01-01",
                "end": "2023-01-01",
                "train_window": "6M",
                "test_window": "3M",
                "step": "3M",
            }
        )
    ]

    assert splits == [
        {
            "split_id": "expanding_0000",
            "mode": "expanding",
            "train_start": "2022-01-01",
            "train_end": "2022-07-01",
            "test_start": "2022-07-01",
            "test_end": "2022-10-01",
        },
        {
            "split_id": "expanding_0001",
            "mode": "expanding",
            "train_start": "2022-01-01",
            "train_end": "2022-10-01",
            "test_start": "2022-10-01",
            "test_end": "2023-01-01",
        },
    ]


def test_generate_splits_accepts_exact_end_boundary_fit() -> None:
    splits = generate_evaluation_splits(
        {
            "mode": "rolling",
            "start": "2022-01-01",
            "end": "2022-10-01",
            "train_window": "6M",
            "test_window": "3M",
            "step": "3M",
        }
    )

    assert len(splits) == 1
    assert splits[0].test_end == "2022-10-01"


def test_generate_splits_rejects_insufficient_data_span() -> None:
    with pytest.raises(EvaluationSplitConfigError, match="enough data to form at least one train/test split"):
        generate_evaluation_splits(
            {
                "mode": "rolling",
                "start": "2022-01-01",
                "end": "2022-08-01",
                "train_window": "6M",
                "test_window": "3M",
                "step": "3M",
            }
        )


def test_generate_splits_rejects_overlapping_fixed_windows() -> None:
    with pytest.raises(EvaluationSplitConfigError, match="must not overlap"):
        generate_evaluation_splits(
            {
                "mode": "fixed",
                "train_start": "2022-01-01",
                "train_end": "2022-07-01",
                "test_start": "2022-06-15",
                "test_end": "2022-09-01",
            }
        )


def test_generate_splits_rejects_bad_config_values() -> None:
    with pytest.raises(ValueError, match="Evaluation mode must be one of"):
        generate_evaluation_splits({"mode": "random"})

    with pytest.raises(EvaluationSplitConfigError, match="train_window must use a positive duration"):
        generate_evaluation_splits(
            {
                "mode": "rolling",
                "start": "2022-01-01",
                "end": "2023-01-01",
                "train_window": "abc",
                "test_window": "3M",
                "step": "3M",
            }
        )

    with pytest.raises(EvaluationSplitConfigError, match="missing required fields"):
        generate_evaluation_splits(
            {
                "mode": "rolling",
                "start": "2022-01-01",
                "end": "2023-01-01",
                "train_window": "6M",
                "test_window": "3M",
            }
        )

    with pytest.raises(EvaluationSplitConfigError, match="Evaluation start must be before evaluation end"):
        generate_evaluation_splits(
            {
                "mode": "rolling",
                "start": "2023-01-01",
                "end": "2023-01-01",
                "train_window": "6M",
                "test_window": "3M",
                "step": "3M",
            }
        )


def test_load_evaluation_config_rejects_missing_evaluation_mapping() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "evaluation.yml"
        write_yaml(config_path, {"not_evaluation": {}})

        with pytest.raises(ValueError, match="must define an 'evaluation' mapping"):
            load_evaluation_config(config_path)
