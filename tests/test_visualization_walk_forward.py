from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.walk_forward import (
    plot_fold_level_metrics,
    plot_walk_forward_results,
    plot_walk_forward_splits,
)


def _split_records() -> list[dict[str, object]]:
    return [
        {
            "fold": "rolling_0000",
            "train_start": "2024-01-01",
            "train_end": "2024-01-10",
            "test_start": "2024-01-10",
            "test_end": "2024-01-15",
        },
        {
            "fold": "rolling_0001",
            "train_start": "2024-01-06",
            "train_end": "2024-01-15",
            "test_start": "2024-01-15",
            "test_end": "2024-01-20",
        },
    ]


def test_plot_walk_forward_splits_saves_png_for_dataframe_input(tmp_path: Path) -> None:
    splits_frame = pd.DataFrame(_split_records())

    output = plot_walk_forward_splits(splits_frame, output_path=tmp_path / "plots" / "walk_forward_splits.png")

    assert output == tmp_path / "plots" / "walk_forward_splits.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_walk_forward_splits_accepts_sequence_of_dicts_and_returns_figure() -> None:
    result = plot_walk_forward_splits(_split_records())

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_ylabel() == "Fold"
    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["rolling_0000", "rolling_0001"]
    assert len(axis.patches) == 4
    result.clf()


def test_plot_fold_level_metrics_saves_png_and_returns_path(tmp_path: Path) -> None:
    metrics_frame = pd.DataFrame(
        {
            "fold": ["rolling_0000", "rolling_0001", "rolling_0002"],
            "sharpe": [0.8, -0.1, 1.25],
        }
    )

    output = plot_fold_level_metrics(
        metrics_frame,
        metric_name="sharpe",
        output_path=tmp_path / "plots" / "fold_metric_sharpe.png",
    )

    assert output == tmp_path / "plots" / "fold_metric_sharpe.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_fold_level_metrics_returns_figure_when_output_path_is_not_provided() -> None:
    result = plot_fold_level_metrics(
        [
            {"fold": "rolling_0000", "total_return": 0.03},
            {"fold": "rolling_0001", "total_return": -0.02},
        ],
        metric_name="total_return",
    )

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_xlabel() == "Fold"
    assert axis.get_ylabel() == "total_return"
    assert len(axis.patches) == 2
    result.clf()


def test_walk_forward_visualization_raises_for_missing_required_columns() -> None:
    invalid_splits = pd.DataFrame(
        {
            "fold": ["rolling_0000"],
            "train_start": ["2024-01-01"],
            "train_end": ["2024-01-10"],
            "test_start": ["2024-01-10"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        plot_walk_forward_splits(invalid_splits)

    with pytest.raises(ValueError, match="missing required columns"):
        plot_fold_level_metrics(pd.DataFrame({"fold": ["rolling_0000"]}), metric_name="sharpe")


def test_walk_forward_visualization_raises_for_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_walk_forward_splits(pd.DataFrame())

    with pytest.raises(ValueError, match="non-empty"):
        plot_fold_level_metrics([], metric_name="sharpe")


def test_plot_walk_forward_results_returns_figure_for_equity_frames() -> None:
    fold_results = [
        pd.DataFrame({"equity": [1.0, 1.02, 1.05]}),
        pd.DataFrame({"equity": [1.0, 0.99, 1.01]}),
    ]

    result = plot_walk_forward_results(fold_results, labels=["fold_a", "fold_b"])

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_ylabel() == "Fold Return"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["fold_a", "fold_b"]
    result.clf()
