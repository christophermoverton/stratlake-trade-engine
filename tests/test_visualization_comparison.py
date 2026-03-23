from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.comparison import (
    align_series_collection,
    plot_equity_comparison,
    plot_metric_comparison,
    plot_strategy_metric_bars,
    plot_strategy_overlays,
)


def _date_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=4, freq="D")


def test_plot_equity_comparison_saves_png_for_dict_input(tmp_path: Path) -> None:
    index = _date_index()
    strategy_data = {
        "Alpha": pd.Series([0.01, -0.02, 0.03, 0.0], index=index, name="alpha_returns"),
        "Beta": pd.DataFrame({"returns": [0.0, 0.01, 0.01, -0.01]}, index=index),
    }

    output = plot_equity_comparison(
        strategy_data,
        input_type="returns",
        output_path=tmp_path / "plots" / "equity_comparison.png",
    )

    assert output == tmp_path / "plots" / "equity_comparison.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_equity_comparison_accepts_sequence_input_and_aligns_on_inner_join() -> None:
    alpha = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    beta = pd.Series(
        [0.9, 1.0, 1.05],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    result = plot_equity_comparison([("Alpha", alpha), ("Beta", beta)])

    assert isinstance(result, Figure)
    axis = result.axes[0]
    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assert pd.Index(axis.lines[0].get_xdata()).equals(expected_index)
    assert pd.Index(axis.lines[1].get_xdata()).equals(expected_index)
    assert [line.get_label() for line in axis.lines] == ["Alpha", "Beta"]
    assert axis.get_ylabel() == "Equity"
    result.clf()


def test_align_series_collection_raises_when_series_do_not_overlap() -> None:
    alpha = pd.Series([1.0, 1.1], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    beta = pd.Series([1.0, 1.05], index=pd.to_datetime(["2024-01-03", "2024-01-04"]))

    with pytest.raises(ValueError, match="aligned index value"):
        align_series_collection([("Alpha", alpha), ("Beta", beta)])


def test_plot_metric_comparison_saves_png_for_record_sequence(tmp_path: Path) -> None:
    metrics = [
        {"strategy": "Alpha", "sharpe": 1.2, "max_drawdown": -0.08},
        {"strategy": "Beta", "sharpe": 0.9, "max_drawdown": -0.05},
    ]

    output = plot_metric_comparison(
        metrics,
        metric_name="sharpe",
        output_path=tmp_path / "plots" / "sharpe_comparison.png",
    )

    assert output == tmp_path / "plots" / "sharpe_comparison.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_metric_comparison_returns_figure_for_dataframe_input() -> None:
    metrics = pd.DataFrame(
        {
            "run_id": ["run-alpha", "run-beta"],
            "total_return": [0.12, 0.07],
        }
    )

    result = plot_metric_comparison(metrics, metric_name="total_return")

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_ylabel() == "total_return"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["run-alpha", "run-beta"]
    result.clf()


def test_plot_metric_comparison_raises_for_missing_or_invalid_metric() -> None:
    metrics = pd.DataFrame({"strategy": ["Alpha", "Beta"], "sharpe": ["high", "low"]})

    with pytest.raises(ValueError, match="missing required column"):
        plot_metric_comparison(metrics, metric_name="max_drawdown")

    with pytest.raises(ValueError, match="must contain only numeric values"):
        plot_metric_comparison(metrics, metric_name="sharpe")


def test_plot_comparison_helpers_raise_for_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plot_equity_comparison({})

    with pytest.raises(ValueError, match="non-empty"):
        plot_metric_comparison(pd.DataFrame(), metric_name="sharpe")

    with pytest.raises(ValueError, match="non-empty"):
        plot_strategy_metric_bars(pd.DataFrame({"strategy": ["Alpha"]}), metrics=[])


def test_plot_equity_comparison_raises_for_non_time_index() -> None:
    strategy_data = {"Alpha": pd.Series([1.0, 1.1, 1.2], index=[0, 1, 2])}

    with pytest.raises(ValueError, match="time-indexed"):
        plot_equity_comparison(strategy_data)


def test_plot_strategy_overlays_accepts_legacy_sequence_and_labels() -> None:
    index = _date_index()
    strategy_frames = [
        pd.DataFrame({"equity": [1.0, 1.02, 1.05, 1.04]}, index=index),
        pd.DataFrame({"equity": [1.0, 1.01, 1.03, 1.02]}, index=index),
    ]

    result = plot_strategy_overlays(strategy_frames, labels=["Alpha", "Beta"])

    assert isinstance(result, Figure)
    assert [line.get_label() for line in result.axes[0].lines] == ["Alpha", "Beta"]
    result.clf()
