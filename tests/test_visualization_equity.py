from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.equity import (
    align_strategy_and_benchmark,
    compute_cumulative_returns,
    plot_cumulative_returns,
    plot_equity_curve,
)


def _date_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=4, freq="D")


def test_plot_cumulative_returns_saves_png_for_series_input(tmp_path: Path) -> None:
    strategy_returns = pd.Series([0.01, -0.02, 0.03, 0.0], index=_date_index(), name="Strategy")

    output = plot_cumulative_returns(strategy_returns, output_path=tmp_path / "plots" / "returns.png")

    assert output == tmp_path / "plots" / "returns.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_equity_curve_accepts_single_column_dataframe_and_benchmark_overlay() -> None:
    index = _date_index()
    strategy_equity = pd.DataFrame({"equity": [1.0, 1.02, 1.01, 1.05]}, index=index)
    benchmark_equity = pd.Series([1.0, 1.01, 1.015, 1.03], index=index, name="SPY")

    result = plot_equity_curve(strategy_equity, benchmark_data=benchmark_equity)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert len(axis.lines) == 2
    assert [line.get_label() for line in axis.lines] == ["equity", "SPY"]
    assert axis.get_ylabel() == "Equity"
    result.clf()


def test_plot_cumulative_returns_returns_figure_when_output_path_is_not_provided() -> None:
    strategy_returns = pd.DataFrame({"strategy_return": [0.01, 0.0, -0.01]}, index=_date_index()[:3])

    result = plot_cumulative_returns(strategy_returns)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    plotted_values = axis.lines[0].get_ydata()
    expected = compute_cumulative_returns(strategy_returns["strategy_return"]) - 1.0
    assert plotted_values.tolist() == pytest.approx(expected.tolist())
    assert axis.get_ylabel() == "Cumulative Return"
    result.clf()


def test_plot_functions_raise_for_empty_inputs() -> None:
    empty_series = pd.Series(dtype="float64")

    with pytest.raises(ValueError, match="non-empty"):
        plot_equity_curve(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        plot_cumulative_returns(empty_series)


def test_align_strategy_and_benchmark_uses_shared_index_values() -> None:
    strategy = pd.Series([1.0, 1.1, 1.2], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    benchmark = pd.Series([0.9, 1.0, 1.05], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))

    aligned_strategy, aligned_benchmark = align_strategy_and_benchmark(strategy, benchmark)

    expected_index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assert aligned_strategy.index.equals(expected_index)
    assert aligned_benchmark is not None
    assert aligned_benchmark.index.equals(expected_index)


def test_plot_equity_curve_can_compound_return_inputs_before_plotting() -> None:
    strategy_returns = pd.Series([0.01, 0.02, -0.01], index=_date_index()[:3], name="Alpha")

    result = plot_equity_curve(strategy_returns, input_type="returns")

    assert isinstance(result, Figure)
    plotted_values = result.axes[0].lines[0].get_ydata()
    expected = compute_cumulative_returns(strategy_returns)
    assert plotted_values.tolist() == pytest.approx(expected.tolist())
    result.clf()
