from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.diagnostics import (
    compute_drawdown_series,
    compute_equity_from_returns,
    plot_drawdown,
    plot_underwater_curve,
)


def _date_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=4, freq="D")


def test_compute_drawdown_series_from_returns_matches_expected_values() -> None:
    returns = pd.Series([0.10, -0.20, 0.05, 0.10], index=_date_index(), name="Alpha")

    result = compute_drawdown_series(returns, input_type="returns")

    expected_equity = compute_equity_from_returns(returns)
    expected = expected_equity / expected_equity.cummax() - 1.0
    assert result.index.equals(returns.index)
    assert result.tolist() == pytest.approx(expected.tolist())
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.min() < 0.0


def test_compute_drawdown_series_from_equity_matches_expected_values() -> None:
    equity = pd.Series([1.0, 1.2, 0.9, 1.05], index=_date_index(), name="Equity")

    result = compute_drawdown_series(equity, input_type="equity")

    expected = [0.0, 0.0, -0.25, -0.125]
    assert result.index.equals(equity.index)
    assert result.tolist() == pytest.approx(expected)


def test_plot_drawdown_accepts_single_column_dataframe_and_returns_figure() -> None:
    equity_frame = pd.DataFrame({"equity": [1.0, 1.1, 0.95, 1.15]}, index=_date_index())

    result = plot_drawdown(equity_frame)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    plotted_values = axis.lines[0].get_ydata()
    expected = compute_drawdown_series(equity_frame, input_type="equity")
    assert plotted_values.tolist() == pytest.approx(expected.tolist())
    assert axis.get_ylabel() == "Drawdown"
    result.clf()


def test_plot_drawdown_saves_png_for_returns_input(tmp_path: Path) -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01], index=_date_index(), name="Strategy")

    output = plot_drawdown(returns, input_type="returns", output_path=tmp_path / "plots" / "drawdown.png")

    assert output == tmp_path / "plots" / "drawdown.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_underwater_curve_returns_figure_when_not_saving() -> None:
    equity = pd.Series([1.0, 1.05, 0.98], index=_date_index()[:3], name="Strategy")

    result = plot_underwater_curve(equity)

    assert isinstance(result, Figure)
    assert result.axes[0].get_title() == "Underwater Curve"
    result.clf()


def test_drawdown_helpers_raise_for_empty_input() -> None:
    empty_series = pd.Series(dtype="float64")

    with pytest.raises(ValueError, match="non-empty"):
        compute_drawdown_series(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        plot_drawdown(empty_series)
