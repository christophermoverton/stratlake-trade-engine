from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.plot_utils import (
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    apply_axis_style,
    create_figure,
    finalize_figure,
    save_or_return_figure,
)


def test_create_figure_uses_standard_default_size() -> None:
    figure, axis = create_figure()

    assert isinstance(figure, Figure)
    assert tuple(figure.get_size_inches()) == DEFAULT_FIGSIZE
    assert axis.figure is figure
    figure.clf()


def test_finalize_figure_formats_time_axes_and_apply_axis_style_adds_legend() -> None:
    figure, axis = create_figure()
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    axis.plot(dates, [1.0, 1.1, 1.05, 1.2], label="Strategy")

    apply_axis_style(axis, title="Equity Curve", x_label="Date", y_label="Equity", legend=True)
    finalize_figure(figure, axis, use_date_axis=True)

    legend = axis.get_legend()
    assert legend is not None
    assert axis.get_title() == "Equity Curve"
    assert axis.get_xlabel() == "Date"
    assert axis.get_ylabel() == "Equity"
    assert axis.xaxis.get_major_formatter().__class__.__name__ == "ConciseDateFormatter"
    figure.clf()


def test_save_or_return_figure_persists_png_with_standard_dpi(tmp_path: Path) -> None:
    figure, axis = create_figure()
    axis.plot([0, 1], [1, 2])
    output_path = tmp_path / "plots" / "shared_defaults.png"

    result = save_or_return_figure(figure, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert figure.dpi == DEFAULT_DPI
