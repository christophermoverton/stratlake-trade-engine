from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.research.visualization.diagnostics import (
    compute_long_short_counts,
    compute_drawdown_series,
    compute_equity_from_returns,
    compute_rolling_sharpe,
    compute_trade_statistics,
    plot_drawdown,
    plot_exposure_over_time,
    plot_long_short_counts,
    plot_rolling_metric,
    plot_rolling_sharpe,
    plot_signal_distribution,
    plot_signal_diagnostics,
    plot_trade_return_distribution,
    plot_underwater_curve,
    plot_win_loss_distribution,
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


def test_compute_rolling_sharpe_matches_expected_values() -> None:
    returns = pd.Series([0.01, 0.03, 0.02, 0.04], index=_date_index(), name="Alpha")

    result = compute_rolling_sharpe(returns, window=2)

    expected = returns.rolling(window=2, min_periods=2).mean() / returns.rolling(window=2, min_periods=2).std()
    assert result.index.equals(returns.index)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1:].tolist() == pytest.approx(expected.iloc[1:].tolist())


def test_compute_rolling_sharpe_accepts_single_column_dataframe() -> None:
    returns_frame = pd.DataFrame({"returns": [0.01, -0.01, 0.02, 0.03]}, index=_date_index())

    result = compute_rolling_sharpe(returns_frame, window=3)
    expected = compute_rolling_sharpe(returns_frame["returns"], window=3)

    assert result.index.equals(returns_frame.index)
    assert result.tolist() == pytest.approx(expected.tolist(), nan_ok=True)


def test_compute_rolling_sharpe_raises_for_invalid_inputs() -> None:
    empty_series = pd.Series(dtype="float64")
    returns = pd.Series([0.01, 0.02], index=_date_index()[:2])

    with pytest.raises(ValueError, match="non-empty"):
        compute_rolling_sharpe(empty_series, window=2)

    with pytest.raises(ValueError, match="greater than zero"):
        compute_rolling_sharpe(returns, window=0)


def test_compute_rolling_sharpe_uses_nan_for_zero_variance_windows() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.02], index=_date_index(), name="Stable")

    result = compute_rolling_sharpe(returns, window=3)

    assert pd.isna(result.iloc[2])
    assert not np.isinf(result.to_numpy(dtype="float64", na_value=np.nan)).any()


def test_plot_rolling_metric_returns_figure_when_not_saving() -> None:
    metric = pd.Series([float("nan"), 0.5, 0.75, -0.25], index=_date_index(), name="Rolling Sharpe")

    result = plot_rolling_metric(metric, metric_name="Rolling Sharpe", window_label="2-period")

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_title() == "Rolling Sharpe (2-period)"
    assert axis.get_ylabel() == "Rolling Sharpe"
    result.clf()


def test_plot_rolling_sharpe_saves_png_and_returns_path(tmp_path: Path) -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, 0.02], index=pd.date_range("2024-01-01", periods=5, freq="D"))

    output = plot_rolling_sharpe(returns, window=3, output_path=tmp_path / "plots" / "rolling_sharpe.png")

    assert output == tmp_path / "plots" / "rolling_sharpe.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_rolling_sharpe_raises_for_empty_input() -> None:
    empty_series = pd.Series(dtype="float64")

    with pytest.raises(ValueError, match="non-empty"):
        plot_rolling_sharpe(empty_series, window=2)


def test_plot_signal_distribution_saves_png_for_series_input(tmp_path: Path) -> None:
    signals = pd.Series([-1, 0, 1, 1, -1, 0], index=_date_index().repeat(2)[:6], name="Signal")

    output = plot_signal_distribution(signals, output_path=tmp_path / "plots" / "signal_distribution.png")

    assert output == tmp_path / "plots" / "signal_distribution.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_compute_trade_statistics_matches_known_dataset() -> None:
    trade_returns = pd.Series([0.10, -0.05, 0.00, 0.02, -0.01], name="Trade Return")

    result = compute_trade_statistics(trade_returns)

    assert result["count"] == pytest.approx(5.0)
    assert result["win_count"] == pytest.approx(2.0)
    assert result["loss_count"] == pytest.approx(3.0)
    assert result["win_rate"] == pytest.approx(0.4)
    assert result["loss_rate"] == pytest.approx(0.6)
    assert result["mean_return"] == pytest.approx(float(trade_returns.mean()))
    assert result["median_return"] == pytest.approx(float(trade_returns.median()))
    assert result["std_return"] == pytest.approx(float(trade_returns.std()))


def test_plot_trade_return_distribution_saves_png_for_series_input(tmp_path: Path) -> None:
    trade_returns = pd.Series([0.04, -0.02, 0.01, -0.03, 0.06, 0.00], name="Per-Trade Return")

    output = plot_trade_return_distribution(
        trade_returns,
        output_path=tmp_path / "plots" / "trade_return_distribution.png",
    )

    assert output == tmp_path / "plots" / "trade_return_distribution.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_trade_return_distribution_accepts_single_column_dataframe_and_returns_figure() -> None:
    trade_returns = pd.DataFrame({"trade_return": [0.03, -0.01, 0.02, -0.04, 0.01]})

    result = plot_trade_return_distribution(trade_returns)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_title() == "Trade Return Distribution"
    assert axis.get_xlabel() == "trade_return"
    assert len(axis.patches) > 0
    result.clf()


def test_plot_win_loss_distribution_saves_png_for_series_input(tmp_path: Path) -> None:
    trade_returns = pd.Series([0.03, -0.02, 0.01, 0.02, -0.01], name="Trade Return")

    output = plot_win_loss_distribution(
        trade_returns,
        output_path=tmp_path / "plots" / "win_loss_distribution.png",
    )

    assert output == tmp_path / "plots" / "win_loss_distribution.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_win_loss_distribution_returns_expected_counts() -> None:
    trade_returns = pd.Series([0.03, -0.02, 0.00, 0.01, -0.01], name="Trade Return")

    result = plot_win_loss_distribution(trade_returns)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    heights = [patch.get_height() for patch in axis.patches]
    assert heights == pytest.approx([2.0, 3.0])
    result.clf()


def test_trade_diagnostic_helpers_raise_for_empty_input() -> None:
    empty_series = pd.Series(dtype="float64")

    with pytest.raises(ValueError, match="non-empty"):
        compute_trade_statistics(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        plot_trade_return_distribution(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        plot_win_loss_distribution(empty_series)


def test_trade_diagnostic_helpers_raise_for_non_numeric_input() -> None:
    invalid_series = pd.Series(["win", "loss"], name="Trade Return")

    with pytest.raises(ValueError, match="numeric"):
        compute_trade_statistics(invalid_series)

    with pytest.raises(ValueError, match="numeric"):
        plot_trade_return_distribution(invalid_series)

    with pytest.raises(ValueError, match="numeric"):
        plot_win_loss_distribution(invalid_series)


def test_plot_signal_diagnostics_accepts_single_column_dataframe_and_returns_figure() -> None:
    signals = pd.DataFrame({"target_weight": [-1.0, -0.5, 0.0, 0.5]}, index=_date_index())

    result = plot_signal_diagnostics(signals)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_title() == "Signal Diagnostics"
    assert axis.get_xlabel() == "target_weight"
    result.clf()


def test_plot_exposure_over_time_saves_png_for_dataframe_input(tmp_path: Path) -> None:
    exposure = pd.DataFrame({"net_exposure": [-0.25, 0.10, 0.35, -0.05]}, index=_date_index())

    output = plot_exposure_over_time(exposure, output_path=tmp_path / "plots" / "exposure_over_time.png")

    assert output == tmp_path / "plots" / "exposure_over_time.png"
    assert output.exists()
    assert output.suffix == ".png"
    assert output.stat().st_size > 0


def test_plot_exposure_over_time_returns_figure_for_series_input() -> None:
    exposure = pd.Series([0.2, 0.4, 0.1, 0.3], index=_date_index(), name="Gross Exposure")

    result = plot_exposure_over_time(exposure)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    assert axis.get_ylabel() == "Gross Exposure"
    assert len(axis.lines) == 1
    result.clf()


def test_compute_long_short_counts_matches_expected_counts_for_position_matrix() -> None:
    positions = pd.DataFrame(
        {
            "AAPL": [1.0, 0.0, -1.0, 2.0],
            "MSFT": [-0.5, 1.0, 0.0, 3.0],
            "NVDA": [0.0, -2.0, -0.25, 0.0],
        },
        index=_date_index(),
    )

    result = compute_long_short_counts(positions)

    expected = pd.DataFrame(
        {
            "long_count": [1, 1, 0, 2],
            "short_count": [1, 1, 2, 0],
        },
        index=_date_index(),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_plot_long_short_counts_handles_long_only_series_predictably() -> None:
    aggregate_positions = pd.Series([0.0, 1.0, 0.5, 0.25], index=_date_index(), name="Net Position")

    result = plot_long_short_counts(aggregate_positions)

    assert isinstance(result, Figure)
    axis = result.axes[0]
    long_values = axis.lines[0].get_ydata()
    short_values = axis.lines[1].get_ydata()
    assert long_values.tolist() == [0, 1, 1, 1]
    assert short_values.tolist() == [0, 0, 0, 0]
    result.clf()


def test_signal_and_exposure_helpers_raise_for_empty_input() -> None:
    empty_series = pd.Series(dtype="float64")
    empty_frame = pd.DataFrame()

    with pytest.raises(ValueError, match="non-empty"):
        plot_signal_distribution(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        plot_exposure_over_time(empty_series)

    with pytest.raises(ValueError, match="non-empty"):
        compute_long_short_counts(empty_frame)
