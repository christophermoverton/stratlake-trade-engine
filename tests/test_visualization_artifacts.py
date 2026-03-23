from __future__ import annotations

from pathlib import Path

import pytest

from src.research.visualization import get_plot_dir, get_plot_filename, get_plot_path


def test_plot_artifact_helpers_return_standardized_run_scoped_paths() -> None:
    run_dir = Path("artifacts/strategies/run-123")

    assert get_plot_dir(run_dir) == run_dir / "plots"
    assert get_plot_path(run_dir, "equity_curve") == run_dir / "plots" / "equity_curve.png"
    assert get_plot_path(run_dir, "drawdown") == run_dir / "plots" / "drawdown.png"
    assert get_plot_path(run_dir, "rolling_sharpe") == run_dir / "plots" / "rolling_sharpe.png"
    assert get_plot_path(run_dir, "signal_distribution") == run_dir / "plots" / "signal_distribution.png"
    assert get_plot_path(run_dir, "exposure_over_time") == run_dir / "plots" / "exposure_over_time.png"
    assert get_plot_path(run_dir, "long_short_counts") == run_dir / "plots" / "long_short_counts.png"
    assert get_plot_path(run_dir, "trade_return_distribution") == run_dir / "plots" / "trade_return_distribution.png"
    assert get_plot_path(run_dir, "win_loss_distribution") == run_dir / "plots" / "win_loss_distribution.png"
    assert get_plot_path(run_dir, "walk_forward_splits") == run_dir / "plots" / "walk_forward_splits.png"
    assert get_plot_path(run_dir, "fold_metric", metric_name="Sharpe Ratio") == (
        run_dir / "plots" / "fold_metric_sharpe_ratio.png"
    )
    assert get_plot_path(run_dir, "equity_comparison") == run_dir / "plots" / "equity_comparison.png"
    assert get_plot_path(run_dir, "metric_comparison", metric_name="max_drawdown") == (
        run_dir / "plots" / "metric_comparison_max_drawdown.png"
    )


def test_plot_artifact_helpers_reject_unknown_or_invalid_plot_names() -> None:
    with pytest.raises(ValueError, match="Unsupported plot artifact name"):
        get_plot_filename("unknown_plot")

    with pytest.raises(ValueError, match="metric_name is required"):
        get_plot_filename("fold_metric")
