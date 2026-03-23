from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment
from src.research.reporting import (
    generate_strategy_plots,
    generate_strategy_report,
    load_run_artifacts,
    print_quick_report,
    summarize_run,
)
from src.research.visualization import get_plot_dir, get_plot_path


def _metrics() -> dict[str, float | None]:
    return {
        "cumulative_return": 0.0098,
        "total_return": 0.0098,
        "volatility": 0.012,
        "annualized_return": 0.19,
        "annualized_volatility": 0.21,
        "sharpe_ratio": 0.41,
        "max_drawdown": 0.05,
        "win_rate": 0.5,
        "hit_rate": 0.5,
        "profit_factor": 1.1,
        "turnover": 0.33,
        "exposure_pct": 66.7,
    }


def _experiment_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "feature_alpha": [0.15, -0.10, 0.25, 0.05],
            "signal": [1, 1, 0, 0],
            "strategy_return": [0.0, 0.02, -0.01, 0.0],
            "equity_curve": [1.0, 1.02, 1.0098, 1.0098],
        }
    )


def _report_results() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=25, freq="D")
    strategy_returns = [
        0.010,
        -0.004,
        0.006,
        0.003,
        -0.002,
        0.005,
        0.004,
        -0.003,
        0.007,
        0.002,
        -0.001,
        0.004,
        0.003,
        -0.002,
        0.006,
        0.005,
        -0.004,
        0.004,
        0.003,
        -0.001,
        0.005,
        0.002,
        -0.002,
        0.004,
        0.003,
    ]
    equity_curve = (1.0 + pd.Series(strategy_returns)).cumprod()
    signal = [1, 1, 1, 0, 0, 1, 1, -1, -1, 0, 1, 1, 0, -1, -1, 1, 1, 0, -1, -1, 1, 1, 0, 1, 0]

    return pd.DataFrame(
        {
            "symbol": ["SPY"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "signal": signal,
            "strategy_return": strategy_returns,
            "equity_curve": equity_curve,
        }
    )


def test_reporting_helpers_load_and_summarize_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "mean_reversion",
        _experiment_results(),
        _metrics(),
        {"strategy_name": "mean_reversion"},
    )

    loaded = load_run_artifacts(run_dir)
    summary = summarize_run(run_dir)
    print_quick_report(run_dir)

    assert loaded["manifest"]["run_id"] == run_dir.name
    assert loaded["metrics"]["sharpe_ratio"] == 0.41
    assert loaded["equity_curve"] is not None
    assert loaded["signals"] is not None
    assert loaded["trades"] is not None
    assert loaded["splits"] == {}

    assert summary == {
        "run_id": run_dir.name,
        "strategy_name": "mean_reversion",
        "evaluation_mode": "single",
        "split_count": None,
        "primary_metric": "sharpe_ratio",
        "primary_metric_value": 0.41,
        "cumulative_return": 0.0098,
        "sharpe_ratio": 0.41,
        "max_drawdown": 0.05,
        "trade_count": 1,
        "artifact_count": len(loaded["manifest"]["artifact_files"]),
        "split_metrics_rows": 0,
    }

    stdout = capsys.readouterr().out
    assert f"run_id: {run_dir.name}" in stdout
    assert "strategy: mean_reversion" in stdout
    assert "mode: single" in stdout
    assert "sharpe_ratio: 0.410000" in stdout


def test_generate_strategy_report_creates_markdown_and_plot_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "momentum",
        _report_results(),
        _metrics(),
        {"strategy_name": "momentum", "parameters": {"lookback": 20, "threshold": 0.5}},
    )

    output_path = generate_strategy_report(run_dir)
    report_text = output_path.read_text(encoding="utf-8")

    assert output_path == run_dir / "report.md"
    assert output_path.exists()
    assert "# Strategy Report: momentum" in report_text
    assert "## Run Metadata" in report_text
    assert "## Performance Summary" in report_text
    assert "## Equity Curve" in report_text
    assert "## Drawdown" in report_text
    assert "## Rolling Metrics" in report_text
    assert "## Trade Analysis" in report_text
    assert "## Observations" in report_text
    assert "| Sharpe | 0.410000 |" in report_text
    assert "![Equity Curve](plots/equity_curve.png)" in report_text
    assert "![Drawdown](plots/drawdown.png)" in report_text
    assert "![Rolling Sharpe](plots/rolling_sharpe.png)" in report_text
    assert "![Trade Return Distribution](plots/trade_return_distribution.png)" in report_text
    assert "![Win Loss Distribution](plots/win_loss_distribution.png)" in report_text

    assert (run_dir / "plots" / "equity_curve.png").exists()
    assert (run_dir / "plots" / "drawdown.png").exists()
    assert (run_dir / "plots" / "rolling_sharpe.png").exists()
    assert (run_dir / "plots" / "trade_return_distribution.png").exists()
    assert (run_dir / "plots" / "win_loss_distribution.png").exists()


def test_strategy_plot_generation_and_reporting_share_standardized_plot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "shared_paths",
        _report_results(),
        _metrics(),
        {"strategy_name": "shared_paths"},
    )

    plot_paths = generate_strategy_plots(run_dir)
    report_path = generate_strategy_report(run_dir)
    report_text = report_path.read_text(encoding="utf-8")

    assert get_plot_dir(run_dir) == run_dir / "plots"
    assert plot_paths["equity_curve"] == get_plot_path(run_dir, "equity_curve")
    assert plot_paths["drawdown"] == get_plot_path(run_dir, "drawdown")
    assert plot_paths["rolling_sharpe"] == get_plot_path(run_dir, "rolling_sharpe")
    assert plot_paths["trade_return_distribution"] == get_plot_path(run_dir, "trade_return_distribution")
    assert plot_paths["win_loss_distribution"] == get_plot_path(run_dir, "win_loss_distribution")
    assert "![Equity Curve](plots/equity_curve.png)" in report_text
    assert sorted(path.name for path in get_plot_dir(run_dir).iterdir()) == [
        "drawdown.png",
        "equity_curve.png",
        "rolling_sharpe.png",
        "trade_return_distribution.png",
        "win_loss_distribution.png",
    ]


def test_generate_strategy_plots_rejects_nonstandard_plot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "nonstandard_plots",
        _report_results(),
        _metrics(),
        {"strategy_name": "nonstandard_plots"},
    )

    with pytest.raises(ValueError, match="standardized run plot directory"):
        generate_strategy_plots(run_dir, plots_dir=run_dir / "custom_plots")


def test_generate_strategy_report_skips_optional_sections_when_artifacts_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_without_optionals"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"max_drawdown": 0.05, "sharpe_ratio": 0.41, "total_return": 0.0098}',
        encoding="utf-8",
    )

    output_path = generate_strategy_report(run_dir)
    report_text = output_path.read_text(encoding="utf-8")

    assert output_path.exists()
    assert "## Performance Summary" in report_text
    assert "## Rolling Metrics" in report_text
    assert "_Rolling Sharpe unavailable for this run._" in report_text
    assert "## Trade Analysis" in report_text
    assert "_Trade data unavailable for this run._" in report_text
    assert "## Equity Curve" not in report_text
    assert "## Drawdown" not in report_text


def test_generate_strategy_report_raises_when_metrics_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing_metrics"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        generate_strategy_report(run_dir)


def test_generate_strategy_report_is_deterministic_for_identical_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "deterministic",
        _report_results(),
        _metrics(),
        {"strategy_name": "deterministic", "parameters": {"lookback": 20}},
    )

    first_output = generate_strategy_report(run_dir)
    first_text = first_output.read_text(encoding="utf-8")
    second_output = generate_strategy_report(run_dir)
    second_text = second_output.read_text(encoding="utf-8")

    assert first_output == second_output
    assert first_text == second_text
