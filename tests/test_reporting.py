from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
import pandas as pd

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics
from src.research.reporting import (
    generate_strategy_plots,
    generate_strategy_report,
    load_run_artifacts,
    print_quick_report,
    summarize_run,
)
from src.research.visualization import get_plot_dir, get_plot_path

matplotlib.use("Agg")


def assert_file_exists(path: Path) -> None:
    assert path.exists(), f"Expected artifact to exist: {path}"
    assert path.is_file(), f"Expected artifact to be a file: {path}"


def assert_in_file(path: Path, substring: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert substring in text, f"Expected '{substring}' to appear in {path}"


def assert_not_in_file(path: Path, substring: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert substring not in text, f"Did not expect '{substring}' to appear in {path}"


def _experiment_metrics() -> dict[str, float | None]:
    return compute_performance_metrics(_experiment_results())


def _report_metrics() -> dict[str, float | None]:
    return compute_performance_metrics(_report_results())


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


def _trade_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_ts_utc": pd.date_range("2022-01-02", periods=4, freq="7D", tz="UTC"),
            "exit_ts_utc": pd.date_range("2022-01-03", periods=4, freq="7D", tz="UTC"),
            "return": [0.015, -0.01, 0.02, 0.005],
        }
    )


def test_reporting_helpers_load_and_summarize_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "mean_reversion",
        _experiment_results(),
        _experiment_metrics(),
        {"strategy_name": "mean_reversion"},
    )

    loaded = load_run_artifacts(run_dir)
    summary = summarize_run(run_dir)
    print_quick_report(run_dir)

    assert loaded["manifest"]["run_id"] == run_dir.name
    assert loaded["metrics"]["sharpe_ratio"] == pytest.approx(_experiment_metrics()["sharpe_ratio"])
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
        "primary_metric_value": _experiment_metrics()["sharpe_ratio"],
        "cumulative_return": _experiment_metrics()["cumulative_return"],
        "sharpe_ratio": _experiment_metrics()["sharpe_ratio"],
        "max_drawdown": _experiment_metrics()["max_drawdown"],
        "trade_count": 1,
        "artifact_count": len(loaded["manifest"]["artifact_files"]),
        "split_metrics_rows": 0,
    }

    stdout = capsys.readouterr().out
    assert f"run_id: {run_dir.name}" in stdout
    assert "strategy: mean_reversion" in stdout
    assert "mode: single" in stdout
    assert f"sharpe_ratio: {_experiment_metrics()['sharpe_ratio']:.6f}" in stdout


def test_generate_strategy_report_creates_markdown_and_plot_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "momentum",
        _report_results(),
        _report_metrics(),
        {"strategy_name": "momentum", "parameters": {"lookback": 20, "threshold": 0.5}},
    )

    output_path = generate_strategy_report(run_dir)
    report_text = output_path.read_text(encoding="utf-8")

    assert output_path == run_dir / "report.md"
    assert output_path.exists()
    assert "# Strategy Report: momentum" in report_text
    assert "## Run Configuration Summary" in report_text
    assert "## Key Metrics" in report_text
    assert "## Visualizations" in report_text
    assert "### Performance Overview" in report_text
    assert "### Trade Summary" in report_text
    assert "## Interpretation" in report_text
    assert "## Artifact References" in report_text
    assert "| Sharpe | 9.775 |" in report_text
    assert "| Total Return | 5.84% |" in report_text
    assert "![Equity Curve](plots/equity_curve.png)" in report_text
    assert "![Drawdown](plots/drawdown.png)" in report_text
    assert "rolling_sharpe_debug.png" not in report_text
    assert "trade_return_distribution_debug.png" not in report_text
    assert "win_loss_distribution_debug.png" not in report_text
    assert "- [metrics.json](metrics.json)" in report_text
    assert "- [plots/](plots)" in report_text

    assert_file_exists(run_dir / "plots" / "equity_curve.png")
    assert_file_exists(run_dir / "plots" / "drawdown.png")
    assert_file_exists(run_dir / "plots" / "rolling_sharpe_debug.png")
    assert_file_exists(run_dir / "plots" / "trade_return_distribution_debug.png")
    assert_file_exists(run_dir / "plots" / "win_loss_distribution_debug.png")


def test_strategy_plot_generation_and_reporting_share_standardized_plot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "shared_paths",
        _report_results(),
        _report_metrics(),
        {"strategy_name": "shared_paths"},
    )

    plot_paths = generate_strategy_plots(run_dir)
    report_path = generate_strategy_report(run_dir)
    report_text = report_path.read_text(encoding="utf-8")

    assert get_plot_dir(run_dir) == run_dir / "plots"
    assert plot_paths["equity_curve"] == get_plot_path(run_dir, "equity_curve")
    assert plot_paths["drawdown"] == get_plot_path(run_dir, "drawdown")
    assert plot_paths["rolling_sharpe_debug"] == get_plot_path(run_dir, "rolling_sharpe_debug")
    assert plot_paths["trade_return_distribution_debug"] == get_plot_path(run_dir, "trade_return_distribution_debug")
    assert plot_paths["win_loss_distribution_debug"] == get_plot_path(run_dir, "win_loss_distribution_debug")
    assert "![Equity Curve](plots/equity_curve.png)" in report_text
    assert sorted(path.name for path in get_plot_dir(run_dir).iterdir()) == [
        "drawdown.png",
        "equity_curve.png",
        "rolling_sharpe_debug.png",
        "trade_return_distribution_debug.png",
        "win_loss_distribution_debug.png",
    ]


def test_generate_strategy_report_preserves_expected_artifact_structure_and_relative_report_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "artifact_contract",
        _report_results(),
        _report_metrics(),
        {"strategy_name": "artifact_contract", "parameters": {"lookback": 20}},
    )
    _trade_results().to_parquet(run_dir / "trades.parquet", index=False)

    report_path = generate_strategy_report(run_dir)
    plots_dir = get_plot_dir(run_dir)

    expected_plot_names = [
        "drawdown.png",
        "equity_curve.png",
        "rolling_sharpe_debug.png",
        "trade_return_distribution_debug.png",
        "win_loss_distribution_debug.png",
    ]

    assert plots_dir.is_dir()
    assert sorted(path.name for path in plots_dir.iterdir()) == expected_plot_names
    for plot_name in expected_plot_names:
        assert_file_exists(plots_dir / plot_name)

    assert_in_file(report_path, "# Strategy Report: artifact_contract")
    assert_in_file(report_path, "## Run Configuration Summary")
    assert_in_file(report_path, "## Key Metrics")
    assert_in_file(report_path, "## Visualizations")
    assert_in_file(report_path, "### Performance Overview")
    assert_in_file(report_path, "### Trade Summary")
    assert_in_file(report_path, "## Interpretation")
    assert_in_file(report_path, "## Artifact References")
    assert_in_file(report_path, "![Equity Curve](plots/equity_curve.png)")
    assert_in_file(report_path, "![Drawdown](plots/drawdown.png)")
    assert_in_file(report_path, "- [plots/](plots)")
    assert_in_file(report_path, "- [equity_curve.png](plots/equity_curve.png)")
    assert_in_file(report_path, "- [drawdown.png](plots/drawdown.png)")
    assert_in_file(report_path, "Trade diagnostics cover `4` closed trades.")
    assert_not_in_file(report_path, "C:/")
    assert_not_in_file(report_path, str(run_dir).replace("\\", "/"))
    assert_not_in_file(report_path, "rolling_sharpe_debug.png")
    assert_not_in_file(report_path, "trade_return_distribution_debug.png")
    assert_not_in_file(report_path, "win_loss_distribution_debug.png")


def test_generate_strategy_plots_rejects_nonstandard_plot_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "nonstandard_plots",
        _report_results(),
        _report_metrics(),
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
    assert "## Key Metrics" in report_text
    assert "## Visualizations" in report_text
    assert "_No visualization artifacts were available for this run._" in report_text
    assert "### Trade Summary" in report_text
    assert "_Trade data unavailable for this run._" in report_text
    assert "### Performance Overview" not in report_text


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
        _report_metrics(),
        {"strategy_name": "deterministic", "parameters": {"lookback": 20}},
    )

    first_output = generate_strategy_report(run_dir)
    first_text = first_output.read_text(encoding="utf-8")
    second_output = generate_strategy_report(run_dir)
    second_text = second_output.read_text(encoding="utf-8")

    assert first_output == second_output
    assert first_text == second_text


def test_generate_strategy_report_keeps_relative_paths_for_custom_output_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "relative_paths",
        _report_results(),
        _report_metrics(),
        {"strategy_name": "relative_paths", "parameters": {"lookback": 20}},
    )

    output_path = tmp_path / "reports" / "custom_report.md"
    generate_strategy_report(run_dir, output_path=output_path)
    report_text = output_path.read_text(encoding="utf-8")

    assert "](../artifacts/strategies/" in report_text
    assert "![Equity Curve](../artifacts/strategies/" in report_text
    assert "- [metrics.json](../artifacts/strategies/" in report_text


def test_generate_strategy_report_includes_walk_forward_context(tmp_path: Path) -> None:
    split_one_results = _report_results().iloc[:6].copy()
    split_two_results = _report_results().iloc[6:13].copy()
    split_results = [
        {
            "split_id": "split_001",
            "results_df": split_one_results,
            "metrics": compute_performance_metrics(split_one_results),
            "split_metadata": {
                "split_id": "split_001",
                "mode": "walk_forward",
                "train_start": "2022-01-01",
                "train_end": "2022-01-06",
                "test_start": "2022-01-07",
                "test_end": "2022-01-12",
            },
            "split_rows": 12,
            "train_rows": 6,
            "test_rows": 6,
        },
        {
            "split_id": "split_002",
            "results_df": split_two_results,
            "metrics": compute_performance_metrics(split_two_results),
            "split_metadata": {
                "split_id": "split_002",
                "mode": "walk_forward",
                "train_start": "2022-01-13",
                "train_end": "2022-01-18",
                "test_start": "2022-01-19",
                "test_end": "2022-01-25",
            },
            "split_rows": 13,
            "train_rows": 6,
            "test_rows": 7,
        },
    ]
    config = {
        "strategy_name": "walk_forward_momentum",
        "dataset": "features_daily",
        "evaluation": {
            "timeframe": "1D",
            "train_start": "2022-01-01",
            "train_end": "2022-01-18",
            "test_end": "2022-01-25",
        },
        "evaluation_config_path": "configs/walk_forward/example.json",
    }

    run_dir = save_walk_forward_experiment(
        "walk_forward_momentum",
        split_results,
        compute_performance_metrics(pd.concat([item["results_df"] for item in split_results], ignore_index=True)),
        config,
    )

    report_text = generate_strategy_report(run_dir).read_text(encoding="utf-8")

    assert "| Evaluation Mode | Walk Forward |" in report_text
    assert "| Split Count | 2 |" in report_text
    assert "| Evaluation Config Path | configs/walk_forward/example.json |" in report_text
    assert "metrics_by_split.csv" in report_text
    assert "Walk-forward artifacts summarize `2` saved split(s)" in report_text
