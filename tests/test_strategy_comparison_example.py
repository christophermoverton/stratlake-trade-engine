from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.strategy_comparison_example import _generate_comparison_plots, parse_args, run_cli
from src.research.compare import ComparisonResult, LeaderboardEntry


def assert_file_exists(path: Path) -> None:
    assert path.exists(), f"Expected artifact to exist: {path}"
    assert path.is_file(), f"Expected artifact to be a file: {path}"


def test_parse_args_supports_bounded_example_workflow() -> None:
    args = parse_args(
        [
            "--strategies",
            "momentum_v1",
            "mean_reversion_v1",
            "--start",
            "2025-01-01",
            "--end",
            "2025-03-01",
            "--report-strategy",
            "mean_reversion_v1",
            "--output-dir",
            "artifacts/comparisons/example",
        ]
    )

    assert args.strategies == ["momentum_v1", "mean_reversion_v1"]
    assert args.start == "2025-01-01"
    assert args.end == "2025-03-01"
    assert args.report_strategy == "mean_reversion_v1"
    assert args.output_dir == "artifacts/comparisons/example"


def test_run_cli_generates_summary_and_comparison_plots(tmp_path: Path, monkeypatch) -> None:
    artifacts_root = tmp_path / "artifacts" / "strategies"
    run_ids = {
        "momentum_v1": "momentum_v1_single_abc123",
        "mean_reversion_v1": "mean_reversion_v1_single_def456",
    }
    for strategy_name, run_id in run_ids.items():
        run_dir = artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ts_utc": pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
                "equity": [1.0, 1.05, 1.1] if strategy_name == "momentum_v1" else [1.0, 0.99, 1.01],
                "strategy_return": [0.0, 0.05, 0.047619] if strategy_name == "momentum_v1" else [0.0, -0.01, 0.020202],
            }
        ).to_csv(run_dir / "equity_curve.csv", index=False)

    comparison_result = ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id=run_ids["momentum_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.2,
                cumulative_return=0.1,
                total_return=0.1,
                sharpe_ratio=1.2,
                max_drawdown=0.02,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
            LeaderboardEntry(
                rank=2,
                strategy_name="mean_reversion_v1",
                run_id=run_ids["mean_reversion_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=0.4,
                cumulative_return=0.01,
                total_return=0.01,
                sharpe_ratio=0.4,
                max_drawdown=0.03,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
        ],
        csv_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.csv",
        json_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.json",
    )
    comparison_result.csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_result.csv_path.write_text("rank,strategy_name\n1,momentum_v1\n", encoding="utf-8")
    comparison_result.json_path.write_text("{}", encoding="utf-8")

    generated_plot_calls: list[Path] = []
    report_calls: list[Path] = []

    monkeypatch.setattr("src.cli.strategy_comparison_example.ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr("src.cli.strategy_comparison_example.compare_strategies", lambda *args, **kwargs: comparison_result)
    monkeypatch.setattr(
        "src.cli.strategy_comparison_example.generate_strategy_plots",
        lambda run_dir: {"equity_curve": run_dir / "plots" / "equity_curve.png"},
    )

    def fake_generate_strategy_report(run_dir: Path) -> Path:
        report_calls.append(run_dir)
        report_path = run_dir / "report.md"
        report_path.write_text("# Report\n", encoding="utf-8")
        return report_path

    monkeypatch.setattr("src.cli.strategy_comparison_example.generate_strategy_report", fake_generate_strategy_report)

    def fake_plot_equity_comparison(*args, output_path: Path, **kwargs) -> Path:
        generated_plot_calls.append(output_path)
        output_path.write_text("equity", encoding="utf-8")
        return output_path

    def fake_plot_metric_comparison(*args, output_path: Path, **kwargs) -> Path:
        generated_plot_calls.append(output_path)
        output_path.write_text("metric", encoding="utf-8")
        return output_path

    monkeypatch.setattr("src.cli.strategy_comparison_example.plot_equity_comparison", fake_plot_equity_comparison)
    monkeypatch.setattr("src.cli.strategy_comparison_example.plot_metric_comparison", fake_plot_metric_comparison)

    summary = run_cli(
        [
            "--strategies",
            "momentum_v1",
            "mean_reversion_v1",
            "--output-dir",
            str(tmp_path / "artifacts" / "comparisons" / "example"),
        ]
    )

    summary_path = tmp_path / "artifacts" / "comparisons" / "example" / "example_summary.json"
    assert summary_path.exists()
    assert report_calls == [artifacts_root / run_ids["momentum_v1"]]
    assert generated_plot_calls == [
        tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "equity_comparison.png",
        tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "equity_comparison_debug.png",
        tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "metric_comparison_sharpe_ratio.png",
    ]
    assert summary["report_path"] == (artifacts_root / run_ids["momentum_v1"] / "report.md").as_posix()
    assert summary["comparison_plots"]["equity_comparison_debug"].endswith("equity_comparison_debug.png")


def test_generate_comparison_plots_creates_expected_artifact_set_and_is_deterministic(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "strategies"
    run_ids = {
        "momentum_v1": "momentum_v1_single_abc123",
        "mean_reversion_v1": "mean_reversion_v1_single_def456",
    }
    run_dirs: dict[str, Path] = {}

    for strategy_name, run_id in run_ids.items():
        run_dir = artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
                "equity_curve": [1.0, 1.04, 1.08, 1.1, 1.12]
                if strategy_name == "momentum_v1"
                else [1.0, 0.99, 1.01, 1.02, 1.03],
            }
        ).to_csv(run_dir / "equity_curve.csv", index=False)
        run_dirs[strategy_name] = run_dir

    comparison_result = ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id=run_ids["momentum_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.2,
                cumulative_return=0.12,
                total_return=0.12,
                sharpe_ratio=1.2,
                max_drawdown=0.02,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
            LeaderboardEntry(
                rank=2,
                strategy_name="mean_reversion_v1",
                run_id=run_ids["mean_reversion_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=0.4,
                cumulative_return=0.03,
                total_return=0.03,
                sharpe_ratio=0.4,
                max_drawdown=0.03,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
        ],
        csv_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.csv",
        json_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.json",
    )

    comparison_dir = tmp_path / "artifacts" / "comparisons" / "example"
    first_paths = _generate_comparison_plots(
        comparison_result=comparison_result,
        run_dirs=run_dirs,
        comparison_dir=comparison_dir,
    )
    second_paths = _generate_comparison_plots(
        comparison_result=comparison_result,
        run_dirs=run_dirs,
        comparison_dir=comparison_dir,
    )

    expected_names = {
        "equity_comparison": "equity_comparison.png",
        "equity_comparison_debug": "equity_comparison_debug.png",
        "metric_comparison": "metric_comparison_sharpe_ratio.png",
    }
    expected_plot_listing = sorted(expected_names.values())

    assert first_paths.keys() == expected_names.keys()
    assert second_paths.keys() == expected_names.keys()
    for key, filename in expected_names.items():
        assert first_paths[key].name == filename
        assert second_paths[key] == first_paths[key]
        assert_file_exists(first_paths[key])
        assert first_paths[key].parent == comparison_dir / "plots"
        assert first_paths[key].stat().st_size > 0

    assert sorted(path.name for path in (comparison_dir / "plots").iterdir()) == expected_plot_listing


def test_generate_comparison_plots_handles_duplicate_timestamps_from_run_artifacts(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "strategies"
    run_ids = {
        "momentum_v1": "momentum_v1_single_abc123",
        "mean_reversion_v1": "mean_reversion_v1_single_def456",
    }
    run_dirs: dict[str, Path] = {}

    for strategy_name, run_id in run_ids.items():
        run_dir = artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ts_utc": [
                    pd.Timestamp("2025-01-01", tz="UTC"),
                    pd.Timestamp("2025-01-01", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                ],
                "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
                "equity": [1.0, 1.0, 1.05, 1.05]
                if strategy_name == "momentum_v1"
                else [1.0, 1.0, 0.99, 0.99],
            }
        ).to_csv(run_dir / "equity_curve.csv", index=False)
        run_dirs[strategy_name] = run_dir

    comparison_result = ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id=run_ids["momentum_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.2,
                cumulative_return=0.12,
                total_return=0.12,
                sharpe_ratio=1.2,
                max_drawdown=0.02,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
            LeaderboardEntry(
                rank=2,
                strategy_name="mean_reversion_v1",
                run_id=run_ids["mean_reversion_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=0.4,
                cumulative_return=0.03,
                total_return=0.03,
                sharpe_ratio=0.4,
                max_drawdown=0.03,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
        ],
        csv_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.csv",
        json_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.json",
    )

    comparison_dir = tmp_path / "artifacts" / "comparisons" / "example"
    paths = _generate_comparison_plots(
        comparison_result=comparison_result,
        run_dirs=run_dirs,
        comparison_dir=comparison_dir,
    )

    assert_file_exists(paths["equity_comparison"])
    assert_file_exists(paths["equity_comparison_debug"])
    assert_file_exists(paths["metric_comparison"])


def test_run_cli_summary_lists_relative_comparison_artifacts_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_root = tmp_path / "artifacts" / "strategies"
    run_ids = {
        "momentum_v1": "momentum_v1_single_abc123",
        "mean_reversion_v1": "mean_reversion_v1_single_def456",
    }
    for strategy_name, run_id in run_ids.items():
        run_dir = artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "ts_utc": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
                "equity_curve": [1.0, 1.03, 1.05, 1.08, 1.1]
                if strategy_name == "momentum_v1"
                else [1.0, 1.0, 1.01, 1.0, 1.02],
                "strategy_return": [0.0, 0.03, 0.019417, 0.028571, 0.018519]
                if strategy_name == "momentum_v1"
                else [0.0, 0.0, 0.01, -0.009901, 0.02],
            }
        ).to_csv(run_dir / "equity_curve.csv", index=False)
        (run_dir / "metrics.json").write_text(
            '{"total_return": 0.1, "sharpe_ratio": 1.0, "max_drawdown": 0.05}',
            encoding="utf-8",
        )

    comparison_result = ComparisonResult(
        metric="sharpe_ratio",
        evaluation_mode="single",
        selection_mode="fresh",
        selection_rule="freshly executed run per strategy",
        leaderboard=[
            LeaderboardEntry(
                rank=1,
                strategy_name="momentum_v1",
                run_id=run_ids["momentum_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=1.2,
                cumulative_return=0.1,
                total_return=0.1,
                sharpe_ratio=1.2,
                max_drawdown=0.02,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
            LeaderboardEntry(
                rank=2,
                strategy_name="mean_reversion_v1",
                run_id=run_ids["mean_reversion_v1"],
                evaluation_mode="single",
                selected_metric_name="sharpe_ratio",
                selected_metric_value=0.4,
                cumulative_return=0.01,
                total_return=0.01,
                sharpe_ratio=0.4,
                max_drawdown=0.03,
                annualized_return=None,
                annualized_volatility=None,
                volatility=None,
                win_rate=None,
                hit_rate=None,
                profit_factor=None,
                turnover=None,
                exposure_pct=None,
            ),
        ],
        csv_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.csv",
        json_path=tmp_path / "artifacts" / "comparisons" / "example" / "leaderboard.json",
    )
    comparison_result.csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_result.csv_path.write_text("rank,strategy_name\n1,momentum_v1\n2,mean_reversion_v1\n", encoding="utf-8")
    comparison_result.json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("src.cli.strategy_comparison_example.ARTIFACTS_ROOT", artifacts_root)
    monkeypatch.setattr("src.cli.strategy_comparison_example.compare_strategies", lambda *args, **kwargs: comparison_result)

    summary = run_cli(
        [
            "--strategies",
            "momentum_v1",
            "mean_reversion_v1",
            "--output-dir",
            str(tmp_path / "artifacts" / "comparisons" / "example"),
        ]
    )

    summary_path = tmp_path / "artifacts" / "comparisons" / "example" / "example_summary.json"
    report_path = artifacts_root / run_ids["momentum_v1"] / "report.md"

    assert_file_exists(summary_path)
    assert_file_exists(report_path)
    assert_file_exists(tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "equity_comparison.png")
    assert_file_exists(tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "equity_comparison_debug.png")
    assert_file_exists(tmp_path / "artifacts" / "comparisons" / "example" / "plots" / "metric_comparison_sharpe_ratio.png")
    assert summary["comparison_plots"]["equity_comparison"].endswith("equity_comparison.png")
    assert summary["comparison_plots"]["equity_comparison_debug"].endswith("equity_comparison_debug.png")
    assert summary["comparison_plots"]["metric_comparison"].endswith("metric_comparison_sharpe_ratio.png")
    assert "equity_comparison_debug.png" not in report_path.read_text(encoding="utf-8")
