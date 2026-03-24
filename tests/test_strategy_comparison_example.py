from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.strategy_comparison_example import parse_args, run_cli
from src.research.compare import ComparisonResult, LeaderboardEntry


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
