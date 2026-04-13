from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.generate_report import parse_args as parse_report_args
from src.cli.generate_milestone_report import parse_args as parse_milestone_report_args
from src.cli.generate_milestone_report import run_cli as run_generate_milestone_report_cli
from src.cli.generate_report import run_cli as run_generate_report_cli
from src.cli.plot_strategy_run import parse_args as parse_plot_args
from src.cli.plot_strategy_run import run_cli as run_plot_strategy_run_cli
from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment
from src.research.metrics import compute_performance_metrics
from src.research.visualization import get_plot_dir, get_plot_path


def _metrics() -> dict[str, float | None]:
    return compute_performance_metrics(_report_results())


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


def test_plot_strategy_run_parse_args_accepts_run_dir() -> None:
    args = parse_plot_args(["--run-dir", "artifacts/strategies/run-123"])

    assert args.run_dir == "artifacts/strategies/run-123"


def test_generate_report_parse_args_accepts_run_dir_and_output_path() -> None:
    args = parse_report_args(
        ["--run-dir", "artifacts/strategies/run-123", "--output-path", "artifacts/reports/custom.md"]
    )

    assert args.run_dir == "artifacts/strategies/run-123"
    assert args.output_path == "artifacts/reports/custom.md"


def test_generate_milestone_report_parse_args_accepts_campaign_artifact_path_and_output_path() -> None:
    args = parse_milestone_report_args(
        [
            "--campaign-artifact-path",
            "artifacts/research_campaigns/run-123",
            "--output-path",
            "artifacts/reports/milestone_pack",
            "--milestone-name",
            "Milestone 18",
            "--title",
            "Milestone 18 Report",
        ]
    )

    assert args.campaign_artifact_path == "artifacts/research_campaigns/run-123"
    assert args.output_path == "artifacts/reports/milestone_pack"
    assert args.milestone_name == "Milestone 18"
    assert args.title == "Milestone 18 Report"


def test_plot_strategy_run_cli_generates_supported_plot_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "momentum",
        _report_results(),
        _metrics(),
        {"strategy_name": "momentum", "parameters": {"lookback": 20}},
    )

    plot_paths = run_plot_strategy_run_cli(["--run-dir", str(run_dir)])

    assert set(plot_paths) == {
        "drawdown",
        "equity_curve",
        "rolling_sharpe_debug",
        "trade_return_distribution_debug",
        "win_loss_distribution_debug",
    }
    for path in plot_paths.values():
        assert path.exists()
        assert path.parent == get_plot_dir(run_dir)

    stdout = capsys.readouterr().out
    assert f"run_dir: {run_dir}" in stdout
    assert "plot_count: 5" in stdout
    assert "equity_curve:" in stdout


def test_plot_strategy_run_cli_generates_only_supported_optional_plots(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"max_drawdown": 0.05, "sharpe_ratio": 0.41, "total_return": 0.0098}',
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=5, freq="D").strftime("%Y-%m-%d"),
            "equity": [1.0, 1.01, 1.02, 1.01, 1.03],
            "strategy_return": [0.0, 0.01, 0.01, -0.01, 0.02],
        }
    ).to_csv(run_dir / "equity_curve.csv", index=False)

    plot_paths = run_plot_strategy_run_cli(["--run-dir", str(run_dir)])

    assert set(plot_paths) == {"drawdown", "equity_curve"}
    assert (run_dir / "plots" / "equity_curve.png").exists()
    assert (run_dir / "plots" / "drawdown.png").exists()
    assert not (run_dir / "plots" / "trade_return_distribution_debug.png").exists()


def test_plot_strategy_run_cli_raises_for_missing_run_dir() -> None:
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        run_plot_strategy_run_cli(["--run-dir", "artifacts/strategies/missing-run"])


def test_plot_strategy_run_cli_raises_when_no_supported_plot_inputs_exist(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"sharpe_ratio": 0.41}', encoding="utf-8")

    with pytest.raises(ValueError, match="No plot artifacts can be generated"):
        run_plot_strategy_run_cli(["--run-dir", str(run_dir)])


def test_generate_report_cli_writes_default_report_and_reuses_reporting_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    run_dir = save_experiment(
        "momentum",
        _report_results(),
        _metrics(),
        {"strategy_name": "momentum", "parameters": {"lookback": 20}},
    )

    report_path = run_generate_report_cli(["--run-dir", str(run_dir)])

    assert report_path == run_dir / "report.md"
    assert report_path.exists()
    assert get_plot_path(run_dir, "equity_curve").exists()
    assert get_plot_path(run_dir, "drawdown").exists()

    stdout = capsys.readouterr().out
    assert f"run_dir: {run_dir}" in stdout
    assert f"report_path: {run_dir / 'report.md'}" in stdout


def test_generate_report_cli_supports_output_path_override(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"max_drawdown": 0.05, "sharpe_ratio": 0.41, "total_return": 0.0098}',
        encoding="utf-8",
    )

    output_path = tmp_path / "reports" / "custom_report.md"
    report_path = run_generate_report_cli(["--run-dir", str(run_dir), "--output-path", str(output_path)])

    assert report_path == output_path
    assert output_path.exists()


def test_generate_report_cli_raises_for_missing_run_dir() -> None:
    with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
        run_generate_report_cli(["--run-dir", "artifacts/strategies/missing-run"])


def test_generate_report_cli_raises_for_missing_metrics_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        run_generate_report_cli(["--run-dir", str(run_dir)])


def test_generate_milestone_report_cli_writes_default_pack(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign_dir = tmp_path / "campaign"
    review_dir = tmp_path / "review"
    campaign_dir.mkdir()
    review_dir.mkdir()
    (review_dir / "review_summary.json").write_text('{"review_id":"review_demo","entry_count":2}', encoding="utf-8")
    (campaign_dir / "summary.json").write_text(
        """
{
  "run_type": "research_campaign",
  "campaign_run_id": "research_campaign_demo",
  "status": "completed",
  "preflight_status": "passed",
  "stage_statuses": {"review": "completed"},
  "stage_state_counts": {"completed": 1, "failed": 0, "partial": 0, "pending": 0, "reused": 0, "skipped": 0},
  "selected_run_ids": {"alpha_run_ids": [], "strategy_run_ids": [], "candidate_selection_run_id": null, "portfolio_run_id": null, "review_id": "review_demo"},
  "targets": {"alpha_names": [], "strategy_names": [], "portfolio_names": [], "candidate_selection_alpha_name": null, "portfolio_name": null},
  "key_metrics": {"alpha_runs": [], "strategy_runs": [], "candidate_selection": null, "portfolio": null, "review": {"review_id": "review_demo", "entry_count": 2}},
  "output_paths": {"review_summary": "%REVIEW_SUMMARY%"},
  "final_outcomes": {"failed_stage_names": [], "partial_stage_names": [], "resumable_stage_names": []},
  "checkpoint": {"stages": []},
  "stages": []
}
""".replace("%REVIEW_SUMMARY%", (review_dir / "review_summary.json").as_posix()).strip(),
        encoding="utf-8",
    )

    summary_path = run_generate_milestone_report_cli(["--campaign-artifact-path", str(campaign_dir)])

    assert summary_path == campaign_dir / "milestone_report" / "summary.json"
    assert summary_path.exists()

    stdout = capsys.readouterr().out
    assert f"campaign_artifact_path: {campaign_dir}" in stdout
    assert f"milestone_report_path: {summary_path}" in stdout
