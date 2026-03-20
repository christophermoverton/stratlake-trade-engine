from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from src.research import experiment_tracker
from src.research.experiment_tracker import save_experiment
from src.research.reporting import load_run_artifacts, print_quick_report, summarize_run


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
