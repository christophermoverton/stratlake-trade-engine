from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.execution import run_strategy
from src.execution.result import ExecutionResult


def test_execution_result_serializes_paths_and_metrics(tmp_path: Path) -> None:
    result = ExecutionResult(
        workflow="strategy",
        run_id="run-123",
        name="momentum_v1",
        artifact_dir=tmp_path / "run-123",
        metrics={"sharpe_ratio": 1.2},
        manifest_path=tmp_path / "run-123" / "manifest.json",
        output_paths={"metrics_json": tmp_path / "run-123" / "metrics.json"},
    )

    payload = result.to_dict()

    assert payload["workflow"] == "strategy"
    assert payload["run_id"] == "run-123"
    assert payload["artifact_dir"].endswith("run-123")
    assert payload["metrics"] == {"sharpe_ratio": 1.2}
    assert payload["output_paths"]["metrics_json"].endswith("metrics.json")


def test_run_strategy_api_delegates_to_same_strategy_execution(monkeypatch) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "src.cli.run_strategy.get_strategy_config",
        lambda strategy_name, path=None: {"dataset": "features_daily", "parameters": {}},
    )

    def fake_resolve_runtime_config(config, cli_overrides=None, cli_strict=False):
        calls["runtime"] = {
            "config": config,
            "cli_overrides": cli_overrides,
            "cli_strict": cli_strict,
        }

        class Runtime:
            strict_mode = type("StrictMode", (), {"enabled": cli_strict})()
            execution = type("Execution", (), {"to_dict": lambda self: {"enabled": False}})()

        return Runtime()

    def fake_run_strategy_experiment(strategy_name, **kwargs):
        calls["strategy_name"] = strategy_name
        calls["kwargs"] = kwargs
        return type(
            "StrategyRunResult",
            (),
            {
                "strategy_name": strategy_name,
                "run_id": "run-123",
                "metrics": {"cumulative_return": 0.1, "sharpe_ratio": 1.2},
                "experiment_dir": Path("artifacts/strategies/run-123"),
                "results_df": pd.DataFrame(),
            },
        )()

    monkeypatch.setattr("src.cli.run_strategy.resolve_runtime_config", fake_resolve_runtime_config)
    monkeypatch.setattr("src.cli.run_strategy.run_strategy_experiment", fake_run_strategy_experiment)

    result = run_strategy(
        "momentum_v1",
        start="2025-01-01",
        end="2025-02-01",
        execution_override={"enabled": True},
        strict=True,
    )

    assert result.workflow == "strategy"
    assert result.run_id == "run-123"
    assert result.name == "momentum_v1"
    assert result.metrics["sharpe_ratio"] == 1.2
    assert result.raw_result.run_id == "run-123"
    assert calls["strategy_name"] == "momentum_v1"
    assert calls["runtime"]["cli_overrides"] == {"execution": {"enabled": True}}
    assert calls["runtime"]["cli_strict"] is True
    assert calls["kwargs"]["start"] == "2025-01-01"
    assert calls["kwargs"]["end"] == "2025-02-01"
    assert calls["kwargs"]["strict"] is True
