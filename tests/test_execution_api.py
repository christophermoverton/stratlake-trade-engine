from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd

from src.execution import run_alpha, run_alpha_evaluation, run_portfolio, run_strategy
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


def test_run_alpha_evaluation_api_returns_notebook_summary(monkeypatch) -> None:
    artifact_dir = Path("artifacts/alpha/alpha-eval-123")

    def fake_resolve_cli_config(args):
        return {
            "alpha_model": args.alpha_model,
            "dataset": args.dataset,
            "target_column": args.target_column,
            "price_column": args.price_column,
            "artifacts_root": "artifacts/alpha",
        }

    def fake_run_resolved_config(config):
        return type(
            "AlphaEvaluationRunResult",
            (),
            {
                "alpha_name": config["alpha_model"],
                "run_id": "alpha-eval-123",
                "artifact_dir": artifact_dir,
                "evaluation_result": type(
                    "Evaluation",
                    (),
                    {"summary": {"mean_ic": 0.2, "ic_ir": 1.5, "n_periods": 12}},
                )(),
            },
        )()

    monkeypatch.setattr("src.cli.run_alpha_evaluation.resolve_cli_config", fake_resolve_cli_config)
    monkeypatch.setattr("src.cli.run_alpha_evaluation.run_resolved_config", fake_run_resolved_config)

    result = run_alpha_evaluation(
        config={
            "alpha_model": "cs_linear_ret_1d",
            "dataset": "features_daily",
            "target_column": "target_ret_1d",
            "price_column": "close",
        }
    )

    assert result.workflow == "alpha_evaluation"
    assert result.run_id == "alpha-eval-123"
    assert result.name == "cs_linear_ret_1d"
    assert result.artifact_dir == artifact_dir
    assert result.manifest_path == artifact_dir / "manifest.json"
    assert result.metrics["mean_ic"] == 0.2
    assert result.output_paths["alpha_metrics_json"] == artifact_dir / "alpha_metrics.json"
    assert result.to_dict()["manifest_path"].endswith("manifest.json")


def test_run_alpha_api_returns_structured_full_run_result(monkeypatch) -> None:
    artifact_dir = Path("artifacts/alpha/alpha-full-123")

    resolved_config = {
        "alpha_name": "cs_linear_ret_1d",
        "alpha_model": "cs_linear_ret_1d",
        "run_mode": "full",
        "price_column": "close",
        "realized_return_column": None,
        "artifacts_root": "artifacts/alpha",
    }

    evaluation_result = type(
        "AlphaEvaluationRunResult",
        (),
        {
            "run_id": "alpha-full-123",
            "artifact_dir": artifact_dir,
            "loaded_frame": pd.DataFrame({"close": [1.0]}),
            "signal_mapping_result": type("Signals", (), {"signals": pd.DataFrame({"signal": [1.0]})})(),
            "evaluation_result": type(
                "Evaluation",
                (),
                {"summary": {"mean_ic": 0.3, "ic_ir": 2.0, "n_periods": 8}},
            )(),
            "resolved_config": resolved_config,
            "manifest": {"artifact_paths": {}},
        },
    )()

    monkeypatch.setattr("src.cli.run_alpha.resolve_cli_config", lambda args: dict(resolved_config))
    monkeypatch.setattr("src.cli.run_alpha.run_resolved_config", lambda config: evaluation_result)
    monkeypatch.setattr(
        "src.cli.run_alpha.generate_alpha_sleeve",
        lambda **kwargs: type("Sleeve", (), {})(),
    )
    monkeypatch.setattr(
        "src.cli.run_alpha.write_alpha_sleeve_artifacts",
        lambda artifact_dir, sleeve_result, update_manifest=True: {"artifact_paths": {}},
    )
    monkeypatch.setattr("src.cli.run_alpha.register_alpha_evaluation_run", lambda **kwargs: None)
    monkeypatch.setattr(
        "src.cli.run_alpha.alpha_evaluation_registry_path",
        lambda artifacts_root: artifacts_root / "registry.jsonl",
    )
    monkeypatch.setattr(
        "src.cli.run_alpha.write_full_run_scaffold",
        lambda **kwargs: artifact_dir / "alpha_run_scaffold.json",
    )

    result = run_alpha("cs_linear_ret_1d")

    assert result.workflow == "alpha"
    assert result.run_id == "alpha-full-123"
    assert result.name == "cs_linear_ret_1d"
    assert result.artifact_dir == artifact_dir
    assert result.metrics == {"mean_ic": 0.3, "ic_ir": 2.0, "n_periods": 8}
    assert result.output_paths["scaffold_path"] == artifact_dir / "alpha_run_scaffold.json"
    assert result.extra["mode"] == "full"


def test_run_portfolio_api_uses_explicit_parameters_and_summarizes_result(monkeypatch) -> None:
    calls: dict[str, object] = {}
    artifact_dir = Path("artifacts/portfolios/portfolio-123")

    def fake_run_portfolio_resolved(**kwargs):
        calls.update(kwargs)
        return type(
            "PortfolioRunResult",
            (),
            {
                "portfolio_name": "core_portfolio",
                "run_id": "portfolio-123",
                "allocator_name": "equal_weight",
                "timeframe": "1D",
                "component_count": 2,
                "metrics": {"total_return": 0.05, "sharpe_ratio": 1.1},
                "experiment_dir": artifact_dir,
                "config": {},
                "components": [
                    {"strategy_name": "alpha_v1", "run_id": "run-alpha"},
                    {"strategy_name": "beta_v1", "run_id": "run-beta"},
                ],
            },
        )()

    monkeypatch.setattr("src.execution.portfolio._run_portfolio_resolved", fake_run_portfolio_resolved)

    result = run_portfolio(
        portfolio_name="core_portfolio",
        run_ids=["run-alpha", "run-beta"],
        timeframe="1D",
        execution_override={"enabled": True},
        strict=True,
    )

    assert result.workflow == "portfolio"
    assert result.run_id == "portfolio-123"
    assert result.name == "core_portfolio"
    assert result.artifact_dir == artifact_dir
    assert result.manifest_path == artifact_dir / "manifest.json"
    assert result.metrics["sharpe_ratio"] == 1.1
    assert result.output_paths["weights_csv"] == artifact_dir / "weights.csv"
    assert result.extra == {
        "allocator_name": "equal_weight",
        "component_count": 2,
        "timeframe": "1D",
    }
    assert calls["portfolio_name"] == "core_portfolio"
    assert calls["explicit_run_ids"] == ["run-alpha", "run-beta"]
    assert calls["execution_override"] == {"enabled": True}
    assert calls["strict"] is True


def test_notebook_execution_api_example_is_import_safe() -> None:
    namespace = runpy.run_path("docs/examples/notebook_execution_api_examples.py")

    assert callable(namespace["strategy_example"])
    assert callable(namespace["alpha_evaluation_example"])
    assert callable(namespace["alpha_full_run_example"])
    assert callable(namespace["portfolio_example"])
    assert callable(namespace["registry_backed_portfolio_example"])
