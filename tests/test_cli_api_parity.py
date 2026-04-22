from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.execution.result import ExecutionResult


EXECUTION_RESULT_FIELDS = {
    "workflow",
    "run_id",
    "name",
    "artifact_dir",
    "metrics",
    "manifest_path",
    "registry_path",
    "output_paths",
    "extra",
}


def _api_contract(result: ExecutionResult) -> dict[str, object]:
    payload = result.to_dict()
    return {
        "shape": set(payload),
        "workflow": payload["workflow"],
        "run_id": payload["run_id"],
        "name": payload["name"],
        "artifact_leaf": _path_leaf(payload["artifact_dir"]),
        "manifest_leaf": _path_leaf(payload["manifest_path"]),
        "metrics": payload["metrics"],
        "output_files": _output_file_names(payload["output_paths"]),
        "extra": payload["extra"],
    }


def _assert_contract_shape(result: ExecutionResult, *, required_output_keys: set[str]) -> None:
    payload = result.to_dict()
    assert EXECUTION_RESULT_FIELDS.issubset(payload)
    assert required_output_keys.issubset(set(payload["output_paths"]))


def _path_leaf(value: object) -> str | None:
    if value is None:
        return None
    return Path(str(value)).name


def _output_file_names(output_paths: dict[str, object]) -> dict[str, str]:
    return {
        key: Path(str(value)).name
        for key, value in sorted(output_paths.items())
        if value is not None
    }


def _strategy_contract_from_cli(raw_result: object) -> dict[str, object]:
    artifact_dir = raw_result.experiment_dir
    return {
        "workflow": "strategy",
        "run_id": raw_result.run_id,
        "name": raw_result.strategy_name,
        "artifact_leaf": artifact_dir.name,
        "manifest_leaf": "manifest.json",
        "metrics": dict(raw_result.metrics),
        "output_files": {
            "equity_curve_csv": "equity_curve.csv",
            "manifest_json": "manifest.json",
            "metrics_json": "metrics.json",
            "qa_summary_json": "qa_summary.json",
        },
        "extra": {"result_type": raw_result.__class__.__name__},
    }


def test_strategy_cli_api_parity_uses_structured_run_summary(monkeypatch) -> None:
    from src.cli import run_strategy as cli
    from src.cli.run_strategy import StrategyRunResult
    from src.execution.strategy import run_strategy_from_argv

    def fake_runtime_config(config, cli_overrides=None, cli_strict=False):
        return SimpleNamespace(
            execution=SimpleNamespace(to_dict=lambda: {"enabled": False}),
            strict_mode=SimpleNamespace(enabled=cli_strict),
        )

    def fake_run_strategy_experiment(strategy_name, **kwargs):
        return StrategyRunResult(
            strategy_name=strategy_name,
            run_id="momentum_v1_strategy_abc123",
            metrics={"cumulative_return": 0.125, "sharpe_ratio": 1.75},
            experiment_dir=Path("artifacts/strategies/momentum_v1_strategy_abc123"),
            results_df=pd.DataFrame({"strategy_return": [0.01], "equity_curve": [1.01]}),
        )

    monkeypatch.setattr(cli, "get_strategy_config", lambda strategy_name, path=None: {"dataset": "features_daily"})
    monkeypatch.setattr(cli, "resolve_runtime_config", fake_runtime_config)
    monkeypatch.setattr(cli, "run_strategy_experiment", fake_run_strategy_experiment)

    argv = ["--strategy", "momentum_v1", "--start", "2025-01-01", "--end", "2025-02-01"]
    api_result = run_strategy_from_argv(argv)
    cli_result = cli.run_cli(argv)

    _assert_contract_shape(
        api_result,
        required_output_keys={"manifest_json", "metrics_json", "qa_summary_json", "equity_curve_csv"},
    )
    api_contract = _api_contract(api_result)
    api_contract.pop("shape")
    assert api_contract == _strategy_contract_from_cli(cli_result)


def _alpha_evaluation_contract_from_cli(raw_result: object) -> dict[str, object]:
    artifact_dir = raw_result.artifact_dir
    return {
        "workflow": "alpha_evaluation",
        "run_id": raw_result.run_id,
        "name": raw_result.alpha_name,
        "artifact_leaf": artifact_dir.name,
        "manifest_leaf": "manifest.json",
        "metrics": dict(raw_result.evaluation_result.summary),
        "output_files": {
            "alpha_metrics_json": "alpha_metrics.json",
            "manifest_json": "manifest.json",
            "predictions_parquet": "predictions.parquet",
            "signals_parquet": "signals.parquet",
            "sleeve_metrics_json": "sleeve_metrics.json",
        },
        "extra": {},
    }


def test_alpha_evaluation_cli_api_parity_uses_metrics_and_artifact_keys(monkeypatch) -> None:
    from src.cli import run_alpha_evaluation as cli
    from src.execution.alpha import run_alpha_evaluation_from_argv

    artifact_dir = Path("artifacts/alpha/cs_linear_ret_1d_alpha_eval_abc123")

    def fake_resolve_cli_config(args):
        return {
            "alpha_model": args.alpha_model,
            "dataset": args.dataset,
            "target_column": args.target_column,
            "price_column": args.price_column,
        }

    def fake_run_resolved_config(config):
        return SimpleNamespace(
            alpha_name=config["alpha_model"],
            run_id="cs_linear_ret_1d_alpha_eval_abc123",
            artifact_dir=artifact_dir,
            evaluation_result=SimpleNamespace(
                summary={"mean_ic": 0.21, "ic_ir": 1.4, "n_periods": 12}
            ),
        )

    monkeypatch.setattr(cli, "resolve_cli_config", fake_resolve_cli_config)
    monkeypatch.setattr(cli, "run_resolved_config", fake_run_resolved_config)

    argv = [
        "--alpha-model",
        "cs_linear_ret_1d",
        "--dataset",
        "features_daily",
        "--target-column",
        "target_ret_1d",
        "--price-column",
        "close",
    ]
    api_result = run_alpha_evaluation_from_argv(argv)
    cli_result = cli.run_cli(argv)

    _assert_contract_shape(
        api_result,
        required_output_keys={"manifest_json", "alpha_metrics_json", "predictions_parquet"},
    )
    api_contract = _api_contract(api_result)
    api_contract.pop("shape")
    assert api_contract == _alpha_evaluation_contract_from_cli(cli_result)


def _portfolio_contract_from_cli(raw_result: object) -> dict[str, object]:
    artifact_dir = raw_result.experiment_dir
    return {
        "workflow": "portfolio",
        "run_id": raw_result.run_id,
        "name": raw_result.portfolio_name,
        "artifact_leaf": artifact_dir.name,
        "manifest_leaf": "manifest.json",
        "metrics": dict(raw_result.metrics),
        "output_files": {
            "manifest_json": "manifest.json",
            "metrics_json": "metrics.json",
            "portfolio_equity_curve_csv": "portfolio_equity_curve.csv",
            "portfolio_returns_csv": "portfolio_returns.csv",
            "qa_summary_json": "qa_summary.json",
            "weights_csv": "weights.csv",
        },
        "extra": {
            "allocator_name": raw_result.allocator_name,
            "component_count": raw_result.component_count,
            "timeframe": raw_result.timeframe,
        },
    }


def test_portfolio_cli_api_parity_uses_metrics_components_and_artifacts(monkeypatch) -> None:
    from src.cli import run_portfolio as cli
    from src.cli.run_portfolio import PortfolioRunResult
    from src.execution import portfolio as execution_portfolio
    from src.execution.portfolio import run_portfolio_from_argv

    def fake_run_portfolio_resolved(**kwargs):
        return PortfolioRunResult(
            portfolio_name=kwargs["portfolio_name"],
            run_id="core_portfolio_abc123",
            allocator_name="equal_weight",
            timeframe=kwargs["timeframe"],
            component_count=2,
            metrics={
                "total_return": 0.04,
                "gross_total_return": 0.04,
                "execution_drag_total_return": 0.0,
                "total_execution_friction": 0.0,
                "sharpe_ratio": 1.1,
                "realized_volatility": 0.08,
                "max_drawdown": -0.02,
                "value_at_risk": -0.01,
                "conditional_value_at_risk": -0.015,
            },
            experiment_dir=Path("artifacts/portfolios/core_portfolio_abc123"),
            portfolio_output=pd.DataFrame(),
            config={"allocator": "equal_weight"},
            components=[
                {"strategy_name": "alpha_v1", "run_id": "run-alpha"},
                {"strategy_name": "beta_v1", "run_id": "run-beta"},
            ],
        )

    monkeypatch.setattr(execution_portfolio, "_run_portfolio_resolved", fake_run_portfolio_resolved)

    argv = ["--portfolio-name", "core_portfolio", "--run-ids", "run-alpha", "run-beta", "--timeframe", "1D"]
    api_result = run_portfolio_from_argv(argv)
    cli_result = cli.run_cli(argv)

    _assert_contract_shape(
        api_result,
        required_output_keys={
            "manifest_json",
            "metrics_json",
            "portfolio_returns_csv",
            "portfolio_equity_curve_csv",
            "weights_csv",
            "qa_summary_json",
        },
    )
    api_contract = _api_contract(api_result)
    api_contract.pop("shape")
    assert api_contract == _portfolio_contract_from_cli(cli_result)


def _pipeline_contract_from_cli(raw_result: object) -> dict[str, object]:
    return {
        "workflow": "pipeline",
        "run_id": raw_result.pipeline_run_id,
        "name": raw_result.pipeline_id,
        "artifact_leaf": raw_result.artifact_dir.name,
        "manifest_leaf": "manifest.json",
        "metrics": {
            "status": raw_result.status,
            "steps_executed": len(raw_result.step_results),
            "duration_seconds": 4.0,
            "status_counts": {"completed": 2, "failed": 0, "reused": 0, "skipped": 0},
            "row_counts": {},
        },
        "output_files": {
            "lineage_json": "lineage.json",
            "manifest_json": "manifest.json",
            "pipeline_metrics_json": "pipeline_metrics.json",
            "state_json": "state.json",
        },
        "extra": {
            "execution_order": list(raw_result.execution_order),
            "state": {},
            "state_path": raw_result.state_path.as_posix(),
            "status": raw_result.status,
        },
    }


def test_pipeline_cli_api_parity_uses_manifest_metrics_lineage_and_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.cli import run_pipeline as cli
    from src.execution.pipeline import run_pipeline_from_argv

    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(
        """
id: parity_pipeline
steps:
  - id: prepare
    adapter: python_module
    module: src.pipeline.testing
    argv: ["--stage", "prepare"]
  - id: evaluate
    depends_on: [prepare]
    adapter: python_module
    module: src.pipeline.testing
    argv: ["--stage", "evaluate"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    argv = ["--config", str(config_path)]
    api_result = run_pipeline_from_argv(argv)
    cli_result = cli.run_cli(argv)

    _assert_contract_shape(
        api_result,
        required_output_keys={"manifest_json", "pipeline_metrics_json", "lineage_json", "state_json"},
    )
    api_contract = _api_contract(api_result)
    api_contract.pop("shape")
    assert api_contract == _pipeline_contract_from_cli(cli_result)

    manifest = json.loads(api_result.output_paths["manifest_json"].read_text(encoding="utf-8"))
    assert [step["step_id"] for step in manifest["steps"]] == ["prepare", "evaluate"]


def test_docs_path_lint_cli_api_parity_allows_cli_exit_policy_difference(tmp_path: Path) -> None:
    from src.cli import run_docs_path_lint as cli
    from src.execution.validation import run_docs_path_lint_from_argv

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "safe.md").write_text("portable relative reference only\n", encoding="utf-8")

    api_output = tmp_path / "api_docs_lint.json"
    cli_output = tmp_path / "cli_docs_lint.json"
    api_result = run_docs_path_lint_from_argv(["--repo-root", str(tmp_path), "--output", str(api_output)])
    cli_report = cli.run_cli(["--repo-root", str(tmp_path), "--output", str(cli_output)])

    _assert_contract_shape(api_result, required_output_keys={"report_json"})
    assert api_result.workflow == "docs_path_lint"
    assert api_result.metrics == cli_report
    assert api_result.extra["finding_count"] == cli_report["finding_count"]
    assert api_result.output_paths["report_json"].name == "api_docs_lint.json"
    assert cli_output.exists()


def test_docs_path_lint_api_returns_result_when_cli_would_exit_nonzero(tmp_path: Path) -> None:
    from src.cli import run_docs_path_lint as cli
    from src.execution.validation import run_docs_path_lint

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "leak.md").write_text("Reference: C:/Users/example/local/path/file.md\n", encoding="utf-8")

    api_result = run_docs_path_lint(repo_root=tmp_path, output=tmp_path / "api_failed_docs_lint.json")
    assert api_result.metrics["status"] == "failed"
    assert api_result.metrics["finding_count"] >= 1

    try:
        cli.run_cli(["--repo-root", str(tmp_path), "--output", str(tmp_path / "cli_failed_docs_lint.json")])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("docs path lint CLI should exit non-zero for failed validation")


def _benchmark_contract_from_cli(raw_result: object) -> dict[str, object]:
    return {
        "workflow": "benchmark_pack",
        "run_id": raw_result.pack_run_id,
        "name": raw_result.pack_run_id,
        "artifact_leaf": raw_result.output_root.name,
        "manifest_leaf": "manifest.json",
        "metrics": dict(raw_result.summary),
        "output_files": {
            "batch_plan_csv": "batch_plan.csv",
            "batch_plan_json": "batch_plan.json",
            "benchmark_matrix_csv": "benchmark_matrix.csv",
            "benchmark_matrix_summary": "benchmark_matrix.json",
            "checkpoint_json": "checkpoint.json",
            "comparison_json": "inventory_comparison.json",
            "config_json": "benchmark_pack_config.json",
            "dataset_summary_json": "dataset_summary.json",
            "inventory_json": "inventory.json",
            "manifest_json": "manifest.json",
            "summary_json": "summary.json",
        },
        "extra": {
            "comparison": raw_result.comparison,
            "pack_id": raw_result.pack_id,
            "status": raw_result.status,
        },
    }


def test_benchmark_pack_cli_api_parity_uses_summary_inventory_and_matrix_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.cli import run_benchmark_pack as cli
    from src.execution.benchmark import run_benchmark_pack_from_argv

    output_root = tmp_path / "benchmark_pack"
    comparison_path = output_root / "comparisons" / "inventory_comparison.json"

    raw_result = SimpleNamespace(
        pack_id="m22_scale_repro",
        pack_run_id="m22_scale_repro_abc123",
        status="completed",
        output_root=output_root,
        checkpoint_path=output_root / "checkpoint.json",
        manifest_path=output_root / "manifest.json",
        summary_path=output_root / "summary.json",
        inventory_path=output_root / "inventory.json",
        batch_plan_path=output_root / "batch_plan.json",
        batch_plan_csv_path=output_root / "batch_plan.csv",
        benchmark_matrix_csv_path=output_root / "benchmark_matrix.csv",
        benchmark_matrix_summary_path=output_root / "benchmark_matrix.json",
        comparison_path=comparison_path,
        summary={
            "run_type": "benchmark_pack",
            "status": "completed",
            "batch_count": 1,
            "batch_status_counts": {"completed": 1},
            "scenario_count": 2,
        },
        comparison={"matches": True},
    )

    monkeypatch.setattr("src.config.benchmark_pack.load_benchmark_pack_config", lambda path: object())
    monkeypatch.setattr(
        "src.research.benchmark_pack.run_benchmark_pack",
        lambda config, output_root, compare_to, stop_after_batches: raw_result,
    )

    argv = [
        "--config",
        "configs/benchmark_packs/m22_scale_repro.yml",
        "--output-root",
        str(output_root),
        "--compare-to",
        str(tmp_path / "reference_inventory.json"),
    ]
    api_result = run_benchmark_pack_from_argv(argv)
    cli_result = cli.run_cli(argv)

    _assert_contract_shape(
        api_result,
        required_output_keys={
            "summary_json",
            "manifest_json",
            "checkpoint_json",
            "inventory_json",
            "batch_plan_json",
            "batch_plan_csv",
            "benchmark_matrix_csv",
            "benchmark_matrix_summary",
            "comparison_json",
        },
    )
    api_contract = _api_contract(api_result)
    api_contract.pop("shape")
    assert api_contract == _benchmark_contract_from_cli(cli_result)
