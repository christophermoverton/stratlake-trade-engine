from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.cli.run_research_campaign import (
    CampaignStageRecord,
    ResearchCampaignOrchestrationResult,
    ResearchCampaignRunResult,
    ScenarioCampaignRunResult,
)
from src.execution import (
    run_alpha,
    run_alpha_evaluation,
    run_campaign,
    run_pipeline,
    run_portfolio,
    run_research_campaign,
    run_strategy,
)
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


def test_run_pipeline_api_exposes_artifact_state_and_lineage_paths(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "pipeline.yml"
    config_path.write_text(
        """
id: notebook_pipeline
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

    result = run_pipeline(config_path)

    assert result.workflow == "pipeline"
    assert result.name == "notebook_pipeline"
    assert result.artifact_dir == tmp_path / "artifacts" / "pipelines" / result.run_id
    assert result.manifest_path == result.artifact_dir / "manifest.json"
    assert result.output_paths["manifest_json"] == result.manifest_path
    assert result.output_paths["pipeline_metrics_json"] == result.artifact_dir / "pipeline_metrics.json"
    assert result.output_paths["lineage_json"] == result.artifact_dir / "lineage.json"
    assert result.output_paths["state_json"] == result.artifact_dir / "state.json"
    assert result.metrics["status"] == "completed"
    assert result.metrics["steps_executed"] == 2
    assert result.metrics["status_counts"]["completed"] == 2
    assert result.extra["execution_order"] == ["prepare", "evaluate"]
    assert result.extra["state"] == {}
    assert result.to_dict()["output_paths"]["lineage_json"].endswith("lineage.json")


def test_run_research_campaign_api_summarizes_campaign_state_and_outputs(monkeypatch, tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign_artifacts" / "research_campaign_123"
    campaign_dir.mkdir(parents=True)
    raw_result = ResearchCampaignRunResult(
        config=SimpleNamespace(to_dict=lambda: {}),
        campaign_run_id="research_campaign_123",
        campaign_artifact_dir=campaign_dir,
        campaign_checkpoint_path=campaign_dir / "checkpoint.json",
        campaign_manifest_path=campaign_dir / "manifest.json",
        campaign_summary_path=campaign_dir / "summary.json",
        campaign_milestone_summary_path=campaign_dir / "milestone_report" / "summary.json",
        campaign_milestone_decision_log_path=campaign_dir / "milestone_report" / "decision_log.json",
        campaign_milestone_manifest_path=campaign_dir / "milestone_report" / "manifest.json",
        campaign_milestone_markdown_path=campaign_dir / "milestone_report" / "report.md",
        preflight_summary_path=campaign_dir / "preflight_summary.json",
        campaign_checkpoint={"stage_states": {"preflight": "completed", "research": "reused"}},
        campaign_manifest={"campaign_run_id": "research_campaign_123"},
        campaign_summary={
            "status": "completed",
            "stage_statuses": {"preflight": "completed", "research": "reused"},
            "selected_run_ids": {"alpha_run_ids": ["alpha_run"]},
            "final_outcomes": {"resumable_stage_names": []},
            "stage_execution": {"research": {"reuse": {"reused": True}}},
            "output_paths": {"alpha_manifest": (tmp_path / "alpha" / "manifest.json").as_posix()},
        },
        preflight_summary={"status": "passed"},
        stage_records=(
            CampaignStageRecord("preflight", "completed", {}),
            CampaignStageRecord("research", "reused", {"reuse_policy": {"action": "reuse"}}),
        ),
        alpha_results=(SimpleNamespace(run_id="alpha_run"),),
        strategy_results=(),
        alpha_comparison_result=None,
        strategy_comparison_result=None,
        candidate_selection_result=None,
        portfolio_result=None,
        candidate_review_result=None,
        review_result=None,
    )

    monkeypatch.setattr("src.execution.orchestration._run_campaign", lambda config: raw_result)

    result = run_research_campaign({"outputs": {"campaign_artifacts_root": campaign_dir.parent.as_posix()}})

    assert result.workflow == "research_campaign"
    assert result.run_id == "research_campaign_123"
    assert result.artifact_dir == campaign_dir
    assert result.manifest_path == campaign_dir / "manifest.json"
    assert result.output_paths["checkpoint_json"] == campaign_dir / "checkpoint.json"
    assert result.output_paths["summary_json"] == campaign_dir / "summary.json"
    assert result.output_paths["alpha_manifest"] == tmp_path / "alpha" / "manifest.json"
    assert result.metrics["status"] == "completed"
    assert result.metrics["stage_counts"] == {"completed": 1, "reused": 1}
    assert result.metrics["reused_stage_count"] == 1
    assert result.extra["stage_statuses"]["research"] == "reused"
    assert result.extra["stage_execution"]["research"]["reuse"]["reused"] is True
    assert result.extra["checkpoint_path"].endswith("checkpoint.json")
    assert result.raw_result is raw_result


def test_run_campaign_api_summarizes_scenario_orchestration(monkeypatch, tmp_path: Path) -> None:
    orchestration_dir = tmp_path / "campaign_artifacts" / "research_campaign_orchestration_123"
    scenario_dir = orchestration_dir / "scenarios" / "scenario_a"
    scenario_result = ResearchCampaignRunResult(
        config=SimpleNamespace(to_dict=lambda: {}),
        campaign_run_id="research_campaign_scenario_a",
        campaign_artifact_dir=scenario_dir,
        campaign_checkpoint_path=scenario_dir / "checkpoint.json",
        campaign_manifest_path=scenario_dir / "manifest.json",
        campaign_summary_path=scenario_dir / "summary.json",
        campaign_milestone_summary_path=None,
        campaign_milestone_decision_log_path=None,
        campaign_milestone_manifest_path=None,
        campaign_milestone_markdown_path=None,
        preflight_summary_path=scenario_dir / "preflight_summary.json",
        campaign_checkpoint={},
        campaign_manifest={},
        campaign_summary={"status": "completed"},
        preflight_summary={"status": "passed"},
        stage_records=(),
        alpha_results=(),
        strategy_results=(),
        alpha_comparison_result=None,
        strategy_comparison_result=None,
        candidate_selection_result=None,
        portfolio_result=None,
        candidate_review_result=None,
        review_result=None,
    )
    raw_result = ResearchCampaignOrchestrationResult(
        config=SimpleNamespace(to_dict=lambda: {}),
        orchestration_run_id="research_campaign_orchestration_123",
        orchestration_artifact_dir=orchestration_dir,
        scenario_catalog_path=orchestration_dir / "scenario_catalog.json",
        orchestration_manifest_path=orchestration_dir / "manifest.json",
        orchestration_summary_path=orchestration_dir / "summary.json",
        scenario_matrix_csv_path=orchestration_dir / "scenario_matrix.csv",
        scenario_matrix_summary_path=orchestration_dir / "scenario_matrix.json",
        expansion_preflight_path=orchestration_dir / "expansion_preflight.json",
        scenario_catalog={"scenarios": []},
        orchestration_manifest={"run_type": "research_campaign_orchestration"},
        orchestration_summary={
            "status": "completed",
            "scenario_status_counts": {"completed": 1},
            "scenarios": [{"scenario_id": "scenario_a", "status": "completed"}],
        },
        expansion_preflight={"status": "passed"},
        scenario_results=(
            ScenarioCampaignRunResult(
                scenario_id="scenario_a",
                description=None,
                source="matrix",
                sweep_values={"lookback": 20},
                fingerprint="abc123",
                result=scenario_result,
            ),
        ),
    )

    monkeypatch.setattr("src.execution.orchestration._run_campaign", lambda config: raw_result)

    result = run_campaign(
        {
            "scenarios": {
                "enabled": True,
                "include": [
                    {
                        "scenario_id": "scenario_a",
                        "overrides": {"dataset_selection": {"evaluation_horizon": 5}},
                    }
                ],
            }
        }
    )

    assert result.workflow == "research_campaign_orchestration"
    assert result.run_id == "research_campaign_orchestration_123"
    assert result.artifact_dir == orchestration_dir
    assert result.manifest_path == orchestration_dir / "manifest.json"
    assert result.output_paths["scenario_catalog_json"] == orchestration_dir / "scenario_catalog.json"
    assert result.output_paths["expansion_preflight_json"] == orchestration_dir / "expansion_preflight.json"
    assert result.metrics == {
        "status": "completed",
        "scenario_count": 1,
        "scenario_status_counts": {"completed": 1},
    }
    assert result.extra["scenario_run_ids"] == ["research_campaign_scenario_a"]
    assert result.extra["scenarios"][0]["scenario_id"] == "scenario_a"


def test_notebook_execution_api_example_is_import_safe() -> None:
    namespace = runpy.run_path("docs/examples/notebook_execution_api_examples.py")

    assert callable(namespace["strategy_example"])
    assert callable(namespace["alpha_evaluation_example"])
    assert callable(namespace["alpha_full_run_example"])
    assert callable(namespace["portfolio_example"])
    assert callable(namespace["registry_backed_portfolio_example"])
    assert callable(namespace["pipeline_example"])
    assert callable(namespace["research_campaign_example"])
