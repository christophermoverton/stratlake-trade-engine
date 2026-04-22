from __future__ import annotations

import json
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
    run_benchmark_pack,
    run_campaign,
    run_docs_path_lint,
    run_deterministic_rerun_validation,
    run_milestone_validation,
    run_pipeline,
    run_portfolio,
    run_research_campaign,
    run_strategy,
)
from src.execution.result import ExecutionResult, load_json_artifact


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


def test_execution_result_helpers_load_artifacts_and_summarize_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run-123"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "manifest.json"
    metrics_path = artifact_dir / "metrics.json"
    summary_path = artifact_dir / "summary.json"
    comparison_path = artifact_dir / "comparison.json"
    manifest_payload = {"run_id": "run-123", "artifact_paths": {"metrics": "metrics.json"}}
    metrics_payload = {"sharpe_ratio": 1.2, "total_return": 0.1}
    summary_payload = {"status": "completed", "row_count": 2}
    comparison_payload = {"matches": True}
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison_payload), encoding="utf-8")

    result = ExecutionResult(
        workflow="strategy_comparison",
        run_id="run-123",
        name="comparison",
        artifact_dir=artifact_dir,
        metrics={"sharpe_ratio": 1.2},
        manifest_path=manifest_path,
        output_paths={
            "summary_json": summary_path,
            "metrics_json": metrics_path,
            "comparison_json": comparison_path,
        },
        extra={"selection_mode": "explicit"},
    )

    assert result.output_keys() == ("comparison_json", "metrics_json", "summary_json")
    assert result.has_output("metrics_json") is True
    assert result.output_path("metrics_json", must_exist=True) == metrics_path
    assert result.artifact_path("metrics.json", must_exist=True) == metrics_path
    assert result.load_manifest() == manifest_payload
    assert load_json_artifact(metrics_path) == metrics_payload
    assert result.load_metrics_json() == metrics_payload
    assert result.load_output_json("summary_json") == summary_payload
    assert result.load_summary_json() == summary_payload
    assert result.load_comparison_json() == comparison_payload
    assert result.notebook_summary() == {
        "workflow": "strategy_comparison",
        "run_id": "run-123",
        "name": "comparison",
        "artifact_dir": artifact_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "metrics": {"sharpe_ratio": 1.2},
        "output_keys": ["comparison_json", "metrics_json", "summary_json"],
        "extra": {"selection_mode": "explicit"},
    }


def test_execution_result_helpers_raise_clearly_for_missing_optional_outputs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run-123"
    artifact_dir.mkdir()
    result = ExecutionResult(
        workflow="strategy",
        run_id="run-123",
        name="momentum_v1",
        artifact_dir=artifact_dir,
        output_paths={"metrics_json": artifact_dir / "missing_metrics.json"},
    )

    try:
        result.output_path("summary_json")
    except KeyError as exc:
        assert "available outputs: metrics_json" in str(exc)
    else:
        raise AssertionError("Expected missing output key to raise KeyError.")

    try:
        result.load_metrics_json()
    except FileNotFoundError as exc:
        assert "missing_metrics.json" in str(exc)
    else:
        raise AssertionError("Expected missing metrics file to raise FileNotFoundError.")

    try:
        result.load_comparison_json()
    except KeyError as exc:
        assert "comparison_json" in str(exc)
    else:
        raise AssertionError("Expected missing comparison output to raise KeyError.")

    try:
        result.artifact_path("..", "outside.json")
    except ValueError as exc:
        assert "escapes artifact_dir" in str(exc)
    else:
        raise AssertionError("Expected artifact path escape to raise ValueError.")


def test_execution_result_helpers_are_side_effect_free(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run-123"
    artifact_dir.mkdir()
    manifest_path = artifact_dir / "manifest.json"
    metrics_path = artifact_dir / "metrics.json"
    manifest_path.write_text('{"run_id": "run-123"}', encoding="utf-8")
    metrics_path.write_text('{"sharpe_ratio": 1.2}', encoding="utf-8")
    before = {path.name: path.stat().st_mtime_ns for path in artifact_dir.iterdir()}
    result = ExecutionResult(
        workflow="strategy",
        run_id="run-123",
        name="momentum_v1",
        artifact_dir=artifact_dir,
        metrics={"sharpe_ratio": 1.2},
        manifest_path=manifest_path,
        output_paths={"metrics_json": metrics_path},
    )

    assert result.load_manifest()["run_id"] == "run-123"
    assert result.load_metrics_json()["sharpe_ratio"] == 1.2
    assert result.to_dict()["metrics"] == {"sharpe_ratio": 1.2}

    after = {path.name: path.stat().st_mtime_ns for path in artifact_dir.iterdir()}
    assert after == before
    assert {path.name for path in artifact_dir.iterdir()} == {"manifest.json", "metrics.json"}


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


def test_run_docs_path_lint_api_writes_report_and_exposes_findings(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "leak.md").write_text(
        "Reference: C:/Users/example/local/path/file.md\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "artifacts" / "qa" / "docs_path_lint.json"

    result = run_docs_path_lint(repo_root=tmp_path, output=output_path)

    assert result.workflow == "docs_path_lint"
    assert result.run_id == "docs_path_lint"
    assert result.artifact_dir == output_path.parent
    assert result.output_paths["report_json"] == output_path
    assert result.metrics["run_type"] == "docs_path_lint"
    assert result.metrics["status"] == "failed"
    assert result.metrics["finding_count"] >= 1
    assert result.extra["finding_count"] == result.metrics["finding_count"]
    assert output_path.exists()
    assert result.raw_result == result.metrics


def test_docs_path_lint_cli_delegates_to_execution_api(monkeypatch, tmp_path: Path) -> None:
    from src.cli.run_docs_path_lint import run_cli

    calls: dict[str, object] = {}
    report_path = tmp_path / "docs_path_lint.json"
    report = {
        "run_type": "docs_path_lint",
        "schema_version": 1,
        "status": "passed",
        "guarded_file_count": 2,
        "finding_count": 0,
        "findings": [],
    }

    def fake_run_from_cli_args(args):
        calls["repo_root"] = args.repo_root
        calls["output"] = args.output
        return ExecutionResult(
            workflow="docs_path_lint",
            run_id="docs_path_lint",
            name="docs_path_lint",
            artifact_dir=report_path.parent,
            metrics=dict(report),
            output_paths={"report_json": report_path},
            raw_result=report,
        )

    monkeypatch.setattr(
        "src.execution.validation.run_docs_path_lint_from_cli_args",
        fake_run_from_cli_args,
    )

    returned = run_cli(["--repo-root", str(tmp_path), "--output", str(report_path)])

    assert returned == report
    assert calls == {"repo_root": str(tmp_path), "output": str(report_path)}


def test_run_deterministic_rerun_validation_api_exposes_report_path(monkeypatch, tmp_path: Path) -> None:
    report = {
        "run_type": "deterministic_rerun_validation",
        "schema_version": 1,
        "status": "passed",
        "target_count": 1,
        "pass_count": 1,
        "targets": [],
    }

    monkeypatch.setattr(
        "src.validation.deterministic_rerun.run_deterministic_rerun_validation",
        lambda repo_root, output_root: report,
    )

    output_path = tmp_path / "deterministic_rerun.json"
    result = run_deterministic_rerun_validation(
        repo_root=tmp_path,
        workdir=tmp_path / "workdir",
        output=output_path,
    )

    assert result.workflow == "deterministic_rerun_validation"
    assert result.output_paths["report_json"] == output_path
    assert result.metrics["run_type"] == "deterministic_rerun_validation"
    assert result.extra == {"target_count": 1, "pass_count": 1}
    assert output_path.exists()


def test_run_milestone_validation_api_exposes_bundle_artifacts(monkeypatch, tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    summary = {
        "run_type": "milestone_validation_bundle",
        "schema_version": 1,
        "status": "passed",
        "started_at_utc": "2026-04-22T00:00:00Z",
        "finished_at_utc": "2026-04-22T00:00:01Z",
        "bundle_dir": bundle_dir.as_posix(),
        "checks": {
            "docs_path_lint": {
                "status": "passed",
                "report_path": "checks/docs_path_lint.json",
                "finding_count": 0,
                "guarded_file_count": 12,
            },
            "deterministic_rerun": {
                "status": "passed",
                "report_path": "checks/deterministic_rerun.json",
                "target_count": 3,
                "pass_count": 3,
            },
            "commands": [],
        },
        "pytest_targets": ["tests/test_docs_path_portability.py"],
        "include_full_pytest": False,
    }

    monkeypatch.setattr(
        "src.validation.milestone_bundle.build_milestone_validation_bundle",
        lambda repo_root, bundle_dir, include_full_pytest: summary,
    )

    result = run_milestone_validation(repo_root=tmp_path, bundle_dir=bundle_dir)

    assert result.workflow == "milestone_validation"
    assert result.artifact_dir == bundle_dir
    assert result.output_paths["summary_json"] == bundle_dir / "summary.json"
    assert result.output_paths["docs_path_lint_json"] == bundle_dir / "checks" / "docs_path_lint.json"
    assert result.output_paths["deterministic_rerun_json"] == bundle_dir / "checks" / "deterministic_rerun.json"
    assert result.metrics["run_type"] == "milestone_validation_bundle"
    assert result.extra["checks"] == summary["checks"]


def test_run_benchmark_pack_api_exposes_machine_readable_artifacts(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "benchmark_pack"
    raw_result = type(
        "BenchmarkPackRunResult",
        (),
        {
            "pack_id": "m22_scale_repro",
            "pack_run_id": "m22_scale_repro_123",
            "status": "completed",
            "output_root": output_root,
            "checkpoint_path": output_root / "checkpoint.json",
            "manifest_path": output_root / "manifest.json",
            "summary_path": output_root / "summary.json",
            "inventory_path": output_root / "inventory.json",
            "batch_plan_path": output_root / "batch_plan.json",
            "batch_plan_csv_path": output_root / "batch_plan.csv",
            "benchmark_matrix_csv_path": output_root / "benchmark_matrix.csv",
            "benchmark_matrix_summary_path": output_root / "benchmark_matrix.json",
            "comparison_path": output_root / "comparisons" / "inventory_comparison.json",
            "summary": {
                "run_type": "benchmark_pack",
                "status": "completed",
                "batch_count": 1,
                "scenario_count": 2,
            },
            "comparison": {"matches": True},
        },
    )()

    monkeypatch.setattr("src.config.benchmark_pack.load_benchmark_pack_config", lambda path: object())
    monkeypatch.setattr(
        "src.research.benchmark_pack.run_benchmark_pack",
        lambda config, output_root, compare_to, stop_after_batches: raw_result,
    )

    result = run_benchmark_pack(
        "configs/benchmark_packs/m22_scale_repro.yml",
        output_root=output_root,
        compare_to=tmp_path / "reference_inventory.json",
    )

    assert result.workflow == "benchmark_pack"
    assert result.run_id == "m22_scale_repro_123"
    assert result.artifact_dir == output_root
    assert result.manifest_path == output_root / "manifest.json"
    assert result.output_paths["summary_json"] == output_root / "summary.json"
    assert result.output_paths["batch_plan_csv"] == output_root / "batch_plan.csv"
    assert result.output_paths["benchmark_matrix_csv"] == output_root / "benchmark_matrix.csv"
    assert result.output_paths["comparison_json"] == output_root / "comparisons" / "inventory_comparison.json"
    assert result.metrics["run_type"] == "benchmark_pack"
    assert result.extra == {
        "pack_id": "m22_scale_repro",
        "status": "completed",
        "comparison": {"matches": True},
    }


def test_notebook_execution_api_example_is_import_safe() -> None:
    namespace = runpy.run_path("docs/examples/notebook_execution_api_examples.py")

    assert callable(namespace["inspect_result"])
    assert callable(namespace["strategy_notebook_cell"])
    assert callable(namespace["strategy_comparison_notebook_cell"])
    assert callable(namespace["alpha_evaluation_notebook_cell"])
    assert callable(namespace["alpha_full_run_notebook_cell"])
    assert callable(namespace["portfolio_notebook_cell"])
    assert callable(namespace["registry_backed_portfolio_notebook_cell"])
    assert callable(namespace["pipeline_notebook_cell"])
    assert callable(namespace["research_campaign_notebook_cell"])
    assert callable(namespace["scenario_orchestration_notebook_cell"])
    assert callable(namespace["validation_notebook_cell"])
    assert callable(namespace["benchmark_pack_notebook_cell"])
    assert callable(namespace["explicit_artifact_load_notebook_cell"])


def test_notebook_execution_api_example_uses_public_imports_and_helpers() -> None:
    source = Path("docs/examples/notebook_execution_api_examples.py").read_text(encoding="utf-8")

    assert "from src.execution import" in source
    assert "from src.execution.result import" not in source
    for helper_name in (
        "output_keys(",
        "output_path(",
        "load_manifest(",
        "load_metrics_json(",
        "load_summary_json(",
        "load_comparison_json(",
        "notebook_summary(",
    ):
        assert helper_name in source


def test_q1_2026_notebook_example_is_valid_and_uses_execution_api() -> None:
    notebook_path = Path("docs/examples/ml_cross_sectional_xgb_2026_q1_notebook.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]
    assert all(cell["outputs"] == [] for cell in notebook["cells"] if cell["cell_type"] == "code")

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "from src.execution import run_alpha, run_docs_path_lint, run_portfolio" in source
    assert "run_alpha(" in source
    assert "run_portfolio(" in source
    assert "run_docs_path_lint(" in source
    assert "notebook_summary(" in source
    assert "output_keys(" in source
    assert "output_path(" in source
    assert "load_manifest(" in source
    assert "load_metrics_json(" in source
    assert "load_summary_json(" in source
    assert "subprocess" not in source
    assert "!python" not in source
    assert "features_daily" in source
    assert "configs/alphas_2026_q1.yml" in source
