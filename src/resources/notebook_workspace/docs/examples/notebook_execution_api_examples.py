from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution import (  # noqa: E402
    ExecutionResult,
    compare_strategies,
    load_json_artifact,
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


def inspect_result(result: ExecutionResult) -> dict[str, Any]:
    """Return the notebook-safe inspection payload users usually want first."""

    inspection: dict[str, Any] = {
        "summary": result.notebook_summary(),
        "output_keys": result.output_keys(),
    }
    if result.manifest_path is not None and result.manifest_path.exists():
        inspection["manifest"] = result.load_manifest()
    if any(result.has_output(key) for key in ("metrics_json", "alpha_metrics_json", "report_json")):
        inspection["metrics_artifact"] = result.load_metrics_json()
    if any(result.has_output(key) for key in ("summary_json", "qa_summary_json")):
        inspection["summary_artifact"] = result.load_summary_json()
    if result.has_output("comparison_json") and result.output_path("comparison_json").exists():
        inspection["comparison_artifact"] = result.load_comparison_json()
    return inspection


def strategy_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run one deterministic strategy and inspect its manifest and metrics."""

    result = run_strategy(
        "momentum_v1",
        start="2022-01-01",
        end="2023-01-01",
        strict=True,
    )

    metrics = result.load_metrics_json()
    manifest = result.load_manifest()
    equity_curve_path = result.output_path("equity_curve_csv", must_exist=True)
    return result, {
        "notebook_summary": result.notebook_summary(),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "manifest_run_id": manifest.get("run_id"),
        "equity_curve_path": equity_curve_path,
    }


def strategy_comparison_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Compare strategy outputs from Python and inspect the leaderboard summary."""

    result = compare_strategies(
        ["momentum_v1", "mean_reversion_v1"],
        start="2022-01-01",
        end="2023-01-01",
        metric="sharpe_ratio",
    )

    summary = result.load_summary_json()
    leaderboard_path = result.output_path("leaderboard_csv", must_exist=True)
    return result, {
        "notebook_summary": result.notebook_summary(),
        "leaderboard_path": leaderboard_path,
        "metric": result.extra.get("metric"),
        "row_count": summary.get("row_count", result.extra.get("row_count")),
    }


def alpha_evaluation_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Evaluate one alpha from an in-memory config mapping."""

    result = run_alpha_evaluation(
        config={
            "alpha_name": "cs_linear_ret_1d",
            "dataset": "features_daily",
            "target_column": "target_ret_1d",
            "price_column": "close",
            "start": "2022-01-01",
            "end": "2023-01-01",
        }
    )

    alpha_metrics = result.load_metrics_json("alpha_metrics_json")
    return result, {
        "notebook_summary": result.notebook_summary(),
        "mean_ic": alpha_metrics.get("mean_ic"),
        "predictions_path": result.output_path("predictions_parquet", must_exist=True),
        "manifest": result.load_manifest(),
    }


def alpha_full_run_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run the full alpha surface and inspect signal and sleeve artifacts."""

    result = run_alpha(
        "cs_linear_ret_1d",
        mode="full",
        start="2022-01-01",
        end="2023-01-01",
    )

    sleeve_metrics = result.load_metrics_json("sleeve_metrics_json")
    return result, {
        "notebook_summary": result.notebook_summary(),
        "mode": result.extra.get("mode"),
        "signals_path": result.output_path("signals_parquet", must_exist=True),
        "sleeve_metrics": sleeve_metrics,
    }


def portfolio_notebook_cell(component_run_ids: list[str]) -> tuple[ExecutionResult, dict[str, Any]]:
    """Build a portfolio from completed strategy or alpha-sleeve component runs."""

    result = run_portfolio(
        portfolio_name="core_portfolio",
        run_ids=component_run_ids,
        timeframe="1D",
        strict=True,
    )

    portfolio_metrics = result.load_metrics_json()
    return result, {
        "notebook_summary": result.notebook_summary(),
        "total_return": portfolio_metrics.get("total_return"),
        "weights_path": result.output_path("weights_csv", must_exist=True),
        "returns_path": result.output_path("portfolio_returns_csv", must_exist=True),
    }


def registry_backed_portfolio_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Build a portfolio from latest matching registry entries."""

    result = run_portfolio(
        portfolio_config_path="configs/portfolios.yml",
        portfolio_name="momentum_meanrev_equal",
        from_registry=True,
        timeframe="1D",
    )

    return result, {
        "notebook_summary": result.notebook_summary(),
        "allocator": result.extra.get("allocator_name"),
        "component_count": result.extra.get("component_count"),
        "manifest": result.load_manifest(),
    }


def pipeline_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run a YAML pipeline spec and inspect lineage, metrics, and state handoff."""

    result = run_pipeline("configs/test_pipeline.yml")

    return result, {
        "notebook_summary": result.notebook_summary(),
        "pipeline_metrics": result.load_metrics_json("pipeline_metrics_json"),
        "lineage": result.load_output_json("lineage_json"),
        "state": result.load_output_json("state_json"),
        "execution_order": result.extra.get("execution_order"),
    }


def research_campaign_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run one campaign config and inspect staged orchestration outputs."""

    result = run_research_campaign(config_path="configs/research_campaign.yml")

    return result, {
        "notebook_summary": result.notebook_summary(),
        "campaign_summary": result.load_summary_json(),
        "checkpoint": result.load_output_json("checkpoint_json"),
        "stage_statuses": result.extra.get("stage_statuses"),
    }


def scenario_orchestration_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run a scenario-enabled campaign and inspect orchestration artifacts."""

    result = run_campaign(
        config_path="configs/research_campaign.yml",
        overrides={
            "scenarios": {
                "enabled": True,
                "max_scenarios": 4,
                "matrix": [
                    {
                        "name": "top_k",
                        "path": "comparison.top_k",
                        "values": [5, 10],
                    }
                ],
                "include": [
                    {
                        "scenario_id": "review_only",
                        "overrides": {"review": {"filters": {"run_types": ["alpha_evaluation"]}}},
                    }
                ],
            }
        },
    )

    return result, {
        "notebook_summary": result.notebook_summary(),
        "orchestration_summary": result.load_summary_json(),
        "scenario_catalog": result.load_output_json("scenario_catalog_json"),
        "scenario_matrix_path": result.output_path("scenario_matrix_csv", must_exist=True),
        "scenario_run_ids": result.extra.get("scenario_run_ids"),
    }


def validation_notebook_cell() -> dict[str, tuple[ExecutionResult, dict[str, Any]]]:
    """Run validation reports from Python and keep failures inspectable."""

    docs_result = run_docs_path_lint(output="artifacts/qa/docs_path_lint.json")
    rerun_result = run_deterministic_rerun_validation(
        workdir="artifacts/qa/rerun_workdir",
        output="artifacts/qa/deterministic_rerun.json",
    )
    bundle_result = run_milestone_validation(
        bundle_dir="artifacts/qa/milestone_validation_bundle",
        include_full_pytest=False,
    )

    return {
        "docs_path_lint": (
            docs_result,
            {
                "status": docs_result.metrics.get("status"),
                "report": docs_result.load_metrics_json("report_json"),
            },
        ),
        "deterministic_rerun": (
            rerun_result,
            {
                "status": rerun_result.metrics.get("status"),
                "report": rerun_result.load_metrics_json("report_json"),
            },
        ),
        "milestone_validation": (
            bundle_result,
            {
                "status": bundle_result.metrics.get("status"),
                "summary": bundle_result.load_summary_json("summary_json"),
                "docs_lint_path": bundle_result.output_path("docs_path_lint_json"),
                "rerun_path": bundle_result.output_path("deterministic_rerun_json"),
            },
        ),
    }


def benchmark_pack_notebook_cell() -> tuple[ExecutionResult, dict[str, Any]]:
    """Run a benchmark pack and inspect summary, inventory, matrix, and comparison."""

    result = run_benchmark_pack(
        "configs/benchmark_packs/m22_scale_repro.yml",
        output_root="artifacts/benchmark_packs/m22_scale_repro",
    )

    inspection: dict[str, Any] = {
        "notebook_summary": result.notebook_summary(),
        "summary": result.load_summary_json("summary_json"),
        "inventory": result.load_output_json("inventory_json"),
        "benchmark_matrix_path": result.output_path("benchmark_matrix_csv", must_exist=True),
    }
    if result.has_output("comparison_json") and result.output_path("comparison_json").exists():
        inspection["comparison"] = result.load_comparison_json()
    return result, inspection


def explicit_artifact_load_notebook_cell(result: ExecutionResult) -> dict[str, Any]:
    """Load a known JSON artifact path explicitly when it is not a named output."""

    manifest = result.load_manifest()
    extra_summary_path = manifest.get("artifact_paths", {}).get("summary")
    if not extra_summary_path:
        return {"manifest": manifest, "extra_summary": None}
    return {
        "manifest": manifest,
        "extra_summary": load_json_artifact(
            result.artifact_path(extra_summary_path, must_exist=True)
        ),
    }
