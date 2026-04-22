from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution import (  # noqa: E402
    run_alpha,
    run_alpha_evaluation,
    run_benchmark_pack,
    run_docs_path_lint,
    run_deterministic_rerun_validation,
    run_milestone_validation,
    run_pipeline,
    run_portfolio,
    run_research_campaign,
    run_strategy,
)
from src.execution.result import ExecutionResult  # noqa: E402


def summarize(result: ExecutionResult) -> dict[str, Any]:
    """Return the JSON-safe fields notebook users usually inspect first."""

    return result.notebook_summary()


def strategy_example() -> ExecutionResult:
    """Run one strategy from Python and inspect deterministic artifacts."""

    result = run_strategy(
        "momentum_v1",
        start="2022-01-01",
        end="2023-01-01",
        strict=True,
    )

    print(result.run_id)
    print(result.metrics.get("sharpe_ratio"))
    print(result.output_path("metrics_json"))
    print(result.manifest_path)
    print(result.load_metrics_json())
    return result


def alpha_evaluation_example() -> ExecutionResult:
    """Run alpha evaluation from an in-memory config mapping."""

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

    print(result.run_id)
    print(result.metrics.get("mean_ic"))
    print(result.output_path("alpha_metrics_json"))
    print(result.output_path("predictions_parquet"))
    print(result.output_keys())
    return result


def alpha_full_run_example() -> ExecutionResult:
    """Run a built-in alpha and produce evaluation plus sleeve artifacts."""

    result = run_alpha(
        "cs_linear_ret_1d",
        mode="full",
        start="2022-01-01",
        end="2023-01-01",
    )

    print(result.run_id)
    print(result.extra.get("mode"))
    print(result.output_path("signals_parquet"))
    print(result.output_path("sleeve_metrics_json"))
    return result


def portfolio_example(component_run_ids: list[str]) -> ExecutionResult:
    """Build a portfolio from completed component run ids."""

    result = run_portfolio(
        portfolio_name="core_portfolio",
        run_ids=component_run_ids,
        timeframe="1D",
        strict=True,
    )

    print(result.run_id)
    print(result.metrics.get("total_return"))
    print(result.output_path("weights_csv"))
    print(result.output_path("portfolio_returns_csv"))
    return result


def registry_backed_portfolio_example() -> ExecutionResult:
    """Build a portfolio from the latest matching registry entries."""

    result = run_portfolio(
        portfolio_config_path="configs/portfolios.yml",
        portfolio_name="momentum_meanrev_equal",
        from_registry=True,
        timeframe="1D",
    )

    print(result.run_id)
    print(result.extra.get("allocator_name"))
    print(result.manifest_path)
    return result


def pipeline_example() -> ExecutionResult:
    """Run a YAML pipeline spec from Python and inspect orchestration artifacts."""

    result = run_pipeline("configs/test_pipeline.yml")

    print(result.run_id)
    print(result.output_path("lineage_json"))
    print(result.output_path("state_json"))
    print(result.extra.get("execution_order"))
    return result


def research_campaign_example() -> ExecutionResult:
    """Run a campaign config through the same staged workflow used by the CLI."""

    result = run_research_campaign(config_path="configs/research_campaign.yml")

    print(result.workflow)
    print(result.run_id)
    print(result.output_path("checkpoint_json"))
    print(result.load_summary_json())
    print(result.extra.get("stage_statuses"))
    return result


def docs_path_lint_example() -> ExecutionResult:
    """Run guarded docs/path lint from Python and inspect the JSON report."""

    result = run_docs_path_lint(output="artifacts/qa/docs_path_lint.json")

    print(result.metrics.get("status"))
    print(result.metrics.get("finding_count"))
    print(result.output_path("report_json"))
    print(result.load_metrics_json("report_json"))
    return result


def deterministic_rerun_validation_example() -> ExecutionResult:
    """Run canonical deterministic rerun validation from Python."""

    result = run_deterministic_rerun_validation(
        workdir="artifacts/qa/rerun_workdir",
        output="artifacts/qa/deterministic_rerun.json",
    )

    print(result.metrics.get("status"))
    print(result.metrics.get("pass_count"))
    print(result.output_path("report_json"))
    return result


def milestone_validation_example() -> ExecutionResult:
    """Build the milestone validation bundle through the notebook API."""

    result = run_milestone_validation(
        bundle_dir="artifacts/qa/milestone_validation_bundle",
        include_full_pytest=False,
    )

    print(result.metrics.get("status"))
    print(result.output_path("summary_json"))
    print(result.output_path("docs_path_lint_json"))
    print(result.output_path("deterministic_rerun_json"))
    return result


def benchmark_pack_example() -> ExecutionResult:
    """Run the scale/repro benchmark pack and inspect generated artifacts."""

    result = run_benchmark_pack(
        "configs/benchmark_packs/m22_scale_repro.yml",
        output_root="artifacts/benchmark_packs/m22_scale_repro",
    )

    print(result.metrics.get("status"))
    print(result.metrics.get("scenario_count"))
    print(result.output_path("summary_json"))
    print(result.output_path("inventory_json"))
    if result.has_output("comparison_json"):
        print(result.load_comparison_json())
    return result
