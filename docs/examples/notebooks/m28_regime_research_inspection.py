from __future__ import annotations

from pathlib import Path
from typing import Any


REGIME_BENCHMARK_CONFIG = "configs/regime_benchmark_packs/m26_regime_policy_benchmark.yml"
REGIME_STRESS_CONFIG = "configs/regime_stress_tests/m26_adaptive_policy_stress.yml"
REGIME_OUTPUT_ROOT = "artifacts/notebooks/m28_regime_research_inspection/attempt_001"


def run_regime_benchmark_notebook_cell(
    output_root: str | Path = REGIME_OUTPUT_ROOT,
) -> tuple[Any, dict[str, Any]]:
    """Run a regime benchmark through the existing execution wrapper."""

    from src.execution.regime_benchmark import run_regime_benchmark_pack

    result = run_regime_benchmark_pack(
        REGIME_BENCHMARK_CONFIG,
        output_root=output_root,
    )

    return result, {
        "notebook_summary": result.notebook_summary(),
        "output_keys": result.output_keys(),
        "manifest": result.load_manifest(),
        "benchmark_summary": result.load_output_json("benchmark_summary_json"),
        "benchmark_matrix": result.output_path("benchmark_matrix_csv", must_exist=True),
        "policy_comparison": result.output_path("policy_comparison_csv", must_exist=True),
    }


def run_regime_policy_stress_notebook_cell(
    source_review_pack: str | Path,
    output_root: str | Path = REGIME_OUTPUT_ROOT,
) -> tuple[Any, dict[str, Any]]:
    """Run policy stress tests with explicit artifact inputs and output root."""

    from src.execution.regime_policy_stress_tests import run_regime_policy_stress_tests

    result = run_regime_policy_stress_tests(
        config_path=REGIME_STRESS_CONFIG,
        source_review_pack=source_review_pack,
        output_root=output_root,
    )

    return result, {
        "notebook_summary": result.notebook_summary(),
        "output_keys": result.output_keys(),
        "manifest": result.load_manifest(),
        "scenario_summary": result.load_output_json("scenario_summary_json"),
        "policy_stress_summary": result.load_output_json("policy_stress_summary_json"),
        "stress_leaderboard": result.output_path("stress_leaderboard_csv", must_exist=True),
    }


def inspect_regime_execution_result(result: Any) -> dict[str, Any]:
    """Inspect regime artifacts without recreating regime workflow logic."""

    payload: dict[str, Any] = {
        "summary": result.notebook_summary(),
        "manifest": result.load_manifest(),
        "named_outputs": result.output_keys(),
        "artifact_root": result.artifact_dir,
    }
    if result.has_output("benchmark_summary_json"):
        payload["benchmark_summary"] = result.load_output_json("benchmark_summary_json")
    if result.has_output("policy_stress_summary_json"):
        payload["policy_stress_summary"] = result.load_output_json("policy_stress_summary_json")
    return payload


if __name__ == "__main__":
    print("Import this script-backed notebook from a repo-root Python session.")
