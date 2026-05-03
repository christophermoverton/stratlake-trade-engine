from __future__ import annotations

from pathlib import Path
from typing import Any


BENCHMARK_PACK_CONFIG = "configs/benchmark_packs/m22_scale_repro.yml"
NOTEBOOK_OUTPUT_ROOT = "artifacts/notebooks/m28_benchmark_pack_execution_api/attempt_001"


def run_benchmark_pack_notebook_cell(
    output_root: str | Path = NOTEBOOK_OUTPUT_ROOT,
) -> tuple[Any, dict[str, Any]]:
    """Run the lightweight benchmark pack through `src.execution`."""

    from src.execution import run_benchmark_pack

    result = run_benchmark_pack(
        BENCHMARK_PACK_CONFIG,
        output_root=output_root,
        stop_after_batches=1,
    )

    inspection: dict[str, Any] = {
        "notebook_summary": result.notebook_summary(),
        "output_keys": result.output_keys(),
        "summary": result.load_summary_json("summary_json"),
        "manifest": result.load_manifest(),
        "inventory": result.load_output_json("inventory_json"),
        "benchmark_matrix_path": result.output_path("benchmark_matrix_csv", must_exist=True),
        "status_markers_to_check_when_reusing": (
            "_RUNNING.json",
            "_SUCCESS.json",
            "_FAILED.json",
        ),
    }
    if result.has_output("comparison_json") and result.output_path("comparison_json").exists():
        inspection["comparison"] = result.load_comparison_json()
    return result, inspection


def inspect_benchmark_artifacts(result: Any) -> dict[str, Any]:
    """Inspect benchmark-pack outputs after execution has completed."""

    return {
        "summary": result.load_summary_json("summary_json"),
        "inventory": result.load_output_json("inventory_json"),
        "batch_plan": result.load_output_json("batch_plan_json"),
        "matrix_csv": result.output_path("benchmark_matrix_csv", must_exist=True),
        "artifact_root": result.artifact_dir,
    }


if __name__ == "__main__":
    _, view = run_benchmark_pack_notebook_cell()
    print(view["notebook_summary"])
