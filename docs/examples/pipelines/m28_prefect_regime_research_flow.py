from __future__ import annotations

from pathlib import Path
from typing import Any

# Prefect is optional for StratLake. The plain Python callable below remains
# importable and testable even when Prefect is not installed.
try:
    from prefect import flow, task
except ImportError:
    flow = None
    task = None


BENCHMARK_PACK_CONFIG = "configs/benchmark_packs/m22_scale_repro.yml"


def build_m28_prefect_output_root(flow_run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(flow_run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return Path("artifacts") / "orchestrator_examples" / "prefect" / safe_run_id / f"attempt_{attempt}"


def run_m28_prefect_example(flow_run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run a small benchmark pack through the public StratLake execution API."""

    from src.execution import run_benchmark_pack

    return run_benchmark_pack(
        BENCHMARK_PACK_CONFIG,
        output_root=build_m28_prefect_output_root(flow_run_id=flow_run_id, attempt=attempt),
    )


if flow is not None and task is not None:

    @task(name="run-stratlake-regime-research")
    def run_regime_research_task(flow_run_id: str = "manual", attempt: str | int = 1) -> dict[str, Any]:
        result = run_m28_prefect_example(flow_run_id=flow_run_id, attempt=attempt)
        return result.to_dict()

    @flow(name="m28-stratlake-regime-research-example")
    def m28_prefect_regime_research_flow(
        flow_run_id: str = "manual",
        attempt: str | int = 1,
    ) -> dict[str, Any]:
        return run_regime_research_task(flow_run_id=flow_run_id, attempt=attempt)

else:
    run_regime_research_task = None
    m28_prefect_regime_research_flow = None
