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


# ---------------------------------------------------------------------------
# M28.6 Capstone: unified regime research case study — Prefect wrapper
# ---------------------------------------------------------------------------


def build_m28_capstone_prefect_output_root(flow_run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(flow_run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return (
        Path("artifacts")
        / "orchestrator_examples"
        / "prefect"
        / safe_run_id
        / f"capstone_attempt_{attempt}"
    )


def run_m28_capstone_prefect_example(flow_run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run the M28.6 unified regime research case study through the public callable."""

    from docs.examples.m28_unified_regime_research_case_study import (
        run_m28_unified_regime_research_case_study,
    )

    return run_m28_unified_regime_research_case_study(
        output_root=build_m28_capstone_prefect_output_root(
            flow_run_id=flow_run_id, attempt=attempt
        ),
        include_cross_layer_validation=False,
    )


if flow is not None and task is not None:

    @task(name="run-stratlake-m28-capstone-regime-research")
    def run_m28_capstone_regime_research_task(
        flow_run_id: str = "manual", attempt: str | int = 1
    ) -> dict[str, Any]:
        return run_m28_capstone_prefect_example(
            flow_run_id=flow_run_id, attempt=attempt
        )

    @flow(name="m28-capstone-stratlake-unified-regime-research")
    def m28_capstone_prefect_regime_research_flow(
        flow_run_id: str = "manual",
        attempt: str | int = 1,
    ) -> dict[str, Any]:
        return run_m28_capstone_regime_research_task(
            flow_run_id=flow_run_id, attempt=attempt
        )

else:
    run_m28_capstone_regime_research_task = None
    m28_capstone_prefect_regime_research_flow = None
