from __future__ import annotations

from pathlib import Path
from typing import Any

# Dagster is optional for StratLake. The job/op objects are only created when
# Dagster is available; the fallback callable has no scheduler dependency.
try:
    from dagster import job, op
except ImportError:
    job = None
    op = None


BENCHMARK_PACK_CONFIG = "configs/benchmark_packs/m22_scale_repro.yml"


def build_m28_dagster_output_root(run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return Path("artifacts") / "orchestrator_examples" / "dagster" / safe_run_id / f"attempt_{attempt}"


def run_m28_dagster_example(run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run a small benchmark pack through the public StratLake execution API."""

    from src.execution import run_benchmark_pack

    return run_benchmark_pack(
        BENCHMARK_PACK_CONFIG,
        output_root=build_m28_dagster_output_root(run_id=run_id, attempt=attempt),
    )


if job is not None and op is not None:

    @op
    def run_regime_research_op() -> dict[str, Any]:
        result = run_m28_dagster_example()
        return result.to_dict()

    @job(name="m28_stratlake_regime_research_example")
    def m28_dagster_regime_research_job() -> None:
        run_regime_research_op()

else:
    run_regime_research_op = None
    m28_dagster_regime_research_job = None


# ---------------------------------------------------------------------------
# M28.6 Capstone: unified regime research case study — Dagster wrapper
# ---------------------------------------------------------------------------


def build_m28_dagster_capstone_output_root(run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return (
        Path("artifacts")
        / "orchestrator_examples"
        / "dagster"
        / safe_run_id
        / f"capstone_attempt_{attempt}"
    )


def run_m28_dagster_capstone_example(run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run the M28.6 unified regime research case study through the public callable."""

    from docs.examples.m28_unified_regime_research_case_study import (
        run_m28_unified_regime_research_case_study,
    )

    return run_m28_unified_regime_research_case_study(
        output_root=build_m28_dagster_capstone_output_root(run_id=run_id, attempt=attempt),
        include_cross_layer_validation=False,
    )


if job is not None and op is not None:

    @op
    def run_m28_capstone_regime_research_op() -> dict[str, Any]:
        return run_m28_dagster_capstone_example()

    @job(name="m28_capstone_stratlake_unified_regime_research")
    def m28_dagster_capstone_regime_research_job() -> None:
        run_m28_capstone_regime_research_op()

else:
    run_m28_capstone_regime_research_op = None
    m28_dagster_capstone_regime_research_job = None
