from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

# Airflow is optional for StratLake. Keep DAG parsing import-safe when the
# scheduler package is not installed in a local research or CI environment.
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None
    PythonOperator = None


BENCHMARK_PACK_CONFIG = "configs/benchmark_packs/m22_scale_repro.yml"


def build_m28_airflow_output_root(run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return Path("artifacts") / "orchestrator_examples" / "airflow" / safe_run_id / f"attempt_{attempt}"


def run_m28_airflow_example(run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run a small benchmark pack through the public StratLake execution API."""

    from src.execution import run_benchmark_pack

    return run_benchmark_pack(
        BENCHMARK_PACK_CONFIG,
        output_root=build_m28_airflow_output_root(run_id=run_id, attempt=attempt),
    )


def _run_from_airflow_context(**context: Any) -> dict[str, Any]:
    dag_run = context.get("dag_run")
    task_instance = context.get("task_instance")
    run_id = getattr(dag_run, "run_id", "manual")
    attempt = getattr(task_instance, "try_number", 1)
    result = run_m28_airflow_example(run_id=run_id, attempt=attempt)
    return result.to_dict()


if DAG is not None and PythonOperator is not None:
    with DAG(
        dag_id="m28_stratlake_regime_research_example",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["stratlake", "m28"],
    ) as m28_airflow_regime_research_dag:
        run_regime_research = PythonOperator(
            task_id="run_regime_research_benchmark_pack",
            python_callable=_run_from_airflow_context,
        )
else:
    m28_airflow_regime_research_dag = None
    run_regime_research = None


# ---------------------------------------------------------------------------
# M28.6 Capstone: unified regime research case study — Airflow wrapper
# ---------------------------------------------------------------------------


def build_m28_capstone_airflow_output_root(run_id: str = "manual", attempt: str | int = 1) -> Path:
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_").replace(":", "_")
    return (
        Path("artifacts")
        / "orchestrator_examples"
        / "airflow"
        / safe_run_id
        / f"capstone_attempt_{attempt}"
    )


def run_m28_capstone_airflow_example(run_id: str = "manual", attempt: str | int = 1) -> Any:
    """Run the M28.6 unified regime research case study through the public callable."""

    from docs.examples.m28_unified_regime_research_case_study import (
        run_m28_unified_regime_research_case_study,
    )

    return run_m28_unified_regime_research_case_study(
        output_root=build_m28_capstone_airflow_output_root(run_id=run_id, attempt=attempt),
        include_cross_layer_validation=False,
    )


def _run_capstone_from_airflow_context(**context: Any) -> dict[str, Any]:
    dag_run = context.get("dag_run")
    task_instance = context.get("task_instance")
    run_id = getattr(dag_run, "run_id", "manual")
    attempt = getattr(task_instance, "try_number", 1)
    return run_m28_capstone_airflow_example(run_id=run_id, attempt=attempt)


if DAG is not None and PythonOperator is not None:
    with DAG(
        dag_id="m28_capstone_stratlake_unified_regime_research",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["stratlake", "m28", "capstone"],
    ) as m28_capstone_airflow_regime_research_dag:
        run_capstone_regime_research = PythonOperator(
            task_id="run_m28_capstone_unified_regime_research",
            python_callable=_run_capstone_from_airflow_context,
        )
else:
    m28_capstone_airflow_regime_research_dag = None
    run_capstone_regime_research = None
