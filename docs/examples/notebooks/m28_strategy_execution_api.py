from __future__ import annotations

from typing import Any


def run_strategy_notebook_cell() -> tuple[Any, dict[str, Any]]:
    """Run a strategy through the public execution API and inspect artifacts."""

    from src.execution import run_strategy

    result = run_strategy(
        "momentum_v1",
        start="2022-01-01",
        end="2023-01-01",
        strict=True,
    )

    return result, {
        "notebook_summary": result.notebook_summary(),
        "output_keys": result.output_keys(),
        "manifest": result.load_manifest(),
        "metrics": result.load_metrics_json(),
        "qa_summary": result.load_summary_json("qa_summary_json"),
        "equity_curve_path": result.output_path("equity_curve_csv", must_exist=True),
    }


def inspect_existing_strategy_result(result: Any) -> dict[str, Any]:
    """Read canonical strategy artifacts without relying on notebook state."""

    return {
        "summary": result.notebook_summary(),
        "manifest": result.load_manifest(),
        "metrics": result.load_metrics_json("metrics_json"),
        "artifact_root": result.artifact_dir,
        "named_outputs": result.output_keys(),
    }


if __name__ == "__main__":
    _, view = run_strategy_notebook_cell()
    print(view["notebook_summary"])
