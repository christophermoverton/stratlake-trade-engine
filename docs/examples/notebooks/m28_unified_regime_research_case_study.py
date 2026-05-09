from __future__ import annotations

from pathlib import Path
from typing import Any


NOTEBOOK_OUTPUT_ROOT = "artifacts/notebooks/m28_unified_regime_research_case_study/attempt_001"


def run_m28_case_study_notebook_cell(
    output_root: str | Path = NOTEBOOK_OUTPUT_ROOT,
    *,
    include_cross_layer_validation: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Run the M28.6 unified regime research case study from a notebook cell.

    This is a thin wrapper around the canonical case-study callable. It does
    not reimplement any workflow logic. Artifacts, manifests, summaries, and
    named output paths are the source of truth.

    Parameters
    ----------
    output_root:
        Notebook-specific output root. Change the attempt suffix to avoid
        colliding with previous notebook runs.
    include_cross_layer_validation:
        When ``True``, also run the M28.5 cross-layer validation stage (slower).
        Defaults to ``False`` for lightweight notebook inspection.

    Returns
    -------
    tuple of (result_dict, inspection)
        ``result_dict`` is the dict returned by the canonical case-study
        callable. ``inspection`` is a notebook-friendly summary dict.
    """
    from docs.examples.m28_unified_regime_research_case_study import (
        run_m28_unified_regime_research_case_study,
    )

    result = run_m28_unified_regime_research_case_study(
        output_root=output_root,
        include_cross_layer_validation=include_cross_layer_validation,
    )

    inspection: dict[str, Any] = {
        "notebook_summary": result["regime_benchmark_result"],
        "case_study_name": result["case_study_name"],
        "milestone": result["milestone"],
        "output_root": result["output_root"],
        "output_keys": sorted(result.keys()),
        "manifest_path": result["manifest_path"],
        "summary_path": result["summary_path"],
        "validation_report_path": result["validation_report_path"],
        "cross_layer_comparison_path": result["cross_layer_comparison_path"],
        "artifact_index_path": result["artifact_index_path"],
        "workflow_stages": result["summary"].get("workflow_stages"),
        "execution_surfaces_called": result["summary"].get("execution_surfaces_called"),
        "cross_layer_status": result["summary"]
        .get("cross_layer_validation", {})
        .get("status", "skipped"),
        "regime_benchmark_variant_count": result["summary"]
        .get("regime_benchmark", {})
        .get("variant_count"),
    }
    return result, inspection


def inspect_case_study_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    """Inspect case study outputs after execution has completed.

    Loads manifests and summaries from the persisted artifact root.
    Artifacts are the source of truth, not notebook cell state.
    """
    import json

    def _load_json(path_str: str) -> Any:
        path = Path(path_str)
        if not path.is_absolute():
            # Resolve relative to repo root (two levels up from docs/examples/notebooks/).
            repo_root = Path(__file__).resolve().parents[3]
            path = repo_root / path
        if not path.exists():
            return {"error": f"path not found: {path_str}"}
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "manifest": _load_json(result["manifest_path"]),
        "summary": _load_json(result["summary_path"]),
        "validation_report": _load_json(result["validation_report_path"]),
        "artifact_index": _load_json(result["artifact_index_path"]),
    }


if __name__ == "__main__":
    result, view = run_m28_case_study_notebook_cell()
    print(view["notebook_summary"])
    print(f"output_root: {view['output_root']}")
    print(f"workflow_stages: {view['workflow_stages']}")
