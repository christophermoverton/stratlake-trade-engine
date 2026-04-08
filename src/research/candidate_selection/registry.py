from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.research.registry import (
    canonicalize_value,
    default_registry_path,
    stable_timestamp_from_run_id,
    upsert_registry_entry,
)

_CANDIDATE_SELECTION_RUN_TYPE = "candidate_selection"


def candidate_selection_registry_path(
    artifacts_root: str | Path,
) -> Path:
    """Return candidate-selection registry path under one artifacts root."""

    return default_registry_path(Path(artifacts_root))


def build_candidate_selection_registry_entry(
    *,
    run_id: str,
    artifact_dir: str | Path,
    manifest: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one deterministic candidate-selection registry entry."""

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping.")
    if not isinstance(selection_summary, Mapping):
        raise ValueError("selection_summary must be a mapping.")

    candidate_stats = manifest.get("candidate_statistics")
    if not isinstance(candidate_stats, Mapping):
        candidate_stats = {}

    config_snapshot = manifest.get("config_snapshot")
    if not isinstance(config_snapshot, Mapping):
        config_snapshot = {}

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}

    allocation_summary = manifest.get("allocation_summary")
    if not isinstance(allocation_summary, Mapping):
        allocation_summary = {}

    thresholds = {
        "eligibility": config_snapshot.get("eligibility_thresholds"),
        "redundancy": config_snapshot.get("redundancy_thresholds"),
        "allocation_constraints": config_snapshot.get("allocation_constraints"),
    }

    resolved_artifact_dir = Path(artifact_dir)
    return {
        "run_id": str(run_id),
        "run_type": _CANDIDATE_SELECTION_RUN_TYPE,
        "timestamp": stable_timestamp_from_run_id(str(run_id)),
        "alpha_name": config_snapshot.get("filters", {}).get("alpha_name")
        if isinstance(config_snapshot.get("filters"), Mapping)
        else None,
        "dataset": provenance.get("dataset"),
        "timeframe": provenance.get("timeframe"),
        "evaluation_horizon": provenance.get("evaluation_horizon"),
        "candidate_count": candidate_stats.get("total_candidates"),
        "selected_count": candidate_stats.get("selected_candidates"),
        "rejected_count": candidate_stats.get("rejected_candidates"),
        "allocation_method": allocation_summary.get("allocation_method"),
        "thresholds": canonicalize_value(thresholds),
        "artifact_path": resolved_artifact_dir.as_posix(),
        "manifest_path": resolved_artifact_dir.joinpath("manifest.json").as_posix(),
        "summary_path": resolved_artifact_dir.joinpath("selection_summary.json").as_posix(),
        "upstream_alpha_run_ids": (
            provenance.get("upstream", {}).get("alpha_run_ids")
            if isinstance(provenance.get("upstream"), Mapping)
            else None
        ),
        "metadata": canonicalize_value(
            {
                "mapping_names": provenance.get("mapping_names"),
                "primary_metric": config_snapshot.get("primary_metric"),
                "row_counts": manifest.get("row_counts"),
                "candidate_statistics": candidate_stats,
                "allocation_summary": allocation_summary,
            }
        ),
    }


def register_candidate_selection_run(
    *,
    artifacts_root: str | Path,
    run_id: str,
    artifact_dir: str | Path,
    manifest: Mapping[str, Any],
    selection_summary: Mapping[str, Any],
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Deterministically upsert one candidate-selection run into registry."""

    entry = build_candidate_selection_registry_entry(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest=manifest,
        selection_summary=selection_summary,
    )
    resolved_registry_path = (
        candidate_selection_registry_path(artifacts_root)
        if registry_path is None
        else Path(registry_path)
    )
    upsert_registry_entry(resolved_registry_path, entry)
    return entry


__all__ = [
    "build_candidate_selection_registry_entry",
    "candidate_selection_registry_path",
    "register_candidate_selection_run",
]
