from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.regimes.conditional import (
    RegimeConditionalResult,
    _ALPHA_METRIC_COLUMNS,
    _PORTFOLIO_METRIC_COLUMNS,
    _STRATEGY_METRIC_COLUMNS,
)
from src.research.regimes.taxonomy import TAXONOMY_VERSION

# ---------------------------------------------------------------------------
# Filename constants
# ---------------------------------------------------------------------------

METRICS_BY_REGIME_FILENAME = "metrics_by_regime.csv"
REGIME_CONDITIONAL_SUMMARY_FILENAME = "regime_conditional_summary.json"
REGIME_CONDITIONAL_MANIFEST_FILENAME = "regime_conditional_manifest.json"

_SCHEMA_VERSION = 1

#: Stable column ordering per surface used when writing the canonical CSV.
_SURFACE_COLUMNS: dict[str, tuple[str, ...]] = {
    "strategy": _STRATEGY_METRIC_COLUMNS,
    "alpha": _ALPHA_METRIC_COLUMNS,
    "portfolio": _PORTFOLIO_METRIC_COLUMNS,
}


class RegimeConditionalArtifactError(ValueError):
    """Raised when conditional artifact persistence fails validation."""


# ---------------------------------------------------------------------------
# Artifact path helper
# ---------------------------------------------------------------------------


def resolve_regime_conditional_artifact_dir(
    run_id: str,
    *,
    artifacts_root: str | Path = Path("artifacts") / "regime_conditional",
) -> Path:
    """Return the deterministic output directory for one conditional evaluation run."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RegimeConditionalArtifactError("run_id must be a non-empty string.")
    return Path(artifacts_root) / normalized_run_id


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_regime_conditional_artifacts(
    output_dir: str | Path,
    result: RegimeConditionalResult,
    *,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist ``metrics_by_regime.csv`` and ``regime_conditional_summary.json``.

    Parameters
    ----------
    output_dir:
        Directory path where artifacts will be written. Created if needed.
    result:
        Evaluated ``RegimeConditionalResult`` from the conditional module.
    run_id:
        Optional run identifier included in manifest metadata.
    extra_metadata:
        Additional metadata merged into the summary and manifest payloads.

    Returns
    -------
    dict[str, Any]
        Manifest payload describing persisted artifacts.
    """

    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    validated_frame = _validate_metrics_frame(result)
    summary_payload = _build_summary_payload(result, run_id=run_id, extra_metadata=extra_metadata)
    manifest_payload = _build_manifest_payload(
        output_dir=resolved_dir,
        result=result,
        summary_payload=summary_payload,
        run_id=run_id,
        extra_metadata=extra_metadata,
    )

    _write_csv(resolved_dir / METRICS_BY_REGIME_FILENAME, validated_frame)
    _write_json(resolved_dir / REGIME_CONDITIONAL_SUMMARY_FILENAME, summary_payload)
    _write_json(resolved_dir / REGIME_CONDITIONAL_MANIFEST_FILENAME, manifest_payload)

    return manifest_payload


def write_regime_conditional_artifacts_multi_dimension(
    output_dir: str | Path,
    results: dict[str, RegimeConditionalResult],
    *,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist regime-conditional artifacts for all dimensions in one directory.

    Produces:

    * ``metrics_by_regime.csv`` — combined across all dimensions with a stable
      ``dimension`` column.
    * ``regime_conditional_summary.json`` — per-dimension alignment summaries
      and coverage breakdowns.
    * ``regime_conditional_manifest.json`` — traceability manifest.

    Parameters
    ----------
    output_dir:
        Directory path where artifacts will be written.
    results:
        Dict of ``{dimension: RegimeConditionalResult}`` as returned by
        ``evaluate_all_dimensions()``.
    run_id:
        Optional run identifier.
    extra_metadata:
        Additional metadata merged into payload.

    Returns
    -------
    dict[str, Any]
        Manifest payload.
    """

    if not results:
        raise RegimeConditionalArtifactError("results dict must not be empty.")

    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: list[pd.DataFrame] = []
    for _dim, result in sorted(results.items()):
        combined_frames.append(result.metrics_by_regime.copy())

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = combined.sort_values(
        ["dimension", "regime_label"], kind="stable"
    ).reset_index(drop=True)
    combined.attrs = {}

    # Determine surface from first result.
    first_result = next(iter(results.values()))
    surface = first_result.surface
    canonical_columns = _SURFACE_COLUMNS.get(surface, ())
    for col in canonical_columns:
        if col not in combined.columns:
            combined[col] = None
    # Reorder to canonical columns, keeping any extras at the end.
    extra_cols = [c for c in combined.columns if c not in canonical_columns]
    combined = combined[list(canonical_columns) + extra_cols]

    summary_payload = _build_multi_dimension_summary_payload(
        results, run_id=run_id, extra_metadata=extra_metadata
    )
    manifest_payload = _build_multi_dimension_manifest_payload(
        output_dir=resolved_dir,
        results=results,
        summary_payload=summary_payload,
        run_id=run_id,
        extra_metadata=extra_metadata,
    )

    _write_csv(resolved_dir / METRICS_BY_REGIME_FILENAME, combined)
    _write_json(resolved_dir / REGIME_CONDITIONAL_SUMMARY_FILENAME, summary_payload)
    _write_json(resolved_dir / REGIME_CONDITIONAL_MANIFEST_FILENAME, manifest_payload)

    return manifest_payload


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_regime_conditional_metrics(path: str | Path) -> pd.DataFrame:
    """Load ``metrics_by_regime.csv`` from a file path or artifact directory."""

    csv_path = _resolve_metrics_path(path)
    frame = pd.read_csv(csv_path)
    frame.attrs = {}
    return frame


def load_regime_conditional_summary(path: str | Path) -> dict[str, Any]:
    """Load ``regime_conditional_summary.json`` from a file path or directory."""

    summary_path = _resolve_summary_path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeConditionalArtifactError(
            f"Regime conditional summary must be an object: {summary_path}"
        )
    return payload


def load_regime_conditional_manifest(path: str | Path) -> dict[str, Any]:
    """Load ``regime_conditional_manifest.json`` from a file path or directory."""

    manifest_path = _resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeConditionalArtifactError(
            f"Regime conditional manifest must be an object: {manifest_path}"
        )
    return payload


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_metrics_frame(result: RegimeConditionalResult) -> pd.DataFrame:
    frame = result.metrics_by_regime.copy()
    if frame.empty:
        return frame
    if "regime_label" not in frame.columns:
        raise RegimeConditionalArtifactError("metrics_by_regime is missing 'regime_label' column.")
    if "observation_count" not in frame.columns:
        raise RegimeConditionalArtifactError("metrics_by_regime is missing 'observation_count' column.")
    return frame


def _build_summary_payload(
    result: RegimeConditionalResult,
    *,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    frame = result.metrics_by_regime
    regime_labels = sorted(frame["regime_label"].unique().tolist()) if not frame.empty else []
    coverage_breakdown = (
        frame.set_index("regime_label")["coverage_status"].to_dict()
        if not frame.empty
        else {}
    )
    observation_counts = (
        frame.set_index("regime_label")["observation_count"].to_dict()
        if not frame.empty
        else {}
    )

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "dimension": result.dimension,
        "taxonomy_version": result.config.taxonomy_version,
        "min_observations": result.config.min_observations,
        "periods_per_year": result.config.periods_per_year,
        "regime_labels": regime_labels,
        "regime_label_count": len(regime_labels),
        "coverage_breakdown": {str(k): str(v) for k, v in coverage_breakdown.items()},
        "observation_counts": {str(k): int(v) for k, v in observation_counts.items()},
        "alignment_summary": result.alignment_summary,
        "metric_columns": list(result.metrics_by_regime.columns),
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    if extra_metadata:
        payload.update({k: v for k, v in extra_metadata.items() if k not in payload})
    return payload


def _build_multi_dimension_summary_payload(
    results: dict[str, RegimeConditionalResult],
    *,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    first_result = next(iter(results.values()))
    dimensions_summary: dict[str, Any] = {}
    for dimension, result in sorted(results.items()):
        frame = result.metrics_by_regime
        regime_labels = sorted(frame["regime_label"].unique().tolist()) if not frame.empty else []
        dimensions_summary[dimension] = {
            "regime_labels": regime_labels,
            "regime_label_count": len(regime_labels),
            "coverage_breakdown": (
                {str(k): str(v) for k, v in frame.set_index("regime_label")["coverage_status"].items()}
                if not frame.empty
                else {}
            ),
            "observation_counts": (
                {str(k): int(v) for k, v in frame.set_index("regime_label")["observation_count"].items()}
                if not frame.empty
                else {}
            ),
            "alignment_summary": result.alignment_summary,
        }

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "surface": first_result.surface,
        "taxonomy_version": first_result.config.taxonomy_version,
        "min_observations": first_result.config.min_observations,
        "periods_per_year": first_result.config.periods_per_year,
        "dimensions": sorted(results.keys()),
        "dimensions_summary": dimensions_summary,
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    if extra_metadata:
        payload.update({k: v for k, v in extra_metadata.items() if k not in payload})
    return payload


def _build_manifest_payload(
    output_dir: Path,
    result: RegimeConditionalResult,
    summary_payload: dict[str, Any],
    *,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "dimension": result.dimension,
        "taxonomy_version": TAXONOMY_VERSION,
        "artifacts": {
            "metrics_by_regime_csv": METRICS_BY_REGIME_FILENAME,
            "regime_conditional_summary_json": REGIME_CONDITIONAL_SUMMARY_FILENAME,
            "regime_conditional_manifest_json": REGIME_CONDITIONAL_MANIFEST_FILENAME,
        },
        "alignment_summary": result.alignment_summary,
        "regime_label_count": summary_payload.get("regime_label_count", 0),
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    if extra_metadata:
        manifest.update({k: v for k, v in extra_metadata.items() if k not in manifest})
    return manifest


def _build_multi_dimension_manifest_payload(
    output_dir: Path,
    results: dict[str, RegimeConditionalResult],
    summary_payload: dict[str, Any],
    *,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    first_result = next(iter(results.values()))
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "surface": first_result.surface,
        "taxonomy_version": TAXONOMY_VERSION,
        "dimensions": sorted(results.keys()),
        "artifacts": {
            "metrics_by_regime_csv": METRICS_BY_REGIME_FILENAME,
            "regime_conditional_summary_json": REGIME_CONDITIONAL_SUMMARY_FILENAME,
            "regime_conditional_manifest_json": REGIME_CONDITIONAL_MANIFEST_FILENAME,
        },
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    if extra_metadata:
        manifest.update({k: v for k, v in extra_metadata.items() if k not in manifest})
    return manifest


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        import math
        if math.isnan(obj) or math.isinf(obj):
            return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def _resolve_metrics_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        return resolved / METRICS_BY_REGIME_FILENAME
    return resolved


def _resolve_summary_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        return resolved / REGIME_CONDITIONAL_SUMMARY_FILENAME
    return resolved


def _resolve_manifest_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        return resolved / REGIME_CONDITIONAL_MANIFEST_FILENAME
    return resolved


__all__ = [
    "METRICS_BY_REGIME_FILENAME",
    "REGIME_CONDITIONAL_MANIFEST_FILENAME",
    "REGIME_CONDITIONAL_SUMMARY_FILENAME",
    "RegimeConditionalArtifactError",
    "load_regime_conditional_manifest",
    "load_regime_conditional_metrics",
    "load_regime_conditional_summary",
    "resolve_regime_conditional_artifact_dir",
    "write_regime_conditional_artifacts",
    "write_regime_conditional_artifacts_multi_dimension",
]
