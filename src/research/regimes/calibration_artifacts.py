from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.research.regimes.calibration import RegimeCalibrationResult
from src.research.registry import canonicalize_value

REGIME_CALIBRATION_FILENAME = "regime_calibration.json"
REGIME_CALIBRATION_SUMMARY_FILENAME = "regime_calibration_summary.json"
REGIME_STABILITY_METRICS_FILENAME = "regime_stability_metrics.json"

_SCHEMA_VERSION = 1


class RegimeCalibrationArtifactError(ValueError):
    """Raised when regime calibration artifacts cannot be persisted or loaded."""


def write_regime_calibration_artifacts(
    output_dir: str | Path,
    result: RegimeCalibrationResult,
    *,
    run_id: str | None = None,
    source_regime_artifact_references: dict[str, Any] | None = None,
    taxonomy_metadata: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    write_stability_metrics: bool = True,
) -> dict[str, Any]:
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    calibration_path = resolved_dir / REGIME_CALIBRATION_FILENAME
    summary_path = resolved_dir / REGIME_CALIBRATION_SUMMARY_FILENAME
    metrics_path = resolved_dir / REGIME_STABILITY_METRICS_FILENAME

    summary_payload = _build_summary_payload(result)
    metrics_payload = canonicalize_value(dict(result.stability_metrics))

    if write_stability_metrics:
        _write_json(metrics_path, metrics_payload)

    calibration_payload = _build_calibration_payload(
        result=result,
        run_id=run_id,
        source_regime_artifact_references=source_regime_artifact_references,
        taxonomy_metadata=taxonomy_metadata,
        extra_metadata=extra_metadata,
        include_metrics_file=write_stability_metrics,
    )
    _write_json(calibration_path, calibration_payload)
    _write_json(summary_path, summary_payload)
    calibration_payload["file_inventory"][REGIME_CALIBRATION_SUMMARY_FILENAME] = {
        "path": REGIME_CALIBRATION_SUMMARY_FILENAME,
        "sha256": _sha256_file(summary_path),
    }
    if write_stability_metrics:
        calibration_payload["file_inventory"][REGIME_STABILITY_METRICS_FILENAME] = {
            "path": REGIME_STABILITY_METRICS_FILENAME,
            "sha256": _sha256_file(metrics_path),
        }
    _write_json(calibration_path, calibration_payload)
    return calibration_payload


def load_regime_calibration(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_CALIBRATION_FILENAME)


def load_regime_calibration_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_CALIBRATION_SUMMARY_FILENAME)


def load_regime_stability_metrics(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_STABILITY_METRICS_FILENAME)


def _build_calibration_payload(
    *,
    result: RegimeCalibrationResult,
    run_id: str | None,
    source_regime_artifact_references: dict[str, Any] | None,
    taxonomy_metadata: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None,
    include_metrics_file: bool,
) -> dict[str, Any]:
    artifacts = {
        "regime_calibration_json": REGIME_CALIBRATION_FILENAME,
        "regime_calibration_summary_json": REGIME_CALIBRATION_SUMMARY_FILENAME,
    }
    if include_metrics_file:
        artifacts["regime_stability_metrics_json"] = REGIME_STABILITY_METRICS_FILENAME
    payload: dict[str, Any] = {
        "artifact_type": "regime_calibration",
        "schema_version": _SCHEMA_VERSION,
        "taxonomy_version": result.taxonomy_version,
        "profile_name": result.profile.name,
        "profile": result.profile.to_dict(),
        "output_row_count": int(len(result.labels)),
        "warnings": list(result.warnings),
        "profile_flags": dict(result.profile_flags),
        "fallback_behavior_applied": dict(result.fallback_summary),
        "stability_metrics": dict(result.stability_metrics),
        "attribution_summary": _frame_rows(result.attribution_summary),
        "source_regime_artifact_references": canonicalize_value(dict(source_regime_artifact_references or {})),
        "taxonomy_metadata": canonicalize_value(dict(taxonomy_metadata or {})),
        "metadata": canonicalize_value({**result.metadata, **dict(extra_metadata or {})}),
        "artifacts": artifacts,
        "file_inventory": {
            REGIME_CALIBRATION_FILENAME: {"path": REGIME_CALIBRATION_FILENAME},
            REGIME_CALIBRATION_SUMMARY_FILENAME: {"path": REGIME_CALIBRATION_SUMMARY_FILENAME},
        },
        "summary": {
            "is_unstable_profile": bool(result.profile_flags.get("is_unstable_profile", False)),
            "output_row_count": int(len(result.labels)),
            "profile_name": result.profile.name,
            "warning_count": int(len(result.warnings)),
        },
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    return canonicalize_value(payload)


def _build_summary_payload(result: RegimeCalibrationResult) -> dict[str, Any]:
    return canonicalize_value(
        {
            "artifact_type": "regime_calibration_summary",
            "schema_version": _SCHEMA_VERSION,
            "taxonomy_version": result.taxonomy_version,
            "profile_name": result.profile.name,
            "parameters": result.profile.to_dict(),
            "output_row_count": int(len(result.labels)),
            "profile_flags": dict(result.profile_flags),
            "stability_metrics": dict(result.stability_metrics),
            "fallback_behavior_applied": dict(result.fallback_summary),
            "warnings": list(result.warnings),
        }
    )


def _frame_rows(frame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy(deep=True)
    normalized = normalized.astype("object").where(normalized.notna(), None)
    return canonicalize_value(normalized.to_dict(orient="records"))


def _load_json(path: str | Path, filename: str) -> dict[str, Any]:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Regime calibration artifact file does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeCalibrationArtifactError(f"Regime calibration payload must be a JSON object: {resolved}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "REGIME_CALIBRATION_FILENAME",
    "REGIME_CALIBRATION_SUMMARY_FILENAME",
    "REGIME_STABILITY_METRICS_FILENAME",
    "RegimeCalibrationArtifactError",
    "load_regime_calibration",
    "load_regime_calibration_summary",
    "load_regime_stability_metrics",
    "write_regime_calibration_artifacts",
]