from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.research.regimes.sensitivity import RegimeCalibrationSensitivityResult
from src.research.registry import canonicalize_value

REGIME_SENSITIVITY_MATRIX_FILENAME = "regime_sensitivity_matrix.csv"
REGIME_SENSITIVITY_SUMMARY_FILENAME = "regime_sensitivity_summary.json"
REGIME_STABILITY_REPORT_FILENAME = "regime_stability_report.md"
CALIBRATION_PROFILE_RESULTS_FILENAME = "calibration_profile_results.json"

_SCHEMA_VERSION = 1


class RegimeSensitivityArtifactError(ValueError):
    """Raised when regime sensitivity artifacts cannot be persisted or loaded."""


def write_regime_sensitivity_artifacts(
    output_dir: str | Path,
    result: RegimeCalibrationSensitivityResult,
    *,
    run_id: str | None = None,
    source_regime_artifact_references: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = resolved_dir / REGIME_SENSITIVITY_MATRIX_FILENAME
    summary_path = resolved_dir / REGIME_SENSITIVITY_SUMMARY_FILENAME
    report_path = resolved_dir / REGIME_STABILITY_REPORT_FILENAME
    profile_results_path = resolved_dir / CALIBRATION_PROFILE_RESULTS_FILENAME

    matrix = _artifact_frame(result.matrix)
    summary_payload = _build_summary_payload(
        result,
        run_id=run_id,
        source_regime_artifact_references=source_regime_artifact_references,
        extra_metadata=extra_metadata,
    )
    profile_results_payload = _build_profile_results_payload(result)
    report_markdown = _render_stability_report(result)

    _write_csv(matrix_path, matrix)
    _write_json(profile_results_path, profile_results_payload)
    report_path.write_text(report_markdown, encoding="utf-8")
    _write_json(summary_path, summary_payload)

    summary_payload["file_inventory"] = {
        REGIME_SENSITIVITY_MATRIX_FILENAME: {
            "path": REGIME_SENSITIVITY_MATRIX_FILENAME,
            "rows": _count_csv_rows(matrix_path),
            "sha256": _sha256_file(matrix_path),
        },
        REGIME_SENSITIVITY_SUMMARY_FILENAME: {
            "path": REGIME_SENSITIVITY_SUMMARY_FILENAME,
        },
        REGIME_STABILITY_REPORT_FILENAME: {
            "path": REGIME_STABILITY_REPORT_FILENAME,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256_file(report_path),
        },
        CALIBRATION_PROFILE_RESULTS_FILENAME: {
            "path": CALIBRATION_PROFILE_RESULTS_FILENAME,
            "sha256": _sha256_file(profile_results_path),
        },
    }
    _write_json(summary_path, summary_payload)
    return summary_payload


def load_regime_sensitivity_matrix(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_SENSITIVITY_MATRIX_FILENAME))
    frame.attrs = {}
    return frame


def load_regime_sensitivity_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_SENSITIVITY_SUMMARY_FILENAME)


def load_calibration_profile_results(path: str | Path) -> dict[str, Any]:
    return _load_json(path, CALIBRATION_PROFILE_RESULTS_FILENAME)


def load_regime_stability_report(path: str | Path) -> str:
    return _resolve_path(path, REGIME_STABILITY_REPORT_FILENAME).read_text(encoding="utf-8")


def _build_summary_payload(
    result: RegimeCalibrationSensitivityResult,
    *,
    run_id: str | None,
    source_regime_artifact_references: Mapping[str, Any] | None,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    recommended_rows = result.matrix.loc[result.matrix["is_recommended_profile"].astype("bool")]
    recommended_profile = None if recommended_rows.empty else str(recommended_rows.iloc[0]["profile_name"])
    payload: dict[str, Any] = {
        "artifact_type": "regime_sensitivity_summary",
        "schema_version": _SCHEMA_VERSION,
        "taxonomy_version": result.taxonomy_version,
        "profile_count": int(len(result.profile_order)),
        "profile_names": list(result.profile_order),
        "recommended_profile": recommended_profile,
        "source_regime_artifact_references": canonicalize_value(dict(source_regime_artifact_references or {})),
        "metadata": canonicalize_value({**dict(result.metadata), **dict(extra_metadata or {})}),
        "artifacts": {
            "regime_sensitivity_matrix_csv": REGIME_SENSITIVITY_MATRIX_FILENAME,
            "regime_sensitivity_summary_json": REGIME_SENSITIVITY_SUMMARY_FILENAME,
            "regime_stability_report_md": REGIME_STABILITY_REPORT_FILENAME,
            "calibration_profile_results_json": CALIBRATION_PROFILE_RESULTS_FILENAME,
        },
        "summary": canonicalize_value(dict(result.summary)),
        "file_inventory": {
            REGIME_SENSITIVITY_SUMMARY_FILENAME: {
                "path": REGIME_SENSITIVITY_SUMMARY_FILENAME,
            }
        },
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    return canonicalize_value(payload)


def _build_profile_results_payload(result: RegimeCalibrationSensitivityResult) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile_name in result.profile_order:
        calibration = result.profile_results[profile_name]
        profile_payload: dict[str, Any] = {
            "profile": calibration.profile.to_dict(),
            "warnings": list(calibration.warnings),
            "profile_flags": dict(calibration.profile_flags),
            "stability_metrics": dict(calibration.stability_metrics),
            "fallback_summary": dict(calibration.fallback_summary),
            "attribution_summary": _frame_rows(calibration.attribution_summary),
        }
        if profile_name in result.profile_performance:
            profile_payload["performance_summary"] = _frame_rows(result.profile_performance[profile_name])
        profiles[profile_name] = canonicalize_value(profile_payload)

    payload = {
        "artifact_type": "calibration_profile_results",
        "schema_version": _SCHEMA_VERSION,
        "taxonomy_version": result.taxonomy_version,
        "profile_count": int(len(result.profile_order)),
        "profiles": profiles,
    }
    if not result.performance_summary.empty:
        payload["profile_level_performance_summary"] = _frame_rows(result.performance_summary)
    return canonicalize_value(payload)


def _render_stability_report(result: RegimeCalibrationSensitivityResult) -> str:
    ranked = result.matrix.sort_values(["stable_profile_rank", "profile_name"], kind="stable")
    recommended_rows = ranked.loc[ranked["is_recommended_profile"].astype("bool")]
    recommended_profile = None if recommended_rows.empty else str(recommended_rows.iloc[0]["profile_name"])

    lines = [
        "# Regime Stability Report",
        "",
        "## Overview",
        "This report compares deterministic calibration profiles on the same regime-label input.",
        "It measures stability, fallback usage, attribution readiness, and optional performance sensitivity.",
        "",
        "## Recommendation",
    ]
    if recommended_profile is None:
        lines.append("- No profile met the downstream decisioning eligibility gate.")
    else:
        lines.append(f"- Recommended profile: `{recommended_profile}`.")
    lines.append("")
    lines.append("## Profile Comparison")
    for row in ranked.to_dict(orient="records"):
        lines.append(
            "- "
            f"`{row['profile_name']}`: rank={row['stable_profile_rank']}, "
            f"unstable={str(bool(row['is_unstable_profile'])).lower()}, "
            f"flip_rate={float(row['flip_rate']):.4f}, "
            f"single_day_flip_share={float(row['single_day_flip_share']):.4f}, "
            f"defined_share={float(row['defined_observation_share']):.4f}, "
            f"warnings={int(row['warning_count'])}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- Higher ranks favor stable, lower-flip, lower-noise profiles with better defined coverage.",
            "- Unstable profiles should remain audit inputs rather than downstream decisioning defaults.",
            "- Optional performance summaries are descriptive and do not imply causality.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    return normalized


def _frame_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy(deep=True)
    normalized = normalized.astype("object").where(normalized.notna(), None)
    return canonicalize_value(normalized.to_dict(orient="records"))


def _resolve_path(path: str | Path, filename: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Regime sensitivity artifact file does not exist: {resolved}")
    return resolved


def _load_json(path: str | Path, filename: str) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeSensitivityArtifactError(f"{filename} must deserialize to a JSON object.")
    return payload


def _count_csv_rows(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return 0
    count = text.count("\n")
    if text.endswith("\n"):
        count -= 1
    return max(count, 0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "CALIBRATION_PROFILE_RESULTS_FILENAME",
    "REGIME_SENSITIVITY_MATRIX_FILENAME",
    "REGIME_SENSITIVITY_SUMMARY_FILENAME",
    "REGIME_STABILITY_REPORT_FILENAME",
    "RegimeSensitivityArtifactError",
    "load_calibration_profile_results",
    "load_regime_sensitivity_matrix",
    "load_regime_sensitivity_summary",
    "load_regime_stability_report",
    "write_regime_sensitivity_artifacts",
]
