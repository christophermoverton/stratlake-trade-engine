from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.research.regimes.policy import RegimePolicyDecisionResult
from src.research.registry import canonicalize_value

REGIME_POLICY_DECISIONS_FILENAME = "regime_policy_decisions.csv"
REGIME_POLICY_SUMMARY_FILENAME = "regime_policy_summary.json"
ADAPTIVE_VS_STATIC_COMPARISON_FILENAME = "adaptive_vs_static_comparison.csv"
ADAPTIVE_POLICY_MANIFEST_FILENAME = "adaptive_policy_manifest.json"

_SCHEMA_VERSION = 1


class RegimePolicyArtifactError(ValueError):
    """Raised when regime policy artifacts cannot be persisted or loaded."""


def write_regime_policy_artifacts(
    output_dir: str | Path,
    result: RegimePolicyDecisionResult,
    *,
    run_id: str | None = None,
    source_regime_artifact_references: Mapping[str, Any] | None = None,
    calibration_profile: str | None = None,
    sensitivity_profile: str | None = None,
    confidence_artifact_references: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, RegimePolicyDecisionResult):
        raise TypeError("result must be a RegimePolicyDecisionResult.")
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = resolved_dir / REGIME_POLICY_DECISIONS_FILENAME
    summary_path = resolved_dir / REGIME_POLICY_SUMMARY_FILENAME
    comparison_path = resolved_dir / ADAPTIVE_VS_STATIC_COMPARISON_FILENAME
    manifest_path = resolved_dir / ADAPTIVE_POLICY_MANIFEST_FILENAME

    decisions = _artifact_decisions_frame(result.decisions)
    comparison = _artifact_comparison_frame(result.comparison)
    summary_payload = canonicalize_value(dict(result.summary))
    if extra_metadata:
        summary_payload["metadata"] = canonicalize_value(
            {**dict(summary_payload.get("metadata", {})), **dict(extra_metadata)}
        )

    _write_csv(decisions_path, decisions)
    _write_json(summary_path, summary_payload)
    _write_csv(comparison_path, comparison)

    manifest = {
        "artifact_type": "regime_policy",
        "schema_version": _SCHEMA_VERSION,
        "taxonomy_version": result.taxonomy_version,
        "artifacts": {
            "regime_policy_decisions_csv": REGIME_POLICY_DECISIONS_FILENAME,
            "regime_policy_summary_json": REGIME_POLICY_SUMMARY_FILENAME,
            "adaptive_vs_static_comparison_csv": ADAPTIVE_VS_STATIC_COMPARISON_FILENAME,
            "adaptive_policy_manifest_json": ADAPTIVE_POLICY_MANIFEST_FILENAME,
        },
        "source_regime_artifact_references": canonicalize_value(dict(source_regime_artifact_references or {})),
        "confidence_artifact_references": canonicalize_value(dict(confidence_artifact_references or {})),
        "calibration_profile": calibration_profile,
        "sensitivity_profile": sensitivity_profile,
        "policy_config": result.config.to_dict(),
        "row_counts": {
            "decision_rows": int(len(decisions)),
            "comparison_rows": int(len(comparison)),
        },
        "fallback_counts": canonicalize_value(dict(summary_payload.get("fallback_counts", {}))),
        "decision_counts_by_policy_key": canonicalize_value(dict(summary_payload.get("decision_counts_by_policy_key", {}))),
        "summary_metrics": canonicalize_value(summary_payload.get("comparison_metrics", [])),
        "metadata": canonicalize_value({**dict(result.metadata), **dict(extra_metadata or {})}),
        "file_inventory": {
            REGIME_POLICY_DECISIONS_FILENAME: {
                "path": REGIME_POLICY_DECISIONS_FILENAME,
                "rows": int(len(decisions)),
                "sha256": _sha256_file(decisions_path),
            },
            REGIME_POLICY_SUMMARY_FILENAME: {
                "path": REGIME_POLICY_SUMMARY_FILENAME,
                "sha256": _sha256_file(summary_path),
            },
            ADAPTIVE_VS_STATIC_COMPARISON_FILENAME: {
                "path": ADAPTIVE_VS_STATIC_COMPARISON_FILENAME,
                "rows": int(len(comparison)),
                "sha256": _sha256_file(comparison_path),
            },
            ADAPTIVE_POLICY_MANIFEST_FILENAME: {
                "path": ADAPTIVE_POLICY_MANIFEST_FILENAME,
            },
        },
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    manifest = canonicalize_value(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def load_regime_policy_decisions(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_POLICY_DECISIONS_FILENAME))
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    return frame


def load_regime_policy_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_POLICY_SUMMARY_FILENAME)


def load_adaptive_vs_static_comparison(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, ADAPTIVE_VS_STATIC_COMPARISON_FILENAME))
    frame.attrs = {}
    return frame


def load_adaptive_policy_manifest(path: str | Path) -> dict[str, Any]:
    return _load_json(path, ADAPTIVE_POLICY_MANIFEST_FILENAME)


def _artifact_decisions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    if "ts_utc" in normalized.columns:
        normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="raise").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return normalized.astype("object").where(pd.notna(normalized), None)


def _artifact_comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "surface",
                "baseline_total_return",
                "adaptive_total_return",
                "baseline_volatility",
                "adaptive_volatility",
                "baseline_max_drawdown",
                "adaptive_max_drawdown",
                "baseline_sharpe",
                "adaptive_sharpe",
                "average_signal_scale",
                "average_allocation_scale",
                "fallback_row_count",
                "low_confidence_fallback_count",
                "unknown_regime_fallback_count",
            ]
        )
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    return normalized.astype("object").where(pd.notna(normalized), None)


def _resolve_path(path: str | Path, filename: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Regime policy artifact file does not exist: {resolved}")
    return resolved


def _load_json(path: str | Path, filename: str) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimePolicyArtifactError(f"{filename} must deserialize to a JSON object.")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(canonicalize_value(payload), indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ADAPTIVE_POLICY_MANIFEST_FILENAME",
    "ADAPTIVE_VS_STATIC_COMPARISON_FILENAME",
    "REGIME_POLICY_DECISIONS_FILENAME",
    "REGIME_POLICY_SUMMARY_FILENAME",
    "RegimePolicyArtifactError",
    "load_adaptive_policy_manifest",
    "load_adaptive_vs_static_comparison",
    "load_regime_policy_decisions",
    "load_regime_policy_summary",
    "write_regime_policy_artifacts",
]
