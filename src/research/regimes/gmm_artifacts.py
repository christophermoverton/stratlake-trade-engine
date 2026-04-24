from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.research.regimes.gmm_classifier import RegimeGMMClassifierResult
from src.research.registry import canonicalize_value

REGIME_GMM_LABELS_FILENAME = "regime_gmm_labels.csv"
REGIME_GMM_POSTERIOR_FILENAME = "regime_gmm_posteriors.csv"
REGIME_GMM_SHIFT_EVENTS_FILENAME = "regime_gmm_shift_events.csv"
REGIME_GMM_SUMMARY_FILENAME = "regime_gmm_summary.json"
REGIME_GMM_MANIFEST_FILENAME = "regime_gmm_manifest.json"

_SCHEMA_VERSION = 1


class RegimeGMMArtifactError(ValueError):
    """Raised when regime GMM artifacts cannot be persisted or loaded."""


def write_regime_gmm_artifacts(
    output_dir: str | Path,
    result: RegimeGMMClassifierResult,
    *,
    run_id: str | None = None,
    source_regime_artifact_references: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(result, RegimeGMMClassifierResult):
        raise TypeError("result must be a RegimeGMMClassifierResult.")

    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    labels_path = resolved_dir / REGIME_GMM_LABELS_FILENAME
    posterior_path = resolved_dir / REGIME_GMM_POSTERIOR_FILENAME
    shift_events_path = resolved_dir / REGIME_GMM_SHIFT_EVENTS_FILENAME
    summary_path = resolved_dir / REGIME_GMM_SUMMARY_FILENAME
    manifest_path = resolved_dir / REGIME_GMM_MANIFEST_FILENAME

    labels = _artifact_frame(result.labels)
    posterior = _artifact_frame(result.posterior_probabilities)
    shift_events = _artifact_frame(result.shift_events)
    summary_payload = canonicalize_value(
        {
            **dict(result.summary),
            "metadata": canonicalize_value({**result.config.metadata, **dict(extra_metadata or {})}),
        }
    )

    _write_csv(labels_path, labels)
    _write_csv(posterior_path, posterior)
    _write_csv(shift_events_path, shift_events)
    _write_json(summary_path, summary_payload)

    manifest = {
        "artifact_type": "regime_gmm_classifier",
        "schema_version": _SCHEMA_VERSION,
        "taxonomy_version": result.taxonomy_version,
        "artifacts": {
            "regime_gmm_labels_csv": REGIME_GMM_LABELS_FILENAME,
            "regime_gmm_posteriors_csv": REGIME_GMM_POSTERIOR_FILENAME,
            "regime_gmm_shift_events_csv": REGIME_GMM_SHIFT_EVENTS_FILENAME,
            "regime_gmm_summary_json": REGIME_GMM_SUMMARY_FILENAME,
            "regime_gmm_manifest_json": REGIME_GMM_MANIFEST_FILENAME,
        },
        "source_regime_artifact_references": canonicalize_value(dict(source_regime_artifact_references or {})),
        "summary": {
            "row_count": int(len(labels)),
            "posterior_row_count": int(len(posterior)),
            "shift_event_count": int(len(shift_events)),
            "cluster_count": int(result.summary.get("cluster_count", 0)),
            "feature_columns": list(result.summary.get("feature_columns", [])),
        },
        "file_inventory": {
            REGIME_GMM_LABELS_FILENAME: {
                "path": REGIME_GMM_LABELS_FILENAME,
                "rows": int(len(labels)),
                "sha256": _sha256_file(labels_path),
            },
            REGIME_GMM_POSTERIOR_FILENAME: {
                "path": REGIME_GMM_POSTERIOR_FILENAME,
                "rows": int(len(posterior)),
                "sha256": _sha256_file(posterior_path),
            },
            REGIME_GMM_SHIFT_EVENTS_FILENAME: {
                "path": REGIME_GMM_SHIFT_EVENTS_FILENAME,
                "rows": int(len(shift_events)),
                "sha256": _sha256_file(shift_events_path),
            },
            REGIME_GMM_SUMMARY_FILENAME: {
                "path": REGIME_GMM_SUMMARY_FILENAME,
                "sha256": _sha256_file(summary_path),
            },
            REGIME_GMM_MANIFEST_FILENAME: {
                "path": REGIME_GMM_MANIFEST_FILENAME,
            },
        },
        "metadata": canonicalize_value(dict(extra_metadata or {})),
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    manifest = canonicalize_value(manifest)
    _write_json(manifest_path, manifest)
    return manifest


def load_regime_gmm_labels(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_GMM_LABELS_FILENAME))
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    return frame


def load_regime_gmm_posteriors(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_GMM_POSTERIOR_FILENAME))
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    return frame


def load_regime_gmm_shift_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_GMM_SHIFT_EVENTS_FILENAME))
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    return frame


def load_regime_gmm_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_GMM_SUMMARY_FILENAME)


def load_regime_gmm_manifest(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_GMM_MANIFEST_FILENAME)


def _artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    if "ts_utc" in normalized.columns:
        normalized["ts_utc"] = pd.to_datetime(normalized["ts_utc"], utc=True, errors="raise").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return normalized.astype("object").where(pd.notna(normalized), None)


def _resolve_path(path: str | Path, filename: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Regime GMM artifact file does not exist: {resolved}")
    return resolved


def _load_json(path: str | Path, filename: str) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeGMMArtifactError(f"{filename} must deserialize to a JSON object.")
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
    "REGIME_GMM_LABELS_FILENAME",
    "REGIME_GMM_MANIFEST_FILENAME",
    "REGIME_GMM_POSTERIOR_FILENAME",
    "REGIME_GMM_SHIFT_EVENTS_FILENAME",
    "REGIME_GMM_SUMMARY_FILENAME",
    "RegimeGMMArtifactError",
    "load_regime_gmm_labels",
    "load_regime_gmm_manifest",
    "load_regime_gmm_posteriors",
    "load_regime_gmm_shift_events",
    "load_regime_gmm_summary",
    "write_regime_gmm_artifacts",
]
