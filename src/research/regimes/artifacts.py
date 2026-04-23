from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.regimes.taxonomy import (
    REGIME_AUDIT_COLUMNS,
    REGIME_DIMENSIONS,
    REGIME_LABEL_COLUMNS,
    REGIME_OUTPUT_COLUMNS,
    REGIME_STATE_COLUMNS,
    REGIME_TAXONOMY,
    TAXONOMY_VERSION,
)
from src.research.regimes.validation import validate_regime_labels
from src.research.registry import canonicalize_value

DEFAULT_REGIME_ARTIFACTS_ROOT = Path("artifacts") / "regimes"
REGIME_LABELS_FILENAME = "regime_labels.csv"
REGIME_SUMMARY_FILENAME = "regime_summary.json"
REGIME_MANIFEST_FILENAME = "manifest.json"
_REGIME_MANIFEST_SCHEMA_VERSION = 1


class RegimeArtifactError(ValueError):
    """Raised when canonical regime artifact persistence fails."""


@dataclass(frozen=True)
class RegimeArtifactPaths:
    """Canonical file paths for one regime artifact bundle."""

    output_dir: Path
    labels_csv_path: Path
    summary_json_path: Path
    manifest_json_path: Path


def resolve_regime_artifact_dir(
    run_id: str,
    *,
    artifacts_root: str | Path = DEFAULT_REGIME_ARTIFACTS_ROOT,
) -> Path:
    """Return a deterministic output directory for one regime-label artifact run."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RegimeArtifactError("run_id must be a non-empty string.")
    return Path(artifacts_root) / normalized_run_id


def write_regime_artifacts(
    output_dir: str | Path,
    labels: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[RegimeArtifactPaths, dict[str, Any]]:
    """Persist canonical regime labels, summary metadata, and manifest metadata."""

    normalized_labels = _artifact_labels_frame(labels)
    summary_payload = _build_regime_summary_payload(normalized_labels, metadata=metadata)

    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    paths = RegimeArtifactPaths(
        output_dir=resolved_output_dir,
        labels_csv_path=resolved_output_dir / REGIME_LABELS_FILENAME,
        summary_json_path=resolved_output_dir / REGIME_SUMMARY_FILENAME,
        manifest_json_path=resolved_output_dir / REGIME_MANIFEST_FILENAME,
    )

    _write_csv(paths.labels_csv_path, normalized_labels)
    _write_json(paths.summary_json_path, summary_payload)

    manifest_payload = _build_regime_manifest_payload(paths=paths, summary_payload=summary_payload)
    _write_json(paths.manifest_json_path, manifest_payload)

    return paths, manifest_payload


def load_regime_labels(path: str | Path) -> pd.DataFrame:
    """Load canonical regime labels from a file path or artifact directory."""

    labels_path = _resolve_labels_path(path)
    frame = pd.read_csv(labels_path)
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    return validate_regime_labels(frame)


def load_regime_summary(path: str | Path) -> dict[str, Any]:
    """Load one persisted regime summary JSON payload."""

    summary_path = _resolve_summary_path(path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeArtifactError(f"Regime summary payload must be an object: {summary_path}")
    return payload


def load_regime_manifest(path: str | Path) -> dict[str, Any]:
    """Load one persisted regime manifest JSON payload."""

    manifest_path = _resolve_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeArtifactError(f"Regime manifest payload must be an object: {manifest_path}")
    return payload


def attach_regime_artifacts_to_manifest(
    manifest_path: str | Path,
    *,
    regime_manifest_path: str | Path,
    section_key: str = "regime_artifacts",
) -> dict[str, Any]:
    """Attach one regime artifact bundle to an existing manifest-style JSON payload."""

    resolved_manifest_path = Path(manifest_path)
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {resolved_manifest_path}")

    manifest_payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise RegimeArtifactError(
            f"Manifest payload must be an object before regime attachment: {resolved_manifest_path}"
        )

    regime_manifest = load_regime_manifest(regime_manifest_path)
    resolved_regime_manifest_path = _resolve_manifest_path(regime_manifest_path)
    relative_regime_manifest_path = _relative_path_or_posix(
        resolved_regime_manifest_path,
        base_dir=resolved_manifest_path.parent,
    )

    manifest_payload[section_key] = canonicalize_value(
        {
            "artifact_type": "regime_labels",
            "manifest_path": relative_regime_manifest_path,
            "taxonomy_version": regime_manifest.get("taxonomy_version"),
            "summary": dict(regime_manifest.get("summary", {})),
            "files": dict(regime_manifest.get("artifacts", {})),
        }
    )
    normalized_manifest = canonicalize_value(manifest_payload)
    _write_json(resolved_manifest_path, normalized_manifest)
    return normalized_manifest


def _artifact_labels_frame(labels: pd.DataFrame) -> pd.DataFrame:
    normalized = validate_regime_labels(labels)
    artifact = normalized.copy(deep=True)
    artifact.attrs = {}
    artifact = artifact.loc[:, list(REGIME_OUTPUT_COLUMNS)]
    artifact["ts_utc"] = pd.to_datetime(artifact["ts_utc"], utc=True, errors="raise").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    artifact = artifact.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return artifact


def _build_regime_summary_payload(
    labels: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    timestamp_series = pd.to_datetime(labels["ts_utc"], utc=True, errors="raise")
    defined_row_count = int(pd.Series(labels["is_defined"]).astype("bool").sum())

    state_distribution: dict[str, dict[str, int]] = {}
    for dimension in REGIME_DIMENSIONS:
        column = REGIME_STATE_COLUMNS[dimension]
        counts = labels[column].astype("string").value_counts(dropna=False)
        state_distribution[dimension] = {
            state: int(counts.get(state, 0))
            for state in REGIME_TAXONOMY[dimension].labels
        }

    payload = {
        "artifact_type": "regime_labels",
        "schema_version": _REGIME_MANIFEST_SCHEMA_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "row_count": int(len(labels)),
        "defined_row_count": defined_row_count,
        "undefined_row_count": int(len(labels)) - defined_row_count,
        "ts_utc_start": None
        if labels.empty
        else timestamp_series.min().isoformat().replace("+00:00", "Z"),
        "ts_utc_end": None
        if labels.empty
        else timestamp_series.max().isoformat().replace("+00:00", "Z"),
        "label_columns": list(REGIME_LABEL_COLUMNS),
        "metric_columns": list(REGIME_AUDIT_COLUMNS),
        "columns": list(REGIME_OUTPUT_COLUMNS),
        "state_distribution": state_distribution,
        "metadata": canonicalize_value(dict(metadata or {})),
    }
    return canonicalize_value(payload)


def _build_regime_manifest_payload(
    *,
    paths: RegimeArtifactPaths,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    labels_rows = _count_csv_data_rows(paths.labels_csv_path)

    manifest_payload = {
        "artifact_type": "regime_labels",
        "schema_version": _REGIME_MANIFEST_SCHEMA_VERSION,
        "run_id": paths.output_dir.name,
        "taxonomy_version": summary_payload.get("taxonomy_version", TAXONOMY_VERSION),
        "artifacts": {
            "regime_labels_csv": REGIME_LABELS_FILENAME,
            "regime_summary_json": REGIME_SUMMARY_FILENAME,
            "manifest_json": REGIME_MANIFEST_FILENAME,
        },
        "file_inventory": {
            REGIME_LABELS_FILENAME: {
                "path": REGIME_LABELS_FILENAME,
                "rows": labels_rows,
                "sha256": _sha256_file(paths.labels_csv_path),
            },
            REGIME_SUMMARY_FILENAME: {
                "path": REGIME_SUMMARY_FILENAME,
                "sha256": _sha256_file(paths.summary_json_path),
            },
            REGIME_MANIFEST_FILENAME: {
                "path": REGIME_MANIFEST_FILENAME,
            },
        },
        "summary": {
            "row_count": int(summary_payload.get("row_count", 0)),
            "defined_row_count": int(summary_payload.get("defined_row_count", 0)),
            "undefined_row_count": int(summary_payload.get("undefined_row_count", 0)),
            "ts_utc_start": summary_payload.get("ts_utc_start"),
            "ts_utc_end": summary_payload.get("ts_utc_end"),
        },
    }
    return canonicalize_value(manifest_payload)


def _resolve_labels_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / REGIME_LABELS_FILENAME
    if not resolved.exists():
        raise FileNotFoundError(f"Regime labels file does not exist: {resolved}")
    return resolved


def _resolve_summary_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / REGIME_SUMMARY_FILENAME
    if not resolved.exists():
        raise FileNotFoundError(f"Regime summary file does not exist: {resolved}")
    return resolved


def _resolve_manifest_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / REGIME_MANIFEST_FILENAME
    if not resolved.exists():
        raise FileNotFoundError(f"Regime manifest file does not exist: {resolved}")
    return resolved


def _count_csv_data_rows(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return 0
    line_count = content.count("\n")
    if content.endswith("\n"):
        line_count -= 1
    return max(line_count, 0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path_or_posix(path: Path, *, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "DEFAULT_REGIME_ARTIFACTS_ROOT",
    "REGIME_LABELS_FILENAME",
    "REGIME_MANIFEST_FILENAME",
    "REGIME_SUMMARY_FILENAME",
    "RegimeArtifactError",
    "RegimeArtifactPaths",
    "attach_regime_artifacts_to_manifest",
    "load_regime_labels",
    "load_regime_manifest",
    "load_regime_summary",
    "resolve_regime_artifact_dir",
    "write_regime_artifacts",
]
