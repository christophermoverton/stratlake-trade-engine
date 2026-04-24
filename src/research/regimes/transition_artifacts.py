from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.regimes.transition import (
    TRANSITION_EVENT_COLUMNS,
    TRANSITION_WINDOW_TAG_COLUMNS,
    RegimeTransitionAnalysisResult,
)
from src.research.regimes.taxonomy import TAXONOMY_VERSION
from src.research.registry import canonicalize_value

REGIME_TRANSITION_EVENTS_FILENAME = "regime_transition_events.csv"
REGIME_TRANSITION_WINDOWS_FILENAME = "regime_transition_windows.csv"
REGIME_TRANSITION_SUMMARY_FILENAME = "regime_transition_summary.json"
REGIME_TRANSITION_MANIFEST_FILENAME = "regime_transition_manifest.json"

_SCHEMA_VERSION = 1


class RegimeTransitionArtifactError(ValueError):
    """Raised when transition artifact persistence fails."""


def write_regime_transition_artifacts(
    output_dir: str | Path,
    result: RegimeTransitionAnalysisResult,
    *,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    events = _artifact_frame(result.events, timestamp_columns=("ts_utc",))
    windows = _artifact_frame(result.windows, timestamp_columns=("ts_utc", "transition_ts_utc"))
    summary_payload = _build_transition_summary_payload(result, run_id=run_id, extra_metadata=extra_metadata)

    events_path = resolved_dir / REGIME_TRANSITION_EVENTS_FILENAME
    windows_path = resolved_dir / REGIME_TRANSITION_WINDOWS_FILENAME
    summary_path = resolved_dir / REGIME_TRANSITION_SUMMARY_FILENAME
    manifest_path = resolved_dir / REGIME_TRANSITION_MANIFEST_FILENAME

    _write_csv(events_path, events)
    _write_csv(windows_path, windows)
    _write_json(summary_path, summary_payload)

    manifest_payload = _build_transition_manifest_payload(
        result=result,
        summary_payload=summary_payload,
        events_path=events_path,
        windows_path=windows_path,
        summary_path=summary_path,
        run_id=run_id,
        extra_metadata=extra_metadata,
    )
    _write_json(manifest_path, manifest_payload)
    return manifest_payload


def load_regime_transition_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_TRANSITION_EVENTS_FILENAME))
    if "ts_utc" in frame.columns:
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True, errors="coerce")
    frame.attrs = {}
    return frame


def load_regime_transition_windows(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, REGIME_TRANSITION_WINDOWS_FILENAME))
    for column in ("ts_utc", "transition_ts_utc"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame.attrs = {}
    return frame


def load_regime_transition_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, REGIME_TRANSITION_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeTransitionArtifactError("Transition summary payload must be a JSON object.")
    return payload


def load_regime_transition_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, REGIME_TRANSITION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeTransitionArtifactError("Transition manifest payload must be a JSON object.")
    return payload


def _artifact_frame(frame: pd.DataFrame, *, timestamp_columns: tuple[str, ...]) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized.attrs = {}
    for column in timestamp_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="raise").dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    return normalized


def _build_transition_summary_payload(
    result: RegimeTransitionAnalysisResult,
    *,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    events = result.events
    windows = result.windows
    summaries = result.event_summaries
    event_counts_by_dimension = (
        events.groupby("transition_dimension", sort=True).size().astype("int64").to_dict()
        if not events.empty
        else {}
    )
    coverage_counts = (
        summaries["coverage_status"].value_counts().sort_index().astype("int64").to_dict()
        if not summaries.empty and "coverage_status" in summaries.columns
        else {}
    )
    payload: dict[str, Any] = {
        "artifact_type": "regime_transition_analysis",
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "taxonomy_version": result.config.taxonomy_version,
        "transition_dimensions": sorted(events["transition_dimension"].unique().tolist()) if not events.empty else [],
        "event_count": int(len(events)),
        "stress_transition_count": int(events["is_stress_transition"].astype("bool").sum()) if not events.empty else 0,
        "window_configuration": {
            "pre_event_rows": result.config.pre_event_rows,
            "post_event_rows": result.config.post_event_rows,
            "allow_window_overlap": result.config.allow_window_overlap,
        },
        "coverage_counts": {str(key): int(value) for key, value in coverage_counts.items()},
        "event_counts_by_dimension": {str(key): int(value) for key, value in event_counts_by_dimension.items()},
        "window_row_count": int(len(windows)),
        "overlap_row_count": int(windows["transition_has_window_overlap"].astype("bool").sum()) if not windows.empty else 0,
        "event_columns": list(TRANSITION_EVENT_COLUMNS),
        "window_columns": list(windows.columns) if not windows.empty else list(TRANSITION_WINDOW_TAG_COLUMNS),
        "summary_columns": list(summaries.columns),
        "event_summaries": canonicalize_value(_summary_rows(summaries)),
        "metadata": canonicalize_value({**result.metadata, **(extra_metadata or {})}),
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    return canonicalize_value(payload)


def _build_transition_manifest_payload(
    *,
    result: RegimeTransitionAnalysisResult,
    summary_payload: dict[str, Any],
    events_path: Path,
    windows_path: Path,
    summary_path: Path,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "artifact_type": "regime_transition_analysis",
        "schema_version": _SCHEMA_VERSION,
        "surface": result.surface,
        "taxonomy_version": TAXONOMY_VERSION,
        "artifacts": {
            "regime_transition_events_csv": REGIME_TRANSITION_EVENTS_FILENAME,
            "regime_transition_windows_csv": REGIME_TRANSITION_WINDOWS_FILENAME,
            "regime_transition_summary_json": REGIME_TRANSITION_SUMMARY_FILENAME,
            "regime_transition_manifest_json": REGIME_TRANSITION_MANIFEST_FILENAME,
        },
        "summary": {
            "event_count": int(summary_payload.get("event_count", 0)),
            "stress_transition_count": int(summary_payload.get("stress_transition_count", 0)),
            "window_row_count": int(summary_payload.get("window_row_count", 0)),
            "coverage_counts": dict(summary_payload.get("coverage_counts", {})),
        },
        "file_inventory": {
            REGIME_TRANSITION_EVENTS_FILENAME: {
                "path": REGIME_TRANSITION_EVENTS_FILENAME,
                "rows": _count_csv_rows(events_path),
                "sha256": _sha256_file(events_path),
            },
            REGIME_TRANSITION_WINDOWS_FILENAME: {
                "path": REGIME_TRANSITION_WINDOWS_FILENAME,
                "rows": _count_csv_rows(windows_path),
                "sha256": _sha256_file(windows_path),
            },
            REGIME_TRANSITION_SUMMARY_FILENAME: {
                "path": REGIME_TRANSITION_SUMMARY_FILENAME,
                "sha256": _sha256_file(summary_path),
            },
            REGIME_TRANSITION_MANIFEST_FILENAME: {
                "path": REGIME_TRANSITION_MANIFEST_FILENAME,
            },
        },
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    if extra_metadata:
        manifest["metadata"] = canonicalize_value(extra_metadata)
    return canonicalize_value(manifest)


def _summary_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.copy(deep=True)
    for column in normalized.columns:
        if str(column).endswith("ts_utc") or column == "ts_utc":
            normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="raise").dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
    normalized = normalized.astype("object").where(pd.notna(normalized), None)
    return normalized.to_dict(orient="records")


def _resolve_path(path: str | Path, filename: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Transition artifact file does not exist: {resolved}")
    return resolved


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
    "REGIME_TRANSITION_EVENTS_FILENAME",
    "REGIME_TRANSITION_MANIFEST_FILENAME",
    "REGIME_TRANSITION_SUMMARY_FILENAME",
    "REGIME_TRANSITION_WINDOWS_FILENAME",
    "RegimeTransitionArtifactError",
    "load_regime_transition_events",
    "load_regime_transition_manifest",
    "load_regime_transition_summary",
    "load_regime_transition_windows",
    "write_regime_transition_artifacts",
]
