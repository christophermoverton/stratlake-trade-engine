from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.regimes.attribution import (
    RegimeAttributionResult,
    RegimeComparisonResult,
    RegimeTransitionAttributionResult,
    render_regime_attribution_report,
)
from src.research.regimes.taxonomy import TAXONOMY_VERSION
from src.research.registry import canonicalize_value

REGIME_ATTRIBUTION_SUMMARY_FILENAME = "regime_attribution_summary.json"
REGIME_ATTRIBUTION_TABLE_FILENAME = "regime_attribution_table.csv"
REGIME_COMPARISON_SUMMARY_FILENAME = "regime_comparison_summary.json"
REGIME_COMPARISON_TABLE_FILENAME = "regime_comparison_table.csv"
REGIME_ATTRIBUTION_REPORT_FILENAME = "regime_attribution_report.md"
REGIME_ATTRIBUTION_MANIFEST_FILENAME = "regime_attribution_manifest.json"

_SCHEMA_VERSION = 1


class RegimeAttributionArtifactError(ValueError):
    """Raised when regime attribution artifacts cannot be persisted or loaded."""


def resolve_regime_attribution_artifact_dir(
    run_id: str,
    *,
    artifacts_root: str | Path = Path("artifacts") / "regime_attribution",
) -> Path:
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RegimeAttributionArtifactError("run_id must be a non-empty string.")
    return Path(artifacts_root) / normalized_run_id


def write_regime_attribution_artifacts(
    output_dir: str | Path,
    attribution: RegimeAttributionResult,
    *,
    transition: RegimeTransitionAttributionResult | None = None,
    comparison: RegimeComparisonResult | None = None,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    report_markdown = render_regime_attribution_report(
        attribution,
        transition=transition,
        comparison=comparison,
    )

    attribution_table = _artifact_frame(attribution.attribution_table)
    _write_csv(resolved_dir / REGIME_ATTRIBUTION_TABLE_FILENAME, attribution_table)
    _write_json(
        resolved_dir / REGIME_ATTRIBUTION_SUMMARY_FILENAME,
        _build_summary_payload(attribution, transition=transition, run_id=run_id, extra_metadata=extra_metadata),
    )

    if comparison is not None:
        _write_csv(resolved_dir / REGIME_COMPARISON_TABLE_FILENAME, _artifact_frame(comparison.comparison_table))
        _write_json(
            resolved_dir / REGIME_COMPARISON_SUMMARY_FILENAME,
            canonicalize_value({**comparison.summary, **dict(extra_metadata or {})}),
        )

    (resolved_dir / REGIME_ATTRIBUTION_REPORT_FILENAME).write_text(report_markdown, encoding="utf-8")
    manifest = _build_manifest_payload(
        output_dir=resolved_dir,
        attribution=attribution,
        comparison=comparison,
        run_id=run_id,
        extra_metadata=extra_metadata,
    )
    _write_json(resolved_dir / REGIME_ATTRIBUTION_MANIFEST_FILENAME, manifest)
    return manifest


def load_regime_attribution_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_ATTRIBUTION_SUMMARY_FILENAME)


def load_regime_comparison_summary(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_COMPARISON_SUMMARY_FILENAME)


def load_regime_attribution_table(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, REGIME_ATTRIBUTION_TABLE_FILENAME)


def load_regime_comparison_table(path: str | Path) -> pd.DataFrame:
    return _load_csv(path, REGIME_COMPARISON_TABLE_FILENAME)


def load_regime_attribution_manifest(path: str | Path) -> dict[str, Any]:
    return _load_json(path, REGIME_ATTRIBUTION_MANIFEST_FILENAME)


def _build_summary_payload(
    attribution: RegimeAttributionResult,
    *,
    transition: RegimeTransitionAttributionResult | None,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        **attribution.summary,
        "transition_summary": transition.summary if transition is not None else None,
        "metadata": canonicalize_value(
            {
                **dict(attribution.metadata),
                **(dict(transition.metadata) if transition is not None else {}),
                **dict(extra_metadata or {}),
            }
        ),
    }
    if run_id is not None:
        payload["run_id"] = str(run_id)
    return canonicalize_value(payload)


def _build_manifest_payload(
    *,
    output_dir: Path,
    attribution: RegimeAttributionResult,
    comparison: RegimeComparisonResult | None,
    run_id: str | None,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    files = [
        REGIME_ATTRIBUTION_MANIFEST_FILENAME,
        REGIME_ATTRIBUTION_REPORT_FILENAME,
        REGIME_ATTRIBUTION_SUMMARY_FILENAME,
        REGIME_ATTRIBUTION_TABLE_FILENAME,
    ]
    if comparison is not None:
        files.extend([REGIME_COMPARISON_SUMMARY_FILENAME, REGIME_COMPARISON_TABLE_FILENAME])
    files = sorted(files)
    manifest: dict[str, Any] = {
        "artifact_type": "regime_attribution_bundle",
        "schema_version": _SCHEMA_VERSION,
        "surface": attribution.surface,
        "dimension": attribution.dimension,
        "taxonomy_version": TAXONOMY_VERSION,
        "artifact_files": files,
        "artifacts": {
            filename: _manifest_file_entry(output_dir / filename)
            for filename in files
            if filename != REGIME_ATTRIBUTION_MANIFEST_FILENAME
        },
        "summary": {
            "fragility_flag": bool(attribution.summary.get("fragility_flag", False)),
            "dominant_regime_label": attribution.summary.get("dominant_regime_label"),
            "regime_count": int(attribution.summary.get("regime_count", 0)),
            "comparison_included": comparison is not None,
        },
        "metadata": canonicalize_value(dict(extra_metadata or {})),
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    return canonicalize_value(manifest)


def _artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.attrs = {}
    return normalized


def _manifest_file_entry(path: Path) -> dict[str, Any]:
    entry = {"path": path.name, "sha256": _sha256_file(path)}
    if path.suffix.lower() == ".csv":
        entry["rows"] = _count_csv_rows(path)
    if path.suffix.lower() == ".md":
        entry["bytes"] = path.stat().st_size
    return entry


def _resolve_path(path: str | Path, filename: str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    if not resolved.exists():
        raise FileNotFoundError(f"Attribution artifact file does not exist: {resolved}")
    return resolved


def _load_json(path: str | Path, filename: str) -> dict[str, Any]:
    payload = json.loads(_resolve_path(path, filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegimeAttributionArtifactError(f"{filename} must deserialize to a JSON object.")
    return payload


def _load_csv(path: str | Path, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_path(path, filename))
    frame.attrs = {}
    return frame


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
    "REGIME_ATTRIBUTION_MANIFEST_FILENAME",
    "REGIME_ATTRIBUTION_REPORT_FILENAME",
    "REGIME_ATTRIBUTION_SUMMARY_FILENAME",
    "REGIME_ATTRIBUTION_TABLE_FILENAME",
    "REGIME_COMPARISON_SUMMARY_FILENAME",
    "REGIME_COMPARISON_TABLE_FILENAME",
    "RegimeAttributionArtifactError",
    "load_regime_attribution_manifest",
    "load_regime_attribution_summary",
    "load_regime_attribution_table",
    "load_regime_comparison_summary",
    "load_regime_comparison_table",
    "resolve_regime_attribution_artifact_dir",
    "write_regime_attribution_artifacts",
]
