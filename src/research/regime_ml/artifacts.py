from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.registry import canonicalize_value

REGIME_CONFIDENCE_FILENAME = "regime_confidence.csv"
REGIME_ML_DIAGNOSTICS_FILENAME = "regime_ml_diagnostics.json"
REGIME_MODEL_MANIFEST_FILENAME = "regime_model_manifest.json"
REGIME_CLUSTER_MAP_FILENAME = "regime_cluster_map.csv"
REGIME_CLUSTER_DIAGNOSTICS_FILENAME = "regime_cluster_diagnostics.json"
REGIME_LABEL_MAPPING_FILENAME = "regime_label_mapping.json"


def write_regime_ml_artifacts(
    output_dir: str | Path,
    *,
    regime_confidence: pd.DataFrame,
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    cluster_map: pd.DataFrame,
    cluster_diagnostics: dict[str, Any],
    label_mapping: dict[str, Any],
) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    confidence_path = resolved_output_dir / REGIME_CONFIDENCE_FILENAME
    diagnostics_path = resolved_output_dir / REGIME_ML_DIAGNOSTICS_FILENAME
    manifest_path = resolved_output_dir / REGIME_MODEL_MANIFEST_FILENAME
    cluster_map_path = resolved_output_dir / REGIME_CLUSTER_MAP_FILENAME
    cluster_diagnostics_path = resolved_output_dir / REGIME_CLUSTER_DIAGNOSTICS_FILENAME
    label_mapping_path = resolved_output_dir / REGIME_LABEL_MAPPING_FILENAME

    regime_confidence.to_csv(confidence_path, index=False, lineterminator="\n")
    cluster_map.to_csv(cluster_map_path, index=False, lineterminator="\n")
    _write_json(diagnostics_path, diagnostics)
    _write_json(cluster_diagnostics_path, cluster_diagnostics)
    _write_json(label_mapping_path, label_mapping)

    enriched_manifest = canonicalize_value(
        {
            **dict(manifest),
            "artifacts": {
                "regime_confidence_csv": REGIME_CONFIDENCE_FILENAME,
                "regime_ml_diagnostics_json": REGIME_ML_DIAGNOSTICS_FILENAME,
                "regime_model_manifest_json": REGIME_MODEL_MANIFEST_FILENAME,
                "regime_cluster_map_csv": REGIME_CLUSTER_MAP_FILENAME,
                "regime_cluster_diagnostics_json": REGIME_CLUSTER_DIAGNOSTICS_FILENAME,
                "regime_label_mapping_json": REGIME_LABEL_MAPPING_FILENAME,
            },
            "file_inventory": {
                REGIME_CONFIDENCE_FILENAME: {
                    "path": REGIME_CONFIDENCE_FILENAME,
                    "rows": int(len(regime_confidence)),
                    "sha256": _sha256_file(confidence_path),
                },
                REGIME_ML_DIAGNOSTICS_FILENAME: {
                    "path": REGIME_ML_DIAGNOSTICS_FILENAME,
                    "sha256": _sha256_file(diagnostics_path),
                },
                REGIME_MODEL_MANIFEST_FILENAME: {
                    "path": REGIME_MODEL_MANIFEST_FILENAME,
                },
                REGIME_CLUSTER_MAP_FILENAME: {
                    "path": REGIME_CLUSTER_MAP_FILENAME,
                    "rows": int(len(cluster_map)),
                    "sha256": _sha256_file(cluster_map_path),
                },
                REGIME_CLUSTER_DIAGNOSTICS_FILENAME: {
                    "path": REGIME_CLUSTER_DIAGNOSTICS_FILENAME,
                    "sha256": _sha256_file(cluster_diagnostics_path),
                },
                REGIME_LABEL_MAPPING_FILENAME: {
                    "path": REGIME_LABEL_MAPPING_FILENAME,
                    "sha256": _sha256_file(label_mapping_path),
                },
            },
        }
    )
    _write_json(manifest_path, enriched_manifest)
    return enriched_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(canonicalize_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
