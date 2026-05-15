"""Deterministic helpers for portable dataset and feature lineage metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a SHA-256 fingerprint over stable JSON serialization."""

    normalized = _normalize_value(payload)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def feature_columns_fingerprint(columns: Sequence[str]) -> str:
    """Fingerprint feature columns after deterministic sorting."""

    return stable_json_fingerprint({"feature_columns": sorted(str(column) for column in columns)})


def dataset_schema_fingerprint(columns: Sequence[tuple[str, str]] | Mapping[str, str]) -> str:
    """Fingerprint a schema after deterministic key ordering."""

    if isinstance(columns, Mapping):
        normalized = sorted((str(name), str(dtype)) for name, dtype in columns.items())
    else:
        normalized = sorted((str(name), str(dtype)) for name, dtype in columns)
    return stable_json_fingerprint({"schema": normalized})


def portable_dataset_path(path: str | Path, *, repo_root: str | Path | None = None) -> str:
    """Return a POSIX-style repository-relative dataset path when possible."""

    candidate = Path(path)
    if candidate.is_absolute():
        if repo_root is None:
            raise ValueError("Absolute dataset paths require repo_root for portable serialization.")
        candidate = candidate.resolve().relative_to(Path(repo_root).resolve())
    return candidate.as_posix()


def build_dataset_lineage(
    *,
    logical_dataset_id: str,
    dataset_role: str,
    dataset_path: str | Path,
    dataset_contract_version: str,
    schema: Mapping[str, str],
    row_count: int,
    symbol_count: int,
    timeframe: str | None,
    start: str | None,
    end: str | None,
    source_payload: Mapping[str, Any],
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build deterministic dataset lineage metadata from explicit inputs."""

    normalized_path = portable_dataset_path(dataset_path, repo_root=repo_root)
    partition_summary = {
        "dataset_path": normalized_path,
        "row_count": int(row_count),
        "symbol_count": int(symbol_count),
        "timeframe": timeframe,
        "start": start,
        "end": end,
    }
    return {
        "logical_dataset_id": logical_dataset_id,
        "dataset_role": dataset_role,
        "dataset_path": normalized_path,
        "dataset_contract_version": dataset_contract_version,
        "schema_fingerprint": dataset_schema_fingerprint(schema),
        "partition_fingerprint": stable_json_fingerprint(partition_summary),
        "row_count": int(row_count),
        "symbol_count": int(symbol_count),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "source_fingerprint": stable_json_fingerprint(source_payload),
    }


def build_feature_lineage(
    *,
    feature_group_names: Sequence[str],
    feature_columns: Sequence[str],
    schema: Mapping[str, str],
    feature_contract_version: str,
    build_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic feature lineage metadata from explicit inputs."""

    normalized_columns = sorted(str(column) for column in feature_columns)
    return {
        "feature_group_names": sorted(str(name) for name in feature_group_names),
        "feature_column_count": len(normalized_columns),
        "feature_columns_fingerprint": feature_columns_fingerprint(normalized_columns),
        "feature_schema_fingerprint": dataset_schema_fingerprint(schema),
        "feature_contract_version": feature_contract_version,
        "feature_build_config_fingerprint": stable_json_fingerprint(build_config),
    }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return value.as_posix()
    return value
