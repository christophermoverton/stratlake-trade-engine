from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.research.registry import canonicalize_value


def build_pipeline_cli_result(
    *,
    identifier: str | None = None,
    name: str | None = None,
    artifact_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    output_paths: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    state_updates: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one pipeline-safe step payload from an existing CLI result."""

    payload: dict[str, Any] = {
        "status": "completed",
    }
    if identifier is not None:
        payload["run_id"] = str(identifier)
    if name is not None:
        payload["name"] = str(name)

    normalized_artifact_dir = _normalize_optional_path(artifact_dir)
    normalized_manifest_path = _normalize_optional_path(manifest_path)
    if normalized_artifact_dir is not None:
        payload["artifact_dir"] = normalized_artifact_dir
    if normalized_manifest_path is not None:
        payload["manifest_path"] = normalized_manifest_path

    if output_paths:
        payload["output_paths"] = {
            str(key): _normalize_path_value(value)
            for key, value in dict(output_paths).items()
            if value is not None
        }
    if metrics:
        payload["metrics"] = dict(metrics)
    if extra:
        payload.update(dict(extra))
    if state_updates:
        payload["state_updates"] = dict(state_updates)
    return canonicalize_value(payload)


def _normalize_optional_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return Path(value).as_posix()


def _normalize_path_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_path_value(item)
            for key, item in dict(value).items()
        }
    if isinstance(value, list):
        return [_normalize_path_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_path_value(item) for item in value]
    return value
