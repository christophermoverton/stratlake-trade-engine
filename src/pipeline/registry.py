from __future__ import annotations

import os
from pathlib import Path

from src.research.registry import (
    RegistryError,
    _registry_lock,
    canonicalize_value,
    load_registry,
    serialize_canonical_json,
)

PIPELINE_REGISTRY_FILENAME = "registry.jsonl"
_VALID_PIPELINE_STATUSES = frozenset({"completed", "failed", "partial"})


def pipeline_registry_path(root: Path | None = None) -> Path:
    """Return the deterministic pipeline registry path."""

    base = Path.cwd() if root is None else Path(root)
    return (base / "artifacts" / "pipelines" / PIPELINE_REGISTRY_FILENAME).resolve()


def normalize_pipeline_artifact_dir(path: str | Path, *, root: Path | None = None) -> str:
    """Normalize one pipeline artifact path to stable POSIX formatting."""

    base = Path.cwd() if root is None else Path(root)
    candidate = Path(path)
    resolved_candidate = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    try:
        relative = resolved_candidate.relative_to(base.resolve())
        return relative.as_posix()
    except ValueError:
        return resolved_candidate.as_posix()


def build_pipeline_registry_entry(
    *,
    pipeline_run_id: str,
    pipeline_name: str | None,
    status: str,
    artifact_dir: str | Path,
    root: Path | None = None,
) -> dict[str, str | None]:
    """Build one canonical pipeline registry entry."""

    normalized_run_id = _normalize_required_string(pipeline_run_id, field_name="pipeline_run_id")
    normalized_name = _normalize_optional_string(pipeline_name)
    normalized_status = _normalize_status(status)
    normalized_artifact_dir = normalize_pipeline_artifact_dir(artifact_dir, root=root)

    entry = {
        "artifact_dir": normalized_artifact_dir,
        "pipeline_name": normalized_name,
        "pipeline_run_id": normalized_run_id,
        "status": normalized_status,
    }
    canonical = canonicalize_value(entry)
    if not isinstance(canonical, dict):
        raise RegistryError("Pipeline registry entries must serialize to a JSON object.")
    return canonical


def register_pipeline_run(
    *,
    pipeline_run_id: str,
    pipeline_name: str | None,
    status: str,
    artifact_dir: str | Path,
    registry_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, str | None]:
    """Append one deterministic pipeline registry entry if absent."""

    resolved_root = Path.cwd() if root is None else Path(root)
    resolved_registry_path = pipeline_registry_path(resolved_root) if registry_path is None else Path(registry_path)
    entry = build_pipeline_registry_entry(
        pipeline_run_id=pipeline_run_id,
        pipeline_name=pipeline_name,
        status=status,
        artifact_dir=artifact_dir,
        root=resolved_root,
    )
    line = serialize_canonical_json(entry) + "\n"

    resolved_registry_path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock(resolved_registry_path):
        existing_entries = load_registry(resolved_registry_path)
        for existing in existing_entries:
            if existing.get("pipeline_run_id") != entry["pipeline_run_id"]:
                continue
            comparable_existing = canonicalize_value(
                {
                    "artifact_dir": existing.get("artifact_dir"),
                    "pipeline_name": existing.get("pipeline_name"),
                    "pipeline_run_id": existing.get("pipeline_run_id"),
                    "status": existing.get("status"),
                }
            )
            if comparable_existing != entry:
                raise RegistryError(
                    f"Pipeline registry already contains conflicting entry for pipeline_run_id "
                    f"'{entry['pipeline_run_id']}'."
                )
            return entry

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        try:
            descriptor = os.open(str(resolved_registry_path), flags)
        except OSError as exc:
            raise RegistryError(
                f"Failed to open pipeline registry path '{resolved_registry_path}' for append."
            ) from exc
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        except OSError as exc:
            raise RegistryError(
                f"Failed to append pipeline registry entry to '{resolved_registry_path}'."
            ) from exc
        finally:
            os.close(descriptor)
    return entry


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_status(value: object) -> str:
    normalized = _normalize_required_string(value, field_name="status").lower()
    if normalized not in _VALID_PIPELINE_STATUSES:
        formatted = ", ".join(sorted(_VALID_PIPELINE_STATUSES))
        raise ValueError(f"status must be one of: {formatted}.")
    return normalized
