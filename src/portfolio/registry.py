from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.registry import (
    RegistryError,
    _registry_lock,
    canonicalize_value,
    stable_timestamp_from_run_id,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO_TEMPLATE_REGISTRY_PATH = REPO_ROOT / "artifacts" / "registry" / "portfolios.jsonl"
_PORTFOLIO_TEMPLATE_RUN_TYPE = "portfolio_template"


def default_portfolio_template_registry_path() -> Path:
    """Return the default append-only registry for versioned portfolio templates."""

    return DEFAULT_PORTFOLIO_TEMPLATE_REGISTRY_PATH


def load_portfolio_template_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load portfolio template entries from a JSONL registry path."""

    resolved_path = default_portfolio_template_registry_path() if path is None else Path(path)
    if not resolved_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                entry = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RegistryError(
                    f"Portfolio template registry '{resolved_path}' contains invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(entry, dict):
                raise RegistryError(
                    f"Portfolio template registry '{resolved_path}' contains non-object entry on line {line_number}."
                )
            run_type = str(entry.get("run_type") or "").strip()
            if run_type != _PORTFOLIO_TEMPLATE_RUN_TYPE:
                continue
            _validate_template_entry(entry, source_path=resolved_path, line_number=line_number)
            entries.append(canonicalize_value(entry))
    return entries


def register_portfolio_template(
    *,
    name: str,
    version: str,
    definition: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append one deterministic portfolio template definition when absent."""

    normalized_name = _normalize_required_string(name, field_name="name")
    normalized_version = _normalize_required_string(version, field_name="version")
    if not isinstance(definition, Mapping):
        raise ValueError("definition must be a mapping.")
    resolved_registry_path = (
        default_portfolio_template_registry_path() if registry_path is None else Path(registry_path)
    )

    run_id = _template_run_id(normalized_name, normalized_version)
    entry = canonicalize_value(
        {
            "run_id": run_id,
            "run_type": _PORTFOLIO_TEMPLATE_RUN_TYPE,
            "timestamp": stable_timestamp_from_run_id(run_id),
            "portfolio_name": normalized_name,
            "version": normalized_version,
            "definition": dict(definition),
            "metadata": {} if metadata is None else dict(metadata),
        }
    )
    if not isinstance(entry, dict):
        raise RegistryError("Portfolio template entry must serialize to a JSON object.")

    resolved_registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, separators=(",", ":"), sort_keys=False) + "\n"

    with _registry_lock(resolved_registry_path):
        existing = load_portfolio_template_registry(resolved_registry_path)
        for candidate in existing:
            if str(candidate.get("portfolio_name") or "") != normalized_name:
                continue
            if str(candidate.get("version") or "") != normalized_version:
                continue
            if canonicalize_value(candidate) != entry:
                raise RegistryError(
                    "Portfolio template registry already contains conflicting entry for "
                    f"name={normalized_name!r}, version={normalized_version!r}."
                )
            return entry

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        try:
            descriptor = os.open(str(resolved_registry_path), flags)
        except OSError as exc:
            raise RegistryError(
                f"Failed to open portfolio template registry '{resolved_registry_path}' for append."
            ) from exc
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        except OSError as exc:
            raise RegistryError(
                f"Failed to append portfolio template registry entry to '{resolved_registry_path}'."
            ) from exc
        finally:
            os.close(descriptor)
    return entry


def resolve_portfolio_template_definition(
    entries: Sequence[Mapping[str, Any]],
    *,
    name: str,
    version: str | None,
) -> dict[str, Any] | None:
    """Resolve one template definition by name and optional version."""

    normalized_name = _normalize_required_string(name, field_name="name")
    matches = [
        entry
        for entry in entries
        if str(entry.get("portfolio_name") or "") == normalized_name
    ]
    if not matches:
        return None

    if version is not None:
        normalized_version = _normalize_required_string(version, field_name="version")
        for entry in matches:
            if str(entry.get("version") or "") == normalized_version:
                definition = entry.get("definition")
                if isinstance(definition, Mapping):
                    return dict(definition)
                break
        return None

    ordered = sorted(matches, key=lambda item: _version_sort_key(str(item.get("version") or "")))
    definition = ordered[-1].get("definition")
    if not isinstance(definition, Mapping):
        return None
    return dict(definition)


def _template_run_id(name: str, version: str) -> str:
    normalized_name = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower()).strip("_") or "portfolio"
    normalized_version = re.sub(r"[^0-9A-Za-z]+", "_", version.strip().lower()).strip("_") or "v"
    return f"portfolio_template_{normalized_name}_{normalized_version}"


def _validate_template_entry(entry: Mapping[str, Any], *, source_path: Path, line_number: int) -> None:
    if not isinstance(entry.get("run_id"), str) or not str(entry.get("run_id") or "").strip():
        raise RegistryError(
            f"Portfolio template registry '{source_path}' entry on line {line_number} is missing run_id."
        )
    if not isinstance(entry.get("portfolio_name"), str) or not str(entry.get("portfolio_name") or "").strip():
        raise RegistryError(
            f"Portfolio template registry '{source_path}' entry on line {line_number} is missing portfolio_name."
        )
    if not isinstance(entry.get("version"), str) or not str(entry.get("version") or "").strip():
        raise RegistryError(
            f"Portfolio template registry '{source_path}' entry on line {line_number} is missing version."
        )
    if not isinstance(entry.get("definition"), Mapping):
        raise RegistryError(
            f"Portfolio template registry '{source_path}' entry on line {line_number} must include object field definition."
        )


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _version_sort_key(value: str) -> tuple[Any, ...]:
    normalized = value.strip()
    parts = re.split(r"[^0-9A-Za-z]+", normalized)
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key) or ((1, normalized.lower()),)


__all__ = [
    "DEFAULT_PORTFOLIO_TEMPLATE_REGISTRY_PATH",
    "default_portfolio_template_registry_path",
    "load_portfolio_template_registry",
    "register_portfolio_template",
    "resolve_portfolio_template_definition",
]