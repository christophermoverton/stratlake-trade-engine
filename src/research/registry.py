from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any

REGISTRY_FILENAME = "registry.jsonl"
LOCK_SUFFIX = ".lock"
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_POLL_INTERVAL_SECONDS = 0.05


class RegistryError(RuntimeError):
    """Raised when the experiment registry cannot be read or updated safely."""


def default_registry_path(artifacts_root: Path) -> Path:
    """Return the JSONL registry path for a strategy artifact root."""

    return artifacts_root / REGISTRY_FILENAME


def canonicalize_value(value: Any) -> Any:
    """Return a recursively normalized value with deterministic mapping key order."""

    if isinstance(value, Mapping):
        return {key: canonicalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, list):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise RegistryError("Registry values must not include NaN or infinite floats.")
        return value
    return value


def serialize_canonical_json(value: Any) -> str:
    """Serialize a value to a canonical JSON string for deterministic snapshots."""

    return json.dumps(canonicalize_value(value), sort_keys=True, separators=(",", ":"))


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load all registry entries from a JSONL registry file."""

    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            payload = line.strip()
            if not payload:
                continue

            try:
                entry = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RegistryError(
                    f"Registry file '{path}' contains invalid JSON on line {line_number}."
                ) from exc

            if not isinstance(entry, dict):
                raise RegistryError(
                    f"Registry file '{path}' contains a non-object entry on line {line_number}."
                )

            entries.append(entry)
    return entries


def filter_by_strategy_name(
    entries: Iterable[dict[str, Any]], strategy_name: str
) -> list[dict[str, Any]]:
    """Return registry entries for one strategy name."""

    return [entry for entry in entries if entry.get("strategy_name") == strategy_name]


def filter_by_metric_threshold(
    entries: Iterable[dict[str, Any]],
    metric_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> list[dict[str, Any]]:
    """Return registry entries whose summary metric lies within the requested bounds."""

    filtered: list[dict[str, Any]] = []
    for entry in entries:
        metrics_summary = entry.get("metrics_summary")
        if not isinstance(metrics_summary, dict):
            continue

        value = metrics_summary.get(metric_name)
        if not isinstance(value, int | float):
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        filtered.append(entry)
    return filtered


def append_registry_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """Append one registry entry after validating run-id uniqueness."""

    canonical_entry = canonicalize_value(entry)
    if not isinstance(canonical_entry, dict):
        raise RegistryError("Registry entries must serialize to a JSON object.")

    run_id = canonical_entry.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RegistryError("Registry entries must include a non-empty string run_id.")

    line = json.dumps(canonical_entry, separators=(",", ":"), sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    with _registry_lock(path):
        existing_ids = {existing.get("run_id") for existing in load_registry(path)}
        if run_id in existing_ids:
            raise RegistryError(f"Registry already contains run_id '{run_id}'.")

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(str(path), flags)
        try:
            os.write(descriptor, line.encode("utf-8"))
            os.fsync(descriptor)
        except OSError as exc:
            raise RegistryError(f"Failed to append registry entry to '{path}'.") from exc
        finally:
            os.close(descriptor)


def upsert_registry_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """Insert or replace one registry entry by deterministic run-id."""

    canonical_entry = canonicalize_value(entry)
    if not isinstance(canonical_entry, dict):
        raise RegistryError("Registry entries must serialize to a JSON object.")

    run_id = canonical_entry.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RegistryError("Registry entries must include a non-empty string run_id.")

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(canonical_entry, separators=(",", ":"), sort_keys=False) + "\n"

    with _registry_lock(path):
        existing_entries = [
            existing
            for existing in load_registry(path)
            if existing.get("run_id") != run_id
        ]
        serialized = [
            json.dumps(canonicalize_value(existing), separators=(",", ":"), sort_keys=False)
            for existing in existing_entries
        ]
        serialized.append(line.rstrip("\n"))
        payload = "\n".join(serialized)
        if payload:
            payload += "\n"

        try:
            path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            raise RegistryError(f"Failed to upsert registry entry to '{path}'.") from exc


@contextmanager
def _registry_lock(path: Path) -> Iterable[None]:
    lock_path = path.with_name(f"{path.name}{LOCK_SUFFIX}")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS

    while True:
        try:
            descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            if time.monotonic() >= deadline:
                raise RegistryError(
                    f"Timed out waiting for registry lock '{lock_path.name}'."
                ) from exc
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)

    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
