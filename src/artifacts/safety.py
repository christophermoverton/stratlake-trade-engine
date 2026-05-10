from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

CollisionPolicy = Literal["fail", "reuse"]

RUNNING_MARKER = "_RUNNING.json"
SUCCESS_MARKER = "_SUCCESS.json"
FAILED_MARKER = "_FAILED.json"
_MARKER_FILENAMES = frozenset({RUNNING_MARKER, SUCCESS_MARKER, FAILED_MARKER})


class ArtifactCollisionError(FileExistsError):
    """Raised when an artifact root cannot be safely reused."""


def portable_path(
    path: str | os.PathLike[str],
    *,
    roots: Sequence[str | os.PathLike[str]] = (),
    placeholder: str | None = None,
) -> str:
    """Render a path-like value as a deterministic portable artifact reference.

    Native ``Path`` objects should still be used for filesystem operations.
    This helper is only for persisted references in JSON, CSV, Markdown, and
    manifests where local absolute roots should not leak.
    """

    raw = os.fspath(path).strip()
    if not raw:
        return raw
    if raw.startswith("file://"):
        return placeholder or "[external-path]"
    if _is_relative_path_text(raw):
        return _normalize_relative_path_text(raw)

    relative = _relative_to_any_root(raw, roots)
    if relative is not None:
        return _normalize_relative_path_text(relative)

    return placeholder or _portable_name(raw)


def safe_create_run_dir(path: str | Path, *, collision_policy: CollisionPolicy = "fail") -> Path:
    """Create a run directory with conservative collision behavior."""

    return ensure_output_root_available(path, collision_policy=collision_policy)


def ensure_output_root_available(
    path: str | Path,
    *,
    collision_policy: CollisionPolicy = "fail",
) -> Path:
    """Create or validate an artifact root before writing into it.

    The default policy allows a missing or empty directory and fails on existing
    content. Workflows with explicit checkpoint/resume semantics may opt into
    ``reuse`` and are responsible for validating their own checkpoint state.
    """

    if collision_policy not in {"fail", "reuse"}:
        raise ValueError("collision_policy must be 'fail' or 'reuse'.")

    root = Path(path).resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False)
        return root
    if not root.is_dir():
        raise ArtifactCollisionError(f"Artifact root exists and is not a directory: {root.as_posix()}")

    entries = [entry for entry in root.iterdir()]
    if not entries:
        return root
    if collision_policy == "reuse":
        return root

    status = read_run_status(root)
    state = status["status"]
    if state == "completed":
        reason = "completed artifact root already exists"
    elif state in {"running", "failed", "incomplete"}:
        reason = f"{state} artifact root already exists"
    else:
        reason = "non-empty artifact root already exists"
    raise ArtifactCollisionError(f"Refusing to write into {root.as_posix()}: {reason}.")


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = True,
) -> Path:
    text = json.dumps(payload, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n"
    return atomic_write_text(path, text)


def atomic_write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    os.replace(temp_path, output)
    return output


def mark_run_started(
    run_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    recorded_at_utc: str | None = None,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / SUCCESS_MARKER)
    _remove_marker(root / FAILED_MARKER)
    return atomic_write_json(root / RUNNING_MARKER, _marker_payload("running", metadata, recorded_at_utc=recorded_at_utc))


def mark_run_completed(
    run_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    recorded_at_utc: str | None = None,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / RUNNING_MARKER)
    _remove_marker(root / FAILED_MARKER)
    return atomic_write_json(
        root / SUCCESS_MARKER,
        _marker_payload("completed", metadata, recorded_at_utc=recorded_at_utc),
    )


def mark_run_failed(
    run_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    recorded_at_utc: str | None = None,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / RUNNING_MARKER)
    return atomic_write_json(root / FAILED_MARKER, _marker_payload("failed", metadata, recorded_at_utc=recorded_at_utc))


def read_run_status(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        return {"status": "missing", "marker_path": None, "metadata": None}
    if not root.is_dir():
        return {
            "status": "not_directory",
            "marker_path": portable_path(root, roots=(Path.cwd(),)),
            "metadata": None,
        }

    for status, marker in (
        ("completed", SUCCESS_MARKER),
        ("failed", FAILED_MARKER),
        ("running", RUNNING_MARKER),
    ):
        marker_path = root / marker
        if marker_path.exists():
            return {
                "status": status,
                "marker_path": portable_path(marker_path, roots=(Path.cwd(), root)),
                "metadata": _read_marker(marker_path),
            }

    has_content = any(entry.name not in _MARKER_FILENAMES for entry in root.iterdir())
    return {
        "status": "incomplete" if has_content else "empty",
        "marker_path": None,
        "metadata": None,
    }


def _marker_payload(
    status: str,
    metadata: Mapping[str, Any] | None,
    *,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "recorded_at_utc": recorded_at_utc or datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "metadata": {} if metadata is None else dict(metadata),
    }


def _read_marker(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _remove_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_relative_path_text(value: str) -> bool:
    return not (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value.replace("\\", "/")).is_absolute()
    )


def _relative_to_any_root(value: str, roots: Sequence[str | os.PathLike[str]]) -> str | None:
    for root in roots:
        native_relative = _native_relative_to_root(value, root)
        if native_relative is not None:
            return native_relative
        windows_relative = _windows_relative_to_root(value, root)
        if windows_relative is not None:
            return windows_relative
        posix_relative = _posix_relative_to_root(value, root)
        if posix_relative is not None:
            return posix_relative
    return None


def _native_relative_to_root(value: str, root: str | os.PathLike[str]) -> str | None:
    if PureWindowsPath(value).is_absolute() and not Path(value).is_absolute():
        return None
    try:
        return Path(value).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def _windows_relative_to_root(value: str, root: str | os.PathLike[str]) -> str | None:
    try:
        value_path = PureWindowsPath(value)
        root_path = PureWindowsPath(os.fspath(root))
        if not value_path.is_absolute() or not root_path.is_absolute():
            return None
        return value_path.relative_to(root_path).as_posix()
    except ValueError:
        return None


def _posix_relative_to_root(value: str, root: str | os.PathLike[str]) -> str | None:
    normalized_value = value.replace("\\", "/")
    normalized_root = os.fspath(root).replace("\\", "/")
    try:
        value_path = PurePosixPath(normalized_value)
        root_path = PurePosixPath(normalized_root)
        if not value_path.is_absolute() or not root_path.is_absolute():
            return None
        return value_path.relative_to(root_path).as_posix()
    except ValueError:
        return None


def _normalize_relative_path_text(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized or "."


def _portable_name(value: str) -> str:
    parts = _portable_external_parts(value)
    if parts:
        return "external/" + "/".join(parts[-2:])
    return "[external-path]"


def _portable_external_parts(value: str) -> list[str]:
    normalized = value.replace("\\", "/")
    windows_path = PureWindowsPath(value)
    path_parts = windows_path.parts if windows_path.drive else PurePosixPath(normalized).parts
    return [
        part.strip(":")
        for part in path_parts
        if part not in {"", "/", "\\", windows_path.anchor, windows_path.drive}
    ]
