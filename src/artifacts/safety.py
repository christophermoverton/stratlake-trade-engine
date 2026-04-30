from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

CollisionPolicy = Literal["fail", "reuse"]

RUNNING_MARKER = "_RUNNING.json"
SUCCESS_MARKER = "_SUCCESS.json"
FAILED_MARKER = "_FAILED.json"
_MARKER_FILENAMES = frozenset({RUNNING_MARKER, SUCCESS_MARKER, FAILED_MARKER})


class ArtifactCollisionError(FileExistsError):
    """Raised when an artifact root cannot be safely reused."""


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
    text = json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"
    return atomic_write_text(path, text)


def atomic_write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    os.replace(temp_path, output)
    return output


def mark_run_started(run_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / SUCCESS_MARKER)
    _remove_marker(root / FAILED_MARKER)
    return atomic_write_json(root / RUNNING_MARKER, _marker_payload("running", metadata))


def mark_run_completed(run_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / RUNNING_MARKER)
    _remove_marker(root / FAILED_MARKER)
    return atomic_write_json(root / SUCCESS_MARKER, _marker_payload("completed", metadata))


def mark_run_failed(run_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    _remove_marker(root / RUNNING_MARKER)
    return atomic_write_json(root / FAILED_MARKER, _marker_payload("failed", metadata))


def read_run_status(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.exists():
        return {"status": "missing", "marker_path": None, "metadata": None}
    if not root.is_dir():
        return {"status": "not_directory", "marker_path": root.as_posix(), "metadata": None}

    for status, marker in (
        ("completed", SUCCESS_MARKER),
        ("failed", FAILED_MARKER),
        ("running", RUNNING_MARKER),
    ):
        marker_path = root / marker
        if marker_path.exists():
            return {
                "status": status,
                "marker_path": marker_path.as_posix(),
                "metadata": _read_marker(marker_path),
            }

    has_content = any(entry.name not in _MARKER_FILENAMES for entry in root.iterdir())
    return {
        "status": "incomplete" if has_content else "empty",
        "marker_path": None,
        "metadata": None,
    }


def _marker_payload(status: str, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "recorded_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
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
