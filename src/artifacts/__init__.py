"""Shared artifact safety helpers."""

from src.artifacts.safety import (
    ArtifactCollisionError,
    RUNNING_MARKER,
    SUCCESS_MARKER,
    FAILED_MARKER,
    atomic_write_json,
    atomic_write_text,
    ensure_output_root_available,
    mark_run_completed,
    mark_run_failed,
    mark_run_started,
    read_run_status,
    safe_create_run_dir,
)

__all__ = [
    "ArtifactCollisionError",
    "FAILED_MARKER",
    "RUNNING_MARKER",
    "SUCCESS_MARKER",
    "atomic_write_json",
    "atomic_write_text",
    "ensure_output_root_available",
    "mark_run_completed",
    "mark_run_failed",
    "mark_run_started",
    "read_run_status",
    "safe_create_run_dir",
]
