from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts.safety import (
    RUNNING_MARKER,
    SUCCESS_MARKER,
    ArtifactCollisionError,
    atomic_write_json,
    atomic_write_text,
    ensure_output_root_available,
    mark_run_completed,
    mark_run_started,
    read_run_status,
    safe_create_run_dir,
)


def test_safe_create_run_dir_fails_on_existing_non_empty_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactCollisionError, match="incomplete artifact root"):
        safe_create_run_dir(run_dir)


def test_safe_create_run_dir_allows_empty_new_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    created = safe_create_run_dir(run_dir)

    assert created == run_dir.resolve()
    assert run_dir.exists()
    assert read_run_status(run_dir)["status"] == "empty"


def test_atomic_write_json_does_not_leave_partial_file_on_success(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "summary.json"

    atomic_write_json(output, {"b": 2, "a": 1})

    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(output.parent.glob("*.tmp"))


def test_run_status_markers_transition_started_completed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    mark_run_started(run_dir, {"run_id": "abc"})
    running = read_run_status(run_dir)
    mark_run_completed(run_dir, {"run_id": "abc"})
    completed = read_run_status(run_dir)

    assert running["status"] == "running"
    assert running["metadata"]["metadata"] == {"run_id": "abc"}
    assert completed["status"] == "completed"
    assert not (run_dir / RUNNING_MARKER).exists()
    assert (run_dir / SUCCESS_MARKER).exists()


def test_existing_completed_output_root_fails_without_reuse_policy(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    mark_run_started(run_dir)
    atomic_write_text(run_dir / "summary.json", "{}\n")
    mark_run_completed(run_dir)

    with pytest.raises(ArtifactCollisionError, match="completed artifact root"):
        ensure_output_root_available(run_dir)


def test_incomplete_output_root_detected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"status":"partial"}\n', encoding="utf-8")

    assert read_run_status(run_dir)["status"] == "incomplete"
    with pytest.raises(ArtifactCollisionError, match="incomplete artifact root"):
        ensure_output_root_available(run_dir)


def test_checkpoint_resume_workflow_can_reuse_when_explicitly_allowed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    mark_run_started(run_dir)
    atomic_write_json(run_dir / "checkpoint.json", {"status": "partial"})

    reused = ensure_output_root_available(run_dir, collision_policy="reuse")

    assert reused == run_dir.resolve()
    assert read_run_status(run_dir)["status"] == "running"
