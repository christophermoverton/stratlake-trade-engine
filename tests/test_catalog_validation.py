from __future__ import annotations

import json
from pathlib import Path

from src.catalog import (
    CatalogRecord,
    CatalogValidationStatus,
    build_catalog,
    validate_catalog,
    validate_record,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def _make_root(
    tmp_path: Path,
    run_id: str = "strategy_001",
    *,
    marker: str | None = "_SUCCESS.json",
    manifest_artifacts: list[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    root = tmp_path / "strategies" / run_id
    root.mkdir(parents=True, exist_ok=True)
    if marker is not None:
        _write_json(root / marker, {"run_id": run_id, "status": "completed"})
    _write_json(root / "metrics.json", {"sharpe_ratio": 1.2})
    _write_json(root / "summary.json", {"strategy_name": "validation_fixture"})
    _write_json(
        root / "manifest.json",
        {
            "run_id": run_id,
            "artifacts": manifest_artifacts
            if manifest_artifacts is not None
            else ["manifest.json", marker or "", "metrics.json", "summary.json"],
        },
    )
    for relative_path, text in (extra_files or {}).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _record(tmp_path: Path, run_id: str = "strategy_001") -> CatalogRecord:
    records = build_catalog(tmp_path, repo_root=tmp_path)
    return next(record for record in records if record.run_id == run_id)


def _bare_record(
    tmp_path: Path,
    *,
    run_id: str = "manual",
    run_type: str = "strategy",
    status: str = "completed",
    artifact_root: str = "strategies/manual",
    source_manifest_path: str | None = None,
    source_marker_path: str | None = None,
    source_files: list[str] | None = None,
) -> CatalogRecord:
    return CatalogRecord(
        catalog_id=f"catalog_{run_id}",
        run_id=run_id,
        run_type=run_type,
        status=status,
        artifact_root=artifact_root,
        source_registry_path=None,
        source_manifest_path=source_manifest_path,
        source_marker_path=source_marker_path,
        created_at=None,
        timeframe=None,
        start_ts=None,
        end_ts=None,
        strategy_name=None,
        portfolio_name=None,
        allocator_name=None,
        alpha_model_name=None,
        regime_method=None,
        campaign_id=None,
        scenario_id=None,
        metrics_summary=None,
        qa_status=None,
        review_status=None,
        promotion_status=None,
        tags=[],
        source_files=source_files or [],
        metadata={},
        validation=CatalogValidationStatus(
            catalog_status=status,
            marker_status="present",
            manifest_status="present",
            artifact_status="ok",
            qa_status=None,
        ),
    )


def _codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def test_clean_record_validation_has_no_issues(tmp_path: Path) -> None:
    _make_root(tmp_path)
    record = _record(tmp_path)

    report = validate_catalog([record], repo_root=tmp_path)

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.issues == ()


def test_missing_artifact_root_emits_artifact_root_missing(tmp_path: Path) -> None:
    record = _bare_record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "artifact_root_missing" in _codes(issues)


def test_missing_source_manifest_emits_source_manifest_missing(tmp_path: Path) -> None:
    _make_root(tmp_path)
    record = _bare_record(
        tmp_path,
        artifact_root="strategies/manual",
        source_manifest_path="strategies/manual/missing_manifest.json",
    )
    (tmp_path / "strategies" / "manual").mkdir(parents=True)
    _write_json(tmp_path / "strategies" / "manual" / "_SUCCESS.json", {"status": "completed"})

    issues = validate_record(record, repo_root=tmp_path)

    assert "source_manifest_missing" in _codes(issues)


def test_missing_source_marker_emits_source_marker_missing(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    (root / "_SUCCESS.json").unlink()
    record = _bare_record(
        tmp_path,
        artifact_root="strategies/manual",
        source_marker_path="strategies/manual/_SUCCESS.json",
    )
    manual_root = tmp_path / "strategies" / "manual"
    manual_root.mkdir(parents=True)
    _write_json(manual_root / "manifest.json", {"artifacts": ["manifest.json"]})

    issues = validate_record(record, repo_root=tmp_path)

    assert "source_marker_missing" in _codes(issues)


def test_missing_manifest_declared_artifact(tmp_path: Path) -> None:
    _make_root(tmp_path, manifest_artifacts=["metrics.json", "missing.csv"])
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "manifest_artifact_missing" in _codes(issues)


def test_undeclared_artifact_excludes_internal_files(tmp_path: Path) -> None:
    _make_root(
        tmp_path,
        manifest_artifacts=["metrics.json"],
        extra_files={"extra.csv": "value\n1\n"},
    )
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "undeclared_artifact" in _codes(issues)
    undeclared_paths = [
        issue.metadata["relative_path"]
        for issue in issues
        if issue.code == "undeclared_artifact"
    ]
    assert undeclared_paths == ["extra.csv"]


def test_multiple_markers_and_failed_marker(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _write_json(root / "_FAILED.json", {"status": "failed"})
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "multiple_markers" in _codes(issues)
    assert "failed_marker_present" in _codes(issues)


def test_running_marker(tmp_path: Path) -> None:
    _make_root(tmp_path, marker="_RUNNING.json")
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "running_marker_present" in _codes(issues)


def test_failed_marker(tmp_path: Path) -> None:
    _make_root(tmp_path, marker="_FAILED.json")
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "failed_marker_present" in _codes(issues)


def test_unknown_status_and_run_type(tmp_path: Path) -> None:
    root = tmp_path / "mystery" / "run_001"
    root.mkdir(parents=True)
    _write_json(root / "_SUCCESS.json", {"status": "completed"})
    record = _bare_record(
        tmp_path,
        run_type="unknown",
        status="unknown",
        artifact_root="mystery/run_001",
    )

    issues = validate_record(record, repo_root=tmp_path)

    assert "unknown_status" in _codes(issues)
    assert "unknown_run_type" in _codes(issues)


def test_registry_only_record_does_not_emit_missing_root(tmp_path: Path) -> None:
    record = _bare_record(
        tmp_path,
        status="registry_only",
        artifact_root="strategies/missing_registry_only",
    )

    issues = validate_record(record, repo_root=tmp_path)

    assert "registry_only_record" in _codes(issues)
    assert "artifact_root_missing" not in _codes(issues)


def test_malformed_json_emits_corrupt_json_without_crashing(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    (root / "metrics.json").write_text("{bad json", encoding="utf-8")
    record = _record(tmp_path)

    issues = validate_record(record, repo_root=tmp_path)

    assert "corrupt_json" in _codes(issues)


def test_deterministic_issue_ordering_and_report_dict(tmp_path: Path) -> None:
    root = _make_root(tmp_path, manifest_artifacts=["missing.csv"])
    _write_json(root / "_FAILED.json", {"status": "failed"})
    record = _record(tmp_path)

    first = validate_catalog([record], repo_root=tmp_path).to_dict()
    second = validate_catalog([record], repo_root=tmp_path).to_dict()

    assert first == second


def test_report_summary_counts(tmp_path: Path) -> None:
    _make_root(tmp_path, manifest_artifacts=["missing.csv"])
    record = _record(tmp_path)

    report = validate_catalog([record], repo_root=tmp_path)

    assert report.total_records == 1
    assert report.total_artifacts >= 1
    assert report.summary["by_code"]["manifest_artifact_missing"] == 1
    assert report.summary["by_severity"]["warning"] >= 1
    assert report.summary["records_with_warnings"] == 1


def test_validation_is_read_only(tmp_path: Path) -> None:
    _make_root(tmp_path, extra_files={"extra.csv": "value\n1\n"})
    record = _record(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    validate_catalog([record], repo_root=tmp_path)

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert before == after
