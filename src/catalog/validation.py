"""Read-only consistency checks for M29 catalog records.

The validation layer observes catalog records, artifact records, marker files,
manifests, source files, and filesystem state. It never writes reports or
repairs artifacts; every result is returned in memory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from src.catalog.indexer import build_artifact_records
from src.catalog.models import ArtifactRecord, CatalogRecord

ERROR = "error"
WARNING = "warning"
INFO = "info"

_VALID_SEVERITIES = frozenset({ERROR, WARNING, INFO})
_MARKER_FILENAMES = ("_FAILED.json", "_SUCCESS.json", "_RUNNING.json")
_COMMON_JSON_FILENAMES = frozenset(
    {
        "manifest.json",
        "metrics.json",
        "alpha_metrics.json",
        "summary.json",
        "qa_summary.json",
        "_SUCCESS.json",
        "_FAILED.json",
        "_RUNNING.json",
        "checkpoint.json",
        "scenario_catalog.json",
        "decision_log.json",
        "robustness_summary.json",
        "robustness_findings.json",
        "sample_size_validation.json",
        "multiple_testing_summary.json",
        "leakage_validation.json",
        "promotion_governance_summary.json",
        "consistency_validation.json",
        "release_validation.json",
        "release_validation_summary.json",
    }
)
_INTERNAL_UNDECLARED_FILENAMES = frozenset(
    {
        "manifest.json",
        "_SUCCESS.json",
        "_FAILED.json",
        "_RUNNING.json",
        "metrics.json",
        "alpha_metrics.json",
        "summary.json",
        "qa_summary.json",
        "signal_diagnostics.json",
        "checkpoint.json",
        "scenario_catalog.json",
        "decision_log.json",
        "robustness_summary.json",
        "robustness_findings.json",
        "sample_size_validation.json",
        "multiple_testing_summary.json",
        "leakage_validation.json",
        "promotion_governance_summary.json",
        "consistency_validation.json",
        "release_validation.json",
        "release_validation_summary.json",
    }
)


@dataclass(frozen=True)
class CatalogValidationIssue:
    """A single deterministic catalog validation finding."""

    severity: str
    code: str
    catalog_id: str | None
    run_id: str | None
    path: str | None
    message: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"Unknown catalog validation severity: {self.severity}")

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "catalog_id": self.catalog_id,
            "run_id": self.run_id,
            "path": self.path,
            "message": self.message,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class CatalogValidationReport:
    """In-memory validation report for a catalog record collection."""

    total_records: int
    total_artifacts: int
    error_count: int
    warning_count: int
    issues: tuple[CatalogValidationIssue, ...]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_records": self.total_records,
            "total_artifacts": self.total_artifacts,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": _sort_jsonable(self.summary),
        }


def validate_catalog(
    records: Iterable[CatalogRecord],
    *,
    repo_root: str | Path = ".",
    include_info: bool = False,
) -> CatalogValidationReport:
    """Validate catalog records against artifact state and return an in-memory report."""

    root = Path(repo_root).resolve()
    sorted_records = sorted(records, key=_record_sort_key)
    issues: list[CatalogValidationIssue] = []
    total_artifacts = 0

    for record in sorted_records:
        artifacts = _safe_build_artifact_records(record, repo_root=root)
        total_artifacts += len(artifacts)
        issues.extend(
            _validate_record_with_artifacts(
                record,
                artifacts,
                repo_root=root,
                include_info=True,
            )
        )

    filtered = _filter_and_sort_issues(issues, include_info=include_info)
    summary = _build_summary(sorted_records, filtered)
    return CatalogValidationReport(
        total_records=len(sorted_records),
        total_artifacts=total_artifacts,
        error_count=sum(1 for issue in filtered if issue.severity == ERROR),
        warning_count=sum(1 for issue in filtered if issue.severity == WARNING),
        issues=tuple(filtered),
        summary=summary,
    )


def validate_record(
    record: CatalogRecord,
    *,
    repo_root: str | Path = ".",
    include_info: bool = False,
) -> list[CatalogValidationIssue]:
    """Validate one catalog record without mutating any source artifact."""

    root = Path(repo_root).resolve()
    artifacts = _safe_build_artifact_records(record, repo_root=root)
    return _validate_record_with_artifacts(
        record,
        artifacts,
        repo_root=root,
        include_info=include_info,
    )


def _validate_record_with_artifacts(
    record: CatalogRecord,
    artifacts: list[ArtifactRecord],
    *,
    repo_root: Path,
    include_info: bool,
) -> list[CatalogValidationIssue]:
    """Validate one record using a prebuilt artifact inventory snapshot."""

    root = repo_root
    issues: list[CatalogValidationIssue] = []

    if record.status == "registry_only":
        issues.append(
            _issue(
                WARNING,
                "registry_only_record",
                record,
                record.artifact_root or None,
                "Catalog record is registry-derived and has no required artifact root.",
            )
        )
        _validate_source_paths(record, root, issues)
        return _filter_and_sort_issues(issues, include_info=include_info)

    artifact_root = _resolve_path(record.artifact_root, root)
    if not artifact_root.exists() or not artifact_root.is_dir():
        issues.append(
            _issue(
                ERROR,
                "artifact_root_missing",
                record,
                _display_path(artifact_root, root),
                "Artifact root referenced by catalog record is missing.",
            )
        )
        _validate_source_paths(record, root, issues)
        return _filter_and_sort_issues(issues, include_info=include_info)

    _validate_source_paths(record, root, issues)
    _validate_markers(record, artifact_root, root, issues)
    _validate_status_fields(record, issues)
    _validate_common_json(record, artifact_root, root, issues)

    issues.extend(
        validate_artifact_records(record, artifacts, repo_root=root, include_info=True)
    )

    if record.source_manifest_path is None and not (artifact_root / "manifest.json").exists():
        issues.append(
            _issue(
                WARNING,
                "manifest_missing",
                record,
                _display_path(artifact_root / "manifest.json", root),
                "Artifact root does not contain manifest.json.",
            )
        )

    return _filter_and_sort_issues(issues, include_info=include_info)


def validate_artifact_records(
    record: CatalogRecord,
    artifacts: Iterable[ArtifactRecord],
    *,
    repo_root: str | Path = ".",
    include_info: bool = False,
) -> list[CatalogValidationIssue]:
    """Validate artifact-record inventory consistency for one catalog record."""

    root = Path(repo_root).resolve()
    issues: list[CatalogValidationIssue] = []

    for artifact in sorted(artifacts, key=lambda item: item.relative_path):
        artifact_path = _resolve_path(artifact.path, root)
        display = _display_path(artifact_path, root)
        if artifact.exists is False:
            issues.append(
                _issue(
                    WARNING,
                    "manifest_artifact_missing",
                    record,
                    display,
                    "Manifest declares an artifact that is missing on disk.",
                    {"relative_path": artifact.relative_path},
                )
            )
            continue

        if not artifact_path.exists():
            issues.append(
                _issue(
                    ERROR,
                    "artifact_path_missing",
                    record,
                    display,
                    "Artifact record references a missing file.",
                    {"relative_path": artifact.relative_path},
                )
            )
        elif (
            artifact.declared_in_manifest is False
            and artifact.filename not in _INTERNAL_UNDECLARED_FILENAMES
        ):
            issues.append(
                _issue(
                    WARNING,
                    "undeclared_artifact",
                    record,
                    display,
                    "Artifact exists on disk but is not declared in manifest.json.",
                    {"relative_path": artifact.relative_path},
                )
            )

    return _filter_and_sort_issues(issues, include_info=include_info)


def _validate_source_paths(
    record: CatalogRecord,
    repo_root: Path,
    issues: list[CatalogValidationIssue],
) -> None:
    if record.source_manifest_path is not None:
        path = _resolve_path(record.source_manifest_path, repo_root)
        if not path.exists():
            issues.append(
                _issue(
                    WARNING,
                    "source_manifest_missing",
                    record,
                    _display_path(path, repo_root),
                    "Catalog source_manifest_path does not exist.",
                )
            )

    if record.source_marker_path is not None:
        path = _resolve_path(record.source_marker_path, repo_root)
        if not path.exists():
            issues.append(
                _issue(
                    WARNING,
                    "source_marker_missing",
                    record,
                    _display_path(path, repo_root),
                    "Catalog source_marker_path does not exist.",
                )
            )

    for source in sorted(set(record.source_files)):
        path = _resolve_path(source, repo_root)
        if not path.exists():
            issues.append(
                _issue(
                    WARNING,
                    "source_file_missing",
                    record,
                    _display_path(path, repo_root),
                    "Catalog source_files entry does not exist.",
                    {"source_file": source},
                )
            )


def _validate_markers(
    record: CatalogRecord,
    artifact_root: Path,
    repo_root: Path,
    issues: list[CatalogValidationIssue],
) -> None:
    present = [name for name in _MARKER_FILENAMES if (artifact_root / name).exists()]
    if not present:
        issues.append(
            _issue(
                WARNING,
                "marker_missing",
                record,
                _display_path(artifact_root, repo_root),
                "Artifact root has no lifecycle marker file.",
            )
        )
        return

    if len(present) > 1:
        issues.append(
            _issue(
                WARNING,
                "multiple_markers",
                record,
                _display_path(artifact_root, repo_root),
                "Artifact root contains multiple lifecycle marker files.",
                {"markers": tuple(present)},
            )
        )

    if "_FAILED.json" in present:
        issues.append(
            _issue(
                WARNING,
                "failed_marker_present",
                record,
                _display_path(artifact_root / "_FAILED.json", repo_root),
                "Failure marker is present; failed marker retains highest precedence.",
            )
        )
    if "_RUNNING.json" in present:
        issues.append(
            _issue(
                WARNING,
                "running_marker_present",
                record,
                _display_path(artifact_root / "_RUNNING.json", repo_root),
                "Running marker is present for this artifact root.",
            )
        )


def _validate_status_fields(record: CatalogRecord, issues: list[CatalogValidationIssue]) -> None:
    if record.status == "unknown":
        issues.append(
            _issue(
                WARNING,
                "unknown_status",
                record,
                record.artifact_root,
                "Catalog record status is unknown.",
            )
        )
    if record.run_type == "unknown":
        issues.append(
            _issue(
                WARNING,
                "unknown_run_type",
                record,
                record.artifact_root,
                "Catalog record run_type is unknown.",
            )
        )


def _validate_common_json(
    record: CatalogRecord,
    artifact_root: Path,
    repo_root: Path,
    issues: list[CatalogValidationIssue],
) -> None:
    for filename in sorted(_COMMON_JSON_FILENAMES):
        path = artifact_root / filename
        if path.exists() and path.is_file() and not _is_valid_json(path):
            issues.append(
                _issue(
                    ERROR,
                    "corrupt_json",
                    record,
                    _display_path(path, repo_root),
                    "JSON artifact exists but cannot be parsed.",
                    {"filename": filename},
                )
            )


def _safe_build_artifact_records(record: CatalogRecord, *, repo_root: Path) -> list[ArtifactRecord]:
    try:
        return build_artifact_records(record, repo_root=repo_root)
    except Exception:  # noqa: BLE001
        return []


def _is_valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return True


def _issue(
    severity: str,
    code: str,
    record: CatalogRecord,
    path: str | None,
    message: str,
    metadata: dict[str, object] | None = None,
) -> CatalogValidationIssue:
    return CatalogValidationIssue(
        severity=severity,
        code=code,
        catalog_id=record.catalog_id,
        run_id=record.run_id,
        path=path,
        message=message,
        metadata=metadata or {},
    )


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _record_sort_key(record: CatalogRecord) -> tuple[str, str, str, str]:
    return (
        record.run_type,
        record.run_id or "",
        record.artifact_root,
        record.catalog_id,
    )


def _issue_sort_key(issue: CatalogValidationIssue) -> tuple[str, str, str, str, str]:
    return (
        issue.catalog_id or "",
        issue.run_id or "",
        issue.path or "",
        issue.code,
        issue.severity,
    )


def _filter_and_sort_issues(
    issues: Iterable[CatalogValidationIssue],
    *,
    include_info: bool,
) -> list[CatalogValidationIssue]:
    filtered = [issue for issue in issues if include_info or issue.severity != INFO]
    return sorted(filtered, key=_issue_sort_key)


def _build_summary(
    records: list[CatalogRecord],
    issues: list[CatalogValidationIssue],
) -> dict[str, object]:
    by_severity = Counter(issue.severity for issue in issues)
    by_code = Counter(issue.code for issue in issues)
    by_run_type = Counter(record.run_type for record in records)
    by_status = Counter(record.status for record in records)
    records_with_errors = {
        issue.catalog_id for issue in issues if issue.catalog_id and issue.severity == ERROR
    }
    records_with_warnings = {
        issue.catalog_id for issue in issues if issue.catalog_id and issue.severity == WARNING
    }
    return {
        "by_severity": _sorted_counter(by_severity),
        "by_code": _sorted_counter(by_code),
        "by_run_type": _sorted_counter(by_run_type),
        "by_status": _sorted_counter(by_status),
        "records_with_errors": len(records_with_errors),
        "records_with_warnings": len(records_with_warnings),
    }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _sort_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sort_jsonable(item) for item in value)
    return value
