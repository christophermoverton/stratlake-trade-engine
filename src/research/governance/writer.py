from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from src.research.registry import canonicalize_value

from .aggregator import (
    OUTCOME_MATRIX_COLUMNS,
    build_governance_outcome_rows,
    build_governance_report_id,
    build_governance_summary,
    build_reason_code_summary,
    build_severity_summary,
    build_workflow_summary,
)
from .loader import load_governance_artifacts
from .models import GovernanceReportResult
from .validator import validate_governance_consistency

DEFAULT_GOVERNANCE_ROOT = Path("artifacts") / "promotion_governance"
SUMMARY_FILENAME = "promotion_governance_summary.json"
OUTCOME_MATRIX_FILENAME = "promotion_outcome_matrix.csv"
REASON_CODE_SUMMARY_FILENAME = "reason_code_summary.csv"
SEVERITY_SUMMARY_FILENAME = "severity_summary.csv"
WORKFLOW_SUMMARY_FILENAME = "workflow_summary.csv"
VALIDATION_FILENAME = "consistency_validation.json"
REPORT_FILENAME = "promotion_governance_report.md"
MANIFEST_FILENAME = "manifest.json"


def run_promotion_governance_report(
    *,
    registry_path: str | Path | None = None,
    artifact_root: str | Path = Path("artifacts"),
    output_dir: str | Path | None = None,
    report_id: str | None = None,
    strict_validation: bool = False,
) -> GovernanceReportResult:
    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    resolved_report_id = report_id or build_governance_report_id(rows)
    resolved_output_dir = (
        DEFAULT_GOVERNANCE_ROOT / resolved_report_id
        if output_dir is None
        else Path(output_dir) / resolved_report_id
        if Path(output_dir).name != resolved_report_id
        else Path(output_dir)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_governance_summary(rows)
    reason_summary = build_reason_code_summary(rows)
    severity_summary = build_severity_summary(rows)
    workflow_summary = build_workflow_summary(rows)
    validation = validate_governance_consistency(dataset.records, rows)
    if strict_validation and validation["status"] != "pass":
        _write_outputs(
            output_dir=resolved_output_dir,
            report_id=resolved_report_id,
            summary=summary,
            rows=rows,
            reason_summary=reason_summary,
            severity_summary=severity_summary,
            workflow_summary=workflow_summary,
            validation=validation,
            sources=dataset.sources,
        )
        raise ValueError(f"Promotion governance validation failed with {validation['finding_count']} finding(s).")

    paths = _write_outputs(
        output_dir=resolved_output_dir,
        report_id=resolved_report_id,
        summary=summary,
        rows=rows,
        reason_summary=reason_summary,
        severity_summary=severity_summary,
        workflow_summary=workflow_summary,
        validation=validation,
        sources=dataset.sources,
    )
    return GovernanceReportResult(
        report_id=resolved_report_id,
        output_dir=resolved_output_dir,
        summary_path=paths[SUMMARY_FILENAME],
        outcome_matrix_path=paths[OUTCOME_MATRIX_FILENAME],
        reason_code_summary_path=paths[REASON_CODE_SUMMARY_FILENAME],
        severity_summary_path=paths[SEVERITY_SUMMARY_FILENAME],
        workflow_summary_path=paths[WORKFLOW_SUMMARY_FILENAME],
        validation_path=paths[VALIDATION_FILENAME],
        markdown_path=paths[REPORT_FILENAME],
        manifest_path=paths[MANIFEST_FILENAME],
        validation=validation,
    )


def _write_outputs(
    *,
    output_dir: Path,
    report_id: str,
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    reason_summary: list[dict[str, Any]],
    severity_summary: list[dict[str, Any]],
    workflow_summary: list[dict[str, Any]],
    validation: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> dict[str, Path]:
    paths = {
        SUMMARY_FILENAME: output_dir / SUMMARY_FILENAME,
        OUTCOME_MATRIX_FILENAME: output_dir / OUTCOME_MATRIX_FILENAME,
        REASON_CODE_SUMMARY_FILENAME: output_dir / REASON_CODE_SUMMARY_FILENAME,
        SEVERITY_SUMMARY_FILENAME: output_dir / SEVERITY_SUMMARY_FILENAME,
        WORKFLOW_SUMMARY_FILENAME: output_dir / WORKFLOW_SUMMARY_FILENAME,
        VALIDATION_FILENAME: output_dir / VALIDATION_FILENAME,
        REPORT_FILENAME: output_dir / REPORT_FILENAME,
        MANIFEST_FILENAME: output_dir / MANIFEST_FILENAME,
    }
    _write_json(paths[SUMMARY_FILENAME], {"report_id": report_id, **dict(summary)})
    _write_csv(paths[OUTCOME_MATRIX_FILENAME], rows, fieldnames=OUTCOME_MATRIX_COLUMNS)
    _write_csv(paths[REASON_CODE_SUMMARY_FILENAME], reason_summary, fieldnames=["reason_code", "count"])
    _write_csv(paths[SEVERITY_SUMMARY_FILENAME], severity_summary, fieldnames=["severity", "highest_severity_count", "triggered_reason_count"])
    _write_csv(paths[WORKFLOW_SUMMARY_FILENAME], workflow_summary, fieldnames=["workflow_type", "row_count", "eligible_count", "blocked_count", "needs_review_count", "rejected_count"])
    _write_json(paths[VALIDATION_FILENAME], dict(validation))
    paths[REPORT_FILENAME].write_text(
        _render_markdown_report(report_id=report_id, summary=summary, validation=validation),
        encoding="utf-8",
        newline="\n",
    )
    _write_json(
        paths[MANIFEST_FILENAME],
        _manifest_payload(report_id=report_id, output_dir=output_dir, sources=sources, rows=rows, validation=validation),
    )
    return paths


def _manifest_payload(
    *,
    report_id: str,
    output_dir: Path,
    sources: Mapping[str, Any],
    rows: list[dict[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        filename: {"path": filename}
        for filename in sorted(
            [
                SUMMARY_FILENAME,
                OUTCOME_MATRIX_FILENAME,
                REASON_CODE_SUMMARY_FILENAME,
                SEVERITY_SUMMARY_FILENAME,
                WORKFLOW_SUMMARY_FILENAME,
                VALIDATION_FILENAME,
                REPORT_FILENAME,
                MANIFEST_FILENAME,
            ]
        )
    }
    artifacts[OUTCOME_MATRIX_FILENAME]["rows"] = len(rows)
    artifacts[VALIDATION_FILENAME]["finding_count"] = validation.get("finding_count")
    return canonicalize_value(
        {
            "report_id": report_id,
            "run_type": "promotion_governance",
            "artifact_files": sorted(artifacts),
            "artifact_groups": {
                "core": sorted(artifacts),
                "governance": sorted(artifacts),
                "validation": [VALIDATION_FILENAME],
            },
            "artifacts": artifacts,
            "output_dir": _relative_to_cwd(output_dir),
            "sources": _sanitize_sources(sources),
            "row_count": len(rows),
            "validation_status": validation.get("status"),
        }
    )


def _render_markdown_report(
    *,
    report_id: str,
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    lines = [
        "# Promotion Governance Report",
        "",
        f"- Report ID: {report_id}",
        f"- Outcome Rows: {summary.get('row_count', 0)}",
        f"- Validation Status: {validation.get('status')}",
        "",
        "## Promotion Status Counts",
        _markdown_counts(summary.get("promotion_status_counts")),
        "",
        "## Workflow Counts",
        _markdown_counts(summary.get("workflow_type_counts")),
        "",
        "## Validation",
        f"- Findings: {validation.get('finding_count', 0)}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _markdown_counts(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "- None"
    return "\n".join(f"- {key}: {value[key]}" for key in sorted(value))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(canonicalize_value(dict(payload)), handle, indent=2, sort_keys=True, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sanitize_sources(sources: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in sorted(sources.items()):
        if isinstance(value, str) and (Path(value).is_absolute() or "\\" in value):
            sanitized[key] = _relative_to_cwd(Path(value))
        else:
            sanitized[key] = value
    return sanitized


def _relative_to_cwd(path: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), start=Path.cwd().resolve())).as_posix()
    except (OSError, ValueError):
        return path.name if path.is_absolute() else path.as_posix()


__all__ = [
    "DEFAULT_GOVERNANCE_ROOT",
    "MANIFEST_FILENAME",
    "OUTCOME_MATRIX_FILENAME",
    "REASON_CODE_SUMMARY_FILENAME",
    "REPORT_FILENAME",
    "SEVERITY_SUMMARY_FILENAME",
    "SUMMARY_FILENAME",
    "VALIDATION_FILENAME",
    "WORKFLOW_SUMMARY_FILENAME",
    "run_promotion_governance_report",
]
