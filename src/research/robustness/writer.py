from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from src.artifacts.safety import portable_path
from src.research.registry import canonicalize_value

from .models import (
    MULTIPLE_TESTING_JSON_FIELDS,
    ROBUSTNESS_REPORT_ARTIFACT_TYPE,
    SCHEMA_VERSION,
    SAMPLE_SIZE_JSON_FIELDS,
    SENSITIVITY_SUMMARY_COLUMNS,
    WALK_FORWARD_EFFICIENCY_COLUMNS,
    RobustnessReport,
    RobustnessReportResult,
    sanitize_portable_value,
)
from .summary import build_robustness_summary

DEFAULT_ROBUSTNESS_ROOT = Path("artifacts") / "robustness"
SUMMARY_FILENAME = "robustness_summary.json"
FINDINGS_FILENAME = "robustness_findings.json"
WALK_FORWARD_EFFICIENCY_FILENAME = "walk_forward_efficiency.csv"
SAMPLE_SIZE_FILENAME = "sample_size_validation.json"
SENSITIVITY_FILENAME = "sensitivity_summary.csv"
MULTIPLE_TESTING_FILENAME = "multiple_testing_summary.json"
REPORT_FILENAME = "robustness_report.md"
MANIFEST_FILENAME = "manifest.json"

CANONICAL_ARTIFACTS: list[str] = [
    SUMMARY_FILENAME,
    FINDINGS_FILENAME,
    WALK_FORWARD_EFFICIENCY_FILENAME,
    SAMPLE_SIZE_FILENAME,
    SENSITIVITY_FILENAME,
    MULTIPLE_TESTING_FILENAME,
    REPORT_FILENAME,
    MANIFEST_FILENAME,
]


def write_robustness_report_bundle(
    report: RobustnessReport,
    *,
    output_root: str | Path = DEFAULT_ROBUSTNESS_ROOT,
) -> RobustnessReportResult:
    output_dir = _resolve_output_dir(output_root, report.report_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    roots = (Path.cwd(), output_dir)
    paths = {filename: output_dir / filename for filename in CANONICAL_ARTIFACTS}
    summary = build_robustness_summary(report, generated_artifacts=CANONICAL_ARTIFACTS)

    _write_json(paths[SUMMARY_FILENAME], summary)
    _write_json(paths[FINDINGS_FILENAME], _findings_payload(report, roots=roots))
    _write_csv(
        paths[WALK_FORWARD_EFFICIENCY_FILENAME],
        [row.to_csv_row(roots=roots) for row in report.walk_forward_efficiency],
        fieldnames=WALK_FORWARD_EFFICIENCY_COLUMNS,
    )
    _write_json(paths[SAMPLE_SIZE_FILENAME], _sample_size_payload(report, roots=roots))
    _write_csv(
        paths[SENSITIVITY_FILENAME],
        [row.to_csv_row(roots=roots) for row in report.sensitivity_summary],
        fieldnames=SENSITIVITY_SUMMARY_COLUMNS,
    )
    _write_json(paths[MULTIPLE_TESTING_FILENAME], _multiple_testing_payload(report, roots=roots))
    paths[REPORT_FILENAME].write_text(
        _render_markdown_report(report=report, summary=summary),
        encoding="utf-8",
        newline="\n",
    )
    _write_json(paths[MANIFEST_FILENAME], _manifest_payload(report=report, output_dir=output_dir, summary=summary, roots=roots))

    return RobustnessReportResult(
        report_id=report.report_id,
        output_dir=output_dir,
        summary_path=paths[SUMMARY_FILENAME],
        findings_path=paths[FINDINGS_FILENAME],
        walk_forward_efficiency_path=paths[WALK_FORWARD_EFFICIENCY_FILENAME],
        sample_size_validation_path=paths[SAMPLE_SIZE_FILENAME],
        sensitivity_summary_path=paths[SENSITIVITY_FILENAME],
        multiple_testing_summary_path=paths[MULTIPLE_TESTING_FILENAME],
        markdown_path=paths[REPORT_FILENAME],
        manifest_path=paths[MANIFEST_FILENAME],
    )


def _findings_payload(report: RobustnessReport, *, roots: tuple[Path, ...]) -> dict[str, Any]:
    findings = sorted(
        [finding.to_dict(roots=roots) for finding in report.findings],
        key=lambda row: (row["severity"], row["workflow_type"], row["run_id"], row["check_id"], row["message"]),
    )
    return canonicalize_value({"finding_count": len(findings), "findings": findings, "report_id": report.report_id})


def _sample_size_payload(report: RobustnessReport, *, roots: tuple[Path, ...]) -> dict[str, Any]:
    checks = sorted(
        [row.to_dict(roots=roots) for row in report.sample_size_validation],
        key=lambda row: (row["workflow_type"], row["run_id"], row["check_id"]),
    )
    return canonicalize_value(
        {
            "checks": checks,
            "column_contract": SAMPLE_SIZE_JSON_FIELDS,
            "report_id": report.report_id,
            "schema_version": SCHEMA_VERSION,
        }
    )


def _multiple_testing_payload(report: RobustnessReport, *, roots: tuple[Path, ...]) -> dict[str, Any]:
    families = sorted(
        [row.to_dict(roots=roots) for row in report.multiple_testing_summary],
        key=lambda row: (row["workflow_type"], row["run_id"], row["family_id"]),
    )
    return canonicalize_value(
        {
            "column_contract": MULTIPLE_TESTING_JSON_FIELDS,
            "families": families,
            "report_id": report.report_id,
            "schema_version": SCHEMA_VERSION,
        }
    )


def _manifest_payload(
    *,
    report: RobustnessReport,
    output_dir: Path,
    summary: Mapping[str, Any],
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    artifacts = {
        filename: {
            "artifact_type": _artifact_type_for(filename),
            "path": filename,
        }
        for filename in sorted(CANONICAL_ARTIFACTS)
    }
    artifacts[WALK_FORWARD_EFFICIENCY_FILENAME]["columns"] = WALK_FORWARD_EFFICIENCY_COLUMNS
    artifacts[SENSITIVITY_FILENAME]["columns"] = SENSITIVITY_SUMMARY_COLUMNS
    artifacts[SAMPLE_SIZE_FILENAME]["fields"] = SAMPLE_SIZE_JSON_FIELDS
    artifacts[MULTIPLE_TESTING_FILENAME]["fields"] = MULTIPLE_TESTING_JSON_FIELDS
    return canonicalize_value(
        {
            "artifact_files": sorted(CANONICAL_ARTIFACTS),
            "artifact_type": ROBUSTNESS_REPORT_ARTIFACT_TYPE,
            "artifacts": artifacts,
            "created_at_utc": report.created_at_utc,
            "manifest_schema_version": SCHEMA_VERSION,
            "output_dir": _portable_output_dir(output_dir),
            "report_id": report.report_id,
            "robustness_status": report.robustness_status,
            "schema_version": SCHEMA_VERSION,
            "source_artifacts": report.upstream_references.source_artifacts(roots=roots),
            "source_run_references": report.source_run_ids(),
            "summary": sanitize_portable_value(dict(summary), roots=roots),
            "writer": {
                "name": report.writer_name,
                "config": sanitize_portable_value(dict(report.metadata), roots=roots),
            },
        }
    )


def _render_markdown_report(*, report: RobustnessReport, summary: Mapping[str, Any]) -> str:
    lines = [
        "# Robustness Report",
        "",
        f"- Report ID: {report.report_id}",
        f"- Workflow Type: {report.workflow_type}",
        f"- Run ID: {report.run_id}",
        f"- Robustness Status: {report.robustness_status}",
        f"- Findings: {summary.get('finding_count', 0)}",
        f"- Highest Severity: {summary.get('highest_severity', 'info')}",
        "",
        "## Checks Present",
        _markdown_list(summary.get("checks_present")),
        "",
        "## Checks Missing",
        _markdown_list(summary.get("checks_missing")),
        "",
        "## Finding Counts",
        _markdown_counts(summary.get("finding_count_by_severity")),
        "",
        "## Canonical Artifacts",
        _markdown_list(CANONICAL_ARTIFACTS),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(canonicalize_value(dict(payload)), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: json.dumps(canonicalize_value(item), sort_keys=True, allow_nan=False)):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _markdown_counts(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "- None"
    return "\n".join(f"- {key}: {value[key]}" for key in sorted(value))


def _markdown_list(value: Any) -> str:
    if not value:
        return "- None"
    return "\n".join(f"- {item}" for item in sorted(value))


def _resolve_output_dir(output_root: str | Path, report_id: str) -> Path:
    root = Path(output_root)
    if root.name == report_id:
        return root
    return root / report_id


def _artifact_type_for(filename: str) -> str:
    return {
        SUMMARY_FILENAME: "robustness_summary",
        FINDINGS_FILENAME: "robustness_findings",
        WALK_FORWARD_EFFICIENCY_FILENAME: "walk_forward_efficiency_contract",
        SAMPLE_SIZE_FILENAME: "sample_size_validation_contract",
        SENSITIVITY_FILENAME: "sensitivity_summary_contract",
        MULTIPLE_TESTING_FILENAME: "multiple_testing_summary_contract",
        REPORT_FILENAME: "markdown_report",
        MANIFEST_FILENAME: "manifest",
    }[filename]


def _portable_output_dir(output_dir: Path) -> str:
    for parent in output_dir.parents:
        if parent.name == "artifacts":
            return portable_path(output_dir, roots=(parent.parent,))
    return portable_path(output_dir, roots=(Path.cwd(),))
