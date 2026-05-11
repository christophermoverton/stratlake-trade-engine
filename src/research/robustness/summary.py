from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.research.registry import canonicalize_value

from .models import (
    ROBUSTNESS_STATUS_ORDER,
    SEVERITY_ORDER,
    RobustnessReport,
    highest_severity,
)


def build_robustness_summary(
    report: RobustnessReport,
    *,
    generated_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    finding_rows = [finding.to_dict(roots=(Path.cwd(),)) for finding in report.findings]
    severity_counts = Counter(row["severity"] for row in finding_rows)
    workflow_counts = Counter([report.workflow_type, *(row["workflow_type"] for row in finding_rows)])
    status_counts = Counter([report.robustness_status])
    for collection in (
        report.walk_forward_efficiency,
        report.sample_size_validation,
        report.sensitivity_summary,
        report.multiple_testing_summary,
    ):
        for row in collection:
            status_counts[getattr(row, "status", "not_evaluated")] += 1

    artifact_names = sorted(generated_artifacts or [])
    status_order = [*ROBUSTNESS_STATUS_ORDER, "not_evaluated"]
    status_order.extend(status for status in sorted(status_counts) if status not in status_order)
    summary = {
        "artifact_count": len(artifact_names),
        "checks_missing": sorted(set(report.checks_missing)),
        "checks_present": sorted(set(report.checks_present)),
        "finding_count": len(finding_rows),
        "finding_count_by_severity": {severity: severity_counts.get(severity, 0) for severity in SEVERITY_ORDER},
        "generated_artifacts": artifact_names,
        "highest_severity": highest_severity([row["severity"] for row in finding_rows]),
        "report_id": report.report_id,
        "robustness_status_counts": {status: status_counts.get(status, 0) for status in status_order if status_counts.get(status, 0)},
        "source_run_count": len(report.source_run_ids()),
        "source_run_ids": report.source_run_ids(),
        "workflow_type_counts": {key: workflow_counts[key] for key in sorted(workflow_counts)},
    }
    return canonicalize_value(summary)
