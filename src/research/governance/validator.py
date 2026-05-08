from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.research.registry import canonicalize_value

from .models import GovernanceSourceRecord

CANONICAL_PROMOTION_STATUSES = frozenset({"eligible", "warn", "needs_review", "rejected", "blocked"})
PROMOTION_TO_REVIEW_STATUS = {
    "eligible": "candidate",
    "warn": "needs_review",
    "needs_review": "needs_review",
    "rejected": "rejected",
    "blocked": "rejected",
}
PATH_FIELDS = ("registry_path", "manifest_path")


def validate_governance_consistency(
    records: list[GovernanceSourceRecord],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return JSON-safe consistency findings for governance observability inputs."""

    findings: list[dict[str, Any]] = []
    by_run_id = {record.run_id: record for record in records}
    for row in rows:
        run_id = str(row.get("run_id") or "")
        record = by_run_id.get(run_id)
        if record is None:
            continue
        findings.extend(_record_findings(record, row))
        findings.extend(_path_findings(row))

    counts_by_severity: dict[str, int] = {}
    counts_by_check: dict[str, int] = {}
    for finding in findings:
        severity = str(finding["severity"])
        check_id = str(finding["check_id"])
        counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        counts_by_check[check_id] = counts_by_check.get(check_id, 0) + 1
    return canonicalize_value(
        {
            "status": "pass" if not findings else "fail",
            "record_count": len(records),
            "finding_count": len(findings),
            "counts_by_severity": dict(sorted(counts_by_severity.items())),
            "counts_by_check": dict(sorted(counts_by_check.items())),
            "findings": sorted(findings, key=lambda item: (item["severity"], item["check_id"], item["run_id"])),
        }
    )


def _record_findings(record: GovernanceSourceRecord, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    entry = record.registry_entry
    summary = record.promotion_gate_summary
    promotion_status = str(row.get("promotion_status") or "").strip()

    if summary is None:
        findings.append(_finding("missing_promotion_summary", record, severity="warning"))
    if promotion_status and promotion_status not in CANONICAL_PROMOTION_STATUSES:
        findings.append(
            _finding(
                "unknown_promotion_status",
                record,
                severity="error",
                details={"promotion_status": promotion_status, "canonical_statuses": sorted(CANONICAL_PROMOTION_STATUSES)},
            )
        )

    registry_status = _coerce_string(entry.get("promotion_status"))
    summary_status = _coerce_string(summary.get("promotion_status") if isinstance(summary, Mapping) else None)
    if registry_status is not None and summary_status is not None and registry_status != summary_status:
        findings.append(
            _finding(
                "registry_promotion_status_mismatch",
                record,
                severity="error",
                details={"registry_promotion_status": registry_status, "summary_promotion_status": summary_status},
            )
        )

    review_status = _coerce_string(row.get("review_status"))
    expected_review = PROMOTION_TO_REVIEW_STATUS.get(promotion_status)
    if review_status is not None and expected_review is not None and review_status != expected_review:
        findings.append(
            _finding(
                "review_status_mismatch",
                record,
                severity="error",
                details={"promotion_status": promotion_status, "review_status": review_status, "expected_review_status": expected_review},
            )
        )

    if record.manifest_path is not None and not record.manifest_path.exists():
        findings.append(
            _finding(
                "missing_or_stale_manifest_link",
                record,
                severity="warning",
                details={"manifest_path": record.manifest_path.as_posix()},
            )
        )
    if record.manifest is not None:
        manifest_run_id = _coerce_string(record.manifest.get("run_id") or record.manifest.get("review_id"))
        if manifest_run_id is not None and manifest_run_id != record.run_id and record.workflow_type != "candidate_review":
            findings.append(
                _finding(
                    "manifest_run_id_mismatch",
                    record,
                    severity="warning",
                    details={"manifest_run_id": manifest_run_id},
                )
            )

    if record.candidate_review_summary is not None:
        context = record.candidate_review_summary.get("promotion_context")
        if isinstance(context, Mapping):
            portfolio_summary = context.get("portfolio_promotion_gate_summary")
            context_status = (
                _coerce_string(portfolio_summary.get("promotion_status"))
                if isinstance(portfolio_summary, Mapping)
                else None
            )
            if context_status is not None and promotion_status and context_status != promotion_status:
                findings.append(
                    _finding(
                        "candidate_review_context_mismatch",
                        record,
                        severity="error",
                        details={"context_promotion_status": context_status, "promotion_status": promotion_status},
                    )
                )
    return findings


def _path_findings(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    run_id = str(row.get("run_id") or "")
    workflow_type = str(row.get("workflow_type") or "")
    for field in PATH_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if path.is_absolute():
            findings.append(
                {
                    "check_id": "non_relative_artifact_path",
                    "severity": "error",
                    "run_id": run_id,
                    "workflow_type": workflow_type,
                    "message": f"{field} must be relative in governance outputs.",
                    "details": {"field": field, "path": value},
                }
            )
    return findings


def _finding(
    check_id: str,
    record: GovernanceSourceRecord,
    *,
    severity: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "run_id": record.run_id,
        "workflow_type": record.workflow_type,
        "message": _message(check_id),
        "details": canonicalize_value(dict(details or {})),
    }


def _message(check_id: str) -> str:
    messages = {
        "candidate_review_context_mismatch": "Candidate-review promotion context does not match the normalized governance row.",
        "manifest_run_id_mismatch": "Manifest run id does not match the source record run id.",
        "missing_or_stale_manifest_link": "Manifest path is missing or stale.",
        "missing_promotion_summary": "Promotion summary is missing; governance row cannot cite canonical promotion status.",
        "registry_promotion_status_mismatch": "Registry promotion status differs from promotion_gate_summary.promotion_status.",
        "review_status_mismatch": "Review status differs from the canonical M31 promotion-to-review mapping.",
        "unknown_promotion_status": "Promotion status is not one of the canonical M31 promotion statuses.",
    }
    return messages.get(check_id, check_id)


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "CANONICAL_PROMOTION_STATUSES",
    "PROMOTION_TO_REVIEW_STATUS",
    "validate_governance_consistency",
]
