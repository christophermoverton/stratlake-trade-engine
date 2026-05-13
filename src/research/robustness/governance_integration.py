from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

from .models import SEVERITY_ORDER, sanitize_portable_path, sanitize_portable_value

ROBUSTNESS_GOVERNANCE_FIELDS: list[str] = [
    "robustness_report_path",
    "robustness_status",
    "wfe_status",
    "sample_size_status",
    "sensitivity_status",
    "multiple_testing_status",
    "temporal_validation_status",
    "robustness_finding_count",
    "highest_robustness_severity",
    "robustness_reason_codes",
    "robustness_available",
]

ROBUSTNESS_REASON_CODE_MAP: dict[str, str] = {
    "multiple_testing.extreme_risk": "extreme_search_space_warning",
    "multiple_testing.high_risk": "large_search_space_warning",
    "multiple_testing.missing_trial_count_metadata": "missing_trial_count_metadata",
    "sample_size.minimum_oos_trades": "insufficient_oos_trades",
    "sample_size.minimum_total_samples": "thin_total_sample",
    "sample_size.minimum_total_trades": "thin_trade_sample",
    "sample_size.missing_trade_count": "missing_trade_count_metadata",
    "sensitivity.fragile": "fragile_parameter_optimum",
    "sensitivity.mildly_sensitive": "sensitive_parameter_region",
    "temporal_validation.embargo_violation": "temporal_validation_embargo_violation",
    "temporal_validation.purged_interval_overlap": "temporal_validation_leakage_risk",
    "temporal_validation.train_validation_overlap": "temporal_validation_overlap",
    "walk_forward_efficiency.broken": "negative_oos_transfer",
    "walk_forward_efficiency.undefined": "undefined_walk_forward_efficiency",
    "walk_forward_efficiency.weak": "weak_walk_forward_efficiency",
}

STATUS_ORDER: tuple[str, ...] = (
    "missing",
    "unavailable",
    "pass",
    "robust",
    "acceptable",
    "low_risk",
    "stable",
    "improved",
    "info",
    "warning",
    "moderate_risk",
    "mildly_sensitive",
    "weak",
    "high_risk",
    "needs_review",
    "fragile",
    "broken",
    "extreme_risk",
    "reject",
    "undefined",
    "blocked",
)


@dataclass(frozen=True)
class RobustnessGovernanceContext:
    workflow_type: str
    run_id: str
    source_run_id: str | None = None
    robustness_report_path: str = ""
    robustness_status: str = "missing"
    wfe_status: str = "missing"
    sample_size_status: str = "missing"
    sensitivity_status: str = "missing"
    multiple_testing_status: str = "missing"
    temporal_validation_status: str = "missing"
    robustness_finding_count: int = 0
    highest_robustness_severity: str = "info"
    robustness_reason_codes: tuple[str, ...] = ()
    robustness_available: bool = False
    robustness_missing_reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(
            {
                "details": dict(self.details),
                "highest_robustness_severity": self.highest_robustness_severity,
                "multiple_testing_status": self.multiple_testing_status,
                "robustness_available": self.robustness_available,
                "robustness_finding_count": self.robustness_finding_count,
                "robustness_missing_reason": self.robustness_missing_reason,
                "robustness_reason_codes": list(self.robustness_reason_codes),
                "robustness_report_path": self.robustness_report_path,
                "robustness_status": self.robustness_status,
                "run_id": self.run_id,
                "sample_size_status": self.sample_size_status,
                "sensitivity_status": self.sensitivity_status,
                "source_run_id": self.source_run_id,
                "temporal_validation_status": self.temporal_validation_status,
                "wfe_status": self.wfe_status,
                "workflow_type": self.workflow_type,
            }
        )

    def to_governance_fields(self) -> dict[str, Any]:
        return {
            "highest_robustness_severity": self.highest_robustness_severity,
            "multiple_testing_status": self.multiple_testing_status,
            "robustness_available": str(self.robustness_available).lower(),
            "robustness_finding_count": self.robustness_finding_count,
            "robustness_reason_codes": "|".join(self.robustness_reason_codes),
            "robustness_report_path": self.robustness_report_path,
            "robustness_status": self.robustness_status,
            "sample_size_status": self.sample_size_status,
            "sensitivity_status": self.sensitivity_status,
            "temporal_validation_status": self.temporal_validation_status,
            "wfe_status": self.wfe_status,
        }


def load_robustness_governance_context(
    robustness_report_path: str | Path | None,
    *,
    workflow_type: str = "unknown",
    run_id: str = "unknown",
    source_run_id: str | None = None,
    roots: tuple[Path, ...] = (),
) -> RobustnessGovernanceContext:
    if robustness_report_path is None or str(robustness_report_path).strip() == "":
        return _missing_context(
            workflow_type=workflow_type,
            run_id=run_id,
            source_run_id=source_run_id,
            reason="robustness_report_not_found",
        )
    path = Path(robustness_report_path)
    report_dir = path if path.is_dir() else path.parent
    if path.is_file() and path.name != "robustness_summary.json":
        report_dir = path.parent
    summary_path = path / "robustness_summary.json" if path.is_dir() else path
    if path.is_dir() or path.name != "robustness_summary.json":
        summary_path = report_dir / "robustness_summary.json"
    if not summary_path.exists():
        return _missing_context(
            workflow_type=workflow_type,
            run_id=run_id,
            source_run_id=source_run_id,
            path=path,
            reason="robustness_report_not_found",
            roots=roots,
        )
    summary = _read_json(summary_path)
    if summary is None:
        return _missing_context(
            workflow_type=workflow_type,
            run_id=run_id,
            source_run_id=source_run_id,
            path=path,
            reason="robustness_report_malformed",
            roots=roots,
        )
    return build_robustness_governance_context(
        summary=summary,
        findings=_findings_from_path(report_dir / "robustness_findings.json"),
        report_dir=report_dir,
        report_path=summary_path,
        workflow_type=workflow_type,
        run_id=run_id,
        source_run_id=source_run_id,
        roots=roots,
    )


def build_robustness_governance_context(
    *,
    summary: Mapping[str, Any] | None = None,
    findings: Sequence[Mapping[str, Any]] | None = None,
    report_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    workflow_type: str = "unknown",
    run_id: str = "unknown",
    source_run_id: str | None = None,
    roots: tuple[Path, ...] = (),
) -> RobustnessGovernanceContext:
    resolved_summary = dict(summary or {})
    resolved_findings = [dict(finding) for finding in findings or [] if isinstance(finding, Mapping)]
    resolved_report_dir = Path(report_dir) if report_dir is not None else None
    diagnostics = _diagnostic_statuses(resolved_summary, resolved_findings, resolved_report_dir)
    reason_codes = map_robustness_findings_to_reason_codes(resolved_findings)
    severities = [str(finding.get("severity", "info")) for finding in resolved_findings]
    details = _context_details(resolved_summary, resolved_findings, resolved_report_dir)
    report_path_text = ""
    if report_path is not None:
        report_path_text = sanitize_portable_path(report_path, roots=roots)
    elif resolved_report_dir is not None:
        report_path_text = sanitize_portable_path(resolved_report_dir, roots=roots)
    return RobustnessGovernanceContext(
        workflow_type=workflow_type,
        run_id=run_id,
        source_run_id=source_run_id,
        robustness_report_path=report_path_text,
        robustness_status=_summary_status(resolved_summary, diagnostics),
        wfe_status=diagnostics["wfe_status"],
        sample_size_status=diagnostics["sample_size_status"],
        sensitivity_status=diagnostics["sensitivity_status"],
        multiple_testing_status=diagnostics["multiple_testing_status"],
        temporal_validation_status=diagnostics["temporal_validation_status"],
        robustness_finding_count=len(resolved_findings),
        highest_robustness_severity=_highest_severity(severities),
        robustness_reason_codes=tuple(reason_codes),
        robustness_available=True,
        details=sanitize_portable_value(details, roots=roots),
    )


def summarize_robustness_for_governance(context: RobustnessGovernanceContext | Mapping[str, Any]) -> dict[str, Any]:
    resolved = context if isinstance(context, RobustnessGovernanceContext) else _context_from_mapping(context)
    return resolved.to_governance_fields()


def map_robustness_findings_to_reason_codes(findings: Sequence[Mapping[str, Any]]) -> list[str]:
    codes = {
        ROBUSTNESS_REASON_CODE_MAP.get(str(finding.get("check_id", "")).strip(), "robustness_review_finding")
        for finding in findings
        if _finding_should_map(finding)
    }
    return sorted(code for code in codes if code)


def attach_robustness_context_to_governance_rows(
    rows: Sequence[Mapping[str, Any]],
    contexts: Mapping[tuple[str, str] | str, RobustnessGovernanceContext | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        key = (str(row.get("workflow_type", "")), str(row.get("run_id", "")))
        context = contexts.get(key) or contexts.get(str(row.get("run_id", "")))
        if context is None:
            context = _missing_context(
                workflow_type=str(row.get("workflow_type", "unknown")),
                run_id=str(row.get("run_id", "unknown")),
                reason="robustness_context_not_attached",
            )
        result.update(summarize_robustness_for_governance(context))
        output.append(result)
    return sorted(output, key=lambda item: (str(item.get("workflow_type", "")), str(item.get("run_id", ""))))


def _diagnostic_statuses(
    summary: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    report_dir: Path | None,
) -> dict[str, str]:
    statuses = {
        "multiple_testing_status": _status_from_multiple_testing(report_dir),
        "sample_size_status": _status_from_sample_size(report_dir),
        "sensitivity_status": _status_from_sensitivity(report_dir),
        "temporal_validation_status": _status_from_temporal_validation(report_dir, findings),
        "wfe_status": _status_from_wfe(report_dir),
    }
    for check_prefix, status_key in (
        ("multiple_testing.", "multiple_testing_status"),
        ("sample_size.", "sample_size_status"),
        ("sensitivity.", "sensitivity_status"),
        ("temporal_validation.", "temporal_validation_status"),
        ("walk_forward_efficiency.", "wfe_status"),
    ):
        finding_statuses = [
            _status_from_finding(finding, check_prefix=check_prefix)
            for finding in findings
            if str(finding.get("check_id", "")).startswith(check_prefix)
        ]
        statuses[status_key] = _max_status([statuses[status_key], *finding_statuses])
    if not findings and isinstance(summary.get("robustness_status_counts"), Mapping):
        counts = summary["robustness_status_counts"]
        if isinstance(counts, Mapping):
            inferred = _max_status([str(key) for key, value in counts.items() if int(value or 0) > 0])
            for key, value in list(statuses.items()):
                if value in {"missing", "unavailable"} and inferred != "missing":
                    statuses[key] = "unavailable"
    return statuses


def _status_from_finding(finding: Mapping[str, Any], *, check_prefix: str) -> str:
    suffix = str(finding.get("check_id", "")).removeprefix(check_prefix)
    if suffix in STATUS_ORDER:
        return suffix
    severity = str(finding.get("severity", "")).strip()
    return severity if severity in STATUS_ORDER else "needs_review"


def _summary_status(summary: Mapping[str, Any], diagnostics: Mapping[str, str]) -> str:
    diagnostic_values = list(diagnostics.values())
    counts = summary.get("robustness_status_counts")
    if isinstance(counts, Mapping) and counts:
        return _max_status([str(key) for key, value in counts.items() if int(value or 0) > 0] + diagnostic_values)
    return _max_status(diagnostic_values)


def _context_details(
    summary: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    report_dir: Path | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "finding_count_from_findings": len(findings),
        "generated_artifacts": summary.get("generated_artifacts", []),
        "report_id": summary.get("report_id"),
        "source_run_ids": summary.get("source_run_ids", []),
    }
    if isinstance(summary.get("finding_count"), int) and summary.get("finding_count") != len(findings):
        details["metadata_conflicts"] = [
            {
                "check_id": "robustness_finding_count_mismatch",
                "findings_file_count": len(findings),
                "summary_finding_count": summary.get("finding_count"),
            }
        ]
    if report_dir is not None:
        expected = ["robustness_summary.json", "robustness_findings.json"]
        missing = [name for name in expected if not (report_dir / name).exists()]
        if missing:
            details.setdefault("metadata_conflicts", [])
            details["metadata_conflicts"].append({"check_id": "robustness_expected_artifact_missing", "missing_artifacts": missing})
    return canonicalize_value(details)


def _findings_from_path(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return []
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return []
    return [dict(item) for item in findings if isinstance(item, Mapping)]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _status_from_wfe(report_dir: Path | None) -> str:
    return _max_status(row.get("status", "") for row in _read_csv_rows(None if report_dir is None else report_dir / "walk_forward_efficiency.csv")) or "missing"


def _status_from_sensitivity(report_dir: Path | None) -> str:
    return _max_status(row.get("status", "") for row in _read_csv_rows(None if report_dir is None else report_dir / "sensitivity_summary.csv")) or "missing"


def _status_from_sample_size(report_dir: Path | None) -> str:
    payload = _read_json(report_dir / "sample_size_validation.json") if report_dir is not None else None
    checks = payload.get("checks") if isinstance(payload, Mapping) else None
    if not isinstance(checks, list):
        return "missing"
    return _max_status(str(item.get("status", "")) for item in checks if isinstance(item, Mapping))


def _status_from_multiple_testing(report_dir: Path | None) -> str:
    payload = _read_json(report_dir / "multiple_testing_summary.json") if report_dir is not None else None
    families = payload.get("families") if isinstance(payload, Mapping) else None
    if not isinstance(families, list):
        return "missing"
    return _max_status(str(item.get("status", "")) for item in families if isinstance(item, Mapping))


def _status_from_temporal_validation(report_dir: Path | None, findings: Sequence[Mapping[str, Any]]) -> str:
    statuses = [
        str(finding.get("details", {}).get("status", ""))
        for finding in findings
        if str(finding.get("check_id", "")).startswith("temporal_validation.")
        and isinstance(finding.get("details", {}), Mapping)
    ]
    payload = _read_json(report_dir / "leakage_validation.json") if report_dir is not None else None
    if isinstance(payload, Mapping):
        statuses.append(str(payload.get("overall_status", "")))
    return _max_status(statuses)


def _max_status(values: Any) -> str:
    statuses = [str(value).strip() for value in values if str(value).strip()]
    if not statuses:
        return "missing"
    return max(statuses, key=lambda value: STATUS_ORDER.index(value) if value in STATUS_ORDER else len(STATUS_ORDER))


def _highest_severity(values: Sequence[str]) -> str:
    severities = [value for value in values if value in SEVERITY_ORDER]
    if not severities:
        return "info"
    return max(severities, key=lambda value: SEVERITY_ORDER.index(value))


def _finding_should_map(finding: Mapping[str, Any]) -> bool:
    severity = str(finding.get("severity", "info"))
    return severity != "info" or str(finding.get("check_id", "")).startswith("temporal_validation.")


def _missing_context(
    *,
    workflow_type: str,
    run_id: str,
    source_run_id: str | None = None,
    path: str | Path | None = None,
    reason: str,
    roots: tuple[Path, ...] = (),
) -> RobustnessGovernanceContext:
    return RobustnessGovernanceContext(
        workflow_type=workflow_type,
        run_id=run_id,
        source_run_id=source_run_id,
        robustness_report_path="" if path is None else sanitize_portable_path(path, roots=roots),
        robustness_status="missing",
        robustness_available=False,
        robustness_missing_reason=reason,
        details={"reason": reason},
    )


def _context_from_mapping(value: Mapping[str, Any]) -> RobustnessGovernanceContext:
    codes = value.get("robustness_reason_codes", [])
    if isinstance(codes, str):
        resolved_codes = tuple(code for code in codes.split("|") if code)
    elif isinstance(codes, list | tuple):
        resolved_codes = tuple(str(code) for code in codes if str(code).strip())
    else:
        resolved_codes = ()
    return RobustnessGovernanceContext(
        workflow_type=str(value.get("workflow_type", "unknown")),
        run_id=str(value.get("run_id", "unknown")),
        source_run_id=None if value.get("source_run_id") is None else str(value.get("source_run_id")),
        robustness_report_path=str(value.get("robustness_report_path", "")),
        robustness_status=str(value.get("robustness_status", "missing")),
        wfe_status=str(value.get("wfe_status", "missing")),
        sample_size_status=str(value.get("sample_size_status", "missing")),
        sensitivity_status=str(value.get("sensitivity_status", "missing")),
        multiple_testing_status=str(value.get("multiple_testing_status", "missing")),
        temporal_validation_status=str(value.get("temporal_validation_status", "missing")),
        robustness_finding_count=int(value.get("robustness_finding_count", 0) or 0),
        highest_robustness_severity=str(value.get("highest_robustness_severity", "info")),
        robustness_reason_codes=tuple(sorted(set(resolved_codes))),
        robustness_available=bool(value.get("robustness_available", False)),
        robustness_missing_reason=str(value.get("robustness_missing_reason", "")),
        details=dict(value.get("details", {})) if isinstance(value.get("details", {}), Mapping) else {},
    )


__all__ = [
    "ROBUSTNESS_GOVERNANCE_FIELDS",
    "ROBUSTNESS_REASON_CODE_MAP",
    "RobustnessGovernanceContext",
    "attach_robustness_context_to_governance_rows",
    "build_robustness_governance_context",
    "load_robustness_governance_context",
    "map_robustness_findings_to_reason_codes",
    "summarize_robustness_for_governance",
]
