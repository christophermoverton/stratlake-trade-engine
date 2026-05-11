from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.artifacts.safety import portable_path
from src.research.registry import canonicalize_value

SCHEMA_VERSION = 1
ROBUSTNESS_REPORT_ARTIFACT_TYPE = "statistical_robustness_report"
DEFAULT_CREATED_AT_UTC = "not_recorded"

SEVERITY_ORDER: tuple[str, ...] = ("info", "warning", "needs_review", "reject", "blocked")
ROBUSTNESS_STATUS_ORDER: tuple[str, ...] = ("pass", "warning", "needs_review", "reject", "blocked")

WALK_FORWARD_EFFICIENCY_COLUMNS: list[str] = [
    "workflow_type",
    "run_id",
    "split_id",
    "train_period",
    "test_period",
    "in_sample_metric",
    "out_of_sample_metric",
    "walk_forward_efficiency",
    "status",
    "details",
]

SENSITIVITY_SUMMARY_COLUMNS: list[str] = [
    "workflow_type",
    "run_id",
    "scenario_id",
    "parameter",
    "baseline_value",
    "scenario_value",
    "metric",
    "baseline_metric_value",
    "scenario_metric_value",
    "delta",
    "status",
    "details",
]

SAMPLE_SIZE_JSON_FIELDS: list[str] = [
    "workflow_type",
    "run_id",
    "check_id",
    "sample_count",
    "trade_count",
    "minimum_sample_count",
    "minimum_trade_count",
    "status",
    "details",
]

MULTIPLE_TESTING_JSON_FIELDS: list[str] = [
    "workflow_type",
    "run_id",
    "family_id",
    "trial_count",
    "effective_trial_count",
    "adjustment_method",
    "status",
    "details",
]


@dataclass(frozen=True)
class ArtifactReference:
    path: str | Path
    artifact_type: str = "artifact"
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return canonicalize_value(
            {
                "artifact_type": self.artifact_type,
                "description": self.description,
                "metadata": sanitize_portable_value(dict(self.metadata), roots=roots),
                "path": sanitize_portable_path(self.path, roots=roots),
            }
        )


@dataclass(frozen=True)
class UpstreamReferences:
    strategy: tuple[ArtifactReference, ...] = ()
    alpha: tuple[ArtifactReference, ...] = ()
    portfolio: tuple[ArtifactReference, ...] = ()
    campaign: tuple[ArtifactReference, ...] = ()
    governance: tuple[ArtifactReference, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return canonicalize_value(
            {
                "alpha": [reference.to_dict(roots=roots) for reference in self.alpha],
                "artifacts": [reference.to_dict(roots=roots) for reference in self.artifacts],
                "campaign": [reference.to_dict(roots=roots) for reference in self.campaign],
                "governance": [reference.to_dict(roots=roots) for reference in self.governance],
                "portfolio": [reference.to_dict(roots=roots) for reference in self.portfolio],
                "strategy": [reference.to_dict(roots=roots) for reference in self.strategy],
            }
        )

    def source_artifacts(self, *, roots: tuple[Path, ...] = ()) -> list[dict[str, Any]]:
        grouped = self.to_dict(roots=roots)
        rows: list[dict[str, Any]] = []
        for group_name in sorted(grouped):
            for reference in grouped[group_name]:
                rows.append({"reference_group": group_name, **reference})
        return rows


@dataclass(frozen=True)
class RobustnessFinding:
    check_id: str
    severity: str
    workflow_type: str
    run_id: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return canonicalize_value(
            {
                "check_id": _non_empty_text(self.check_id, "check_id"),
                "details": sanitize_portable_value(dict(self.details), roots=roots),
                "message": _non_empty_text(self.message, "message"),
                "run_id": _non_empty_text(self.run_id, "run_id"),
                "severity": normalize_severity(self.severity),
                "workflow_type": _non_empty_text(self.workflow_type, "workflow_type"),
            }
        )


@dataclass(frozen=True)
class WalkForwardEfficiencyRow:
    workflow_type: str
    run_id: str
    split_id: str
    train_period: str = ""
    test_period: str = ""
    in_sample_metric: float | str | None = None
    out_of_sample_metric: float | str | None = None
    walk_forward_efficiency: float | str | None = None
    status: str = "not_evaluated"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_csv_row(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return _stable_csv_row(self, WALK_FORWARD_EFFICIENCY_COLUMNS, roots=roots)


@dataclass(frozen=True)
class SensitivitySummaryRow:
    workflow_type: str
    run_id: str
    scenario_id: str
    parameter: str = ""
    baseline_value: Any = None
    scenario_value: Any = None
    metric: str = ""
    baseline_metric_value: float | str | None = None
    scenario_metric_value: float | str | None = None
    delta: float | str | None = None
    status: str = "not_evaluated"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_csv_row(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return _stable_csv_row(self, SENSITIVITY_SUMMARY_COLUMNS, roots=roots)


@dataclass(frozen=True)
class SampleSizeValidation:
    workflow_type: str
    run_id: str
    check_id: str
    sample_count: int | None = None
    trade_count: int | None = None
    minimum_sample_count: int | None = None
    minimum_trade_count: int | None = None
    status: str = "not_evaluated"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return _stable_json_row(self, SAMPLE_SIZE_JSON_FIELDS, roots=roots)


@dataclass(frozen=True)
class MultipleTestingSummary:
    workflow_type: str
    run_id: str
    family_id: str
    trial_count: int | None = None
    effective_trial_count: int | None = None
    adjustment_method: str = "not_evaluated"
    status: str = "not_evaluated"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, Any]:
        return _stable_json_row(self, MULTIPLE_TESTING_JSON_FIELDS, roots=roots)


@dataclass(frozen=True)
class RobustnessReport:
    report_id: str
    workflow_type: str
    run_id: str
    source_run_id: str | None = None
    robustness_status: str = "not_evaluated"
    upstream_references: UpstreamReferences = field(default_factory=UpstreamReferences)
    findings: tuple[RobustnessFinding, ...] = ()
    walk_forward_efficiency: tuple[WalkForwardEfficiencyRow, ...] = ()
    sample_size_validation: tuple[SampleSizeValidation, ...] = ()
    sensitivity_summary: tuple[SensitivitySummaryRow, ...] = ()
    multiple_testing_summary: tuple[MultipleTestingSummary, ...] = ()
    checks_present: tuple[str, ...] = ()
    checks_missing: tuple[str, ...] = ()
    created_at_utc: str = DEFAULT_CREATED_AT_UTC
    writer_name: str = "robustness_report_writer"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def source_run_ids(self) -> list[str]:
        values = {self.run_id}
        if self.source_run_id:
            values.add(self.source_run_id)
        for finding in self.findings:
            values.add(finding.run_id)
        for row in self.walk_forward_efficiency:
            values.add(row.run_id)
        for row in self.sample_size_validation:
            values.add(row.run_id)
        for row in self.sensitivity_summary:
            values.add(row.run_id)
        for row in self.multiple_testing_summary:
            values.add(row.run_id)
        return sorted(value for value in values if value)


@dataclass(frozen=True)
class RobustnessReportResult:
    report_id: str
    output_dir: Path
    summary_path: Path
    findings_path: Path
    walk_forward_efficiency_path: Path
    sample_size_validation_path: Path
    sensitivity_summary_path: Path
    multiple_testing_summary_path: Path
    markdown_path: Path
    manifest_path: Path


def normalize_severity(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in SEVERITY_ORDER:
        raise ValueError(f"Unknown robustness severity '{value}'. Expected one of: {', '.join(SEVERITY_ORDER)}.")
    return normalized


def highest_severity(severities: list[str]) -> str:
    if not severities:
        return "info"
    normalized = [normalize_severity(value) for value in severities]
    return max(normalized, key=lambda value: SEVERITY_ORDER.index(value))


def sanitize_portable_path(path: str | Path, *, roots: tuple[Path, ...] = ()) -> str:
    return portable_path(path, roots=(Path.cwd(), *roots))


def sanitize_portable_value(value: Any, *, roots: tuple[Path, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        return canonicalize_value({str(key): sanitize_portable_value(item, roots=roots) for key, item in value.items()})
    if isinstance(value, tuple):
        return [sanitize_portable_value(item, roots=roots) for item in value]
    if isinstance(value, list):
        return [sanitize_portable_value(item, roots=roots) for item in value]
    if isinstance(value, Path):
        return sanitize_portable_path(value, roots=roots)
    if isinstance(value, str) and _looks_path_like(value):
        return sanitize_portable_path(value, roots=roots)
    return value


def _stable_json_row(instance: Any, fields: list[str], *, roots: tuple[Path, ...]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in fields:
        value = getattr(instance, field_name)
        payload[field_name] = sanitize_portable_value(value, roots=roots)
    return canonicalize_value(payload)


def _stable_csv_row(instance: Any, fields: list[str], *, roots: tuple[Path, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field_name in fields:
        value = getattr(instance, field_name)
        if value is None:
            row[field_name] = ""
        elif isinstance(value, Mapping | list | tuple):
            row[field_name] = _stable_inline_value(sanitize_portable_value(value, roots=roots))
        else:
            row[field_name] = sanitize_portable_value(value, roots=roots)
    return row


def _stable_inline_value(value: Any) -> str:
    import json

    return json.dumps(canonicalize_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _non_empty_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Robustness {field_name} must be non-empty.")
    return normalized


def _looks_path_like(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith("file://")
