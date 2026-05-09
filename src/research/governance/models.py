from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GovernanceSourceRecord:
    run_id: str
    workflow_type: str
    registry_entry: dict[str, Any] = field(default_factory=dict)
    registry_path: Path | None = None
    artifact_dir: Path | None = None
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    promotion_gate_path: Path | None = None
    promotion_gates: dict[str, Any] | None = None
    promotion_gate_summary: dict[str, Any] | None = None
    review_summary_path: Path | None = None
    review_summary: dict[str, Any] | None = None
    candidate_review_summary_path: Path | None = None
    candidate_review_summary: dict[str, Any] | None = None
    governance_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceDataset:
    records: list[GovernanceSourceRecord]
    sources: dict[str, Any]


@dataclass(frozen=True)
class GovernanceReportResult:
    report_id: str
    output_dir: Path
    summary_path: Path
    outcome_matrix_path: Path
    reason_code_summary_path: Path
    severity_summary_path: Path
    workflow_summary_path: Path
    validation_path: Path
    markdown_path: Path
    manifest_path: Path
    validation: dict[str, Any]


__all__ = [
    "GovernanceDataset",
    "GovernanceReportResult",
    "GovernanceSourceRecord",
]
