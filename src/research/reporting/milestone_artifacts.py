from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value, serialize_canonical_json, stable_timestamp_from_run_id

MILESTONE_REPORT_SCHEMA_VERSION = 2
MILESTONE_REPORT_RUN_TYPE = "milestone_report"
MILESTONE_DECISION_LOG_RUN_TYPE = "milestone_decision_log"
MILESTONE_SUMMARY_FILENAME = "summary.json"
MILESTONE_DECISION_LOG_FILENAME = "decision_log.json"
MILESTONE_MANIFEST_FILENAME = "manifest.json"
_VALID_REPORT_STATUSES = frozenset({"draft", "final"})
_VALID_DECISION_STATUSES = frozenset({"accepted", "deferred", "rejected", "superseded"})


class MilestoneArtifactValidationError(ValueError):
    """Raised when milestone report artifacts do not satisfy the contract."""


@dataclass(frozen=True)
class MilestoneSourceArtifact:
    artifact_id: str
    path: str
    label: str | None = None
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(asdict(self))


@dataclass(frozen=True)
class MilestoneDecisionEntry:
    decision_id: str
    title: str
    status: str
    summary: str
    rationale: str
    impact: str | None = None
    owner: str | None = None
    category: str | None = None
    timestamp: str | None = None
    follow_up_actions: tuple[str, ...] = ()
    evidence_artifacts: tuple[str, ...] = ()
    source_artifacts: tuple[MilestoneSourceArtifact, ...] = ()
    related_stage_names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(asdict(self))


@dataclass(frozen=True)
class MilestoneReport:
    milestone_id: str
    milestone_name: str
    title: str
    status: str
    summary: str
    owner: str | None = None
    reporting_window: dict[str, Any] = field(default_factory=dict)
    scope: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    related_artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize_value(asdict(self))


def build_milestone_report_id(
    *,
    milestone_name: str,
    title: str,
    reporting_window: Mapping[str, Any] | None = None,
    scope: Sequence[str] | None = None,
) -> str:
    payload = {
        "milestone_name": milestone_name,
        "title": title,
        "reporting_window": canonicalize_value(dict(reporting_window or {})),
        "scope": list(scope or []),
    }
    digest = hashlib.sha256(serialize_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"milestone_report_{digest}"


def resolve_milestone_artifact_dir(output_path: str | Path) -> Path:
    resolved = Path(output_path)
    if resolved.suffix.lower() == ".json":
        if resolved.name != MILESTONE_SUMMARY_FILENAME:
            raise MilestoneArtifactValidationError(
                f"JSON output path must point to '{MILESTONE_SUMMARY_FILENAME}', observed '{resolved.name}'."
            )
        return resolved.parent
    return resolved


def write_milestone_report_artifacts(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
    output_path: str | Path,
) -> tuple[Path, Path, Path]:
    validate_milestone_report(report=report, decisions=decisions)
    artifact_dir = resolve_milestone_artifact_dir(output_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = build_milestone_report_summary_payload(report=report, decisions=decisions)
    decision_log_payload = build_milestone_decision_log_payload(report=report, decisions=decisions)
    manifest_payload = build_milestone_report_manifest_payload(
        report=report,
        summary_payload=summary_payload,
        decision_log_payload=decision_log_payload,
    )

    summary_path = artifact_dir / MILESTONE_SUMMARY_FILENAME
    decision_log_path = artifact_dir / MILESTONE_DECISION_LOG_FILENAME
    manifest_path = artifact_dir / MILESTONE_MANIFEST_FILENAME
    _write_json(summary_path, summary_payload)
    _write_json(decision_log_path, decision_log_payload)
    _write_json(manifest_path, manifest_payload)
    return summary_path, decision_log_path, manifest_path


def build_milestone_report_summary_payload(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
) -> dict[str, Any]:
    validate_milestone_report(report=report, decisions=decisions)
    decision_rows = [decision.to_dict() for decision in decisions]
    payload = {
        "run_type": MILESTONE_REPORT_RUN_TYPE,
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "milestone_id": report.milestone_id,
        "milestone_name": report.milestone_name,
        "title": report.title,
        "status": report.status,
        "owner": report.owner,
        "summary": report.summary,
        "reporting_window": canonicalize_value(dict(report.reporting_window)),
        "scope": list(report.scope),
        "key_findings": list(report.key_findings),
        "recommendations": list(report.recommendations),
        "open_questions": list(report.open_questions),
        "related_artifacts": canonicalize_value(dict(report.related_artifacts)),
        "decision_log_path": MILESTONE_DECISION_LOG_FILENAME,
        "decision_ids": [row["decision_id"] for row in decision_rows],
        "decision_counts_by_status": _decision_counts_by_status(decisions),
        "decision_count": len(decision_rows),
        "metadata": canonicalize_value(dict(report.metadata)),
    }
    return canonicalize_value(payload)


def build_milestone_decision_log_payload(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
) -> dict[str, Any]:
    validate_milestone_report(report=report, decisions=decisions)
    decision_rows = [decision.to_dict() for decision in decisions]
    payload = {
        "run_type": MILESTONE_DECISION_LOG_RUN_TYPE,
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "milestone_id": report.milestone_id,
        "milestone_name": report.milestone_name,
        "title": report.title,
        "decision_count": len(decision_rows),
        "decision_ids": [row["decision_id"] for row in decision_rows],
        "decision_counts_by_status": _decision_counts_by_status(decisions),
        "decisions": decision_rows,
        "rendered": _render_milestone_decision_log(
            report=report,
            decision_rows=decision_rows,
        ),
    }
    return canonicalize_value(payload)


def build_milestone_report_manifest_payload(
    *,
    report: MilestoneReport,
    summary_payload: Mapping[str, Any],
    decision_log_payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_milestone_report_payload(summary_payload)
    validate_milestone_decision_log_payload(decision_log_payload)
    artifact_files = sorted(
        [
            MILESTONE_DECISION_LOG_FILENAME,
            MILESTONE_MANIFEST_FILENAME,
            MILESTONE_SUMMARY_FILENAME,
        ]
    )
    payload = {
        "run_type": MILESTONE_REPORT_RUN_TYPE,
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "milestone_id": report.milestone_id,
        "milestone_name": report.milestone_name,
        "status": summary_payload["status"],
        "artifact_files": artifact_files,
        "artifact_groups": {
            "core": artifact_files,
            "decision_log": [MILESTONE_DECISION_LOG_FILENAME],
            "report": [MILESTONE_SUMMARY_FILENAME],
            "summary": [MILESTONE_SUMMARY_FILENAME],
        },
        "artifacts": {
            MILESTONE_DECISION_LOG_FILENAME: {
                "path": MILESTONE_DECISION_LOG_FILENAME,
                "decision_count": int(decision_log_payload["decision_count"]),
            },
            MILESTONE_MANIFEST_FILENAME: {"path": MILESTONE_MANIFEST_FILENAME},
            MILESTONE_SUMMARY_FILENAME: {
                "path": MILESTONE_SUMMARY_FILENAME,
                "decision_count": int(summary_payload["decision_count"]),
                "finding_count": len(list(summary_payload["key_findings"])),
                "recommendation_count": len(list(summary_payload["recommendations"])),
            },
        },
        "decision_counts_by_status": canonicalize_value(dict(summary_payload["decision_counts_by_status"])),
        "decision_ids": list(summary_payload["decision_ids"]),
        "decision_log_path": MILESTONE_DECISION_LOG_FILENAME,
        "summary_path": MILESTONE_SUMMARY_FILENAME,
        "timestamp": stable_timestamp_from_run_id(report.milestone_id),
    }
    return canonicalize_value(payload)


def validate_milestone_report(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
) -> None:
    _require_text(report.milestone_id, "report.milestone_id")
    _require_text(report.milestone_name, "report.milestone_name")
    _require_text(report.title, "report.title")
    _require_text(report.summary, "report.summary")
    if report.status not in _VALID_REPORT_STATUSES:
        formatted = ", ".join(sorted(_VALID_REPORT_STATUSES))
        raise MilestoneArtifactValidationError(f"report.status must be one of: {formatted}.")

    seen_decision_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        _require_text(decision.decision_id, f"decisions[{index}].decision_id")
        _require_text(decision.title, f"decisions[{index}].title")
        _require_text(decision.summary, f"decisions[{index}].summary")
        _require_text(decision.rationale, f"decisions[{index}].rationale")
        if decision.status not in _VALID_DECISION_STATUSES:
            formatted = ", ".join(sorted(_VALID_DECISION_STATUSES))
            raise MilestoneArtifactValidationError(
                f"decisions[{index}].status must be one of: {formatted}."
            )
        if decision.decision_id in seen_decision_ids:
            raise MilestoneArtifactValidationError(
                f"Duplicate decision_id detected: {decision.decision_id!r}."
            )
        seen_decision_ids.add(decision.decision_id)
        _require_posix_relative_paths(decision.evidence_artifacts, owner=f"decisions[{index}].evidence_artifacts")
        _require_non_empty_strings(decision.follow_up_actions, owner=f"decisions[{index}].follow_up_actions")
        _require_source_artifacts(decision.source_artifacts, owner=f"decisions[{index}].source_artifacts")


def validate_milestone_report_payload(payload: Mapping[str, Any]) -> None:
    if str(payload.get("run_type")) != MILESTONE_REPORT_RUN_TYPE:
        raise MilestoneArtifactValidationError("Milestone report summary payload has an invalid run_type.")
    if int(payload.get("schema_version") or 0) != MILESTONE_REPORT_SCHEMA_VERSION:
        raise MilestoneArtifactValidationError("Milestone report summary payload has an invalid schema_version.")
    _require_text(payload.get("milestone_id"), "payload.milestone_id")
    _require_text(payload.get("milestone_name"), "payload.milestone_name")
    _require_text(payload.get("title"), "payload.title")
    _require_text(payload.get("summary"), "payload.summary")
    if str(payload.get("decision_log_path")) != MILESTONE_DECISION_LOG_FILENAME:
        raise MilestoneArtifactValidationError("Milestone report summary must reference decision_log.json.")


def validate_milestone_decision_log_payload(payload: Mapping[str, Any]) -> None:
    if str(payload.get("run_type")) != MILESTONE_DECISION_LOG_RUN_TYPE:
        raise MilestoneArtifactValidationError("Milestone decision log payload has an invalid run_type.")
    if int(payload.get("schema_version") or 0) != MILESTONE_REPORT_SCHEMA_VERSION:
        raise MilestoneArtifactValidationError("Milestone decision log payload has an invalid schema_version.")
    _require_text(payload.get("milestone_id"), "payload.milestone_id")
    _require_text(payload.get("milestone_name"), "payload.milestone_name")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise MilestoneArtifactValidationError("Milestone decision log payload must contain a decisions list.")
    decision_count = int(payload.get("decision_count") or 0)
    if decision_count != len(decisions):
        raise MilestoneArtifactValidationError("Milestone decision log decision_count must match len(decisions).")
    rendered = payload.get("rendered")
    if not isinstance(rendered, Mapping):
        raise MilestoneArtifactValidationError("Milestone decision log payload must contain rendered human-readable content.")
    if not str(rendered.get("markdown") or "").strip():
        raise MilestoneArtifactValidationError("Milestone decision log rendered.markdown must be populated.")
    if not str(rendered.get("text") or "").strip():
        raise MilestoneArtifactValidationError("Milestone decision log rendered.text must be populated.")


def _decision_counts_by_status(decisions: Sequence[MilestoneDecisionEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.status] = counts.get(decision.status, 0) + 1
    return dict(sorted(counts.items()))


def _require_text(value: Any, owner: str) -> None:
    text = None if value is None else str(value).strip()
    if not text:
        raise MilestoneArtifactValidationError(f"{owner} must be a non-empty string.")


def _require_posix_relative_paths(paths: Sequence[str], *, owner: str) -> None:
    for index, path in enumerate(paths):
        text = str(path).strip()
        if not text:
            raise MilestoneArtifactValidationError(f"{owner}[{index}] must be a non-empty string.")
        if Path(text).is_absolute():
            raise MilestoneArtifactValidationError(f"{owner}[{index}] must be relative, observed absolute path.")
        if "\\" in text:
            raise MilestoneArtifactValidationError(
                f"{owner}[{index}] must use forward slashes for deterministic artifact references."
            )


def _require_non_empty_strings(values: Sequence[str], *, owner: str) -> None:
    for index, value in enumerate(values):
        _require_text(value, f"{owner}[{index}]")


def _require_source_artifacts(
    source_artifacts: Sequence[MilestoneSourceArtifact],
    *,
    owner: str,
) -> None:
    for index, source_artifact in enumerate(source_artifacts):
        _require_text(source_artifact.artifact_id, f"{owner}[{index}].artifact_id")
        _require_text(source_artifact.path, f"{owner}[{index}].path")
        _require_posix_relative_paths((source_artifact.path,), owner=f"{owner}[{index}].path")


def _render_milestone_decision_log(
    *,
    report: MilestoneReport,
    decision_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lines = [
        f"# {report.title}",
        "",
        f"Milestone: {report.milestone_name}",
        f"Milestone ID: {report.milestone_id}",
        f"Status: {report.status}",
        f"Decision Count: {len(decision_rows)}",
        "",
    ]
    text_lines = [
        f"{report.title}",
        f"Milestone: {report.milestone_name}",
        f"Milestone ID: {report.milestone_id}",
        f"Status: {report.status}",
        f"Decision Count: {len(decision_rows)}",
        "",
    ]

    for index, row in enumerate(decision_rows, start=1):
        lines.extend(_render_decision_markdown(index=index, row=row))
        text_lines.extend(_render_decision_text(index=index, row=row))
        if index != len(decision_rows):
            lines.append("")
            text_lines.append("")

    return {
        "markdown": "\n".join(lines).strip(),
        "text": "\n".join(text_lines).strip(),
    }


def _render_decision_markdown(*, index: int, row: Mapping[str, Any]) -> list[str]:
    lines = [
        f"## {index}. {row['title']}",
        f"- Decision ID: `{row['decision_id']}`",
        f"- Status: `{row['status']}`",
        f"- Summary: {row['summary']}",
        f"- Rationale: {row['rationale']}",
    ]
    if row.get("impact"):
        lines.append(f"- Impact: {row['impact']}")
    follow_up_actions = row.get("follow_up_actions")
    if isinstance(follow_up_actions, list) and follow_up_actions:
        lines.append("- Follow-Up Actions:")
        lines.extend(f"  - {action}" for action in follow_up_actions)
    source_artifacts = row.get("source_artifacts")
    if isinstance(source_artifacts, list) and source_artifacts:
        lines.append("- Linked Source Artifacts:")
        for source_artifact in source_artifacts:
            label = source_artifact.get("label") or source_artifact.get("artifact_id") or source_artifact.get("path")
            lines.append(f"  - {label}: `{source_artifact.get('path')}`")
    return lines


def _render_decision_text(*, index: int, row: Mapping[str, Any]) -> list[str]:
    lines = [
        f"{index}. {row['title']}",
        f"   Decision ID: {row['decision_id']}",
        f"   Status: {row['status']}",
        f"   Summary: {row['summary']}",
        f"   Rationale: {row['rationale']}",
    ]
    if row.get("impact"):
        lines.append(f"   Impact: {row['impact']}")
    follow_up_actions = row.get("follow_up_actions")
    if isinstance(follow_up_actions, list) and follow_up_actions:
        lines.append("   Follow-Up Actions:")
        lines.extend(f"   - {action}" for action in follow_up_actions)
    source_artifacts = row.get("source_artifacts")
    if isinstance(source_artifacts, list) and source_artifacts:
        lines.append("   Linked Source Artifacts:")
        for source_artifact in source_artifacts:
            label = source_artifact.get("label") or source_artifact.get("artifact_id") or source_artifact.get("path")
            lines.append(f"   - {label}: {source_artifact.get('path')}")
    return lines


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(canonicalize_value(dict(payload)), indent=2, sort_keys=True), encoding="utf-8", newline="\n")


__all__ = [
    "MILESTONE_DECISION_LOG_FILENAME",
    "MILESTONE_MANIFEST_FILENAME",
    "MILESTONE_REPORT_RUN_TYPE",
    "MILESTONE_REPORT_SCHEMA_VERSION",
    "MILESTONE_SUMMARY_FILENAME",
    "MilestoneArtifactValidationError",
    "MilestoneDecisionEntry",
    "MilestoneReport",
    "MilestoneSourceArtifact",
    "build_milestone_decision_log_payload",
    "build_milestone_report_id",
    "build_milestone_report_manifest_payload",
    "build_milestone_report_summary_payload",
    "resolve_milestone_artifact_dir",
    "validate_milestone_decision_log_payload",
    "validate_milestone_report",
    "validate_milestone_report_payload",
    "write_milestone_report_artifacts",
]
