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
MILESTONE_MARKDOWN_REPORT_FILENAME = "report.md"
_VALID_REPORT_STATUSES = frozenset({"draft", "final"})
_VALID_DECISION_STATUSES = frozenset({"accepted", "deferred", "rejected", "superseded"})
_VALID_DECISION_LOG_RENDER_FORMATS = frozenset({"markdown", "text"})
_REPORT_SECTION_KEYS = (
    "campaign_scope",
    "selections",
    "key_findings",
    "key_metrics",
    "gate_outcomes",
    "risks",
    "next_steps",
    "open_questions",
    "decision_snapshot",
    "related_artifacts",
)


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


def _default_generation_options() -> dict[str, Any]:
    return {
        "include_markdown_report": True,
        "decision_log_render_formats": ("markdown", "text"),
        "sections": {key: True for key in _REPORT_SECTION_KEYS},
        "decision_categories": (),
        "summary": {
            "include_stage_counts": True,
            "include_review_outcome": True,
        },
    }


def resolve_milestone_generation_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved = _default_generation_options()
    if options is None:
        return resolved
    if not isinstance(options, Mapping):
        raise MilestoneArtifactValidationError("Milestone generation options must be a mapping when provided.")

    include_markdown = options.get("include_markdown_report")
    if include_markdown is not None:
        if not isinstance(include_markdown, bool):
            raise MilestoneArtifactValidationError(
                "Milestone generation option 'include_markdown_report' must be a boolean."
            )
        resolved["include_markdown_report"] = include_markdown

    render_formats = options.get("decision_log_render_formats")
    if render_formats is not None:
        normalized_formats = _normalize_string_sequence(
            render_formats,
            owner="decision_log_render_formats",
        )
        invalid_formats = [value for value in normalized_formats if value not in _VALID_DECISION_LOG_RENDER_FORMATS]
        if invalid_formats:
            raise MilestoneArtifactValidationError(
                "Milestone generation option 'decision_log_render_formats' must contain only "
                f"{sorted(_VALID_DECISION_LOG_RENDER_FORMATS)}."
            )
        if not normalized_formats:
            raise MilestoneArtifactValidationError(
                "Milestone generation option 'decision_log_render_formats' must not be empty."
            )
        resolved["decision_log_render_formats"] = normalized_formats

    sections = options.get("sections")
    if sections is not None:
        if not isinstance(sections, Mapping):
            raise MilestoneArtifactValidationError("Milestone generation option 'sections' must be a mapping.")
        unknown_sections = sorted(set(sections) - set(_REPORT_SECTION_KEYS))
        if unknown_sections:
            raise MilestoneArtifactValidationError(
                f"Milestone generation option 'sections' contains unsupported keys: {unknown_sections}."
            )
        for key in _REPORT_SECTION_KEYS:
            value = sections.get(key)
            if value is None:
                continue
            if not isinstance(value, bool):
                raise MilestoneArtifactValidationError(
                    f"Milestone generation option 'sections.{key}' must be a boolean."
                )
            resolved["sections"][key] = value

    categories = options.get("decision_categories")
    if categories is not None:
        resolved["decision_categories"] = _normalize_string_sequence(
            categories,
            owner="decision_categories",
        )

    summary = options.get("summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            raise MilestoneArtifactValidationError("Milestone generation option 'summary' must be a mapping.")
        unknown_summary_keys = sorted(set(summary) - {"include_stage_counts", "include_review_outcome"})
        if unknown_summary_keys:
            raise MilestoneArtifactValidationError(
                f"Milestone generation option 'summary' contains unsupported keys: {unknown_summary_keys}."
            )
        for key in ("include_stage_counts", "include_review_outcome"):
            value = summary.get(key)
            if value is None:
                continue
            if not isinstance(value, bool):
                raise MilestoneArtifactValidationError(
                    f"Milestone generation option 'summary.{key}' must be a boolean."
                )
            resolved["summary"][key] = value

    return resolved


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
    options: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    resolved_options = resolve_milestone_generation_options(options)
    filtered_decisions = _filter_decisions(
        decisions,
        allowed_categories=resolved_options["decision_categories"],
    )
    validate_milestone_report(report=report, decisions=filtered_decisions)
    artifact_dir = resolve_milestone_artifact_dir(output_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = build_milestone_report_summary_payload(report=report, decisions=filtered_decisions, options=resolved_options)
    decision_log_payload = build_milestone_decision_log_payload(
        report=report,
        decisions=filtered_decisions,
        options=resolved_options,
    )
    manifest_payload = build_milestone_report_manifest_payload(
        report=report,
        summary_payload=summary_payload,
        decision_log_payload=decision_log_payload,
        options=resolved_options,
    )

    summary_path = artifact_dir / MILESTONE_SUMMARY_FILENAME
    decision_log_path = artifact_dir / MILESTONE_DECISION_LOG_FILENAME
    manifest_path = artifact_dir / MILESTONE_MANIFEST_FILENAME
    markdown_report_path = artifact_dir / MILESTONE_MARKDOWN_REPORT_FILENAME
    _write_json(summary_path, summary_payload)
    _write_json(decision_log_path, decision_log_payload)
    _write_json(manifest_path, manifest_payload)
    if resolved_options["include_markdown_report"]:
        markdown_report_path.write_text(
            build_milestone_markdown_report(report=report, decisions=filtered_decisions, options=resolved_options),
            encoding="utf-8",
            newline="\n",
        )
    elif markdown_report_path.exists():
        markdown_report_path.unlink()
    return summary_path, decision_log_path, manifest_path


def build_milestone_report_summary_payload(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_options = resolve_milestone_generation_options(options)
    filtered_decisions = _filter_decisions(
        decisions,
        allowed_categories=resolved_options["decision_categories"],
    )
    validate_milestone_report(report=report, decisions=filtered_decisions)
    decision_rows = [decision.to_dict() for decision in filtered_decisions]
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
        "report_markdown_path": (
            MILESTONE_MARKDOWN_REPORT_FILENAME if resolved_options["include_markdown_report"] else None
        ),
        "decision_ids": [row["decision_id"] for row in decision_rows],
        "decision_counts_by_status": _decision_counts_by_status(filtered_decisions),
        "decision_count": len(decision_rows),
        "metadata": canonicalize_value(dict(report.metadata)),
    }
    return canonicalize_value(payload)


def build_milestone_decision_log_payload(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_options = resolve_milestone_generation_options(options)
    filtered_decisions = _filter_decisions(
        decisions,
        allowed_categories=resolved_options["decision_categories"],
    )
    validate_milestone_report(report=report, decisions=filtered_decisions)
    decision_rows = [decision.to_dict() for decision in filtered_decisions]
    payload = {
        "run_type": MILESTONE_DECISION_LOG_RUN_TYPE,
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "milestone_id": report.milestone_id,
        "milestone_name": report.milestone_name,
        "title": report.title,
        "decision_count": len(decision_rows),
        "decision_ids": [row["decision_id"] for row in decision_rows],
        "decision_counts_by_status": _decision_counts_by_status(filtered_decisions),
        "decisions": decision_rows,
        "rendered": _render_milestone_decision_log(
            report=report,
            decision_rows=decision_rows,
            render_formats=resolved_options["decision_log_render_formats"],
        ),
    }
    return canonicalize_value(payload)


def build_milestone_report_manifest_payload(
    *,
    report: MilestoneReport,
    summary_payload: Mapping[str, Any],
    decision_log_payload: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_options = resolve_milestone_generation_options(options)
    validate_milestone_report_payload(summary_payload)
    validate_milestone_decision_log_payload(decision_log_payload)
    artifact_files = [
        MILESTONE_DECISION_LOG_FILENAME,
        MILESTONE_MANIFEST_FILENAME,
        MILESTONE_SUMMARY_FILENAME,
    ]
    if resolved_options["include_markdown_report"]:
        artifact_files.append(MILESTONE_MARKDOWN_REPORT_FILENAME)
    artifact_files = sorted(artifact_files)
    artifact_groups = {
        "core": artifact_files,
        "decision_log": [MILESTONE_DECISION_LOG_FILENAME],
        "report": [MILESTONE_SUMMARY_FILENAME],
        "summary": [MILESTONE_SUMMARY_FILENAME],
    }
    artifacts = {
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
    }
    if resolved_options["include_markdown_report"]:
        artifact_groups["markdown"] = [MILESTONE_MARKDOWN_REPORT_FILENAME]
        artifact_groups["report"] = [MILESTONE_SUMMARY_FILENAME, MILESTONE_MARKDOWN_REPORT_FILENAME]
        artifacts[MILESTONE_MARKDOWN_REPORT_FILENAME] = {
            "path": MILESTONE_MARKDOWN_REPORT_FILENAME,
            "section_count": _markdown_section_count(report=report, options=resolved_options),
        }
    payload = {
        "run_type": MILESTONE_REPORT_RUN_TYPE,
        "schema_version": MILESTONE_REPORT_SCHEMA_VERSION,
        "milestone_id": report.milestone_id,
        "milestone_name": report.milestone_name,
        "status": summary_payload["status"],
        "artifact_files": artifact_files,
        "artifact_groups": artifact_groups,
        "artifacts": artifacts,
        "decision_counts_by_status": canonicalize_value(dict(summary_payload["decision_counts_by_status"])),
        "decision_ids": list(summary_payload["decision_ids"]),
        "decision_log_path": MILESTONE_DECISION_LOG_FILENAME,
        "report_markdown_path": (
            MILESTONE_MARKDOWN_REPORT_FILENAME if resolved_options["include_markdown_report"] else None
        ),
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
    report_markdown_path = payload.get("report_markdown_path")
    if report_markdown_path not in {None, MILESTONE_MARKDOWN_REPORT_FILENAME}:
        raise MilestoneArtifactValidationError("Milestone report summary must reference report.md.")


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
    render_keys = sorted(rendered)
    if not render_keys:
        raise MilestoneArtifactValidationError("Milestone decision log rendered content must not be empty.")
    unsupported = sorted(set(rendered) - _VALID_DECISION_LOG_RENDER_FORMATS)
    if unsupported:
        raise MilestoneArtifactValidationError(
            "Milestone decision log rendered content contains unsupported formats: "
            f"{unsupported}."
        )
    for render_key in render_keys:
        if not str(rendered.get(render_key) or "").strip():
            raise MilestoneArtifactValidationError(
                f"Milestone decision log rendered.{render_key} must be populated."
            )


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
    render_formats: Sequence[str],
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

    rendered: dict[str, Any] = {}
    if "markdown" in render_formats:
        rendered["markdown"] = "\n".join(lines).strip()
    if "text" in render_formats:
        rendered["text"] = "\n".join(text_lines).strip()
    return rendered


def build_milestone_markdown_report(
    *,
    report: MilestoneReport,
    decisions: Sequence[MilestoneDecisionEntry],
    options: Mapping[str, Any] | None = None,
) -> str:
    resolved_options = resolve_milestone_generation_options(options)
    filtered_decisions = _filter_decisions(
        decisions,
        allowed_categories=resolved_options["decision_categories"],
    )
    validate_milestone_report(report=report, decisions=filtered_decisions)
    decision_rows = [decision.to_dict() for decision in filtered_decisions]
    metadata = _mapping(report.metadata)
    selected_run_ids = _mapping(metadata.get("selected_run_ids"))
    key_metrics = _mapping(metadata.get("key_metrics"))
    final_outcomes = _mapping(metadata.get("final_outcomes"))
    promotion_gates = _mapping(metadata.get("promotion_gates"))
    stage_state_counts = _mapping(metadata.get("stage_state_counts"))

    lines = [
        f"# {report.title}",
        "",
        "## Milestone Summary",
        f"- Milestone: `{report.milestone_name}`",
        f"- Milestone ID: `{report.milestone_id}`",
        f"- Status: `{report.status}`",
    ]
    if report.owner:
        lines.append(f"- Owner: `{report.owner}`")
    lines.append(f"- Reporting Window: {_format_reporting_window(report.reporting_window)}")
    lines.append(f"- Summary: {report.summary}")
    sections = resolved_options["sections"]
    if sections["campaign_scope"]:
        lines.extend(_render_string_list_section("Campaign Scope", report.scope, empty_text="No explicit campaign scope was recorded."))
    if sections["selections"]:
        lines.extend(_render_mapping_section("Selections", _selection_rows(selected_run_ids), empty_text="No run selections were recorded."))
    if sections["key_findings"]:
        lines.extend(_render_string_list_section("Key Findings", report.key_findings, empty_text="No key findings were recorded."))
    if sections["key_metrics"]:
        lines.extend(_render_mapping_section("Key Metrics", _metric_rows(key_metrics), empty_text="No key metrics were recorded."))
    if sections["gate_outcomes"]:
        lines.extend(
            _render_mapping_section(
                "Gate Outcomes",
                _gate_outcome_rows(
                    decision_rows=decision_rows,
                    promotion_gates=promotion_gates,
                    stage_state_counts=stage_state_counts,
                ),
                empty_text="No gate outcomes were recorded.",
            )
        )
    if sections["risks"]:
        lines.extend(_render_string_list_section("Risks", _risk_rows(final_outcomes, promotion_gates, decision_rows), empty_text="No immediate risks were detected."))
    if sections["next_steps"]:
        lines.extend(
            _render_string_list_section(
                "Next Steps",
                _next_step_rows(report=report, decision_rows=decision_rows),
                empty_text="No additional next steps were recorded.",
            )
        )
    if sections["open_questions"]:
        lines.extend(_render_string_list_section("Open Questions", report.open_questions, empty_text="No open questions were recorded."))
    if sections["decision_snapshot"]:
        lines.extend(_render_decision_snapshot_section(decision_rows))
    if sections["related_artifacts"]:
        lines.extend(
            _render_mapping_section(
                "Related Artifacts",
                sorted(report.related_artifacts.items()),
                empty_text="No related artifacts were recorded.",
            )
        )
    return "\n".join(lines).strip() + "\n"


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


def _render_string_list_section(title: str, values: Sequence[str], *, empty_text: str) -> list[str]:
    lines = ["", f"## {title}"]
    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        lines.append(f"- {empty_text}")
        return lines
    lines.extend(f"- {item}" for item in items)
    return lines


def _render_mapping_section(
    title: str,
    rows: Sequence[tuple[str, str]],
    *,
    empty_text: str,
) -> list[str]:
    lines = ["", f"## {title}"]
    normalized_rows = [(str(label).strip(), str(value).strip()) for label, value in rows if str(label).strip() and str(value).strip()]
    if not normalized_rows:
        lines.append(f"- {empty_text}")
        return lines
    lines.extend(f"- {label}: {value}" for label, value in normalized_rows)
    return lines


def _render_decision_snapshot_section(decision_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["", "## Decision Snapshot"]
    if not decision_rows:
        lines.append("- No milestone decisions were recorded.")
        return lines

    for index, row in enumerate(decision_rows, start=1):
        lines.append(f"### {index}. {row['title']}")
        lines.append(f"- Decision ID: `{row['decision_id']}`")
        lines.append(f"- Status: `{row['status']}`")
        lines.append(f"- Summary: {row['summary']}")
        follow_up_actions = row.get("follow_up_actions")
        if isinstance(follow_up_actions, list) and follow_up_actions:
            lines.append("- Follow-Up Actions:")
            lines.extend(f"  - {action}" for action in follow_up_actions if str(action).strip())
    return lines


def _selection_rows(selected_run_ids: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in (
        ("alpha_run_ids", "Alpha Runs"),
        ("strategy_run_ids", "Strategy Runs"),
        ("candidate_selection_run_id", "Candidate Selection Run"),
        ("portfolio_run_id", "Portfolio Run"),
        ("review_id", "Review"),
    ):
        value = selected_run_ids.get(key)
        rendered = _render_selection_value(value)
        if rendered is not None:
            rows.append((label, rendered))
    return rows


def _metric_rows(key_metrics: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    alpha_runs = key_metrics.get("alpha_runs")
    if isinstance(alpha_runs, list) and alpha_runs:
        rows.append(("Alpha Runs", "; ".join(_render_metric_mapping(_mapping(row)) for row in alpha_runs if _mapping(row))))

    strategy_runs = key_metrics.get("strategy_runs")
    if isinstance(strategy_runs, list) and strategy_runs:
        rows.append(("Strategy Runs", "; ".join(_render_metric_mapping(_mapping(row)) for row in strategy_runs if _mapping(row))))

    for key, label in (
        ("candidate_selection", "Candidate Selection"),
        ("portfolio", "Portfolio"),
        ("review", "Review"),
    ):
        value = _mapping(key_metrics.get(key))
        if value:
            rows.append((label, _render_metric_mapping(value)))
    return rows


def _gate_outcome_rows(
    *,
    decision_rows: Sequence[Mapping[str, Any]],
    promotion_gates: Mapping[str, Any],
    stage_state_counts: Mapping[str, Any],
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if stage_state_counts:
        rows.append(
            (
                "Stage State Counts",
                ", ".join(f"{key}={_format_metric_value(stage_state_counts[key])}" for key in sorted(stage_state_counts)),
            )
        )
    if promotion_gates:
        rows.append(("Promotion Gates", _render_metric_mapping(promotion_gates)))
    for row in decision_rows:
        rows.append((str(row["title"]), f"status={row['status']}; summary={row['summary']}"))
    return rows


def _risk_rows(
    final_outcomes: Mapping[str, Any],
    promotion_gates: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    risks: list[str] = []
    risks.extend(_stage_name_risks(final_outcomes.get("failed_stage_names"), "Failed stages require follow-up"))
    risks.extend(_stage_name_risks(final_outcomes.get("partial_stage_names"), "Partial stages require review before reuse"))
    risks.extend(_stage_name_risks(final_outcomes.get("resumable_stage_names"), "Resumable stages depend on preserved checkpoint continuity"))

    promotion_status = str(promotion_gates.get("promotion_status") or "").strip()
    evaluation_status = str(promotion_gates.get("evaluation_status") or "").strip()
    if promotion_status in {"blocked", "rejected", "deferred", "pending", "review_ready"}:
        risks.append(
            "Promotion status requires follow-up: "
            f"promotion_status={promotion_status or 'NA'}, evaluation_status={evaluation_status or 'NA'}."
        )
    elif evaluation_status in {"blocked", "failed", "warn", "pending"}:
        risks.append(f"Promotion evaluation requires follow-up: evaluation_status={evaluation_status}.")

    for row in decision_rows:
        if row.get("status") in {"deferred", "rejected", "superseded"}:
            risks.append(f"Decision `{row['decision_id']}` is `{row['status']}` and still needs resolution.")
    return _dedupe_preserve_order(risks)


def _next_step_rows(
    *,
    report: MilestoneReport,
    decision_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    steps = list(report.recommendations)
    for row in decision_rows:
        follow_up_actions = row.get("follow_up_actions")
        if isinstance(follow_up_actions, list):
            steps.extend(str(action) for action in follow_up_actions if str(action).strip())
    return _dedupe_preserve_order(steps)


def _stage_name_risks(stage_names: Any, prefix: str) -> list[str]:
    if not isinstance(stage_names, list) or not stage_names:
        return []
    return [f"{prefix}: {', '.join(sorted(str(stage_name) for stage_name in stage_names))}."]


def _render_selection_value(value: Any) -> str | None:
    if isinstance(value, list):
        rendered = ", ".join(str(item) for item in value if str(item).strip())
        return rendered or None
    text = str(value).strip() if value is not None else ""
    return text or None


def _render_metric_mapping(metrics: Mapping[str, Any]) -> str:
    return "; ".join(f"{key}={_format_metric_value(metrics[key])}" for key in sorted(metrics))


def _format_reporting_window(reporting_window: Mapping[str, Any]) -> str:
    start = reporting_window.get("start")
    end = reporting_window.get("end")
    if start is None and end is None:
        return "Not recorded"
    if start is None:
        return f"through {end}"
    if end is None:
        return f"from {start}"
    return f"{start} to {end}"


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "NA"
    return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _markdown_section_count(*, report: MilestoneReport, options: Mapping[str, Any] | None = None) -> int:
    _ = report
    resolved_options = resolve_milestone_generation_options(options)
    return 1 + sum(1 for value in resolved_options["sections"].values() if value)


def _normalize_string_sequence(value: Any, *, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, Sequence):
        raw_values = list(value)
    else:
        raise MilestoneArtifactValidationError(f"Milestone generation option '{owner}' must be a string or sequence of strings.")

    normalized: list[str] = []
    for index, item in enumerate(raw_values):
        if not isinstance(item, str):
            raise MilestoneArtifactValidationError(
                f"Milestone generation option '{owner}[{index}]' must be a string."
            )
        text = item.strip()
        if not text:
            raise MilestoneArtifactValidationError(
                f"Milestone generation option '{owner}[{index}]' must not be empty."
            )
        if text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _filter_decisions(
    decisions: Sequence[MilestoneDecisionEntry],
    *,
    allowed_categories: Sequence[str],
) -> list[MilestoneDecisionEntry]:
    if not allowed_categories:
        return list(decisions)
    allowed = {str(value) for value in allowed_categories}
    return [decision for decision in decisions if decision.category in allowed]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(canonicalize_value(dict(payload)), indent=2, sort_keys=True), encoding="utf-8", newline="\n")


__all__ = [
    "MILESTONE_DECISION_LOG_FILENAME",
    "MILESTONE_MARKDOWN_REPORT_FILENAME",
    "MILESTONE_MANIFEST_FILENAME",
    "MILESTONE_REPORT_RUN_TYPE",
    "MILESTONE_REPORT_SCHEMA_VERSION",
    "MILESTONE_SUMMARY_FILENAME",
    "MilestoneArtifactValidationError",
    "MilestoneDecisionEntry",
    "MilestoneReport",
    "MilestoneSourceArtifact",
    "build_milestone_markdown_report",
    "build_milestone_decision_log_payload",
    "build_milestone_report_id",
    "build_milestone_report_manifest_payload",
    "build_milestone_report_summary_payload",
    "resolve_milestone_generation_options",
    "resolve_milestone_artifact_dir",
    "validate_milestone_decision_log_payload",
    "validate_milestone_report",
    "validate_milestone_report_payload",
    "write_milestone_report_artifacts",
]
