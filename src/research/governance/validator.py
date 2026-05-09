from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from src.artifacts.safety import portable_path
from src.research.registry import canonicalize_value

from .models import GovernanceSourceRecord
from .normalization import (
    CANONICAL_PROMOTION_STATUSES,
    CANONICAL_REVIEW_STATUSES,
    PROMOTION_TO_REVIEW_STATUS,
    normalize_promotion_status,
    normalize_review_status,
    status_was_normalized,
)
PATH_FIELDS = ("registry_path", "manifest_path")
SEVERITY_RANK = {"warn": 0, "review": 1, "reject": 2, "block": 3}


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
    findings.extend(_campaign_rollup_findings(records))
    findings.extend(_candidate_review_findings(records))

    counts_by_severity: dict[str, int] = {}
    counts_by_check: dict[str, int] = {}
    for finding in findings:
        severity = str(finding["severity"])
        check_id = str(finding["check_id"])
        counts_by_severity[severity] = counts_by_severity.get(severity, 0) + 1
        counts_by_check[check_id] = counts_by_check.get(check_id, 0) + 1
    return canonicalize_value(
        {
            "status": "fail" if any(finding["severity"] != "info" for finding in findings) else "pass",
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
    promotion_status = normalize_promotion_status(row.get("promotion_status")) or ""
    raw_promotion_status = _coerce_string(row.get("promotion_status"))

    if summary is None:
        findings.append(_finding("missing_promotion_summary", record, severity="warning"))
    if raw_promotion_status is not None and raw_promotion_status != promotion_status:
        findings.append(
            _finding(
                "legacy_status_normalized",
                record,
                severity="info",
                details={
                    "field": "promotion_status",
                    "raw_status": raw_promotion_status,
                    "normalized_status": promotion_status,
                },
            )
        )
    if promotion_status and promotion_status not in CANONICAL_PROMOTION_STATUSES:
        findings.append(
            _finding(
                "unknown_promotion_status",
                record,
                severity="error",
                details={
                    "promotion_status": promotion_status,
                    "raw_promotion_status": raw_promotion_status,
                    "canonical_statuses": sorted(CANONICAL_PROMOTION_STATUSES),
                },
            )
        )

    registry_raw_status = _coerce_string(entry.get("promotion_status"))
    summary_raw_status = _coerce_string(summary.get("promotion_status") if isinstance(summary, Mapping) else None)
    registry_status = normalize_promotion_status(registry_raw_status)
    summary_status = normalize_promotion_status(summary_raw_status)
    findings.extend(
        _normalization_findings(
            record,
            field="registry.promotion_status",
            raw_status=registry_raw_status,
            normalized_status=registry_status,
        )
    )
    findings.extend(
        _normalization_findings(
            record,
            field="promotion_gate_summary.promotion_status",
            raw_status=summary_raw_status,
            normalized_status=summary_status,
        )
    )
    if registry_status is not None and summary_status is not None and registry_status != summary_status:
        findings.append(
            _finding(
                "registry_promotion_status_mismatch",
                record,
                severity="error",
                details={
                    "registry_promotion_status": registry_status,
                    "registry_raw_promotion_status": registry_raw_status,
                    "summary_promotion_status": summary_status,
                    "summary_raw_promotion_status": summary_raw_status,
                },
            )
        )
    findings.extend(_equivalent_promotion_summary_findings(record))

    review_raw_status = _coerce_string(entry.get("review_status"))
    if review_raw_status is None:
        review_metadata = entry.get("review_metadata")
        if isinstance(review_metadata, Mapping):
            review_raw_status = _coerce_string(review_metadata.get("status"))
    if review_raw_status is None:
        review_raw_status = _coerce_string(row.get("review_status"))
    review_status = normalize_review_status(review_raw_status)
    findings.extend(
        _normalization_findings(
            record,
            field="review_status",
            raw_status=review_raw_status,
            normalized_status=review_status,
        )
    )
    if review_status is not None and review_status not in CANONICAL_REVIEW_STATUSES:
        findings.append(
            _finding(
                "unknown_review_status",
                record,
                severity="error",
                details={
                    "review_status": review_status,
                    "raw_review_status": review_raw_status,
                    "canonical_statuses": sorted(CANONICAL_REVIEW_STATUSES),
                },
            )
        )
    expected_review = PROMOTION_TO_REVIEW_STATUS.get(promotion_status)
    if review_status is not None and expected_review is not None and review_status != expected_review:
        findings.append(
            _finding(
                "review_status_mismatch",
                record,
                severity="error",
                details={
                    "promotion_status": promotion_status,
                    "review_status": review_status,
                    "raw_review_status": review_raw_status,
                    "expected_review_status": expected_review,
                },
            )
        )

    if record.manifest_path is not None and not record.manifest_path.exists():
        findings.append(
            _finding(
                "missing_or_stale_manifest_link",
                record,
                severity="warning",
                details={"manifest_path": _safe_path(record.manifest_path)},
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
        if record.workflow_type == "candidate_review":
            candidate_selection_run_id = _metadata_string(record, "candidate_selection_run_id")
            manifest_candidate_selection_run_id = _coerce_string(record.manifest.get("candidate_selection_run_id"))
            if (
                candidate_selection_run_id is not None
                and manifest_candidate_selection_run_id is not None
                and manifest_candidate_selection_run_id != candidate_selection_run_id
            ):
                findings.append(
                    _finding(
                        "candidate_review_manifest_run_id_mismatch",
                        record,
                        severity="warning",
                        details={
                            "candidate_selection_run_id": candidate_selection_run_id,
                            "manifest_candidate_selection_run_id": manifest_candidate_selection_run_id,
                        },
                    )
                )

    if record.candidate_review_summary is not None:
        context = record.candidate_review_summary.get("promotion_context")
        if isinstance(context, Mapping):
            portfolio_summary = context.get("portfolio_promotion_gate_summary")
            context_status = (
                normalize_promotion_status(portfolio_summary.get("promotion_status"))
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


def _candidate_review_findings(records: list[GovernanceSourceRecord]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    known_run_ids = {record.run_id for record in records}
    known_run_ids.update(
        str(record.registry_entry.get("run_id"))
        for record in records
        if record.registry_entry.get("run_id") is not None
    )
    for record in records:
        if record.workflow_type != "candidate_review":
            continue
        metadata = record.governance_metadata
        if not bool(metadata.get("promotion_context_present")):
            findings.append(_finding("candidate_review_missing_promotion_context", record, severity="warning"))
        context = record.candidate_review_summary.get("promotion_context") if record.candidate_review_summary else None
        portfolio_summary = context.get("portfolio_promotion_gate_summary") if isinstance(context, Mapping) else None
        context_status = (
            normalize_promotion_status(portfolio_summary.get("promotion_status"))
            if isinstance(portfolio_summary, Mapping)
            else None
        )
        if context_status is not None and context_status not in CANONICAL_PROMOTION_STATUSES:
            findings.append(
                _finding(
                    "candidate_review_context_unknown_promotion_status",
                    record,
                    severity="error",
                    details={"context_promotion_status": context_status},
                )
            )
        raw_upstream_run_ids = _metadata_list_preserve_duplicates(metadata.get("upstream_run_ids_raw"))
        duplicate_upstream_ids = sorted(
            {run_id for run_id in raw_upstream_run_ids if raw_upstream_run_ids.count(run_id) > 1}
        )
        if duplicate_upstream_ids:
            findings.append(
                _finding(
                    "candidate_review_duplicate_upstream_run_ids",
                    record,
                    severity="warning",
                    details={"upstream_run_ids": duplicate_upstream_ids},
                )
            )
        selected_run_ids = _metadata_list(metadata.get("selected_run_ids"))
        missing_upstream_ids = [run_id for run_id in selected_run_ids if run_id not in known_run_ids]
        if missing_upstream_ids:
            findings.append(
                _finding(
                    "candidate_review_missing_upstream_run_reference",
                    record,
                    severity="warning",
                    details={"missing_upstream_run_ids": missing_upstream_ids},
                )
            )
        if selected_run_ids and not _metadata_string(record, "selected_candidate_id"):
            findings.append(
                _finding(
                    "candidate_review_missing_selected_candidate_id",
                    record,
                    severity="warning",
                    details={"selected_run_ids": selected_run_ids},
                )
            )
        summary = record.candidate_review_summary or {}
        manifest = record.manifest or {}
        summary_candidate_ids = _candidate_identifier_values(summary)
        manifest_candidate_ids = _candidate_identifier_values(manifest)
        if summary_candidate_ids and manifest_candidate_ids and summary_candidate_ids != manifest_candidate_ids:
            findings.append(
                _finding(
                    "candidate_review_selected_candidate_id_mismatch",
                    record,
                    severity="warning",
                    details={
                        "manifest_selected_candidate_ids": manifest_candidate_ids,
                        "summary_selected_candidate_ids": summary_candidate_ids,
                    },
                )
            )
        summary_run_ids = _selected_run_identifier_values(summary)
        manifest_run_ids = _selected_run_identifier_values(manifest)
        if summary_run_ids and manifest_run_ids and summary_run_ids != manifest_run_ids:
            findings.append(
                _finding(
                    "candidate_review_selected_run_id_mismatch",
                    record,
                    severity="warning",
                    details={
                        "manifest_selected_run_ids": manifest_run_ids,
                        "summary_selected_run_ids": summary_run_ids,
                    },
                )
            )
        # Candidate-review stale-path validation is intentionally limited to
        # required evidence. Future best-effort paths belong under
        # optional_artifact_evidence_paths and do not warn when absent.
        for field, path_value in sorted(_mapping(metadata.get("artifact_evidence_paths")).items()):
            path = Path(str(path_value))
            if path.exists():
                continue
            findings.append(
                _finding(
                    "candidate_review_stale_artifact_evidence_path",
                    record,
                    severity="warning",
                    details={"field": field, "path": _sanitize_detail_path(str(path_value))},
                )
            )
    return findings


def _campaign_rollup_findings(records: list[GovernanceSourceRecord]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    campaign_records = [record for record in records if record.workflow_type == "campaign"]
    scenario_records = [record for record in records if record.workflow_type == "campaign_scenario"]
    scenarios_by_campaign: dict[str, list[GovernanceSourceRecord]] = {}
    for scenario in scenario_records:
        campaign_id = _metadata_string(scenario, "campaign_id")
        if campaign_id is None:
            continue
        scenarios_by_campaign.setdefault(campaign_id, []).append(scenario)
        findings.extend(_scenario_artifact_findings(scenario))

    for campaign in campaign_records:
        campaign_id = _metadata_string(campaign, "campaign_id") or campaign.run_id
        if isinstance(campaign.manifest, Mapping):
            manifest_campaign_id = _coerce_string(
                campaign.manifest.get("orchestration_run_id") or campaign.manifest.get("campaign_run_id")
            )
            if manifest_campaign_id is not None and manifest_campaign_id != campaign_id:
                findings.append(
                    _finding(
                        "campaign_id_mismatch",
                        campaign,
                        severity="warning",
                        details={"campaign_id": campaign_id, "manifest_campaign_id": manifest_campaign_id},
                    )
                )
        metadata = campaign.governance_metadata
        for scenario_id in _metadata_list(metadata.get("scenario_catalog_missing_ids")):
            findings.append(
                _finding(
                    "scenario_catalog_missing_scenario_dir",
                    campaign,
                    severity="warning",
                    details={"campaign_id": campaign_id, "scenario_id": scenario_id},
                )
            )
        child_scenarios = sorted(
            scenarios_by_campaign.get(campaign_id, []),
            key=lambda item: (_metadata_string(item, "scenario_id") or "", item.run_id),
        )
        if not child_scenarios:
            continue
        campaign_summary = campaign.promotion_gate_summary
        child_summaries = [
            scenario.promotion_gate_summary
            for scenario in child_scenarios
            if isinstance(scenario.promotion_gate_summary, Mapping)
        ]
        if not child_summaries or not isinstance(campaign_summary, Mapping):
            continue
        expected_highest = _highest_severity(
            _coerce_string(summary.get("highest_severity"))
            for summary in child_summaries
        )
        observed_highest = _coerce_string(campaign_summary.get("highest_severity"))
        if expected_highest is not None and observed_highest is not None and observed_highest != expected_highest:
            findings.append(
                _finding(
                    "campaign_highest_severity_mismatch",
                    campaign,
                    severity="error",
                    details={
                        "campaign_id": campaign_id,
                        "campaign_highest_severity": observed_highest,
                        "scenario_highest_severity": expected_highest,
                    },
                )
            )
        campaign_codes = set(_string_list(campaign_summary.get("decision_reason_codes")))
        scenario_codes = sorted(
            {
                code
                for summary in child_summaries
                for code in _string_list(summary.get("decision_reason_codes"))
            }
        )
        missing_codes = [code for code in scenario_codes if code not in campaign_codes]
        if missing_codes:
            findings.append(
                _finding(
                    "campaign_missing_scenario_reason_codes",
                    campaign,
                    severity="warning",
                    details={"campaign_id": campaign_id, "missing_reason_codes": missing_codes},
                )
            )
    return findings


def _scenario_artifact_findings(record: GovernanceSourceRecord) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    metadata = record.governance_metadata
    scenario_id = _metadata_string(record, "scenario_id")
    if isinstance(record.manifest, Mapping):
        manifest_scenario_id = _coerce_string(record.manifest.get("scenario_id"))
        if manifest_scenario_id is not None and scenario_id is not None and manifest_scenario_id != scenario_id:
            findings.append(
                _finding(
                    "scenario_id_mismatch",
                    record,
                    severity="warning",
                    details={"manifest_scenario_id": manifest_scenario_id, "scenario_id": scenario_id},
                )
            )
    stage_states = metadata.get("checkpoint_stage_states")
    completed_like = False
    if isinstance(stage_states, Mapping):
        completed_like = any(str(value) in {"completed", "reused"} for value in stage_states.values())
    if not completed_like:
        completed_like = str(metadata.get("scenario_status") or "") in {"completed", "reused"}
    if completed_like and not bool(metadata.get("scenario_summary_present", True)):
        findings.append(
            _finding(
                "checkpoint_completed_scenario_missing_summary",
                record,
                severity="warning",
                details={"scenario_id": scenario_id},
            )
        )
    if completed_like and not bool(metadata.get("scenario_manifest_present", True)):
        findings.append(
            _finding(
                "checkpoint_completed_scenario_missing_manifest",
                record,
                severity="warning",
                details={"scenario_id": scenario_id},
            )
        )

    manifest_summary = None
    if isinstance(record.manifest, Mapping) and isinstance(record.manifest.get("promotion_gate_summary"), Mapping):
        manifest_summary = record.manifest["promotion_gate_summary"]
    if isinstance(manifest_summary, Mapping) and isinstance(record.promotion_gate_summary, Mapping):
        row_status = normalize_promotion_status(record.promotion_gate_summary.get("promotion_status"))
        manifest_status = normalize_promotion_status(manifest_summary.get("promotion_status"))
        if row_status is not None and manifest_status is not None and row_status != manifest_status:
            findings.append(
                _finding(
                    "scenario_promotion_status_mismatch",
                    record,
                    severity="error",
                    details={
                        "scenario_id": scenario_id,
                        "scenario_promotion_status": row_status,
                        "manifest_promotion_status": manifest_status,
                    },
                )
            )
    missing_child_paths = _metadata_list(metadata.get("missing_child_artifact_paths"))
    if missing_child_paths:
        findings.append(
            _finding(
                "scenario_summary_missing_child_artifacts",
                record,
                severity="warning",
                details={
                    "scenario_id": scenario_id,
                    "missing_child_artifact_count": len(missing_child_paths),
                    "missing_child_artifacts": [_sanitize_detail_path(path) for path in missing_child_paths],
                },
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
        if path.is_absolute() or PureWindowsPath(value).is_absolute():
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
        "candidate_review_context_unknown_promotion_status": "Candidate-review promotion context has an unknown promotion status.",
        "candidate_review_duplicate_upstream_run_ids": "Candidate-review upstream run ids contain duplicates.",
        "candidate_review_manifest_run_id_mismatch": "Candidate-review manifest candidate selection run id does not match the review summary.",
        "candidate_review_missing_promotion_context": "Candidate-review summary is missing promotion context.",
        "candidate_review_missing_selected_candidate_id": "Candidate-review metadata includes selected runs but no selected candidate id.",
        "candidate_review_missing_upstream_run_reference": "Candidate-review selected run id is not present in governance records.",
        "candidate_review_selected_candidate_id_mismatch": "Candidate-review summary and manifest disagree on selected candidate ids.",
        "candidate_review_selected_run_id_mismatch": "Candidate-review summary and manifest disagree on selected run ids.",
        "candidate_review_stale_artifact_evidence_path": "Candidate-review artifact evidence path is missing or stale.",
        "campaign_id_mismatch": "Campaign manifest identity differs from governance campaign id.",
        "campaign_highest_severity_mismatch": "Campaign rollup highest severity does not match child scenario severity.",
        "campaign_missing_scenario_reason_codes": "Campaign rollup omits decision reason codes observed in child scenarios.",
        "checkpoint_completed_scenario_missing_manifest": "Scenario is completed or reused but its manifest is missing.",
        "checkpoint_completed_scenario_missing_summary": "Scenario is completed or reused but its summary is missing.",
        "legacy_status_normalized": "Legacy status alias was normalized for governance validation.",
        "manifest_promotion_gates_summary_mismatch": "Manifest promotion_gate_summary differs from promotion_gates.json summary.",
        "manifest_registry_promotion_summary_mismatch": "Manifest promotion_gate_summary differs from registry promotion_gate_summary.",
        "manifest_run_id_mismatch": "Manifest run id does not match the source record run id.",
        "missing_or_stale_manifest_link": "Manifest path is missing or stale.",
        "missing_promotion_summary": "Promotion summary is missing; governance row cannot cite canonical promotion status.",
        "registry_promotion_status_mismatch": "Registry promotion status differs from promotion_gate_summary.promotion_status.",
        "review_status_mismatch": "Review status differs from the canonical M31 promotion-to-review mapping.",
        "scenario_catalog_missing_scenario_dir": "Scenario catalog references a missing scenario directory.",
        "scenario_id_mismatch": "Scenario manifest identity differs from governance scenario id.",
        "scenario_promotion_status_mismatch": "Scenario summary promotion status differs from scenario manifest promotion_gate_summary.",
        "scenario_summary_missing_child_artifacts": "Scenario summary references child artifact paths that are missing.",
        "unknown_promotion_status": "Promotion status is not one of the canonical M31 promotion statuses.",
        "unknown_review_status": "Review status is not one of the canonical M31 review statuses.",
    }
    return messages.get(check_id, check_id)


def _equivalent_promotion_summary_findings(record: GovernanceSourceRecord) -> list[dict[str, Any]]:
    # Equivalent-summary validation is intentionally status-focused. Older or
    # partial summaries may omit reason codes, severity counts, or gate result
    # metadata while still preserving the canonical promotion decision status.
    manifest_status = _promotion_summary_status(record.manifest)
    registry_status = _promotion_summary_status(record.registry_entry)
    gates_status = _promotion_summary_status(record.promotion_gates)
    findings: list[dict[str, Any]] = []
    if manifest_status[0] is not None and registry_status[0] is not None and manifest_status[0] != registry_status[0]:
        findings.append(
            _finding(
                "manifest_registry_promotion_summary_mismatch",
                record,
                severity="error",
                details={
                    "manifest_promotion_status": manifest_status[0],
                    "manifest_raw_promotion_status": manifest_status[1],
                    "registry_summary_promotion_status": registry_status[0],
                    "registry_summary_raw_promotion_status": registry_status[1],
                },
            )
        )
    if manifest_status[0] is not None and gates_status[0] is not None and manifest_status[0] != gates_status[0]:
        findings.append(
            _finding(
                "manifest_promotion_gates_summary_mismatch",
                record,
                severity="error",
                details={
                    "manifest_promotion_status": manifest_status[0],
                    "manifest_raw_promotion_status": manifest_status[1],
                    "promotion_gates_promotion_status": gates_status[0],
                    "promotion_gates_raw_promotion_status": gates_status[1],
                },
            )
        )
    return findings


def _promotion_summary_status(payload: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    summary = _promotion_summary_payload(payload)
    raw_status = _coerce_string(summary.get("promotion_status") if isinstance(summary, Mapping) else None)
    return normalize_promotion_status(raw_status), raw_status


def _promotion_summary_payload(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    summary = payload.get("promotion_gate_summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    if "promotion_status" in payload and "gate_count" in payload:
        return dict(payload)
    return None


def _normalization_findings(
    record: GovernanceSourceRecord,
    *,
    field: str,
    raw_status: str | None,
    normalized_status: str | None,
) -> list[dict[str, Any]]:
    if not status_was_normalized(raw_status, normalized_status):
        return []
    return [
        _finding(
            "legacy_status_normalized",
            record,
            severity="info",
            details={
                "field": field,
                "raw_status": raw_status,
                "normalized_status": normalized_status,
            },
        )
    ]


def _coerce_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_string(record: GovernanceSourceRecord, key: str) -> str | None:
    return _coerce_string(record.governance_metadata.get(key))


def _metadata_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _metadata_list_preserve_duplicates(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _candidate_identifier_values(payload: Mapping[str, Any] | None) -> list[str]:
    values: list[str] = []
    for key in ("candidate_id", "candidate_ids", "selected_candidate_id", "selected_candidate_ids"):
        values.extend(_identifier_values(payload, key))
        values.extend(_identifier_values(_nested_mapping(payload, "selected_run_ids"), key))
    return sorted(set(values))


def _selected_run_identifier_values(payload: Mapping[str, Any] | None) -> list[str]:
    values: list[str] = []
    for key in ("selected_run_id", "selected_run_ids"):
        values.extend(_identifier_values(payload, key))
    return sorted(set(values))


def _identifier_values(payload: Any, key: str) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if isinstance(value, Mapping):
        values: set[str] = set()
        for item in value.values():
            if isinstance(item, list):
                values.update(str(part).strip() for part in item if str(part).strip())
            elif item is not None and str(item).strip():
                values.add(str(item).strip())
        return sorted(values)
    if value is not None and str(value).strip():
        return [str(value).strip()]
    return []


def _nested_mapping(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _highest_severity(values: Any) -> str | None:
    severities = [
        value
        for value in values
        if value in SEVERITY_RANK
    ]
    if not severities:
        return None
    return max(severities, key=lambda severity: SEVERITY_RANK[severity])


def _sanitize_detail_path(value: str) -> str:
    path_fragment = value.split(":", 1)
    if len(path_fragment) == 2 and not PureWindowsPath(value).drive:
        return f"{path_fragment[0]}:{portable_path(path_fragment[1], roots=(Path.cwd(),))}"
    return portable_path(value, roots=(Path.cwd(),))


def _safe_path(path: Path) -> str:
    value = str(path)
    if PureWindowsPath(value).is_absolute():
        return portable_path(value)
    return portable_path(path, roots=(Path.cwd(),))


__all__ = [
    "CANONICAL_PROMOTION_STATUSES",
    "CANONICAL_REVIEW_STATUSES",
    "PROMOTION_TO_REVIEW_STATUS",
    "normalize_promotion_status",
    "normalize_review_status",
    "validate_governance_consistency",
]
