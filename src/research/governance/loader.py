from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.research.promotion import DEFAULT_PROMOTION_ARTIFACT_FILENAME
from src.research.registry import canonicalize_value, default_registry_path, load_registry

from .models import GovernanceDataset, GovernanceSourceRecord


def load_governance_artifacts(
    *,
    registry_path: str | Path | None = None,
    artifact_root: str | Path = Path("artifacts"),
) -> GovernanceDataset:
    """Load governance-relevant artifacts without modifying promotion policy outputs."""

    resolved_artifact_root = Path(artifact_root)
    resolved_registry_path = (
        default_registry_path(resolved_artifact_root)
        if registry_path is None
        else Path(registry_path)
    )
    registry_entries = load_registry(resolved_registry_path)
    records = [
        _record_from_registry_entry(
            entry,
            registry_path=resolved_registry_path,
            artifact_root=resolved_artifact_root,
        )
        for entry in registry_entries
    ]
    records.extend(
        _standalone_review_records(
            artifact_root=resolved_artifact_root,
            known_run_ids={record.run_id for record in records},
        )
    )
    records.extend(
        _candidate_review_context_records(
            artifact_root=resolved_artifact_root,
            known_run_ids={record.run_id for record in records},
        )
    )
    records.extend(
        _campaign_records(
            artifact_root=resolved_artifact_root,
            known_keys={(record.workflow_type, record.run_id) for record in records},
        )
    )
    return GovernanceDataset(
        records=sorted(records, key=lambda item: (item.workflow_type, item.run_id)),
        sources={
            "artifact_root": resolved_artifact_root.as_posix(),
            "registry_path": resolved_registry_path.as_posix(),
            "registry_entry_count": len(registry_entries),
            "campaign_record_count": sum(record.workflow_type == "campaign" for record in records),
            "campaign_scenario_record_count": sum(record.workflow_type == "campaign_scenario" for record in records),
        },
    )


def _record_from_registry_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path,
    artifact_root: Path,
) -> GovernanceSourceRecord:
    run_id = str(entry.get("run_id") or "")
    workflow_type = str(entry.get("run_type") or "strategy")
    artifact_dir = _resolve_artifact_dir(entry, artifact_root=artifact_root)
    manifest_path = _resolve_manifest_path(entry, artifact_dir=artifact_dir)
    manifest = _load_json_if_exists(manifest_path)
    promotion_gate_path = None if artifact_dir is None else artifact_dir / DEFAULT_PROMOTION_ARTIFACT_FILENAME
    promotion_gates = _load_json_if_exists(promotion_gate_path)
    summary = _promotion_gate_summary(entry, manifest, promotion_gates)
    return GovernanceSourceRecord(
        run_id=run_id,
        workflow_type=workflow_type,
        registry_entry=dict(entry),
        registry_path=registry_path,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        promotion_gate_path=promotion_gate_path,
        promotion_gates=promotion_gates,
        promotion_gate_summary=summary,
        governance_metadata={
            "artifact_evidence_paths": _evidence_paths(
                artifact_dir=artifact_dir,
                manifest_path=manifest_path,
                promotion_gate_path=promotion_gate_path,
            )
        },
    )


def _standalone_review_records(*, artifact_root: Path, known_run_ids: set[str]) -> list[GovernanceSourceRecord]:
    review_root = artifact_root / "reviews"
    if not review_root.exists():
        return []
    records: list[GovernanceSourceRecord] = []
    for summary_path in sorted(review_root.rglob("review_summary.json")):
        summary = _load_json_if_exists(summary_path)
        if summary is None:
            continue
        review_id = str(summary.get("review_id") or summary_path.parent.name)
        if review_id in known_run_ids:
            continue
        manifest_path = summary_path.parent / "manifest.json"
        manifest = _load_json_if_exists(manifest_path)
        promotion_gate_path = summary_path.parent / DEFAULT_PROMOTION_ARTIFACT_FILENAME
        promotion_gates = _load_json_if_exists(promotion_gate_path)
        records.append(
            GovernanceSourceRecord(
                run_id=review_id,
                workflow_type="review",
                artifact_dir=summary_path.parent,
                manifest_path=manifest_path,
                manifest=manifest,
                promotion_gate_path=promotion_gate_path,
                promotion_gates=promotion_gates,
                promotion_gate_summary=_promotion_gate_summary({}, manifest, promotion_gates),
                review_summary_path=summary_path,
                review_summary=summary,
                governance_metadata={
                    "artifact_evidence_paths": _evidence_paths(
                        artifact_dir=summary_path.parent,
                        manifest_path=manifest_path,
                        promotion_gate_path=promotion_gate_path,
                        review_summary_path=summary_path,
                    )
                },
            )
        )
    return records


def _candidate_review_context_records(*, artifact_root: Path, known_run_ids: set[str]) -> list[GovernanceSourceRecord]:
    records: list[GovernanceSourceRecord] = []
    for summary_path in sorted(artifact_root.rglob("candidate_review_summary.json")):
        summary = _load_json_if_exists(summary_path)
        if summary is None:
            continue
        run_id = str(summary.get("candidate_selection_run_id") or summary_path.parent.name)
        governance_id = f"candidate_review:{run_id}"
        if governance_id in known_run_ids:
            continue
        manifest_path = summary_path.parent / "manifest.json"
        manifest = _load_json_if_exists(manifest_path)
        promotion_context = summary.get("promotion_context")
        promotion_gate_summary = None
        if isinstance(promotion_context, Mapping):
            portfolio_summary = promotion_context.get("portfolio_promotion_gate_summary")
            if isinstance(portfolio_summary, Mapping):
                promotion_gate_summary = dict(portfolio_summary)
        metadata = _candidate_review_metadata(
            summary=summary,
            manifest=manifest,
            summary_path=summary_path,
            manifest_path=manifest_path,
        )
        records.append(
            GovernanceSourceRecord(
                run_id=governance_id,
                workflow_type="candidate_review",
                artifact_dir=summary_path.parent,
                manifest_path=manifest_path,
                manifest=manifest,
                promotion_gate_summary=promotion_gate_summary,
                candidate_review_summary_path=summary_path,
                candidate_review_summary=summary,
                governance_metadata=metadata,
            )
        )
    return records


def _candidate_review_metadata(
    *,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    summary_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    promotion_context = summary.get("promotion_context")
    selected_candidate_ids = _selected_candidate_ids_from_payloads(summary, manifest)
    selected_run_ids = _selected_run_ids_from_payloads(summary, manifest)
    upstream_run_ids = _upstream_run_ids_from_payloads(summary, manifest, selected_run_ids=selected_run_ids)
    raw_upstream_run_ids = _raw_upstream_run_ids_from_payloads(summary, manifest)
    candidate_selection_run_id = _nested_string(summary, "candidate_selection_run_id") or _nested_string(
        manifest,
        "candidate_selection_run_id",
    )
    selected_candidate_id = (
        _nested_string(summary, "selected_candidate_id")
        or _nested_string(manifest, "selected_candidate_id")
        or (selected_candidate_ids[0] if selected_candidate_ids else None)
    )
    selected_run_id = (
        _nested_string(summary, "selected_run_id")
        or _nested_string(manifest, "selected_run_id")
        or (selected_run_ids[0] if selected_run_ids else None)
    )
    metadata = {
        "candidate_selection_run_id": candidate_selection_run_id,
        "candidate_id": selected_candidate_id or candidate_selection_run_id or summary_path.parent.name,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_ids": selected_candidate_ids,
        "selected_run_id": selected_run_id,
        "selected_run_ids": selected_run_ids,
        "portfolio_run_id": _nested_string(summary, "portfolio_run_id") or _nested_string(manifest, "portfolio_run_id"),
        "strategy_run_id": _nested_string(summary, "strategy_run_id") or _nested_string(manifest, "strategy_run_id"),
        "alpha_run_id": _nested_string(summary, "alpha_run_id") or _nested_string(manifest, "alpha_run_id"),
        "upstream_run_ids": upstream_run_ids,
        "upstream_run_ids_raw": raw_upstream_run_ids,
        "candidate_promotion_status_counts": (
            dict(promotion_context.get("candidate_promotion_status_counts"))
            if isinstance(promotion_context, Mapping) and isinstance(promotion_context.get("candidate_promotion_status_counts"), Mapping)
            else {}
        ),
        "promotion_context_present": isinstance(promotion_context, Mapping),
        "candidate_review_summary_path": summary_path.as_posix(),
        "candidate_review_manifest_path": manifest_path.as_posix(),
        "artifact_evidence_paths": _evidence_paths(
            artifact_dir=summary_path.parent,
            manifest_path=manifest_path,
            candidate_review_summary_path=summary_path,
        ),
        "optional_artifact_evidence_paths": {},
    }
    return canonicalize_value(metadata)


def _campaign_records(*, artifact_root: Path, known_keys: set[tuple[str, str]]) -> list[GovernanceSourceRecord]:
    records_by_key: dict[tuple[str, str], GovernanceSourceRecord] = {}
    for campaign_dir in _discover_campaign_dirs(artifact_root):
        summary_path = campaign_dir / "summary.json"
        summary = _load_json_if_exists(summary_path)
        if summary is None:
            continue
        manifest_path = campaign_dir / "manifest.json"
        manifest = _load_json_if_exists(manifest_path)
        checkpoint_path = campaign_dir / "checkpoint.json"
        checkpoint = _load_json_if_exists(checkpoint_path)
        config_path = campaign_dir / "campaign_config.json"
        config = _load_json_if_exists(config_path)
        scenario_catalog_path = campaign_dir / "scenario_catalog.json"
        scenario_catalog = _load_json_if_exists(scenario_catalog_path)

        run_type = str(summary.get("run_type") or (manifest.get("run_type") if isinstance(manifest, dict) else ""))
        if run_type == "research_campaign_orchestration":
            campaign_record = _build_campaign_record(
                campaign_dir=campaign_dir,
                summary_path=summary_path,
                summary=summary,
                manifest_path=manifest_path,
                manifest=manifest,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                config_path=config_path,
                config=config,
                scenario_catalog_path=scenario_catalog_path,
                scenario_catalog=scenario_catalog,
            )
            _add_record(records_by_key, known_keys, campaign_record)
            for record in _scenario_records_from_orchestration(
                campaign_dir=campaign_dir,
                campaign_summary=summary,
                campaign_manifest_path=manifest_path,
                campaign_manifest=manifest,
                scenario_catalog=scenario_catalog,
                scenario_catalog_path=scenario_catalog_path,
            ):
                _add_record(records_by_key, known_keys, record)
            continue
        if run_type == "research_campaign" and isinstance(summary.get("scenario"), dict):
            _add_record(
                records_by_key,
                known_keys,
                _build_scenario_record(
                    campaign_dir=campaign_dir,
                    summary_path=summary_path,
                    summary=summary,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    config_path=config_path,
                    config=config,
                    parent_campaign_id=_nested_string(summary.get("scenario"), "orchestration_run_id"),
                    parent_campaign_manifest_path=None,
                ),
            )
            continue
        if run_type == "research_campaign":
            _add_record(
                records_by_key,
                known_keys,
                _build_campaign_record(
                    campaign_dir=campaign_dir,
                    summary_path=summary_path,
                    summary=summary,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    config_path=config_path,
                    config=config,
                    scenario_catalog_path=scenario_catalog_path,
                    scenario_catalog=scenario_catalog,
                ),
            )
    return [records_by_key[key] for key in sorted(records_by_key)]


def _discover_campaign_dirs(artifact_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    roots = [
        artifact_root,
        artifact_root / "research_campaigns",
        artifact_root / "campaign_artifacts",
    ]
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for summary_path in root.rglob("summary.json"):
            payload = _load_json_if_exists(summary_path)
            if not isinstance(payload, dict):
                continue
            run_type = str(payload.get("run_type") or "").strip()
            if run_type in {"research_campaign", "research_campaign_orchestration"}:
                candidates.add(summary_path.parent)
    return sorted(candidates, key=lambda path: path.as_posix())


def _build_campaign_record(
    *,
    campaign_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    checkpoint_path: Path,
    checkpoint: dict[str, Any] | None,
    config_path: Path,
    config: dict[str, Any] | None,
    scenario_catalog_path: Path,
    scenario_catalog: dict[str, Any] | None,
) -> GovernanceSourceRecord:
    campaign_id = str(summary.get("campaign_run_id") or summary.get("orchestration_run_id") or campaign_dir.name)
    return GovernanceSourceRecord(
        run_id=campaign_id,
        workflow_type="campaign",
        artifact_dir=campaign_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        promotion_gate_summary=_campaign_promotion_summary(summary, manifest),
        governance_metadata=canonicalize_value(
            {
                "campaign_id": campaign_id,
                "campaign_status": summary.get("status"),
                "checkpoint_status": _checkpoint_status(checkpoint),
                "campaign_summary_path": summary_path.as_posix(),
                "campaign_manifest_path": manifest_path.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "campaign_config_path": config_path.as_posix(),
                "scenario_catalog_path": scenario_catalog_path.as_posix() if scenario_catalog is not None else None,
                "child_run_ids": _child_run_ids(summary),
                "selected_candidate_ids": _selected_candidate_ids(summary),
                "scenario_catalog_ids": _scenario_catalog_ids(scenario_catalog),
                "scenario_catalog_missing_ids": _missing_catalog_scenario_ids(campaign_dir, scenario_catalog),
                "scenario_count": _scenario_count(summary, scenario_catalog),
                "campaign_summary": _campaign_summary_context(summary),
                "campaign_config_present": config is not None,
                "artifact_evidence_paths": _evidence_paths(
                    artifact_dir=campaign_dir,
                    manifest_path=manifest_path,
                    campaign_summary_path=summary_path,
                    checkpoint_path=checkpoint_path,
                    scenario_catalog_path=scenario_catalog_path if scenario_catalog is not None else None,
                    campaign_config_path=config_path if config is not None else None,
                ),
            }
        ),
    )


def _scenario_records_from_orchestration(
    *,
    campaign_dir: Path,
    campaign_summary: Mapping[str, Any],
    campaign_manifest_path: Path | None,
    campaign_manifest: Mapping[str, Any] | None,
    scenario_catalog: Mapping[str, Any] | None,
    scenario_catalog_path: Path,
) -> list[GovernanceSourceRecord]:
    records: list[GovernanceSourceRecord] = []
    campaign_id = str(campaign_summary.get("orchestration_run_id") or campaign_summary.get("campaign_run_id") or campaign_dir.name)
    entries_by_id = {
        scenario_id: entry
        for scenario_id, entry in (
            (_scenario_id(entry), entry)
            for entry in _mapping_list(campaign_summary.get("scenarios"))
        )
        if scenario_id is not None
    }
    for scenario_id in sorted(set(entries_by_id) | set(_scenario_catalog_ids(scenario_catalog))):
        entry = entries_by_id.get(scenario_id, {})
        scenario_dir = campaign_dir / "scenarios" / scenario_id
        summary_path = scenario_dir / "summary.json"
        manifest_path = scenario_dir / "manifest.json"
        checkpoint_path = scenario_dir / "checkpoint.json"
        config_path = scenario_dir / "campaign_config.json"
        summary = _load_json_if_exists(summary_path)
        manifest = _load_json_if_exists(manifest_path)
        checkpoint = _load_json_if_exists(checkpoint_path)
        config = _load_json_if_exists(config_path)
        if summary is None:
            summary = _scenario_summary_from_parent_entry(entry, campaign_id=campaign_id, scenario_id=scenario_id)
        records.append(
            _build_scenario_record(
                campaign_dir=scenario_dir,
                summary_path=summary_path,
                summary=summary,
                manifest_path=manifest_path,
                manifest=manifest,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                config_path=config_path,
                config=config,
                parent_campaign_id=campaign_id,
                parent_campaign_status=_nested_string(campaign_summary, "status"),
                parent_scenario_entry=entry,
                parent_campaign_manifest_path=campaign_manifest_path,
                parent_campaign_manifest=campaign_manifest,
                scenario_catalog_path=scenario_catalog_path,
            )
        )
    return records


def _build_scenario_record(
    *,
    campaign_dir: Path,
    summary_path: Path,
    summary: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any] | None,
    checkpoint_path: Path,
    checkpoint: dict[str, Any] | None,
    config_path: Path,
    config: dict[str, Any] | None,
    parent_campaign_id: str | None,
    parent_campaign_status: str | None = None,
    parent_scenario_entry: Mapping[str, Any] | None = None,
    parent_campaign_manifest_path: Path | None = None,
    parent_campaign_manifest: Mapping[str, Any] | None = None,
    scenario_catalog_path: Path | None = None,
) -> GovernanceSourceRecord:
    scenario = summary.get("scenario") if isinstance(summary.get("scenario"), Mapping) else {}
    scenario_id = (
        _nested_string(scenario, "scenario_id")
        or _nested_string(parent_scenario_entry, "scenario_id")
        or campaign_dir.name
    )
    campaign_id = (
        parent_campaign_id
        or _nested_string(scenario, "orchestration_run_id")
        or str(summary.get("campaign_run_id") or "")
    )
    selected_candidate_ids = _selected_candidate_ids(summary)
    return GovernanceSourceRecord(
        run_id=f"{campaign_id}:{scenario_id}" if campaign_id else scenario_id,
        workflow_type="campaign_scenario",
        artifact_dir=campaign_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        promotion_gate_summary=_campaign_promotion_summary(summary, manifest),
        governance_metadata=canonicalize_value(
            {
                "campaign_id": campaign_id,
                "scenario_id": scenario_id,
                "scenario_name": _nested_string(parent_scenario_entry, "description") or _nested_string(scenario, "description"),
                "scenario_status": summary.get("status") or _nested_string(parent_scenario_entry, "status"),
                "campaign_status": parent_campaign_status,
                "checkpoint_status": _checkpoint_status(checkpoint),
                "selected_run_id": _first_child_run_id(summary),
                "selected_candidate_id": selected_candidate_ids[0] if selected_candidate_ids else None,
                "selected_candidate_ids": selected_candidate_ids,
                "child_run_ids": _child_run_ids(summary),
                "scenario_manifest_path": manifest_path.as_posix(),
                "scenario_summary_path": summary_path.as_posix(),
                "campaign_manifest_path": parent_campaign_manifest_path.as_posix() if parent_campaign_manifest_path is not None else None,
                "campaign_summary_path": summary_path.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "campaign_config_path": config_path.as_posix(),
                "scenario_catalog_path": scenario_catalog_path.as_posix() if scenario_catalog_path is not None else None,
                "parent_scenario_entry": canonicalize_value(dict(parent_scenario_entry or {})),
                "parent_campaign_manifest_has_scenario": _parent_manifest_has_scenario(
                    parent_campaign_manifest,
                    scenario_id=scenario_id,
                ),
                "scenario_summary_present": summary_path.exists(),
                "scenario_manifest_present": manifest_path.exists(),
                "checkpoint_stage_states": _checkpoint_stage_states(checkpoint),
                "missing_child_artifact_paths": _missing_child_artifact_paths(summary),
                "campaign_config_present": config is not None,
                "artifact_evidence_paths": _evidence_paths(
                    artifact_dir=campaign_dir,
                    campaign_manifest_path=parent_campaign_manifest_path,
                    manifest_path=manifest_path if manifest is not None else None,
                    campaign_summary_path=summary_path if summary_path.exists() else None,
                    checkpoint_path=checkpoint_path if checkpoint is not None else None,
                    campaign_config_path=config_path if config is not None else None,
                    scenario_catalog_path=scenario_catalog_path,
                ),
            }
        ),
    )


def _scenario_summary_from_parent_entry(
    entry: Mapping[str, Any],
    *,
    campaign_id: str,
    scenario_id: str,
) -> dict[str, Any]:
    return canonicalize_value(
        {
            "run_type": "research_campaign",
            "campaign_run_id": entry.get("campaign_run_id"),
            "status": entry.get("status"),
            "scenario": {
                "orchestration_run_id": campaign_id,
                "scenario_id": scenario_id,
                "description": entry.get("description"),
                "source": entry.get("source"),
            },
            "selected_run_ids": dict(entry.get("selected_run_ids") or {}),
            "final_outcomes": dict(entry.get("final_outcomes") or {}),
        }
    )


def _add_record(
    records_by_key: dict[tuple[str, str], GovernanceSourceRecord],
    known_keys: set[tuple[str, str]],
    record: GovernanceSourceRecord,
) -> None:
    key = (record.workflow_type, record.run_id)
    if key in known_keys or key in records_by_key:
        return
    records_by_key[key] = record


def _resolve_artifact_dir(entry: Mapping[str, Any], *, artifact_root: Path) -> Path | None:
    raw_path = entry.get("artifact_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return Path(raw_path)
    run_id = entry.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return artifact_root / run_id
    return None


def _campaign_promotion_summary(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    for payload in (
        _nested_mapping(summary, "final_outcomes", "review_promotion_gate_summary"),
        _nested_mapping(summary, "promotion_gate_summary"),
        _nested_mapping(manifest, "promotion_gate_summary"),
        _nested_mapping(manifest, "final_outcomes", "review_promotion_gate_summary"),
    ):
        if isinstance(payload, Mapping):
            return dict(payload)
    final_outcomes = summary.get("final_outcomes")
    if isinstance(final_outcomes, Mapping):
        promotion_status = final_outcomes.get("review_promotion_status")
        gate_status = final_outcomes.get("review_promotion_gate_status")
        highest_severity = final_outcomes.get("review_promotion_highest_severity")
        reason_codes = final_outcomes.get("review_promotion_decision_reason_codes")
        severity_counts = final_outcomes.get("review_promotion_severity_counts")
        if promotion_status is not None or gate_status is not None or highest_severity is not None:
            return canonicalize_value(
                {
                    "promotion_status": promotion_status,
                    "evaluation_status": gate_status,
                    "highest_severity": highest_severity,
                    "decision_reason_codes": reason_codes if isinstance(reason_codes, list) else [],
                    "severity_counts": severity_counts if isinstance(severity_counts, Mapping) else {},
                }
            )
    return None


def _checkpoint_status(checkpoint: Mapping[str, Any] | None) -> str | None:
    if not isinstance(checkpoint, Mapping):
        return None
    if isinstance(checkpoint.get("status"), str):
        return str(checkpoint["status"])
    stage_states = checkpoint.get("stage_states")
    if isinstance(stage_states, Mapping):
        if any(str(value) in {"failed", "partial"} for value in stage_states.values()):
            return "failed"
        if stage_states:
            return "completed"
    return None


def _checkpoint_stage_states(checkpoint: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(checkpoint, Mapping):
        return {}
    stage_states = checkpoint.get("stage_states")
    if not isinstance(stage_states, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in sorted(stage_states.items())
    }


def _child_run_ids(summary: Mapping[str, Any]) -> list[str]:
    selected = summary.get("selected_run_ids")
    if not isinstance(selected, Mapping):
        return []
    run_ids: set[str] = set()
    for value in selected.values():
        if isinstance(value, list):
            run_ids.update(str(item) for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            run_ids.add(str(value))
    return sorted(run_ids)


def _first_child_run_id(summary: Mapping[str, Any]) -> str | None:
    child_run_ids = _child_run_ids(summary)
    return child_run_ids[0] if child_run_ids else None


def _selected_candidate_ids(summary: Mapping[str, Any]) -> list[str]:
    selected = summary.get("selected_run_ids")
    if not isinstance(selected, Mapping):
        return []
    values = []
    for key in ("candidate_id", "candidate_ids", "selected_candidate_id", "selected_candidate_ids"):
        value = selected.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            values.append(str(value))
    return sorted(set(values))


def _selected_candidate_ids_from_payloads(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> list[str]:
    values: list[str] = []
    for payload in (summary, manifest):
        if not isinstance(payload, Mapping):
            continue
        values.extend(_candidate_id_values(payload, "candidate_id"))
        values.extend(_candidate_id_values(payload, "candidate_ids"))
        values.extend(_candidate_id_values(payload, "selected_candidate_id"))
        values.extend(_candidate_id_values(payload, "selected_candidate_ids"))
        values.extend(_candidate_id_values(payload.get("selected_run_ids"), "candidate_id"))
        values.extend(_candidate_id_values(payload.get("selected_run_ids"), "candidate_ids"))
        values.extend(_candidate_id_values(payload.get("selected_run_ids"), "selected_candidate_id"))
        values.extend(_candidate_id_values(payload.get("selected_run_ids"), "selected_candidate_ids"))
    return sorted(set(values))


def _selected_run_ids_from_payloads(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> list[str]:
    values: list[str] = []
    for payload in (summary, manifest):
        if not isinstance(payload, Mapping):
            continue
        values.extend(_candidate_id_values(payload, "selected_run_id"))
        values.extend(_candidate_id_values(payload, "selected_run_ids"))
    return sorted(set(values))


def _upstream_run_ids_from_payloads(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    *,
    selected_run_ids: list[str],
) -> list[str]:
    values = list(selected_run_ids)
    for payload in (summary, manifest):
        if not isinstance(payload, Mapping):
            continue
        for key in ("upstream_run_id", "upstream_run_ids", "portfolio_run_id", "strategy_run_id", "alpha_run_id"):
            values.extend(_candidate_id_values(payload, key))
        input_artifacts = payload.get("input_artifacts")
        if isinstance(input_artifacts, Mapping):
            for key in ("portfolio_run_id", "strategy_run_id", "alpha_run_id", "upstream_run_ids"):
                values.extend(_candidate_id_values(input_artifacts, key))
    return sorted(set(values))


def _raw_upstream_run_ids_from_payloads(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> list[str]:
    values = _raw_upstream_run_id_values(summary)
    if values:
        return values
    return _raw_upstream_run_id_values(manifest)


def _raw_upstream_run_id_values(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    selected_values: list[str] = []
    for key in ("selected_run_id", "selected_run_ids"):
        selected_values.extend(_raw_candidate_id_values(payload, key))
    if selected_values:
        return selected_values
    upstream_values: list[str] = []
    for key in ("upstream_run_id", "upstream_run_ids"):
        upstream_values.extend(_raw_candidate_id_values(payload, key))
    return upstream_values


def _raw_candidate_id_values(payload: Any, key: str) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, Mapping):
        values: list[str] = []
        for item in value.values():
            if isinstance(item, list):
                values.extend(str(part).strip() for part in item if str(part).strip())
            elif item is not None and str(item).strip():
                values.append(str(item).strip())
        return values
    if value is not None and str(value).strip():
        return [str(value).strip()]
    return []


def _candidate_id_values(payload: Any, key: str) -> list[str]:
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


def _scenario_count(
    summary: Mapping[str, Any],
    scenario_catalog: Mapping[str, Any] | None,
) -> int:
    if isinstance(summary.get("scenario_count"), int):
        return int(summary["scenario_count"])
    if isinstance(scenario_catalog, Mapping) and isinstance(scenario_catalog.get("scenario_count"), int):
        return int(scenario_catalog["scenario_count"])
    scenarios = summary.get("scenarios")
    return len(scenarios) if isinstance(scenarios, list) else 0


def _scenario_catalog_ids(scenario_catalog: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(scenario_catalog, Mapping):
        return []
    ids = [
        scenario_id
        for scenario_id in (
            _scenario_id(entry)
            for entry in _mapping_list(scenario_catalog.get("scenarios"))
        )
        if scenario_id is not None
    ]
    return sorted(set(ids))


def _missing_catalog_scenario_ids(campaign_dir: Path, scenario_catalog: Mapping[str, Any] | None) -> list[str]:
    return [
        scenario_id
        for scenario_id in _scenario_catalog_ids(scenario_catalog)
        if not (campaign_dir / "scenarios" / scenario_id).exists()
    ]


def _scenario_id(entry: Mapping[str, Any]) -> str | None:
    value = entry.get("scenario_id")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _campaign_summary_context(summary: Mapping[str, Any]) -> dict[str, Any]:
    return canonicalize_value(
        {
            "run_type": summary.get("run_type"),
            "campaign_run_id": summary.get("campaign_run_id"),
            "orchestration_run_id": summary.get("orchestration_run_id"),
            "status": summary.get("status"),
            "scenario_count": summary.get("scenario_count"),
            "scenario_status_counts": (
                dict(summary.get("scenario_status_counts"))
                if isinstance(summary.get("scenario_status_counts"), Mapping)
                else {}
            ),
            "stage_state_counts": (
                dict(summary.get("stage_state_counts"))
                if isinstance(summary.get("stage_state_counts"), Mapping)
                else {}
            ),
        }
    )


def _parent_manifest_has_scenario(parent_manifest: Mapping[str, Any] | None, *, scenario_id: str) -> bool | None:
    if not isinstance(parent_manifest, Mapping):
        return None
    scenario_path = f"scenarios/{scenario_id}/summary.json"
    artifact_files = parent_manifest.get("artifact_files")
    if isinstance(artifact_files, list):
        return scenario_path in {str(item) for item in artifact_files}
    return None


def _missing_child_artifact_paths(summary: Mapping[str, Any]) -> list[str]:
    output_paths = summary.get("output_paths")
    if not isinstance(output_paths, Mapping):
        return []
    missing = []
    for key, value in sorted(output_paths.items()):
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if path.is_absolute() and not path.exists():
            missing.append(f"{key}:{path.as_posix()}")
    return missing


def _evidence_paths(**paths: Path | None) -> dict[str, str]:
    return {
        key: value.as_posix()
        for key, value in sorted(paths.items())
        if value is not None
    }


def _nested_mapping(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_string(value: Any, key: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    item = value.get(key)
    if item is None:
        return None
    text = str(item).strip()
    return text or None


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _resolve_manifest_path(entry: Mapping[str, Any], *, artifact_dir: Path | None) -> Path | None:
    raw_path = entry.get("manifest_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return Path(raw_path)
    if artifact_dir is None:
        return None
    return artifact_dir / "manifest.json"


def _promotion_gate_summary(
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    promotion_gates: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    for payload in (entry, manifest, promotion_gates):
        if not isinstance(payload, Mapping):
            continue
        summary = payload.get("promotion_gate_summary")
        if isinstance(summary, Mapping):
            return dict(summary)
        if "promotion_status" in payload and "gate_count" in payload:
            return dict(payload)
    review_metadata = entry.get("review_metadata")
    if isinstance(review_metadata, Mapping):
        nested = review_metadata.get("promotion_gate_summary")
        if isinstance(nested, Mapping):
            return dict(nested)
    return None


def _load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


__all__ = ["load_governance_artifacts"]
