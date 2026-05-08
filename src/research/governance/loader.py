from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.research.promotion import DEFAULT_PROMOTION_ARTIFACT_FILENAME
from src.research.registry import default_registry_path, load_registry

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
    return GovernanceDataset(
        records=sorted(records, key=lambda item: (item.workflow_type, item.run_id)),
        sources={
            "artifact_root": resolved_artifact_root.as_posix(),
            "registry_path": resolved_registry_path.as_posix(),
            "registry_entry_count": len(registry_entries),
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
            )
        )
    return records


def _resolve_artifact_dir(entry: Mapping[str, Any], *, artifact_root: Path) -> Path | None:
    raw_path = entry.get("artifact_path")
    if isinstance(raw_path, str) and raw_path.strip():
        return Path(raw_path)
    run_id = entry.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return artifact_root / run_id
    return None


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
