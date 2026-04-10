from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.research.registry import canonicalize_value

CAMPAIGN_CHECKPOINT_SCHEMA_VERSION = 1
CAMPAIGN_STAGE_ORDER = (
    "preflight",
    "research",
    "comparison",
    "candidate_selection",
    "portfolio",
    "candidate_review",
    "review",
)
VALID_CAMPAIGN_STAGE_STATES = frozenset(
    {"completed", "failed", "skipped", "reused", "partial", "pending"}
)

_STAGE_STATE_DEFAULTS: dict[str, dict[str, Any]] = {
    "completed": {"terminal": True, "resumable": False},
    "failed": {"terminal": True, "resumable": True},
    "skipped": {"terminal": True, "resumable": False},
    "reused": {"terminal": True, "resumable": False},
    "partial": {"terminal": False, "resumable": True},
    "pending": {"terminal": False, "resumable": True},
}


class CampaignCheckpointError(ValueError):
    """Raised when a campaign checkpoint payload is malformed."""


def build_campaign_stage_checkpoint(
    *,
    stage_name: str,
    state: str,
    state_reason: str | None = None,
    source: str | None = None,
    selected_run_ids: Mapping[str, Any] | None = None,
    key_metrics: Mapping[str, Any] | None = None,
    output_paths: Mapping[str, Any] | None = None,
    outcomes: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_stage_name = _normalize_stage_name(stage_name)
    normalized_state = _normalize_stage_state(state)
    if state_reason is not None and not str(state_reason).strip():
        state_reason = None
    source_text = None if source is None else str(source).strip() or None
    defaults = _STAGE_STATE_DEFAULTS[normalized_state]
    payload = {
        "stage_name": normalized_stage_name,
        "state": normalized_state,
        "state_reason": state_reason,
        "source": source_text,
        "terminal": defaults["terminal"],
        "resumable": defaults["resumable"],
        "selected_run_ids": canonicalize_value(dict(selected_run_ids or {})),
        "key_metrics": canonicalize_value(dict(key_metrics or {})),
        "output_paths": canonicalize_value(dict(output_paths or {})),
        "outcomes": canonicalize_value(dict(outcomes or {})),
        "details": canonicalize_value(dict(details or {})),
    }
    return canonicalize_value(payload)


def build_campaign_checkpoint_payload(
    *,
    campaign_run_id: str,
    status: str,
    checkpoint_path: str | Path,
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_run_id = _normalize_required_string(campaign_run_id, field_name="campaign_run_id")
    normalized_status = _normalize_required_string(status, field_name="status")
    normalized_checkpoint_path = Path(checkpoint_path).as_posix()
    stage_payloads = [canonicalize_value(dict(stage)) for stage in stages]
    payload = {
        "schema_version": CAMPAIGN_CHECKPOINT_SCHEMA_VERSION,
        "run_type": "research_campaign_checkpoint",
        "campaign_run_id": normalized_run_id,
        "status": normalized_status,
        "checkpoint_path": normalized_checkpoint_path,
        "stage_order": list(CAMPAIGN_STAGE_ORDER),
        "stage_states": {
            str(stage.get("stage_name")): str(stage.get("state"))
            for stage in stage_payloads
        },
        "stages": stage_payloads,
        "completed_stage_count": sum(1 for stage in stage_payloads if stage.get("state") == "completed"),
        "terminal_stage_count": sum(1 for stage in stage_payloads if bool(stage.get("terminal"))),
        "pending_stage_count": sum(1 for stage in stage_payloads if stage.get("state") == "pending"),
        "resumable_stage_names": [
            str(stage["stage_name"])
            for stage in stage_payloads
            if bool(stage.get("resumable"))
        ],
    }
    validate_campaign_checkpoint_payload(payload)
    return canonicalize_value(payload)


def validate_campaign_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise CampaignCheckpointError("Campaign checkpoint payload must be a mapping.")
    schema_version = payload.get("schema_version")
    if schema_version != CAMPAIGN_CHECKPOINT_SCHEMA_VERSION:
        raise CampaignCheckpointError(
            f"Campaign checkpoint schema_version must be {CAMPAIGN_CHECKPOINT_SCHEMA_VERSION}."
        )
    run_type = payload.get("run_type")
    if run_type != "research_campaign_checkpoint":
        raise CampaignCheckpointError("Campaign checkpoint run_type must be 'research_campaign_checkpoint'.")
    _normalize_required_string(payload.get("campaign_run_id"), field_name="campaign_run_id")
    _normalize_required_string(payload.get("status"), field_name="status")
    _normalize_required_string(payload.get("checkpoint_path"), field_name="checkpoint_path")

    stage_order = payload.get("stage_order")
    if list(stage_order or []) != list(CAMPAIGN_STAGE_ORDER):
        raise CampaignCheckpointError(
            f"Campaign checkpoint stage_order must equal {list(CAMPAIGN_STAGE_ORDER)}."
        )

    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise CampaignCheckpointError("Campaign checkpoint stages must be a list.")
    if len(stages) != len(CAMPAIGN_STAGE_ORDER):
        raise CampaignCheckpointError(
            f"Campaign checkpoint must contain exactly {len(CAMPAIGN_STAGE_ORDER)} stage entries."
        )

    seen_stage_names: list[str] = []
    for index, stage in enumerate(stages):
        validate_campaign_stage_payload(stage, index=index)
        seen_stage_names.append(str(stage["stage_name"]))
    if seen_stage_names != list(CAMPAIGN_STAGE_ORDER):
        raise CampaignCheckpointError(
            f"Campaign checkpoint stages must appear in canonical order {list(CAMPAIGN_STAGE_ORDER)}."
        )

    stage_states = payload.get("stage_states")
    if not isinstance(stage_states, Mapping):
        raise CampaignCheckpointError("Campaign checkpoint stage_states must be a mapping.")
    expected_stage_states = {
        str(stage["stage_name"]): str(stage["state"])
        for stage in stages
    }
    if canonicalize_value(stage_states) != canonicalize_value(expected_stage_states):
        raise CampaignCheckpointError("Campaign checkpoint stage_states must mirror stages[].state values.")


def validate_campaign_stage_payload(payload: Mapping[str, Any], *, index: int | None = None) -> None:
    if not isinstance(payload, Mapping):
        raise CampaignCheckpointError("Campaign checkpoint stage payload must be a mapping.")
    prefix = f"stages[{index}]" if index is not None else "stage"
    stage_name = _normalize_stage_name(payload.get("stage_name"))
    state = _normalize_stage_state(payload.get("state"))

    terminal = payload.get("terminal")
    resumable = payload.get("resumable")
    defaults = _STAGE_STATE_DEFAULTS[state]
    if terminal is not defaults["terminal"]:
        raise CampaignCheckpointError(
            f"{prefix}.terminal must be {defaults['terminal']} for state {state!r}."
        )
    if resumable is not defaults["resumable"]:
        raise CampaignCheckpointError(
            f"{prefix}.resumable must be {defaults['resumable']} for state {state!r}."
        )

    for field_name in ("selected_run_ids", "key_metrics", "output_paths", "outcomes", "details"):
        value = payload.get(field_name)
        if not isinstance(value, Mapping):
            raise CampaignCheckpointError(f"{prefix}.{field_name} must be a mapping.")

    state_reason = payload.get("state_reason")
    if state_reason is not None and not isinstance(state_reason, str):
        raise CampaignCheckpointError(f"{prefix}.state_reason must be a string when provided.")
    source = payload.get("source")
    if source is not None and not isinstance(source, str):
        raise CampaignCheckpointError(f"{prefix}.source must be a string when provided.")

    if stage_name == "preflight" and state == "pending":
        raise CampaignCheckpointError("Preflight stage cannot be pending in a persisted checkpoint.")
    if state == "failed" and not str(state_reason or "").strip():
        raise CampaignCheckpointError(f"{prefix}.state_reason is required when state is 'failed'.")


def serialize_campaign_checkpoint(payload: Mapping[str, Any]) -> str:
    validate_campaign_checkpoint_payload(payload)
    return json.dumps(canonicalize_value(payload), indent=2, sort_keys=True)


def write_campaign_checkpoint(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_campaign_checkpoint_payload(payload)
    canonical_payload = canonicalize_value(dict(payload))
    path.write_text(serialize_campaign_checkpoint(canonical_payload), encoding="utf-8", newline="\n")
    return canonical_payload


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise CampaignCheckpointError(f"Campaign checkpoint field '{field_name}' must be a non-empty string.")
    normalized = value.strip()
    if not normalized:
        raise CampaignCheckpointError(f"Campaign checkpoint field '{field_name}' must be a non-empty string.")
    return normalized


def _normalize_stage_name(value: Any) -> str:
    normalized = _normalize_required_string(value, field_name="stage_name")
    if normalized not in CAMPAIGN_STAGE_ORDER:
        raise CampaignCheckpointError(
            f"Campaign checkpoint stage_name must be one of {list(CAMPAIGN_STAGE_ORDER)}."
        )
    return normalized


def _normalize_stage_state(value: Any) -> str:
    normalized = _normalize_required_string(value, field_name="state")
    if normalized not in VALID_CAMPAIGN_STAGE_STATES:
        raise CampaignCheckpointError(
            f"Campaign checkpoint state must be one of {sorted(VALID_CAMPAIGN_STAGE_STATES)}."
        )
    return normalized

