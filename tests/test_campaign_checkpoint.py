from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.campaign_checkpoint import (
    CAMPAIGN_STAGE_ORDER,
    CampaignCheckpointError,
    build_campaign_checkpoint_payload,
    build_campaign_stage_checkpoint,
    write_campaign_checkpoint,
)


def _build_stage(stage_name: str, state: str, **overrides: object) -> dict[str, object]:
    return build_campaign_stage_checkpoint(
        stage_name=stage_name,
        state=state,
        state_reason=overrides.pop("state_reason", None),
        source=overrides.pop("source", "executed"),
        input_fingerprint=overrides.pop("input_fingerprint", None),
        fingerprint_inputs=overrides.pop("fingerprint_inputs", {}),
        selected_run_ids=overrides.pop("selected_run_ids", {}),
        key_metrics=overrides.pop("key_metrics", {}),
        output_paths=overrides.pop("output_paths", {}),
        outcomes=overrides.pop("outcomes", {}),
        details=overrides.pop("details", {}),
    )


def test_checkpoint_payload_accepts_all_canonical_stage_states(tmp_path: Path) -> None:
    states = [
        "failed",
        "completed",
        "reused",
        "partial",
        "skipped",
        "pending",
        "completed",
    ]
    stages = [
        _build_stage(
            stage_name,
            state,
            state_reason="failed stage" if state == "failed" else None,
        )
        for stage_name, state in zip(CAMPAIGN_STAGE_ORDER, states, strict=True)
    ]

    payload = build_campaign_checkpoint_payload(
        campaign_run_id="research_campaign_demo",
        status="partial",
        checkpoint_path=tmp_path / "checkpoint.json",
        stages=stages,
    )

    assert payload["stage_states"] == {
        stage_name: state
        for stage_name, state in zip(CAMPAIGN_STAGE_ORDER, states, strict=True)
    }
    assert payload["stage_input_fingerprints"] == {
        stage_name: None for stage_name in CAMPAIGN_STAGE_ORDER
    }
    assert payload["resumable_stage_names"] == [
        "preflight",
        "candidate_selection",
        "candidate_review",
    ]


def test_checkpoint_payload_persists_stage_input_fingerprints(tmp_path: Path) -> None:
    stages = [
        _build_stage(
            stage_name,
            "completed",
            fingerprint_inputs={"stage": stage_name, "version": 1},
            input_fingerprint=f"fingerprint-{stage_name}",
        )
        for stage_name in CAMPAIGN_STAGE_ORDER
    ]

    payload = build_campaign_checkpoint_payload(
        campaign_run_id="research_campaign_demo",
        status="completed",
        checkpoint_path=tmp_path / "checkpoint.json",
        stages=stages,
    )

    assert payload["stage_input_fingerprints"] == {
        stage_name: f"fingerprint-{stage_name}"
        for stage_name in CAMPAIGN_STAGE_ORDER
    }
    assert payload["stages"][0]["fingerprint_inputs"] == {
        "stage": "preflight",
        "version": 1,
    }


def test_checkpoint_write_is_canonical_and_round_trips(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    payload = build_campaign_checkpoint_payload(
        campaign_run_id="research_campaign_demo",
        status="completed",
        checkpoint_path=checkpoint_path,
        stages=[
            _build_stage(stage_name, "completed")
            for stage_name in CAMPAIGN_STAGE_ORDER
        ],
    )

    written = write_campaign_checkpoint(checkpoint_path, payload)

    assert written == payload
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == payload


def test_checkpoint_validation_rejects_missing_failed_reason(tmp_path: Path) -> None:
    stages = [
        _build_stage(stage_name, "completed")
        for stage_name in CAMPAIGN_STAGE_ORDER
    ]
    stages[1] = _build_stage("research", "failed")

    with pytest.raises(CampaignCheckpointError, match="state_reason is required"):
        build_campaign_checkpoint_payload(
            campaign_run_id="research_campaign_demo",
            status="failed",
            checkpoint_path=tmp_path / "checkpoint.json",
            stages=stages,
        )


def test_checkpoint_validation_rejects_non_canonical_stage_order(tmp_path: Path) -> None:
    stages = [
        _build_stage(stage_name, "completed")
        for stage_name in reversed(CAMPAIGN_STAGE_ORDER)
    ]

    with pytest.raises(CampaignCheckpointError, match="canonical order"):
        build_campaign_checkpoint_payload(
            campaign_run_id="research_campaign_demo",
            status="completed",
            checkpoint_path=tmp_path / "checkpoint.json",
            stages=stages,
        )
