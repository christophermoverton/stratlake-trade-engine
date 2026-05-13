from __future__ import annotations

import json
from pathlib import Path

from src.research.robustness import (
    LEAKAGE_VALIDATION_FILENAME,
    PURGED_SPLIT_PLAN_FILENAME,
    PURGED_SPLIT_SUMMARY_FILENAME,
    LeakageValidationResult,
    PurgedSplit,
    PurgedSplitConfig,
    PurgedSplitPlan,
    build_purged_split_evidence,
    build_purged_split_findings,
    build_purged_split_plan,
    intervals_overlap,
    validate_purged_split_plan,
    write_purged_split_artifacts,
)


def _obs(
    observation_id: str,
    timestamp: str,
    label_start: str | None = None,
    label_end: str | None = None,
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "timestamp": timestamp,
        "label_start": timestamp if label_start is None else label_start,
        "label_end": timestamp if label_end is None else label_end,
    }


def test_interval_overlap_helper_uses_half_open_boundaries() -> None:
    assert not intervals_overlap("2025-01-01", "2025-01-03", "2025-01-03", "2025-01-04")
    assert not intervals_overlap("2025-01-04", "2025-01-05", "2025-01-03", "2025-01-04")
    assert intervals_overlap("2025-01-01", "2025-01-04", "2025-01-03", "2025-01-05")
    assert intervals_overlap("2025-01-04", "2025-01-06", "2025-01-03", "2025-01-05")
    assert intervals_overlap("2025-01-03", "2025-01-04", "2025-01-01", "2025-01-05")
    assert intervals_overlap("2025-01-01", "2025-01-06", "2025-01-03", "2025-01-04")


def test_basic_deterministic_split_generation_sorts_by_timestamp_and_id() -> None:
    records = [
        _obs("obs_b", "2025-01-02"),
        _obs("obs_a", "2025-01-01"),
        _obs("obs_c", "2025-01-02"),
        _obs("obs_d", "2025-01-03"),
    ]

    first = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=2, min_train_observations=1),
        workflow_type="strategy",
        run_id="run_001",
    )
    second = build_purged_split_plan(
        list(reversed(records)),
        config=PurgedSplitConfig(n_splits=2, validation_window_size=2, min_train_observations=1),
        workflow_type="strategy",
        run_id="run_001",
    )

    assert first.to_dict() == second.to_dict()
    assert first.splits[0].validation_observation_ids == ("obs_a", "obs_b")
    assert first.splits[1].validation_observation_ids == ("obs_c", "obs_d")
    for split in first.splits:
        assert set(split.train_observation_ids).isdisjoint(split.validation_observation_ids)
    assert first.overall_status == "pass"


def test_purge_removes_overlapping_label_intervals() -> None:
    records = [
        _obs("before_overlap", "2025-01-01", "2025-01-01", "2025-01-04"),
        _obs("validation", "2025-01-02", "2025-01-02", "2025-01-03"),
        _obs("after", "2025-01-04", "2025-01-04", "2025-01-05"),
    ]

    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1, min_train_observations=1),
    )

    split = plan.splits[1]
    assert split.validation_observation_ids == ("validation",)
    assert split.purged_observation_ids == ("before_overlap",)
    assert "before_overlap" not in split.train_observation_ids
    assert split.leakage_status == "pass"


def test_boundary_non_overlap_does_not_purge() -> None:
    records = [
        _obs("ends_at_validation_start", "2025-01-01", "2025-01-01", "2025-01-03"),
        _obs("validation", "2025-01-03", "2025-01-03", "2025-01-04"),
        _obs("starts_at_validation_end", "2025-01-04", "2025-01-04", "2025-01-05"),
    ]

    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1, min_train_observations=1),
    )

    split = plan.splits[1]
    assert split.purged_observation_ids == ()
    assert set(split.train_observation_ids) == {"ends_at_validation_start", "starts_at_validation_end"}


def test_embargo_excludes_observations_after_validation_window() -> None:
    records = [
        _obs("train_before", "2025-01-01"),
        _obs("validation", "2025-01-02"),
        _obs("embargoed", "2025-01-03"),
        _obs("eligible_after_embargo", "2025-01-04"),
    ]

    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1, embargo_window="2D"),
    )

    split = plan.splits[1]
    assert split.embargoed_observation_ids == ("embargoed",)
    assert "embargoed" not in split.train_observation_ids
    assert "eligible_after_embargo" in split.train_observation_ids


def test_zero_embargo_leaves_post_validation_observations_eligible() -> None:
    records = [
        _obs("train_before", "2025-01-01"),
        _obs("validation", "2025-01-02"),
        _obs("post_validation", "2025-01-03"),
    ]

    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1, embargo_window="0D"),
    )

    assert "post_validation" in plan.splits[1].train_observation_ids
    assert plan.splits[1].embargoed_observation_ids == ()


def test_missing_timestamp_and_label_interval_emit_structured_findings() -> None:
    records = [
        {"observation_id": "missing_timestamp", "label_start": "2025-01-01", "label_end": "2025-01-02"},
        {"observation_id": "missing_label", "timestamp": "2025-01-02", "label_start": "2025-01-02"},
    ]

    plan, findings = build_purged_split_evidence(records)

    assert plan.valid_observation_count == 0
    assert {finding.check_id for finding in findings} == {"temporal_validation.missing_timestamp"}
    assert all(finding.severity == "needs_review" for finding in findings)


def test_label_end_before_label_start_and_duplicate_ids_emit_invalid_findings() -> None:
    records = [
        _obs("dup", "2025-01-01", "2025-01-02", "2025-01-01"),
        _obs("dup", "2025-01-03", "2025-01-03", "2025-01-04"),
        _obs("dup", "2025-01-04", "2025-01-04", "2025-01-05"),
    ]

    plan, findings = build_purged_split_evidence(
        records,
        config=PurgedSplitConfig(n_splits=1, validation_window_size=1),
    )

    assert plan.overall_status == "blocked"
    assert "temporal_validation.invalid_split_config" in {finding.check_id for finding in findings}


def test_insufficient_train_and_validation_observations_emit_findings() -> None:
    train_plan, train_findings = build_purged_split_evidence(
        [_obs("only", "2025-01-01")],
        config=PurgedSplitConfig(n_splits=1, validation_window_size=1, min_train_observations=1),
    )
    validation_plan, validation_findings = build_purged_split_evidence(
        [_obs("only", "2025-01-01")],
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1),
    )

    assert train_plan.splits[0].n_train_observations == 0
    assert "temporal_validation.insufficient_train_observations" in {
        finding.check_id for finding in train_findings
    }
    assert validation_plan.splits == ()
    assert "temporal_validation.insufficient_validation_observations" in {
        finding.check_id for finding in validation_findings
    }


def test_label_horizon_can_fill_label_end_and_zero_horizon_is_supported() -> None:
    records = [
        {"observation_id": "a", "timestamp": "2025-01-01", "label_start": "2025-01-01"},
        {"observation_id": "b", "timestamp": "2025-01-02", "label_start": "2025-01-02"},
    ]

    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=1, validation_window_size=1, label_horizon="0D"),
    )

    assert plan.valid_observation_count == 2
    assert plan.splits[0].leakage_status == "pass"


def test_leakage_validation_catches_manually_constructed_train_validation_overlap() -> None:
    split = PurgedSplit(
        split_id="manual_0001",
        train_start="2025-01-01T00:00:00Z",
        train_end="2025-01-03T00:00:00Z",
        validation_start="2025-01-03T00:00:00Z",
        validation_end="2025-01-04T00:00:00Z",
        purge_window="[2025-01-03T00:00:00Z, 2025-01-04T00:00:00Z)",
        embargo_window="[2025-01-04T00:00:00Z, 2025-01-04T00:00:00Z)",
        train_observation_ids=("shared",),
        validation_observation_ids=("shared",),
        n_train_observations=1,
        n_validation_observations=1,
    )
    plan = PurgedSplitPlan("strategy", "run_manual", PurgedSplitConfig(), splits=(split,))

    checks = validate_purged_split_plan(plan)

    assert "temporal_validation.train_validation_overlap" in {check.check_id for check in checks}


def test_leakage_validation_catches_manual_interval_and_embargo_violations() -> None:
    split = PurgedSplit(
        split_id="manual_0002",
        train_start="2025-01-01T00:00:00Z",
        train_end="2025-01-05T00:00:00Z",
        validation_start="2025-01-03T00:00:00Z",
        validation_end="2025-01-04T00:00:00Z",
        purge_window="[2025-01-03T00:00:00Z, 2025-01-04T00:00:00Z)",
        embargo_window="[2025-01-04T00:00:00Z, 2025-01-06T00:00:00Z)",
        train_observation_ids=("overlap", "embargo"),
        validation_observation_ids=("validation",),
        n_train_observations=2,
        n_validation_observations=1,
        details={
            "embargo_end": "2025-01-06T00:00:00Z",
            "train_label_intervals": {
                "overlap": {
                    "label_start": "2025-01-02T00:00:00Z",
                    "label_end": "2025-01-03T12:00:00Z",
                }
            },
            "train_timestamps": {
                "embargo": "2025-01-05T00:00:00Z",
                "overlap": "2025-01-01T00:00:00Z",
            },
            "validation_label_intervals": {
                "validation": {
                    "label_start": "2025-01-03T00:00:00Z",
                    "label_end": "2025-01-04T00:00:00Z",
                }
            },
        },
    )
    plan = PurgedSplitPlan("strategy", "run_manual", PurgedSplitConfig(), splits=(split,))

    checks = validate_purged_split_plan(plan)

    assert {check.check_id for check in checks} >= {
        "temporal_validation.purged_interval_overlap",
        "temporal_validation.embargo_violation",
    }


def test_artifact_writer_emits_deterministic_portable_outputs(tmp_path: Path) -> None:
    records = [
        _obs("a", "2025-01-01"),
        _obs("b", "2025-01-02"),
        _obs("c", "2025-01-03"),
        _obs("d", "2025-01-04"),
    ]
    plan = build_purged_split_plan(
        records,
        config=PurgedSplitConfig(n_splits=2, validation_window_size=1, embargo_window="1D"),
        workflow_type="campaign",
        run_id="campaign_run",
    )

    first = write_purged_split_artifacts(plan, output_root=tmp_path / "artifacts" / "purged")
    first_snapshot = {
        path.relative_to(first.output_dir).as_posix(): path.read_bytes()
        for path in sorted(first.output_dir.iterdir())
        if path.is_file()
    }
    second = write_purged_split_artifacts(plan, output_root=tmp_path / "artifacts" / "purged")
    second_snapshot = {
        path.relative_to(second.output_dir).as_posix(): path.read_bytes()
        for path in sorted(second.output_dir.iterdir())
        if path.is_file()
    }

    assert sorted(first_snapshot) == [
        LEAKAGE_VALIDATION_FILENAME,
        PURGED_SPLIT_PLAN_FILENAME,
        PURGED_SPLIT_SUMMARY_FILENAME,
    ]
    assert first_snapshot == second_snapshot

    plan_payload = json.loads(first.purged_split_plan_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(first.leakage_validation_path.read_text(encoding="utf-8"))
    assert plan_payload["split_count"] == 2
    assert validation_payload["overall_status"] == "pass"
    assert first.purged_split_summary_path.read_text(encoding="utf-8").splitlines()[0].startswith("split_id,")
    assert not _contains_absolute_path(plan_payload)
    assert "nan" not in json.dumps(plan_payload).lower()
    assert "inf" not in json.dumps(plan_payload).lower()


def test_findings_are_robustness_finding_compatible() -> None:
    plan = PurgedSplitPlan(
        workflow_type="strategy",
        run_id="run_findings",
        config=PurgedSplitConfig(),
        validation_results=(
            LeakageValidationResult(
                check_id="temporal_validation.embargo_violation",
                status="blocked",
                split_id="split_001",
                message="Embargo violation.",
                details={"observation_id": "obs_001"},
            ),
        ),
    )

    findings = build_purged_split_findings(plan)
    payload = findings[0].to_dict()

    assert payload["check_id"] == "temporal_validation.embargo_violation"
    assert payload["severity"] == "blocked"
    assert payload["details"]["split_id"] == "split_001"


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        return (
            "C:/Users/" in normalized
            or normalized.startswith("file://")
            or normalized.startswith("/Users/")
            or normalized.startswith("/home/")
        )
    return False
