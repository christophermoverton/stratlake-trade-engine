from __future__ import annotations

import csv
import json
from pathlib import Path

from src.research.governance import (
    build_governance_outcome_rows,
    build_governance_summary,
    load_governance_artifacts,
    run_promotion_governance_report,
    validate_governance_consistency,
)
from src.research.registry import append_registry_entry


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _artifact_fixture(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    eligible_dir = artifact_root / "strategy_eligible"
    blocked_dir = artifact_root / "strategy_blocked"

    eligible_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "highest_severity": None,
        "decision_reason_codes": [],
        "gate_count": 1,
        "passed_gate_count": 1,
        "failed_gate_count": 0,
        "missing_gate_count": 0,
        "severity_counts": {"block": 0, "reject": 0, "review": 0, "warn": 0},
    }
    blocked_summary = {
        "promotion_status": "blocked",
        "evaluation_status": "fail",
        "highest_severity": "block",
        "decision_reason_codes": ["gate_failed_threshold", "severity_block"],
        "gate_count": 1,
        "passed_gate_count": 0,
        "failed_gate_count": 1,
        "missing_gate_count": 0,
        "severity_counts": {"block": 1, "reject": 0, "review": 0, "warn": 0},
    }
    _write_json(
        eligible_dir / "manifest.json",
        {
            "run_id": "strategy_eligible",
            "strategy_name": "eligible_strategy",
            "metric_summary": {"effective_n": 40, "p_value": 0.01},
            "promotion_gate_summary": eligible_summary,
        },
    )
    _write_json(
        eligible_dir / "promotion_gates.json",
        {
            **eligible_summary,
            "results": [
                {"gate_id": "min_effective_n", "status": "pass", "reason_codes": ["gate_passed"]},
            ],
        },
    )
    _write_json(
        blocked_dir / "manifest.json",
        {
            "run_id": "strategy_blocked",
            "strategy_name": "blocked_strategy",
            "metric_summary": {"effective_n": 10, "p_value": 0.20},
            "promotion_gate_summary": blocked_summary,
        },
    )
    _write_json(
        blocked_dir / "promotion_gates.json",
        {
            **blocked_summary,
            "results": [
                {
                    "gate_id": "min_effective_n",
                    "status": "fail",
                    "reason_codes": ["gate_failed_threshold", "severity_block"],
                },
            ],
        },
    )
    append_registry_entry(
        registry_path,
        {
            "run_id": "strategy_eligible",
            "run_type": "strategy",
            "strategy_name": "eligible_strategy",
            "artifact_path": eligible_dir.as_posix(),
            "metrics_summary": {"effective_n": 40, "p_value": 0.01},
            "promotion_status": "eligible",
            "review_status": "candidate",
            "promotion_gate_summary": eligible_summary,
        },
    )
    append_registry_entry(
        registry_path,
        {
            "run_id": "strategy_blocked",
            "run_type": "strategy",
            "strategy_name": "blocked_strategy",
            "artifact_path": blocked_dir.as_posix(),
            "metrics_summary": {"effective_n": 10, "p_value": 0.20},
            "promotion_status": "eligible",
            "review_status": "candidate",
            "promotion_gate_summary": blocked_summary,
        },
    )
    return artifact_root


def test_governance_loader_handles_missing_optional_files_gracefully(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    append_registry_entry(
        registry_path,
        {
            "run_id": "missing_optional",
            "run_type": "strategy",
            "artifact_path": (artifact_root / "missing_optional").as_posix(),
            "promotion_status": "eligible",
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert len(dataset.records) == 1
    assert rows[0]["run_id"] == "missing_optional"
    assert validation["counts_by_check"]["missing_promotion_summary"] == 1
    assert validation["counts_by_check"]["missing_or_stale_manifest_link"] == 1


def test_governance_aggregation_and_row_order_are_deterministic(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    dataset = load_governance_artifacts(registry_path=artifact_root / "registry.jsonl", artifact_root=artifact_root)

    first_rows = build_governance_outcome_rows(dataset.records)
    second_rows = build_governance_outcome_rows(list(reversed(dataset.records)))
    summary = build_governance_summary(first_rows)

    assert [row["run_id"] for row in first_rows] == ["strategy_blocked", "strategy_eligible"]
    assert first_rows == second_rows
    assert summary["promotion_status_counts"] == {"blocked": 1, "eligible": 1}
    assert summary["severity_counts"] == {"block": 1, "reject": 0, "review": 0, "warn": 0}
    assert summary["blocked_fraction"] == 0.5


def test_governance_validation_detects_status_and_review_mismatches(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    dataset = load_governance_artifacts(registry_path=artifact_root / "registry.jsonl", artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    validation = validate_governance_consistency(dataset.records, rows)

    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["registry_promotion_status_mismatch"] == 1
    assert validation["counts_by_check"]["review_status_mismatch"] == 1


def test_governance_writer_emits_expected_relative_artifact_bundle(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)

    result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance",
        report_id="demo_report",
    )

    expected_files = {
        "consistency_validation.json",
        "manifest.json",
        "promotion_governance_report.md",
        "promotion_governance_summary.json",
        "promotion_outcome_matrix.csv",
        "reason_code_summary.csv",
        "severity_summary.csv",
        "workflow_summary.csv",
    }
    assert {path.name for path in result.output_dir.iterdir()} == expected_files
    assert not (result.output_dir / "promotion_decision.json").exists()
    assert not (result.output_dir / "promotion_readiness.json").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    summary_text = result.summary_path.read_text(encoding="utf-8")
    validation_text = result.validation_path.read_text(encoding="utf-8")
    assert manifest["artifact_files"] == sorted(expected_files)
    assert not any(str(tmp_path) in payload for payload in (summary_text, validation_text, json.dumps(manifest)))

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["strategy_blocked", "strategy_eligible"]
    assert rows[0]["triggered_gate_names"] == "min_effective_n"
