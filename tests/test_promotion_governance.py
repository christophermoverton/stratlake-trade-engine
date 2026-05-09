from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.cli.run_promotion_governance_report import parse_args
from src.research.governance import (
    OUTCOME_MATRIX_COLUMNS,
    build_governance_report_id,
    build_governance_outcome_rows,
    build_reason_code_summary,
    build_severity_summary,
    build_governance_summary,
    build_workflow_summary,
    load_governance_artifacts,
    normalize_promotion_status,
    normalize_review_status,
    run_promotion_governance_report,
    validate_governance_consistency,
)
from src.research.governance.models import GovernanceSourceRecord
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


def test_governance_loader_handles_missing_registry_consistently(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    dataset = load_governance_artifacts(registry_path=artifact_root / "missing_registry.jsonl", artifact_root=artifact_root)

    assert dataset.records == []
    assert dataset.sources["registry_entry_count"] == 0
    assert dataset.sources["campaign_record_count"] == 0
    assert dataset.sources["campaign_scenario_record_count"] == 0


def test_governance_loader_resolves_promotion_summary_sources(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    manifest_dir = artifact_root / "manifest_only"
    registry_dir = artifact_root / "registry_only"
    gates_dir = artifact_root / "gates_only"
    manifest_summary = {
        "promotion_status": "warn",
        "evaluation_status": "pass",
        "highest_severity": "warn",
        "decision_reason_codes": ["severity_warn"],
        "gate_count": 1,
    }
    registry_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    gates_summary = {
        "promotion_status": "blocked",
        "evaluation_status": "fail",
        "highest_severity": "block",
        "decision_reason_codes": ["severity_block", "gate_failed_threshold"],
        "gate_count": 1,
        "results": [{"gate_id": "min_effective_n", "status": "fail"}],
    }
    _write_json(manifest_dir / "manifest.json", {"run_id": "manifest_only", "promotion_gate_summary": manifest_summary})
    _write_json(gates_dir / "manifest.json", {"run_id": "gates_only"})
    _write_json(gates_dir / "promotion_gates.json", gates_summary)
    for run_id, run_dir, summary in (
        ("manifest_only", manifest_dir, None),
        ("registry_only", registry_dir, registry_summary),
        ("gates_only", gates_dir, None),
    ):
        entry = {
            "run_id": run_id,
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "candidate",
        }
        if summary is not None:
            entry["promotion_gate_summary"] = summary
        append_registry_entry(registry_path, entry)

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    statuses = {row["run_id"]: row["promotion_status"] for row in rows}
    assert statuses == {
        "gates_only": "blocked",
        "manifest_only": "warn",
        "registry_only": "eligible",
    }
    gates_row = next(row for row in rows if row["run_id"] == "gates_only")
    assert gates_row["triggered_gate_names"] == "min_effective_n"


def test_manifest_promotion_summary_precedes_equivalent_registry_and_gate_fields(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "summary_precedence"
    manifest_summary = {
        "promotion_status": "warn",
        "evaluation_status": "pass",
        "highest_severity": "warn",
        "decision_reason_codes": ["severity_warn"],
        "gate_count": 1,
        "results": [{"gate_id": "manifest_gate", "status": "fail"}],
    }
    registry_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
        "results": [{"gate_id": "registry_gate", "status": "fail"}],
    }
    gates_summary = {
        "promotion_status": "blocked",
        "evaluation_status": "fail",
        "highest_severity": "block",
        "decision_reason_codes": ["severity_block"],
        "gate_count": 1,
        "results": [{"gate_id": "promotion_gates_artifact_gate", "status": "fail"}],
    }
    _write_json(run_dir / "manifest.json", {"run_id": "summary_precedence", "promotion_gate_summary": manifest_summary})
    _write_json(run_dir / "promotion_gates.json", gates_summary)
    append_registry_entry(
        registry_path,
        {
            "run_id": "summary_precedence",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "needs_review",
            "promotion_gate_summary": registry_summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "warn"
    assert rows[0]["highest_severity"] == "warn"
    assert rows[0]["decision_reason_codes"] == "severity_warn"
    assert rows[0]["triggered_gate_names"] == "manifest_gate"
    assert validation["counts_by_check"]["registry_promotion_status_mismatch"] == 1
    assert validation["counts_by_check"]["manifest_registry_promotion_summary_mismatch"] == 1
    assert validation["counts_by_check"]["manifest_promotion_gates_summary_mismatch"] == 1


def test_equivalent_promotion_summary_validation_is_status_focused(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "status_only_equivalence"
    manifest_summary = {
        "promotion_status": "needs_review",
        "evaluation_status": "fail",
        "highest_severity": "review",
        "decision_reason_codes": ["manifest_reason"],
        "gate_count": 1,
    }
    registry_summary = {
        "promotion_status": "needs_review",
        "evaluation_status": "fail",
        "highest_severity": "warn",
        "decision_reason_codes": ["registry_reason"],
        "gate_count": 1,
    }
    gates_summary = {
        "promotion_status": "needs_review",
        "evaluation_status": "fail",
        "highest_severity": "block",
        "decision_reason_codes": ["gates_reason"],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "status_only_equivalence", "promotion_gate_summary": manifest_summary})
    _write_json(run_dir / "promotion_gates.json", gates_summary)
    append_registry_entry(
        registry_path,
        {
            "run_id": "status_only_equivalence",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "needs_review",
            "review_status": "needs_review",
            "promotion_gate_summary": registry_summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "needs_review"
    assert rows[0]["highest_severity"] == "review"
    assert rows[0]["decision_reason_codes"] == "manifest_reason"
    assert "manifest_registry_promotion_summary_mismatch" not in validation["counts_by_check"]
    assert "manifest_promotion_gates_summary_mismatch" not in validation["counts_by_check"]


def test_governance_loader_resolves_artifact_paths_aliases_deterministically(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "artifact_path_alias"
    summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "artifact_path_alias", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "artifact_path_alias",
            "run_type": "strategy",
            "artifact_paths": {
                "artifact_dir": run_dir.as_posix(),
                "manifest_path": (run_dir / "manifest.json").as_posix(),
            },
            "promotion_status": "eligible",
            "review_status": "candidate",
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    result = run_promotion_governance_report(
        registry_path=registry_path,
        artifact_root=artifact_root,
        output_dir=tmp_path / "alias_governance",
        report_id="alias_report",
    )

    assert rows[0]["promotion_status"] == "eligible"
    assert rows[0]["manifest_path"].endswith("artifact_path_alias/manifest.json")
    for payload in (
        result.summary_path.read_text(encoding="utf-8"),
        result.validation_path.read_text(encoding="utf-8"),
        result.manifest_path.read_text(encoding="utf-8"),
        result.outcome_matrix_path.read_text(encoding="utf-8"),
    ):
        assert str(tmp_path) not in payload


def test_governance_loader_reads_mixed_separator_relative_registry_paths_on_linux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "mixed" / "nested"
    summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": ["manifest_summary_observed"],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "mixed_separator", "promotion_gate_summary": summary})
    _write_json(
        run_dir / "promotion_gates.json",
        {
            **summary,
            "results": [{"gate_id": "existing_gate_artifact", "status": "fail"}],
        },
    )
    append_registry_entry(
        registry_path,
        {
            "run_id": "mixed_separator",
            "run_type": "strategy",
            "artifact_path": r"mixed\nested",
            "manifest_path": r"mixed\nested/manifest.json",
            "promotion_status": "eligible",
            "review_status": "candidate",
        },
    )
    monkeypatch.chdir(artifact_root)

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=Path("."))
    rows = build_governance_outcome_rows(dataset.records)
    result = run_promotion_governance_report(
        registry_path=registry_path,
        artifact_root=Path("."),
        output_dir=tmp_path / "mixed_separator_governance",
        report_id="mixed_separator_report",
    )

    assert len(dataset.records) == 1
    assert dataset.records[0].manifest is not None
    assert dataset.records[0].promotion_gates is not None
    assert rows[0]["manifest_path"] == "mixed/nested/manifest.json"
    assert rows[0]["promotion_status"] == "eligible"
    assert rows[0]["decision_reason_codes"] == "manifest_summary_observed"
    assert rows[0]["triggered_gate_names"] == "existing_gate_artifact"

    payloads = [
        result.manifest_path.read_text(encoding="utf-8"),
        result.summary_path.read_text(encoding="utf-8"),
        result.validation_path.read_text(encoding="utf-8"),
        result.outcome_matrix_path.read_text(encoding="utf-8"),
    ]
    forbidden = ["\\", "/tmp/", "/home/", "/Users/", "C:/Users/", "C:\\Users\\", "file://"]
    assert not any(token in payload for token in forbidden for payload in payloads)

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["manifest_path"] == "mixed/nested/manifest.json"
    assert row["registry_path"] == "registry.jsonl"


def test_governance_loader_discovers_review_and_candidate_review_contexts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    review_dir = artifact_root / "reviews" / "review_a"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_a"
    review_summary = {
        "promotion_status": "needs_review",
        "evaluation_status": "fail",
        "highest_severity": "review",
        "decision_reason_codes": ["severity_review"],
        "gate_count": 1,
    }
    candidate_summary = {
        "promotion_status": "blocked",
        "evaluation_status": "fail",
        "highest_severity": "block",
        "decision_reason_codes": ["severity_block"],
        "gate_count": 1,
    }
    _write_json(review_dir / "review_summary.json", {"review_id": "review_a"})
    _write_json(review_dir / "manifest.json", {"review_id": "review_a", "promotion_gate_summary": review_summary})
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_a",
            "selected_candidate_id": "cand_a",
            "selected_run_ids": {"strategy_run_ids": ["strategy_eligible"]},
            "upstream_run_ids": ["strategy_eligible"],
            "promotion_context": {"portfolio_promotion_gate_summary": candidate_summary},
        },
    )

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    assert [(row["workflow_type"], row["run_id"], row["promotion_status"]) for row in rows] == [
        ("candidate_review", "candidate_review:candidate_a", "blocked"),
        ("review", "review_a", "needs_review"),
    ]
    candidate_row = rows[0]
    assert candidate_row["candidate_selection_run_id"] == "candidate_a"
    assert candidate_row["selected_candidate_id"] == "cand_a"
    assert candidate_row["selected_run_id"] == "strategy_eligible"
    assert candidate_row["upstream_run_ids"] == "strategy_eligible"


def test_governance_loader_extracts_candidate_review_visibility_metadata(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    strategy_dir = artifact_root / "strategy_selected"
    candidate_dir = artifact_root / "candidate_review" / "candidate_visibility"
    promotion_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    append_registry_entry(
        artifact_root / "registry.jsonl",
        {
            "run_id": "strategy_selected",
            "run_type": "strategy",
            "artifact_path": strategy_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "candidate",
            "promotion_gate_summary": promotion_summary,
        },
    )
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_visibility",
            "selected_candidate_ids": ["cand_b", "cand_a"],
            "selected_run_ids": {"strategy_run_ids": ["strategy_selected"]},
            "portfolio_run_id": "portfolio_selected",
            "strategy_run_id": "strategy_selected",
            "alpha_run_id": "alpha_selected",
            "promotion_context": {
                "candidate_promotion_status_counts": {"eligible": 1},
                "portfolio_promotion_gate_summary": promotion_summary,
            },
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "run_type": "candidate_selection_review",
            "candidate_selection_run_id": "candidate_visibility",
            "selected_candidate_id": "cand_a",
            "upstream_run_ids": ["strategy_selected"],
            "promotion_gate_summary": promotion_summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=artifact_root / "registry.jsonl", artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    summary = build_governance_summary(rows)
    candidate_row = next(row for row in rows if row["workflow_type"] == "candidate_review")
    candidate_record = next(record for record in dataset.records if record.workflow_type == "candidate_review")

    assert candidate_row["candidate_id"] == "cand_a"
    assert candidate_row["candidate_selection_run_id"] == "candidate_visibility"
    assert candidate_row["selected_candidate_id"] == "cand_a"
    assert candidate_row["selected_run_id"] == "strategy_selected"
    assert candidate_row["upstream_run_ids"] == "alpha_selected|portfolio_selected|strategy_selected"
    assert candidate_record.governance_metadata["portfolio_run_id"] == "portfolio_selected"
    assert candidate_record.governance_metadata["strategy_run_id"] == "strategy_selected"
    assert candidate_record.governance_metadata["alpha_run_id"] == "alpha_selected"
    assert candidate_record.governance_metadata["candidate_promotion_status_counts"] == {"eligible": 1}
    assert candidate_record.governance_metadata["promotion_context_present"] is True
    assert summary["candidate_review_count"] == 1
    assert summary["candidate_selection_count"] == 1
    assert summary["selected_candidate_count"] == 1
    assert summary["candidate_status_counts"] == {"eligible": 1}


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


def test_governance_aggregation_summaries_are_stable_for_reason_severity_and_workflow() -> None:
    rows = [
        {
            "run_id": "z",
            "workflow_type": "strategy",
            "promotion_status": "blocked",
            "highest_severity": "block",
            "decision_reason_codes": "severity_block|gate_failed_threshold",
        },
        {
            "run_id": "a",
            "workflow_type": "campaign_scenario",
            "promotion_status": "needs_review",
            "highest_severity": "review",
            "decision_reason_codes": "severity_review|gate_failed_threshold",
        },
        {
            "run_id": "b",
            "workflow_type": "strategy",
            "promotion_status": "eligible",
            "highest_severity": "",
            "decision_reason_codes": "",
        },
    ]

    assert build_reason_code_summary(rows) == [
        {"reason_code": "gate_failed_threshold", "count": 2},
        {"reason_code": "severity_block", "count": 1},
        {"reason_code": "severity_review", "count": 1},
    ]
    assert build_severity_summary(rows) == [
        {"severity": "warn", "highest_severity_count": 0, "triggered_reason_count": 0},
        {"severity": "review", "highest_severity_count": 1, "triggered_reason_count": 1},
        {"severity": "reject", "highest_severity_count": 0, "triggered_reason_count": 0},
        {"severity": "block", "highest_severity_count": 1, "triggered_reason_count": 1},
    ]
    assert build_workflow_summary(rows) == [
        {
            "workflow_type": "campaign_scenario",
            "row_count": 1,
            "eligible_count": 0,
            "blocked_count": 0,
            "needs_review_count": 1,
            "rejected_count": 0,
        },
        {
            "workflow_type": "strategy",
            "row_count": 2,
            "eligible_count": 1,
            "blocked_count": 1,
            "needs_review_count": 0,
            "rejected_count": 0,
        },
    ]


def test_governance_rows_omit_missing_and_non_finite_numeric_metrics() -> None:
    record = GovernanceSourceRecord(
        run_id="numeric_case",
        workflow_type="strategy",
        registry_entry={
            "run_id": "numeric_case",
            "metrics_summary": {
                "effective_n": "not-a-number",
                "p_value": float("nan"),
                "hit_rate_p_value": float("inf"),
                "sharpe_stability_ratio": None,
            },
        },
        promotion_gate_summary={"promotion_status": "eligible", "decision_reason_codes": []},
        governance_metadata={"campaign_status": "completed", "scenario_status": "reused"},
    )

    rows = build_governance_outcome_rows([record])

    assert rows[0]["effective_n"] == ""
    assert rows[0]["p_value"] == ""
    assert rows[0]["hit_rate_p_value"] == ""
    assert rows[0]["sharpe_stability_ratio"] == ""
    assert rows[0]["campaign_status"] == "completed"
    assert rows[0]["scenario_status"] == "reused"
    assert "campaign_status" in OUTCOME_MATRIX_COLUMNS
    assert "scenario_status" in OUTCOME_MATRIX_COLUMNS


def test_governance_report_id_is_stable_for_identical_logical_rows(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    dataset = load_governance_artifacts(registry_path=artifact_root / "registry.jsonl", artifact_root=artifact_root)

    first_rows = build_governance_outcome_rows(dataset.records)
    second_rows = build_governance_outcome_rows(list(reversed(dataset.records)))

    assert build_governance_report_id(first_rows) == build_governance_report_id(second_rows)


def test_governance_validation_detects_status_and_review_mismatches(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    dataset = load_governance_artifacts(registry_path=artifact_root / "registry.jsonl", artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    validation = validate_governance_consistency(dataset.records, rows)

    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["registry_promotion_status_mismatch"] == 1
    assert validation["counts_by_check"]["review_status_mismatch"] == 1


def test_governance_validation_finding_order_and_non_relative_path_detection() -> None:
    record = GovernanceSourceRecord(
        run_id="path_case",
        workflow_type="strategy",
        registry_entry={"promotion_status": "mystery", "review_status": "approved"},
        promotion_gate_summary={"promotion_status": "mystery", "gate_count": 1},
    )
    rows = [
        {
            "run_id": "path_case",
            "workflow_type": "strategy",
            "promotion_status": "mystery",
            "review_status": "approved",
            "registry_path": "C:/absolute/registry.jsonl",
            "manifest_path": "C:/absolute/manifest.json",
        }
    ]

    first_validation = validate_governance_consistency([record], rows)
    second_validation = validate_governance_consistency([record], rows)

    assert first_validation == second_validation
    assert first_validation["status"] == "fail"
    assert first_validation["counts_by_check"]["non_relative_artifact_path"] == 2
    assert first_validation["counts_by_check"]["unknown_promotion_status"] == 1
    assert first_validation["counts_by_check"]["unknown_review_status"] == 1
    assert first_validation["findings"] == sorted(
        first_validation["findings"],
        key=lambda item: (item["severity"], item["check_id"], item["run_id"]),
    )


def test_candidate_review_context_mismatch_is_reported(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_mismatch"
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_mismatch",
            "promotion_context": {
                "portfolio_promotion_gate_summary": {
                    "promotion_status": "blocked",
                    "evaluation_status": "fail",
                    "gate_count": 1,
                }
            },
        },
    )

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    record = dataset.records[0]
    rows = build_governance_outcome_rows(
        [
            GovernanceSourceRecord(
                run_id=record.run_id,
                workflow_type=record.workflow_type,
                promotion_gate_summary={"promotion_status": "eligible", "gate_count": 1},
                candidate_review_summary=record.candidate_review_summary,
            )
        ]
    )
    validation = validate_governance_consistency(
        [
            GovernanceSourceRecord(
                run_id=record.run_id,
                workflow_type=record.workflow_type,
                promotion_gate_summary={"promotion_status": "eligible", "gate_count": 1},
                candidate_review_summary=record.candidate_review_summary,
            )
        ],
        rows,
    )

    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["candidate_review_context_mismatch"] == 1


def test_candidate_review_validation_reports_missing_context_duplicate_and_missing_upstream(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_warning"
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_warning",
            "selected_run_ids": {"strategy_run_ids": ["missing_strategy", "missing_strategy"]},
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "candidate_selection_run_id": "candidate_warning",
            "selected_run_ids": {"strategy_run_ids": ["missing_strategy"]},
        },
    )

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["candidate_selection_run_id"] == "candidate_warning"
    assert rows[0]["selected_run_id"] == "missing_strategy"
    assert rows[0]["upstream_run_ids"] == "missing_strategy"
    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["missing_promotion_summary"] == 1
    assert validation["counts_by_check"]["candidate_review_missing_promotion_context"] == 1
    assert validation["counts_by_check"]["candidate_review_duplicate_upstream_run_ids"] == 1
    assert validation["counts_by_check"]["candidate_review_missing_upstream_run_reference"] == 1
    assert validation["counts_by_check"]["candidate_review_missing_selected_candidate_id"] == 1


def test_candidate_review_validation_reports_unknown_context_and_manifest_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_error"
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_error",
            "selected_candidate_id": "cand_error",
            "promotion_context": {
                "portfolio_promotion_gate_summary": {
                    "promotion_status": "mystery",
                    "evaluation_status": "fail",
                    "gate_count": 1,
                }
            },
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "candidate_selection_run_id": "different_candidate_error",
        },
    )

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "mystery"
    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["unknown_promotion_status"] == 1
    assert validation["counts_by_check"]["candidate_review_context_unknown_promotion_status"] == 1
    assert validation["counts_by_check"]["candidate_review_manifest_run_id_mismatch"] == 1


def test_candidate_review_selected_and_upstream_ids_validate_summary_manifest_conflicts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    registry_path = artifact_root / "registry.jsonl"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_conflict"
    promotion_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    for run_id in ("strategy_a", "strategy_b"):
        append_registry_entry(
            registry_path,
            {
                "run_id": run_id,
                "run_type": "strategy",
                "promotion_status": "eligible",
                "review_status": "candidate",
                "promotion_gate_summary": promotion_summary,
            },
        )
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_conflict",
            "selected_candidate_id": "cand_a",
            "selected_run_id": "strategy_a",
            "promotion_context": {
                "portfolio_promotion_gate_summary": promotion_summary,
            },
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "candidate_selection_run_id": "candidate_conflict",
            "selected_candidate_id": "cand_b",
            "selected_run_ids": ["strategy_b"],
            "promotion_gate_summary": promotion_summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    candidate_row = next(row for row in rows if row["workflow_type"] == "candidate_review")
    validation = validate_governance_consistency(dataset.records, rows)

    assert candidate_row["selected_candidate_id"] == "cand_a"
    assert candidate_row["selected_run_id"] == "strategy_a"
    assert candidate_row["upstream_run_ids"] == "strategy_a|strategy_b"
    assert validation["counts_by_check"]["candidate_review_selected_candidate_id_mismatch"] == 1
    assert validation["counts_by_check"]["candidate_review_selected_run_id_mismatch"] == 1


def test_candidate_review_required_evidence_warns_while_optional_evidence_is_ignored(tmp_path: Path) -> None:
    required_path = tmp_path / "missing_required" / "candidate_review_summary.json"
    optional_path = tmp_path / "missing_optional" / "candidate_explainability.json"
    record = GovernanceSourceRecord(
        run_id="candidate_review:evidence_case",
        workflow_type="candidate_review",
        promotion_gate_summary={"promotion_status": "eligible", "gate_count": 1},
        governance_metadata={
            "promotion_context_present": True,
            "artifact_evidence_paths": {
                "candidate_review_summary_path": required_path.as_posix(),
            },
            "optional_artifact_evidence_paths": {
                "candidate_explainability_path": optional_path.as_posix(),
            },
        },
    )
    rows = build_governance_outcome_rows([record])

    validation = validate_governance_consistency([record], rows)

    assert validation["status"] == "fail"
    assert validation["counts_by_check"] == {"candidate_review_stale_artifact_evidence_path": 1}
    finding = validation["findings"][0]
    assert finding["details"] == {
        "field": "candidate_review_summary_path",
        "path": "candidate_review_summary.json",
    }
    assert str(tmp_path) not in json.dumps(validation)
    assert optional_path.name not in json.dumps(validation)


def test_candidate_review_loader_marks_required_and_optional_evidence_separately(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    candidate_dir = artifact_root / "candidate_contexts" / "candidate_evidence"
    promotion_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "candidate_evidence",
            "selected_candidate_id": "cand_evidence",
            "promotion_context": {
                "portfolio_promotion_gate_summary": promotion_summary,
            },
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "candidate_selection_run_id": "candidate_evidence",
            "promotion_gate_summary": promotion_summary,
        },
    )

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    candidate_record = next(record for record in dataset.records if record.workflow_type == "candidate_review")

    assert sorted(candidate_record.governance_metadata["artifact_evidence_paths"]) == [
        "artifact_dir",
        "candidate_review_summary_path",
        "manifest_path",
    ]
    assert candidate_record.governance_metadata["optional_artifact_evidence_paths"] == {}
    assert validate_governance_consistency(dataset.records, build_governance_outcome_rows(dataset.records))["status"] == "pass"


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
    assert manifest["artifact_groups"]["core"] == sorted(expected_files)
    assert manifest["artifact_groups"]["governance"] == sorted(expected_files)
    assert manifest["artifact_groups"]["validation"] == ["consistency_validation.json"]
    assert all(artifact["path"] == filename for filename, artifact in manifest["artifacts"].items())
    assert not any(str(tmp_path) in payload for payload in (summary_text, validation_text, json.dumps(manifest)))
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTCOME_MATRIX_COLUMNS
        rows = list(reader)
    assert [row["run_id"] for row in rows] == ["strategy_blocked", "strategy_eligible"]
    assert rows[0]["triggered_gate_names"] == "min_effective_n"


def test_governance_writer_outputs_are_deterministic_across_reruns(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    output_root = tmp_path / "governance"

    first_result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=output_root,
    )
    first_payloads = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(first_result.output_dir.iterdir(), key=lambda item: item.name)
        if path.suffix in {".csv", ".json", ".md"}
    }
    second_result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=output_root,
    )
    second_payloads = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(second_result.output_dir.iterdir(), key=lambda item: item.name)
        if path.suffix in {".csv", ".json", ".md"}
    }

    assert first_result.report_id == second_result.report_id
    assert first_payloads == second_payloads


def test_strict_validation_writes_artifacts_before_raising(tmp_path: Path) -> None:
    artifact_root = _artifact_fixture(tmp_path)
    output_root = tmp_path / "strict_governance"
    report_dir = output_root / "strict_report"

    with pytest.raises(ValueError, match="validation failed"):
        run_promotion_governance_report(
            registry_path=artifact_root / "registry.jsonl",
            artifact_root=artifact_root,
            output_dir=output_root,
            report_id="strict_report",
            strict_validation=True,
        )

    assert (report_dir / "consistency_validation.json").exists()
    assert (report_dir / "promotion_governance_summary.json").exists()
    assert (report_dir / "promotion_outcome_matrix.csv").exists()
    validation = json.loads((report_dir / "consistency_validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "fail"


def test_status_normalization_helpers_are_deterministic() -> None:
    for status in ("eligible", "warn", "needs_review", "rejected", "blocked"):
        assert normalize_promotion_status(status) == status
        assert normalize_promotion_status(status.upper()) == status
    for alias in ("review", "needs work", "needs_work", "needs-work"):
        assert normalize_promotion_status(alias) == "needs_review"
        assert normalize_review_status(alias) == "needs_review"
    for status in ("candidate", "needs_review", "rejected"):
        assert normalize_review_status(status.upper()) == status
    assert normalize_promotion_status("candidate") == "candidate"
    assert normalize_promotion_status("mystery") == "mystery"
    assert normalize_review_status("approved") == "approved"


def test_legacy_status_aliases_normalize_without_changing_canonical_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "legacy_alias"
    summary = {
        "promotion_status": "review",
        "evaluation_status": "fail",
        "highest_severity": "review",
        "decision_reason_codes": ["severity_review"],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "legacy_alias", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "legacy_alias",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "needs work",
            "review_status": "needs_work",
            "promotion_gate_summary": summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "needs_review"
    assert rows[0]["review_status"] == "needs_review"
    assert validation["status"] == "pass"
    assert validation["counts_by_check"]["legacy_status_normalized"] >= 3
    assert all(finding["severity"] == "info" for finding in validation["findings"])


def test_dash_space_and_underscore_status_aliases_normalize_before_comparison(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "alias_variants"
    summary = {
        "promotion_status": "needs-work",
        "evaluation_status": "fail",
        "highest_severity": "review",
        "decision_reason_codes": ["severity_review"],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "alias_variants", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "alias_variants",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "needs work",
            "review_status": "needs_work",
            "promotion_gate_summary": summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "needs_review"
    assert rows[0]["review_status"] == "needs_review"
    assert validation["status"] == "pass"
    assert validation["counts_by_check"]["legacy_status_normalized"] == 3
    assert {
        finding["details"]["field"]
        for finding in validation["findings"]
        if finding["check_id"] == "legacy_status_normalized"
    } == {
        "registry.promotion_status",
        "promotion_gate_summary.promotion_status",
        "review_status",
    }


def test_unknown_review_status_fails_validation_after_normalization(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "unknown_review_status"
    summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "unknown_review_status", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "unknown_review_status",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "eligible",
            "review_status": "approved",
            "promotion_gate_summary": summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["review_status"] == "approved"
    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["unknown_review_status"] == 1
    assert validation["counts_by_check"]["review_status_mismatch"] == 1


def test_candidate_promotion_status_is_not_silently_mapped_to_eligible(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "candidate_promotion_status"
    summary = {
        "promotion_status": "candidate",
        "evaluation_status": "pass",
        "decision_reason_codes": [],
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "candidate_promotion_status", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "candidate_promotion_status",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "candidate",
            "review_status": "candidate",
            "promotion_gate_summary": summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "candidate"
    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["unknown_promotion_status"] == 1


def test_unknown_statuses_still_produce_validation_findings(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts" / "strategies"
    registry_path = artifact_root / "registry.jsonl"
    run_dir = artifact_root / "unknown_status"
    summary = {
        "promotion_status": "mystery",
        "evaluation_status": "fail",
        "gate_count": 1,
    }
    _write_json(run_dir / "manifest.json", {"run_id": "unknown_status", "promotion_gate_summary": summary})
    append_registry_entry(
        registry_path,
        {
            "run_id": "unknown_status",
            "run_type": "strategy",
            "artifact_path": run_dir.as_posix(),
            "promotion_status": "mystery",
            "review_status": "candidate",
            "promotion_gate_summary": summary,
        },
    )

    dataset = load_governance_artifacts(registry_path=registry_path, artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert rows[0]["promotion_status"] == "mystery"
    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["unknown_promotion_status"] == 1


def test_path_sanitization_handles_windows_mixed_and_external_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "input_artifacts"
    registry_path = artifact_root / "registry.jsonl"
    promotion_summary = {
        "promotion_status": "eligible",
        "evaluation_status": "pass",
        "gate_count": 1,
        "decision_reason_codes": [],
    }
    raw_paths = [
        r"C:\external\strategy_win_abs",
        r"artifacts\strategies/mixed_slashes",
        (tmp_path.parent / "outside_artifacts" / "external_run").as_posix(),
    ]
    for index, raw_path in enumerate(raw_paths):
        append_registry_entry(
            registry_path,
            {
                "run_id": f"path_case_{index}",
                "run_type": "strategy",
                "artifact_path": raw_path,
                "promotion_status": "eligible",
                "review_status": "candidate",
                "promotion_gate_summary": promotion_summary,
            },
        )

    result = run_promotion_governance_report(
        registry_path=registry_path,
        artifact_root=artifact_root,
        output_dir=tmp_path / "path_governance",
        report_id="path_report",
    )

    payloads = [
        result.manifest_path.read_text(encoding="utf-8"),
        result.summary_path.read_text(encoding="utf-8"),
        result.validation_path.read_text(encoding="utf-8"),
        result.outcome_matrix_path.read_text(encoding="utf-8"),
    ]
    forbidden = [str(tmp_path), tmp_path.as_posix(), "C:\\", "C:/"]
    assert not any(token in payload for token in forbidden for payload in payloads)

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        assert not Path(row["registry_path"]).is_absolute()
        assert not Path(row["manifest_path"]).is_absolute()
        assert "\\" not in row["registry_path"]
        assert "\\" not in row["manifest_path"]


def test_default_output_root_is_canonical_even_with_custom_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_artifact_root = _artifact_fixture(tmp_path / "custom")
    monkeypatch.chdir(tmp_path)

    result = run_promotion_governance_report(
        registry_path=custom_artifact_root / "registry.jsonl",
        artifact_root=custom_artifact_root,
        report_id="default_root_report",
    )

    assert result.output_dir == Path("artifacts") / "promotion_governance" / "default_root_report"
    assert result.manifest_path.exists()


def test_promotion_governance_cli_parse_args_defaults_and_strict_flag() -> None:
    default_args = parse_args([])
    strict_args = parse_args(
        [
            "--registry-path",
            "custom_registry.jsonl",
            "--artifact-root",
            "custom_artifacts",
            "--output-dir",
            "governance_output",
            "--report-id",
            "custom_report",
            "--strict-validation",
        ]
    )

    assert default_args.registry_path is None
    assert default_args.artifact_root == "artifacts"
    assert default_args.output_dir is None
    assert default_args.report_id is None
    assert default_args.strict_validation is False
    assert strict_args.registry_path == "custom_registry.jsonl"
    assert strict_args.artifact_root == "custom_artifacts"
    assert strict_args.output_dir == "governance_output"
    assert strict_args.report_id == "custom_report"
    assert strict_args.strict_validation is True


def _campaign_fixture(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "artifacts"
    campaign_dir = artifact_root / "research_campaigns" / "campaign_orchestration"
    scenario_a_dir = campaign_dir / "scenarios" / "scenario_a"
    campaign_summary = {
        "run_type": "research_campaign_orchestration",
        "orchestration_run_id": "campaign_orchestration",
        "status": "completed",
        "scenario_count": 2,
        "scenario_status_counts": {"completed": 2},
        "final_outcomes": {
            "review_promotion_status": "needs_review",
            "review_promotion_gate_status": "fail",
            "review_promotion_highest_severity": "review",
            "review_promotion_decision_reason_codes": ["severity_review"],
            "review_promotion_gate_summary": {
                "promotion_status": "needs_review",
                "evaluation_status": "fail",
                "highest_severity": "review",
                "decision_reason_codes": ["severity_review"],
                "gate_count": 2,
            },
        },
        "scenarios": [
            {
                "scenario_id": "scenario_a",
                "description": "blocked scenario",
                "status": "completed",
                "campaign_run_id": "campaign_a",
                "selected_run_ids": {"strategy_run_ids": ["strategy_blocked"]},
                "final_outcomes": {
                    "review_promotion_gate_summary": {
                        "promotion_status": "blocked",
                        "evaluation_status": "fail",
                        "highest_severity": "block",
                        "decision_reason_codes": ["severity_block", "gate_failed_threshold"],
                        "gate_count": 1,
                    }
                },
            },
            {
                "scenario_id": "scenario_missing",
                "description": "missing scenario artifacts",
                "status": "completed",
                "campaign_run_id": "campaign_missing",
                "selected_run_ids": {"strategy_run_ids": ["strategy_missing"]},
                "final_outcomes": {
                    "review_promotion_gate_summary": {
                        "promotion_status": "needs_review",
                        "evaluation_status": "fail",
                        "highest_severity": "review",
                        "decision_reason_codes": ["severity_review"],
                        "gate_count": 1,
                    }
                },
            },
        ],
    }
    scenario_catalog = {
        "scenario_count": 2,
        "scenarios": [
            {"scenario_id": "scenario_a", "description": "blocked scenario"},
            {"scenario_id": "scenario_missing", "description": "missing scenario artifacts"},
        ],
    }
    _write_json(campaign_dir / "summary.json", campaign_summary)
    _write_json(
        campaign_dir / "manifest.json",
        {
            "run_type": "research_campaign_orchestration",
            "orchestration_run_id": "campaign_orchestration",
            "artifact_files": [
                "summary.json",
                "manifest.json",
                "scenario_catalog.json",
                "scenarios/scenario_a/summary.json",
                "scenarios/scenario_a/manifest.json",
            ],
        },
    )
    _write_json(campaign_dir / "scenario_catalog.json", scenario_catalog)
    _write_json(campaign_dir / "checkpoint.json", {"stage_states": {"review": "completed"}})
    scenario_summary = {
        "run_type": "research_campaign",
        "campaign_run_id": "campaign_a",
        "status": "completed",
        "scenario": {
            "orchestration_run_id": "campaign_orchestration",
            "scenario_id": "scenario_a",
            "description": "blocked scenario",
        },
        "selected_run_ids": {"strategy_run_ids": ["strategy_blocked"]},
        "output_paths": {"strategy_artifact": (tmp_path / "missing" / "strategy_blocked").as_posix()},
        "final_outcomes": {
            "review_promotion_gate_summary": {
                "promotion_status": "blocked",
                "evaluation_status": "fail",
                "highest_severity": "block",
                "decision_reason_codes": ["severity_block", "gate_failed_threshold"],
                "gate_count": 1,
            }
        },
    }
    _write_json(scenario_a_dir / "summary.json", scenario_summary)
    _write_json(
        scenario_a_dir / "manifest.json",
        {
            "run_type": "research_campaign",
            "campaign_run_id": "campaign_a",
            "promotion_gate_summary": {
                "promotion_status": "blocked",
                "evaluation_status": "fail",
                "highest_severity": "block",
                "decision_reason_codes": ["severity_block", "gate_failed_threshold"],
                "gate_count": 1,
            },
        },
    )
    _write_json(scenario_a_dir / "checkpoint.json", {"stage_states": {"review": "completed"}})
    return artifact_root


def test_campaign_artifacts_are_discovered_and_normalized_into_governance_rows(tmp_path: Path) -> None:
    artifact_root = _campaign_fixture(tmp_path)

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    campaign_rows = [row for row in rows if row["workflow_type"] == "campaign"]
    scenario_rows = [row for row in rows if row["workflow_type"] == "campaign_scenario"]
    assert [row["run_id"] for row in campaign_rows] == ["campaign_orchestration"]
    assert [row["scenario_id"] for row in scenario_rows] == ["scenario_a", "scenario_missing"]
    assert campaign_rows[0]["campaign_status"] == "completed"
    assert scenario_rows[0]["campaign_status"] == "completed"
    assert scenario_rows[0]["scenario_status"] == "completed"
    assert campaign_rows[0]["promotion_status"] == "needs_review"
    assert campaign_rows[0]["highest_severity"] == "review"
    assert campaign_rows[0]["decision_reason_codes"] == "severity_review"
    assert scenario_rows[0]["promotion_status"] == "blocked"
    assert scenario_rows[0]["highest_severity"] == "block"
    assert scenario_rows[0]["decision_reason_codes"] == "gate_failed_threshold|severity_block"
    scenario_record = next(
        record
        for record in dataset.records
        if record.workflow_type == "campaign_scenario" and record.governance_metadata["scenario_id"] == "scenario_a"
    )
    assert scenario_record.governance_metadata["scenario_manifest_path"].endswith(
        "research_campaigns/campaign_orchestration/scenarios/scenario_a/manifest.json"
    )
    assert scenario_record.governance_metadata["campaign_manifest_path"].endswith(
        "research_campaigns/campaign_orchestration/manifest.json"
    )
    assert scenario_record.governance_metadata["scenario_manifest_path"] != scenario_record.governance_metadata["campaign_manifest_path"]


def test_campaign_validation_detects_rollup_and_missing_scenario_artifact_issues(tmp_path: Path) -> None:
    artifact_root = _campaign_fixture(tmp_path)
    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    validation = validate_governance_consistency(dataset.records, rows)

    assert validation["status"] == "fail"
    assert validation["counts_by_check"]["campaign_highest_severity_mismatch"] == 1
    assert validation["counts_by_check"]["campaign_missing_scenario_reason_codes"] == 1
    assert validation["counts_by_check"]["scenario_catalog_missing_scenario_dir"] == 1
    assert validation["counts_by_check"]["checkpoint_completed_scenario_missing_summary"] == 1
    assert validation["counts_by_check"]["checkpoint_completed_scenario_missing_manifest"] == 1
    assert validation["counts_by_check"]["scenario_summary_missing_child_artifacts"] == 1


def test_campaign_validation_detects_scenario_manifest_promotion_status_mismatch(tmp_path: Path) -> None:
    artifact_root = _campaign_fixture(tmp_path)
    scenario_manifest_path = (
        artifact_root
        / "research_campaigns"
        / "campaign_orchestration"
        / "scenarios"
        / "scenario_a"
        / "manifest.json"
    )
    scenario_manifest = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
    scenario_manifest["promotion_gate_summary"]["promotion_status"] = "eligible"
    _write_json(scenario_manifest_path, scenario_manifest)
    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)

    validation = validate_governance_consistency(dataset.records, rows)

    assert validation["counts_by_check"]["scenario_promotion_status_mismatch"] == 1


def test_campaign_validation_detects_campaign_and_scenario_id_mismatches(tmp_path: Path) -> None:
    artifact_root = _campaign_fixture(tmp_path)
    campaign_manifest_path = artifact_root / "research_campaigns" / "campaign_orchestration" / "manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_manifest["orchestration_run_id"] = "different_campaign"
    _write_json(campaign_manifest_path, campaign_manifest)
    scenario_manifest_path = (
        artifact_root
        / "research_campaigns"
        / "campaign_orchestration"
        / "scenarios"
        / "scenario_a"
        / "manifest.json"
    )
    scenario_manifest = json.loads(scenario_manifest_path.read_text(encoding="utf-8"))
    scenario_manifest["scenario_id"] = "different_scenario"
    _write_json(scenario_manifest_path, scenario_manifest)

    dataset = load_governance_artifacts(artifact_root=artifact_root)
    rows = build_governance_outcome_rows(dataset.records)
    validation = validate_governance_consistency(dataset.records, rows)

    assert validation["counts_by_check"]["campaign_id_mismatch"] == 1
    assert validation["counts_by_check"]["scenario_id_mismatch"] == 1


def test_campaign_records_are_written_to_existing_governance_outputs(tmp_path: Path) -> None:
    artifact_root = _campaign_fixture(tmp_path)

    result = run_promotion_governance_report(
        artifact_root=artifact_root,
        output_dir=tmp_path / "campaign_governance",
        report_id="campaign_report",
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    summary_text = result.summary_path.read_text(encoding="utf-8")
    validation_text = result.validation_path.read_text(encoding="utf-8")
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    matrix_text = result.outcome_matrix_path.read_text(encoding="utf-8")
    assert summary["campaign_count"] == 1
    assert summary["campaign_scenario_count"] == 2
    assert summary["workflow_type_counts"]["campaign"] == 1
    assert summary["workflow_type_counts"]["campaign_scenario"] == 2
    assert "campaign_governance_summary.csv" not in manifest_text
    assert "promotion_decision" not in manifest_text
    assert "promotion_readiness" not in manifest_text
    assert not any(str(tmp_path) in payload for payload in [summary_text, validation_text, manifest_text, matrix_text])

    with result.outcome_matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "campaign_status" in (reader.fieldnames or [])
        assert "scenario_status" in (reader.fieldnames or [])
        rows = list(reader)
    assert [row["workflow_type"] for row in rows] == ["campaign", "campaign_scenario", "campaign_scenario"]
    assert [row["run_id"] for row in rows] == [
        "campaign_orchestration",
        "campaign_orchestration:scenario_a",
        "campaign_orchestration:scenario_missing",
    ]
    assert rows[0]["campaign_status"] == "completed"
    assert rows[1]["campaign_status"] == "completed"
    assert rows[1]["scenario_status"] == "completed"


def test_campaign_loader_gracefully_handles_artifact_root_without_campaigns(tmp_path: Path) -> None:
    artifact_root = tmp_path / "empty_artifacts"
    artifact_root.mkdir()

    dataset = load_governance_artifacts(artifact_root=artifact_root)

    assert dataset.records == []
    assert dataset.sources["campaign_record_count"] == 0
    assert dataset.sources["campaign_scenario_record_count"] == 0
