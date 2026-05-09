from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from src.research.governance import run_promotion_governance_report
from src.research.promotion import evaluate_promotion_gates, write_promotion_gate_artifact
from src.research.registry import append_registry_entry, build_review_metadata, canonicalize_value


EXPECTED_GOVERNANCE_FILES = {
    "consistency_validation.json",
    "manifest.json",
    "promotion_governance_report.md",
    "promotion_governance_summary.json",
    "promotion_outcome_matrix.csv",
    "reason_code_summary.csv",
    "severity_summary.csv",
    "workflow_summary.csv",
}

M31_SCENARIO_METRICS: dict[str, dict[str, float]] = {
    "eligible": {
        "effective_n": 120.0,
        "p_value": 0.01,
        "hit_rate_p_value": 0.02,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.40,
        "sharpe_stability_ratio": 1.35,
    },
    "warn": {
        "effective_n": 120.0,
        "p_value": 0.01,
        "hit_rate_p_value": 0.02,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.01,
        "sharpe_stability_ratio": 0.70,
    },
    "needs_review": {
        "effective_n": 120.0,
        "p_value": 0.12,
        "hit_rate_p_value": 0.20,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.40,
        "sharpe_stability_ratio": 1.35,
    },
    "blocked": {
        "effective_n": 12.0,
        "p_value": 0.12,
        "hit_rate_p_value": 0.20,
        "autocorr_lag1": 0.05,
        "split_mean_diff_p": 0.01,
        "sharpe_stability_ratio": 0.70,
    },
}


def test_m31_readiness_style_governance_report_end_to_end_is_deterministic(tmp_path: Path) -> None:
    artifact_root = _m31_readiness_governance_fixture(tmp_path)
    first_result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance",
    )
    first_payloads = _report_payloads(first_result.output_dir)
    second_result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance",
    )
    second_payloads = _report_payloads(second_result.output_dir)

    assert first_result.report_id == second_result.report_id
    assert first_payloads == second_payloads
    assert set(first_result.output_dir.iterdir()) == set(second_result.output_dir.iterdir())
    assert {path.name for path in first_result.output_dir.iterdir()} == EXPECTED_GOVERNANCE_FILES
    assert not (first_result.output_dir / "promotion_decision.json").exists()
    assert not (first_result.output_dir / "promotion_readiness.json").exists()
    assert not (first_result.output_dir / "campaign_governance_summary.csv").exists()

    summary = json.loads(first_result.summary_path.read_text(encoding="utf-8"))
    validation = json.loads(first_result.validation_path.read_text(encoding="utf-8"))
    manifest = json.loads(first_result.manifest_path.read_text(encoding="utf-8"))
    outcome_rows = _csv_rows(first_result.outcome_matrix_path)
    reason_rows = _csv_rows(first_result.reason_code_summary_path)
    severity_rows = _csv_rows(first_result.severity_summary_path)
    workflow_rows = _csv_rows(first_result.workflow_summary_path)

    assert summary["row_count"] == 11
    assert summary["promotion_status_counts"] == {
        "blocked": 3,
        "eligible": 2,
        "needs_review": 3,
        "warn": 3,
    }
    assert summary["workflow_type_counts"] == {
        "campaign": 1,
        "campaign_scenario": 4,
        "candidate_review": 1,
        "review": 1,
        "strategy": 4,
    }
    assert summary["campaign_status_counts"] == {"completed": 1}
    assert summary["scenario_status_counts"] == {"completed": 4}
    assert validation["status"] == "pass"
    assert validation["finding_count"] == 0
    assert manifest["artifact_files"] == sorted(EXPECTED_GOVERNANCE_FILES)
    assert manifest["artifact_groups"]["validation"] == ["consistency_validation.json"]

    rows_by_key = {(row["workflow_type"], row["run_id"]): row for row in outcome_rows}
    assert rows_by_key[("strategy", "m31_eligible_strategy")]["promotion_status"] == "eligible"
    assert rows_by_key[("strategy", "m31_warn_strategy")]["promotion_status"] == "warn"
    assert rows_by_key[("strategy", "m31_needs_review_strategy")]["promotion_status"] == "needs_review"
    assert rows_by_key[("strategy", "m31_blocked_strategy")]["promotion_status"] == "blocked"
    assert rows_by_key[("strategy", "m31_blocked_strategy")]["triggered_gate_names"]
    assert rows_by_key[("review", "m31_review")]["review_status"] == ""
    assert rows_by_key[("candidate_review", "candidate_review:m31_candidate_selection")]["promotion_status"] == "warn"
    scenario_row = rows_by_key[("campaign_scenario", "m31_readiness_campaign:blocked")]
    assert scenario_row["campaign_id"] == "m31_readiness_campaign"
    assert scenario_row["scenario_id"] == "blocked"
    assert scenario_row["campaign_status"] == "completed"
    assert scenario_row["scenario_status"] == "completed"

    assert {row["workflow_type"] for row in workflow_rows} == {
        "campaign",
        "campaign_scenario",
        "candidate_review",
        "review",
        "strategy",
    }
    assert {"severity": "block", "highest_severity_count": "3", "triggered_reason_count": "3"} in severity_rows
    reason_counts = {row["reason_code"]: int(row["count"]) for row in reason_rows}
    assert reason_counts["severity_block"] == 3
    assert reason_counts["severity_warn"] == 6
    assert reason_counts["severity_review"] == 6
    _assert_no_absolute_path_leaks(first_result.output_dir, tmp_path)


def test_m31_readiness_style_governance_validation_reports_predictable_mismatch(tmp_path: Path) -> None:
    artifact_root = _m31_readiness_governance_fixture(tmp_path)
    blocked_manifest_path = artifact_root / "research_campaigns" / "m31_readiness_campaign" / "scenarios" / "blocked" / "manifest.json"
    blocked_manifest = json.loads(blocked_manifest_path.read_text(encoding="utf-8"))
    blocked_manifest["promotion_gate_summary"]["promotion_status"] = "eligible"
    _write_json(blocked_manifest_path, blocked_manifest)

    result = run_promotion_governance_report(
        registry_path=artifact_root / "registry.jsonl",
        artifact_root=artifact_root,
        output_dir=tmp_path / "governance_mismatch",
        report_id="m31_mismatch_report",
    )

    validation = json.loads(result.validation_path.read_text(encoding="utf-8"))
    assert validation["status"] == "fail"
    assert validation["counts_by_check"] == {"scenario_promotion_status_mismatch": 1}
    assert [finding["check_id"] for finding in validation["findings"]] == ["scenario_promotion_status_mismatch"]
    assert validation["findings"] == sorted(
        validation["findings"],
        key=lambda item: (item["severity"], item["check_id"], item["run_id"]),
    )
    _assert_no_absolute_path_leaks(result.output_dir, tmp_path)


def _m31_readiness_governance_fixture(tmp_path: Path) -> Path:
    artifact_root = tmp_path / "m31_readiness_artifacts"
    gate_config = _load_m31_gate_config()
    registry_path = artifact_root / "registry.jsonl"
    summaries: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for scenario_id, metrics in sorted(M31_SCENARIO_METRICS.items()):
        run_id = f"m31_{scenario_id}_strategy"
        run_dir = artifact_root / "runs" / scenario_id
        evaluation = evaluate_promotion_gates(run_type="strategy", config=gate_config, sources={"metrics": metrics})
        assert evaluation is not None
        promotion_gate_path = write_promotion_gate_artifact(run_dir, evaluation)
        assert promotion_gate_path is not None
        promotion_summary = evaluation.summary()
        review_metadata = build_review_metadata(
            promotion_status=evaluation.promotion_status,
            promotion_gate_summary=promotion_summary,
        )
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(
            run_dir / "manifest.json",
            {
                "run_id": run_id,
                "run_type": "strategy",
                "metric_summary": metrics,
                "promotion_gate_summary": promotion_summary,
                "review_status": review_metadata["status"],
            },
        )
        entry = canonicalize_value(
            {
                "run_id": run_id,
                "run_type": "strategy",
                "artifact_path": run_dir.as_posix(),
                "manifest_path": (run_dir / "manifest.json").as_posix(),
                "metrics_summary": metrics,
                "promotion_status": evaluation.promotion_status,
                "review_status": review_metadata["status"],
                "review_metadata": review_metadata,
                "promotion_gate_summary": promotion_summary,
            }
        )
        append_registry_entry(registry_path, entry)
        summaries[scenario_id] = promotion_summary
        entries[scenario_id] = entry

    _write_review_artifacts(artifact_root, summaries)
    _write_candidate_review_artifacts(artifact_root, summaries)
    _write_campaign_artifacts(artifact_root, summaries, entries)
    return artifact_root


def _write_review_artifacts(artifact_root: Path, summaries: dict[str, dict[str, Any]]) -> None:
    review_dir = artifact_root / "reviews" / "m31_review"
    _write_json(
        review_dir / "review_summary.json",
        {
            "review_id": "m31_review",
            "promotion_status_counts": _counts(summary["promotion_status"] for summary in summaries.values()),
        },
    )
    _write_json(
        review_dir / "manifest.json",
        {
            "review_id": "m31_review",
            "promotion_gate_summary": summaries["needs_review"],
        },
    )


def _write_candidate_review_artifacts(artifact_root: Path, summaries: dict[str, dict[str, Any]]) -> None:
    candidate_dir = artifact_root / "candidate_review" / "m31_candidate_selection"
    _write_json(
        candidate_dir / "candidate_review_summary.json",
        {
            "candidate_selection_run_id": "m31_candidate_selection",
            "portfolio_run_id": "m31_portfolio",
            "promotion_context": {
                "candidate_promotion_status_counts": _counts(summary["promotion_status"] for summary in summaries.values()),
                "portfolio_promotion_gate_summary": summaries["warn"],
            },
        },
    )
    _write_json(
        candidate_dir / "manifest.json",
        {
            "run_type": "candidate_selection_review",
            "candidate_selection_run_id": "m31_candidate_selection",
            "promotion_gate_summary": summaries["warn"],
        },
    )


def _write_campaign_artifacts(
    artifact_root: Path,
    summaries: dict[str, dict[str, Any]],
    entries: dict[str, dict[str, Any]],
) -> None:
    campaign_dir = artifact_root / "research_campaigns" / "m31_readiness_campaign"
    scenario_entries = []
    scenario_catalog = {"scenario_count": len(summaries), "scenarios": []}
    for scenario_id, summary in sorted(summaries.items()):
        run_id = entries[scenario_id]["run_id"]
        scenario_dir = campaign_dir / "scenarios" / scenario_id
        scenario_entry = {
            "scenario_id": scenario_id,
            "description": f"M31 readiness {scenario_id}",
            "status": "completed",
            "campaign_run_id": f"m31_campaign_{scenario_id}",
            "selected_run_ids": {"strategy_run_ids": [run_id]},
            "final_outcomes": {"review_promotion_gate_summary": summary},
        }
        scenario_entries.append(scenario_entry)
        scenario_catalog["scenarios"].append({"scenario_id": scenario_id, "description": scenario_entry["description"]})
        _write_json(
            scenario_dir / "summary.json",
            {
                "run_type": "research_campaign",
                "campaign_run_id": f"m31_campaign_{scenario_id}",
                "status": "completed",
                "scenario": {
                    "orchestration_run_id": "m31_readiness_campaign",
                    "scenario_id": scenario_id,
                    "description": scenario_entry["description"],
                },
                "selected_run_ids": {"strategy_run_ids": [run_id]},
                "final_outcomes": {"review_promotion_gate_summary": summary},
            },
        )
        _write_json(
            scenario_dir / "manifest.json",
            {
                "run_type": "research_campaign",
                "campaign_run_id": f"m31_campaign_{scenario_id}",
                "promotion_gate_summary": summary,
            },
        )
        _write_json(scenario_dir / "checkpoint.json", {"stage_states": {"review": "completed"}})

    _write_json(
        campaign_dir / "summary.json",
        {
            "run_type": "research_campaign_orchestration",
            "orchestration_run_id": "m31_readiness_campaign",
            "status": "completed",
            "scenario_count": len(summaries),
            "scenario_status_counts": {"completed": len(summaries)},
            "scenarios": scenario_entries,
            "final_outcomes": {
                "review_promotion_status": summaries["blocked"]["promotion_status"],
                "review_promotion_gate_status": summaries["blocked"]["evaluation_status"],
                "review_promotion_highest_severity": summaries["blocked"]["highest_severity"],
                "review_promotion_severity_counts": summaries["blocked"]["severity_counts"],
                "review_promotion_decision_reason_codes": sorted(
                    {code for summary in summaries.values() for code in summary["decision_reason_codes"]}
                ),
                "review_promotion_gate_summary": {
                    **summaries["blocked"],
                    "decision_reason_codes": sorted(
                        {code for summary in summaries.values() for code in summary["decision_reason_codes"]}
                    ),
                },
            },
        },
    )
    _write_json(
        campaign_dir / "manifest.json",
        {
            "run_type": "research_campaign_orchestration",
            "orchestration_run_id": "m31_readiness_campaign",
            "artifact_files": [
                "summary.json",
                "manifest.json",
                "scenario_catalog.json",
                *[
                    f"scenarios/{scenario_id}/summary.json"
                    for scenario_id in sorted(summaries)
                ],
                *[
                    f"scenarios/{scenario_id}/manifest.json"
                    for scenario_id in sorted(summaries)
                ],
            ],
        },
    )
    _write_json(campaign_dir / "scenario_catalog.json", scenario_catalog)
    _write_json(campaign_dir / "checkpoint.json", {"stage_states": {"review": "completed"}})


def _load_m31_gate_config() -> dict[str, Any]:
    payload = yaml.safe_load(Path("configs/statistical_readiness_promotion_gates_example.yml").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    gates = payload.get("promotion_gates")
    assert isinstance(gates, dict)
    return gates


def _report_payloads(output_dir: Path) -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.suffix in {".csv", ".json", ".md"}
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_no_absolute_path_leaks(output_dir: Path, tmp_path: Path) -> None:
    payloads = _report_payloads(output_dir)
    forbidden_tokens = (str(tmp_path), tmp_path.as_posix(), "C:\\", "C:/")
    for payload in payloads.values():
        assert not any(token in payload for token in forbidden_tokens)
    outcome_rows = _csv_rows(output_dir / "promotion_outcome_matrix.csv")
    for row in outcome_rows:
        assert "\\" not in row["registry_path"]
        assert "\\" not in row["manifest_path"]
        assert not Path(row["registry_path"]).is_absolute()
        assert not Path(row["manifest_path"]).is_absolute()


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonicalize_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path
