from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.portfolio import compute_portfolio_metrics, write_portfolio_artifacts
from src.research import experiment_tracker
from src.research.alpha_eval import evaluate_alpha_predictions, write_alpha_evaluation_artifacts
from src.research.experiment_tracker import save_experiment, save_walk_forward_experiment
from src.research.metrics import compute_performance_metrics
from src.research.promotion import (
    PromotionGateError,
    PromotionGateEvaluation,
    PromotionState,
    build_promotion_state_from_evaluation,
    build_unconfigured_promotion_state,
    evaluate_promotion_gates,
    promotion_gate_config_digest,
    serialize_promotion_state,
    write_promotion_gate_artifact,
    write_promotion_state_artifact,
)
from src.research.registry import load_registry


def _assert_state_mutation_blocks_serialization_and_write(
    tmp_path: Path,
    state: PromotionState,
    *,
    match: str,
) -> None:
    with pytest.raises(PromotionGateError, match=match):
        serialize_promotion_state(state)
    with pytest.raises(PromotionGateError, match=match):
        write_promotion_state_artifact(tmp_path, state)


def test_evaluate_promotion_gates_handles_pass_fail_borderline_missing_and_split_statistics() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "borderline_sharpe",
                    "source": "metrics",
                    "metric": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                },
                {
                    "gate_id": "drawdown_limit",
                    "source": "metrics",
                    "metric": "max_drawdown",
                    "comparator": "lte",
                    "threshold": 0.10,
                },
                {
                    "gate_id": "split_stability",
                    "source": "split_metrics",
                    "metric": "sharpe_ratio",
                    "statistic": "min",
                    "comparator": "gte",
                    "threshold": 0.50,
                },
                {
                    "gate_id": "missing_ic",
                    "source": "metrics",
                    "metric": "ic_ir",
                    "comparator": "gte",
                    "threshold": 0.20,
                },
            ]
        },
        sources={
            "metrics": {"sharpe_ratio": 1.0, "max_drawdown": 0.12},
            "split_metrics": [{"sharpe_ratio": 0.50}, {"sharpe_ratio": 0.80}],
        },
    )

    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.passed_gate_count == 2
    assert evaluation.failed_gate_count == 1
    assert evaluation.missing_gate_count == 1
    assert [result.status for result in evaluation.results] == ["pass", "fail", "pass", "missing"]
    assert evaluation.promotion_status == "blocked"
    assert evaluation.highest_severity is None
    assert evaluation.severity_counts == {"block": 0, "reject": 0, "review": 0, "warn": 0}
    assert evaluation.decision_reason_codes == ["gate_failed_threshold", "gate_missing"]


def test_legacy_promotion_gate_status_on_fail_remains_unchanged_without_severity() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_pass": "promoted",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ],
        },
        sources={"metrics": {"sharpe_ratio": 0.5}},
    )

    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == "manual_review"
    assert evaluation.highest_severity is None
    assert evaluation.results[0].severity is None
    assert evaluation.results[0].reason_codes == ["gate_failed_threshold"]


@pytest.mark.parametrize(
    ("severity", "expected_status", "expected_count_key", "expected_reason_code"),
    [
        ("warn", "warn", "warning_gate_count", "severity_warn"),
        ("review", "needs_review", "review_gate_count", "severity_review"),
        ("reject", "rejected", "rejected_gate_count", "severity_reject"),
        ("block", "blocked", "blocked_gate_count", "severity_block"),
    ],
)
def test_promotion_gate_severity_maps_to_promotion_status(
    severity: str,
    expected_status: str,
    expected_count_key: str,
    expected_reason_code: str,
) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_fail": "legacy_fail",
            "gates": [
                {
                    "gate_id": f"{severity}_gate",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": severity,
                }
            ],
        },
        sources={"metrics": {"effective_n": 12}},
    )

    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == expected_status
    assert evaluation.highest_severity == severity
    assert getattr(evaluation, expected_count_key) == 1
    assert expected_reason_code in evaluation.decision_reason_codes
    assert evaluation.results[0].reason_codes == ["gate_failed_threshold", expected_reason_code]


def test_promotion_gate_mixed_severities_resolve_by_highest_severity() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "stability_warn",
                    "source": "metrics",
                    "metric_path": "split_mean_diff_p",
                    "comparator": "gte",
                    "threshold": 0.05,
                    "severity": "warn",
                },
                {
                    "gate_id": "return_review",
                    "source": "metrics",
                    "metric_path": "p_value",
                    "comparator": "lte",
                    "threshold": 0.05,
                    "severity": "review",
                },
                {
                    "gate_id": "sample_block",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": "block",
                },
                {
                    "gate_id": "legacy_failure",
                    "source": "metrics",
                    "metric_path": "max_drawdown",
                    "comparator": "lte",
                    "threshold": 0.10,
                },
            ]
        },
        sources={
            "metrics": {
                "split_mean_diff_p": 0.01,
                "p_value": 0.20,
                "effective_n": 10,
                "max_drawdown": 0.25,
            }
        },
    )

    assert evaluation is not None
    assert evaluation.promotion_status == "blocked"
    assert evaluation.highest_severity == "block"
    assert evaluation.severity_counts == {"block": 1, "reject": 0, "review": 1, "warn": 1}
    assert evaluation.failed_gate_count == 4


def test_promotion_gate_missing_metric_with_severity_resolves_and_skip_does_not_trigger() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "missing_review",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": "review",
                },
                {
                    "gate_id": "missing_block_skipped",
                    "source": "metrics",
                    "metric_path": "sharpe_stability_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                    "missing_behavior": "skip",
                    "severity": "block",
                },
            ]
        },
        sources={"metrics": {}},
    )

    assert evaluation is not None
    assert evaluation.promotion_status == "needs_review"
    assert evaluation.highest_severity == "review"
    assert evaluation.missing_gate_count == 1
    assert evaluation.passed_gate_count == 1
    assert evaluation.results[0].reason_codes == ["gate_missing", "severity_review"]
    assert evaluation.results[1].reason_codes == ["gate_missing_skipped"]


def test_invalid_promotion_gate_severity_fails_clearly() -> None:
    with pytest.raises(PromotionGateError, match="severity.*block.*reject.*review.*warn"):
        evaluate_promotion_gates(
            run_type="strategy",
            config={
                "gates": [
                    {
                        "gate_id": "bad_severity",
                        "source": "metrics",
                        "metric_path": "p_value",
                        "comparator": "lte",
                        "threshold": 0.05,
                        "severity": "critical",
                    }
                ]
            },
            sources={"metrics": {"p_value": 0.01}},
        )


def test_m30_statistical_readiness_metrics_can_be_gated_through_metrics_source() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "minimum_effective_n",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": "block",
                },
                {
                    "gate_id": "return_p_value",
                    "source": "metrics",
                    "metric_path": "p_value",
                    "comparator": "lte",
                    "threshold": 0.05,
                    "severity": "review",
                },
                {
                    "gate_id": "hit_rate_significance",
                    "source": "metrics",
                    "metric_path": "hit_rate_p_value",
                    "comparator": "lte",
                    "threshold": 0.05,
                    "severity": "review",
                },
                {
                    "gate_id": "split_stability",
                    "source": "metrics",
                    "metric_path": "split_mean_diff_p",
                    "comparator": "gte",
                    "threshold": 0.05,
                    "severity": "warn",
                },
                {
                    "gate_id": "sharpe_stability",
                    "source": "metrics",
                    "metric_path": "sharpe_stability_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                    "severity": "warn",
                },
            ]
        },
        sources={
            "metrics": {
                "effective_n": 40,
                "p_value": 0.01,
                "hit_rate_p_value": 0.03,
                "split_mean_diff_p": 0.20,
                "sharpe_stability_ratio": 1.5,
            }
        },
    )

    assert evaluation is not None
    assert evaluation.evaluation_status == "pass"
    assert evaluation.promotion_status == "eligible"
    assert evaluation.highest_severity is None
    assert evaluation.decision_reason_codes == []


def test_promotion_gate_config_digest_changes_when_severity_changes() -> None:
    base_config = {
        "gates": [
            {
                "gate_id": "return_p_value",
                "source": "metrics",
                "metric_path": "p_value",
                "comparator": "lte",
                "threshold": 0.05,
            }
        ]
    }
    review_config = {
        "gates": [
            {
                **base_config["gates"][0],
                "severity": "review",
            }
        ]
    }
    block_config = {
        "gates": [
            {
                **base_config["gates"][0],
                "severity": "block",
            }
        ]
    }

    assert promotion_gate_config_digest(base_config) == promotion_gate_config_digest(dict(base_config))
    assert promotion_gate_config_digest(base_config) != promotion_gate_config_digest(review_config)
    assert promotion_gate_config_digest(review_config) != promotion_gate_config_digest(block_config)


def test_severity_promotion_gate_artifact_is_json_safe_and_deterministic(tmp_path: Path) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "minimum_effective_n",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": "block",
                }
            ]
        },
        sources={"metrics": {"effective_n": 12}},
    )

    artifact_path = write_promotion_gate_artifact(tmp_path, evaluation)
    first_payload = artifact_path.read_text(encoding="utf-8")
    write_promotion_gate_artifact(tmp_path, evaluation)
    second_payload = artifact_path.read_text(encoding="utf-8")
    parsed = json.loads(first_payload)

    assert first_payload == second_payload
    assert parsed["promotion_status"] == "blocked"
    assert parsed["highest_severity"] == "block"
    assert parsed["decision_reason_codes"] == ["gate_failed_threshold", "severity_block"]
    assert parsed["definitions"][0]["severity"] == "block"
    assert parsed["results"][0]["reason_codes"] == ["gate_failed_threshold", "severity_block"]


@pytest.mark.parametrize(
    ("run_type", "object_type", "object_id", "identity_field"),
    [
        ("review", "review", "registry_review_123", "review_id"),
        (
            "research_campaign",
            "research_campaign",
            "research_campaign_123",
            "campaign_run_id",
        ),
    ],
)
def test_unconfigured_promotion_state_is_canonical_v2_and_deterministic(
    tmp_path: Path,
    run_type: str,
    object_type: str,
    object_id: str,
    identity_field: str,
) -> None:
    provenance = {
        "object_id": object_id,
        "object_type": object_type,
        identity_field: object_id,
        "source_artifacts": {
            "manifest": "manifest.json",
            "summary": "summary.json",
        },
    }
    first = build_unconfigured_promotion_state(run_type=run_type, provenance=provenance)
    second = build_unconfigured_promotion_state(run_type=run_type, provenance=dict(provenance))

    first_payload = serialize_promotion_state(first)
    second_payload = serialize_promotion_state(second)
    artifact_path = write_promotion_state_artifact(tmp_path / run_type, first)
    first_bytes = artifact_path.read_bytes()
    second_path = write_promotion_state_artifact(tmp_path / run_type, second)
    parsed = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert first_payload == second_payload == parsed
    assert second_path.read_bytes() == first_bytes
    assert first_payload["schema_version"] == 2
    assert first_payload["artifact_type"] == "promotion_state"
    assert first_payload["run_type"] == run_type
    assert first_payload["configured"] is False
    assert first_payload["configuration_state"] == "not_configured"
    assert first_payload["evaluation_status"] == "not_configured"
    assert first_payload["promotion_status"] == "not_reviewed"
    assert first_payload["decision_authority"] == "none"
    assert first_payload["human_decision"] is None
    assert first_payload["decision_reason_codes"] == ["promotion_policy_not_configured"]
    assert first_payload["gate_counts"] == {
        "blocked": 0,
        "failed": 0,
        "missing": 0,
        "passed": 0,
        "rejected": 0,
        "review": 0,
        "skipped": 0,
        "total": 0,
        "warning": 0,
    }
    assert first_payload["gate_definitions"] == []
    assert first_payload["gate_results"] == []
    assert first_payload["provenance"] == {**provenance, "run_type": run_type}
    assert first_payload["artifact_metadata"] == {
        "artifact_filename": "promotion_gates.json",
        "deterministic": True,
        "generated_by": "src.research.promotion",
        "writer": "engine",
    }


def test_configured_pass_promotion_state_preserves_evaluation_semantics() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ]
        },
        sources={"metrics": {"sharpe_ratio": 1.5}},
    )

    state = build_promotion_state_from_evaluation(
        evaluation,
        provenance={"object_id": "strategy_1", "object_type": "strategy"},
    )
    payload = state.to_payload()

    assert payload["schema_version"] == 2
    assert payload["artifact_type"] == "promotion_state"
    assert payload["configured"] is True
    assert payload["configuration_state"] == "configured"
    assert payload["evaluation_status"] == "pass"
    assert payload["promotion_status"] == "eligible"
    assert payload["decision_authority"] == "engine"
    assert payload["human_decision"] is None
    assert payload["decision_reason_codes"] == []
    assert payload["gate_counts"] == {
        "blocked": 0,
        "failed": 0,
        "missing": 0,
        "passed": 1,
        "rejected": 0,
        "review": 0,
        "skipped": 0,
        "total": 1,
        "warning": 0,
    }
    assert payload["gate_definitions"] == payload["definitions"]
    assert payload["gate_results"] == payload["results"]
    assert payload["gate_count"] == 1
    assert payload["passed_gate_count"] == 1
    assert payload["artifact_filename"] == "promotion_gates.json"


def test_configured_failure_promotion_state_preserves_reason_codes_and_compatibility_fields() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="portfolio",
        config={
            "gates": [
                {
                    "gate_id": "minimum_effective_n",
                    "source": "metrics",
                    "metric_path": "effective_n",
                    "comparator": "gte",
                    "threshold": 30,
                    "severity": "review",
                },
                {
                    "gate_id": "missing_block_skipped",
                    "source": "metrics",
                    "metric_path": "sharpe_stability_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                    "missing_behavior": "skip",
                    "severity": "block",
                },
            ]
        },
        sources={"metrics": {"effective_n": 12}},
    )

    state = build_promotion_state_from_evaluation(
        evaluation,
        provenance={
            "object_id": "portfolio_1",
            "object_type": "portfolio",
            "source_artifacts": {"manifest": "manifest.json"},
        },
    )
    payload = state.to_payload()

    assert payload["evaluation_status"] == "fail"
    assert payload["promotion_status"] == "needs_review"
    assert payload["decision_authority"] == "engine"
    assert payload["decision_reason_codes"] == ["gate_failed_threshold", "severity_review"]
    assert payload["gate_counts"]["total"] == 2
    assert payload["gate_counts"]["failed"] == 1
    assert payload["gate_counts"]["skipped"] == 1
    assert payload["gate_counts"]["review"] == 1
    assert payload["highest_severity"] == "review"
    assert payload["review_gate_count"] == 1
    assert payload["results"][0]["reason_codes"] == ["gate_failed_threshold", "severity_review"]
    assert payload["gate_results"][0]["reason_codes"] == ["gate_failed_threshold", "severity_review"]


def test_promotion_state_rejects_invalid_combinations_and_unknown_values() -> None:
    state = build_unconfigured_promotion_state(run_type="review")
    payload = state.to_payload()

    with pytest.raises(PromotionGateError, match="promotion_status='not_reviewed'"):
        PromotionState({**payload, "promotion_status": "eligible"})
    with pytest.raises(PromotionGateError, match="cannot include gate definitions or results"):
        PromotionState({**payload, "gate_results": [{"gate_id": "fabricated"}]})
    with pytest.raises(PromotionGateError, match="decision_authority='none'"):
        PromotionState({**payload, "decision_authority": "engine"})
    with pytest.raises(PromotionGateError, match="run_type must be one of"):
        build_unconfigured_promotion_state(run_type="notebook")
    with pytest.raises(PromotionGateError, match="state must be a PromotionState"):
        serialize_promotion_state(payload)
    configured_payload = build_promotion_state_from_evaluation(
        evaluate_promotion_gates(
            run_type="strategy",
            config={
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric_path": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": 1.0,
                    }
                ]
            },
            sources={"metrics": {"sharpe_ratio": 2.0}},
        )
    ).to_payload()
    with pytest.raises(PromotionGateError, match="evaluation_status must be pass or fail"):
        PromotionState({**configured_payload, "evaluation_status": "not_configured"})
    with pytest.raises(PromotionGateError, match="promotion_status must be a non-empty string"):
        PromotionState({**configured_payload, "promotion_status": ""})


def test_configured_promotion_state_preserves_legacy_status_on_pass() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_pass": "approved",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ],
        },
        sources={"metrics": {"sharpe_ratio": 2.0}},
    )
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()

    assert payload["promotion_status"] == "approved"
    assert payload["promotion_status"] != "eligible"
    assert payload["status_on_pass"] == "approved"
    assert payload["status_on_fail"] == "manual_review"
    assert payload["evaluation_status"] == "pass"
    assert payload["decision_authority"] == "engine"
    assert payload["gate_count"] == 1
    assert payload["gate_results"] == payload["results"]
    assert not any(key.startswith("_") for key in payload)
    with pytest.raises(PromotionGateError, match="promotion_status must be one of"):
        PromotionState(payload)


def test_configured_promotion_state_preserves_legacy_status_on_fail() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_pass": "approved",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ],
        },
        sources={"metrics": {"sharpe_ratio": 0.5}},
    )
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()

    assert payload["promotion_status"] == "manual_review"
    assert payload["promotion_status"] != "needs_review"
    assert payload["status_on_pass"] == "approved"
    assert payload["status_on_fail"] == "manual_review"
    assert payload["evaluation_status"] == "fail"
    assert payload["decision_authority"] == "engine"
    assert payload["decision_reason_codes"] == ["gate_failed_threshold"]
    assert payload["failed_gate_count"] == 1
    assert not any(key.startswith("_") for key in payload)
    with pytest.raises(PromotionGateError, match="promotion_status must be one of"):
        PromotionState(payload)


@pytest.mark.parametrize("status", ["approved", "manual_review", "definitely_promote_everything"])
def test_direct_configured_promotion_state_rejects_noncanonical_statuses(status: str) -> None:
    payload = build_promotion_state_from_evaluation(
        evaluate_promotion_gates(
            run_type="strategy",
            config={
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric_path": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": 1.0,
                    }
                ]
            },
            sources={"metrics": {"sharpe_ratio": 2.0}},
        )
    ).to_payload()
    with pytest.raises(PromotionGateError, match="promotion_status must be one of"):
        PromotionState({**payload, "promotion_status": status})


def test_direct_promotion_state_cannot_enable_legacy_status_mode() -> None:
    payload = build_promotion_state_from_evaluation(
        evaluate_promotion_gates(
            run_type="strategy",
            config={
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric_path": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": 1.0,
                    }
                ]
            },
            sources={"metrics": {"sharpe_ratio": 2.0}},
        )
    ).to_payload()
    altered_payload = {**payload, "promotion_status": "definitely_promote_everything"}

    with pytest.raises(TypeError, match="_allow_legacy_promotion_status"):
        PromotionState(altered_payload, _allow_legacy_promotion_status=True)
    with pytest.raises(PromotionGateError, match="promotion_status must be one of"):
        PromotionState(altered_payload)


def test_removed_evaluation_payload_bypass_is_not_callable() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_pass": "approved",
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ],
        },
        sources={"metrics": {"sharpe_ratio": 2.0}},
    )
    arbitrary_payload = build_promotion_state_from_evaluation(evaluation).to_payload()
    arbitrary_payload["promotion_status"] = "definitely_promote_everything"

    with pytest.raises(AttributeError):
        PromotionState._from_evaluation_payload(evaluation, arbitrary_payload)


def test_promotion_state_writer_filename_override_updates_metadata_without_mutation(tmp_path: Path) -> None:
    state = build_unconfigured_promotion_state(run_type="review")
    original_payload = state.to_payload()

    first_path = write_promotion_state_artifact(tmp_path, state, artifact_filename="custom_state.json")
    second_path = write_promotion_state_artifact(tmp_path, state, artifact_filename="custom_state.json")
    parsed = json.loads(first_path.read_text(encoding="utf-8"))

    assert first_path == second_path == tmp_path / "custom_state.json"
    assert parsed["artifact_metadata"]["artifact_filename"] == "custom_state.json"
    assert state.to_payload() == original_payload
    assert original_payload["artifact_metadata"]["artifact_filename"] == "promotion_gates.json"
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_legacy_evaluation_state_filename_override_updates_metadata(tmp_path: Path) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="strategy",
        config={
            "status_on_pass": "approved",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "min_sharpe",
                    "source": "metrics",
                    "metric_path": "sharpe_ratio",
                    "comparator": "gte",
                    "threshold": 1.0,
                }
            ],
        },
        sources={"metrics": {"sharpe_ratio": 2.0}},
    )
    state = build_promotion_state_from_evaluation(evaluation)
    original_payload = state.to_payload()

    path = write_promotion_state_artifact(tmp_path, state, artifact_filename="legacy_state.json")
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "legacy_state.json"
    assert parsed["promotion_status"] == "approved"
    assert parsed["artifact_metadata"]["artifact_filename"] == "legacy_state.json"
    assert state.to_payload() == original_payload
    assert not any(key.startswith("_") for key in parsed)


def test_promotion_state_payload_copies_and_write_resist_mutation(tmp_path: Path) -> None:
    provenance = {
        "object_id": "review_a",
        "object_type": "review",
        "source_artifacts": {"manifest": "manifest.json"},
    }
    state = build_unconfigured_promotion_state(run_type="review", provenance=provenance)
    expected_payload = state.to_payload()

    returned_payload = state.to_payload()
    returned_payload["promotion_status"] = "eligible"
    returned_payload["provenance"]["object_id"] = "mutated"
    provenance["object_id"] = "mutated_after_construction"
    state.payload["promotion_status"] = "eligible"

    assert state.to_payload() == expected_payload

    state._payload["promotion_status"] = "eligible"
    _assert_state_mutation_blocks_serialization_and_write(
        tmp_path,
        state,
        match="promotion_status='not_reviewed'",
    )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.pop("provenance"), "provenance must be a mapping"),
        (
            lambda payload: payload["provenance"].__setitem__("run_type", "strategy"),
            "provenance.run_type must match run_type",
        ),
        (lambda payload: payload.pop("artifact_metadata"), "artifact_metadata must be a mapping"),
        (
            lambda payload: payload["artifact_metadata"].__setitem__("artifact_filename", " "),
            "artifact_metadata.artifact_filename",
        ),
        (
            lambda payload: payload["artifact_metadata"].__setitem__("writer", "notebook"),
            "artifact_metadata.writer must be 'engine'",
        ),
        (
            lambda payload: payload["artifact_metadata"].__setitem__("generated_by", "notebook"),
            "artifact_metadata.generated_by must be 'src.research.promotion'",
        ),
        (
            lambda payload: payload["artifact_metadata"].__setitem__("deterministic", "true"),
            "artifact_metadata.deterministic must be True",
        ),
    ],
)
def test_promotion_state_rejects_tampered_provenance_and_metadata(
    tmp_path: Path,
    mutator: Any,
    match: str,
) -> None:
    state = build_unconfigured_promotion_state(run_type="review")
    mutator(state._payload)

    _assert_state_mutation_blocks_serialization_and_write(tmp_path, state, match=match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["gate_counts"].pop("total"),
            "gate_counts missing required keys: total",
        ),
        (
            lambda payload: payload.__setitem__("gate_counts", {}),
            "gate_counts missing required keys",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("extra", 0),
            "gate_counts has unexpected keys: extra",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("total", "0"),
            "gate_counts.total must be a nonnegative integer",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("total", 0.0),
            "gate_counts.total must be a nonnegative integer",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("total", True),
            "gate_counts.total must be a nonnegative integer",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("total", -1),
            "gate_counts.total must be a nonnegative integer",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("passed", 1),
            "requires zero gate counts",
        ),
    ],
)
def test_unconfigured_promotion_state_rejects_malformed_gate_counts(
    tmp_path: Path,
    mutator: Any,
    match: str,
) -> None:
    state = build_unconfigured_promotion_state(run_type="review")
    mutator(state._payload)

    _assert_state_mutation_blocks_serialization_and_write(tmp_path, state, match=match)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["gate_counts"].__setitem__("total", 2),
            "gate_counts.total must equal gate_results length",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("passed", 0),
            "pass/fail/missing gate counts must sum to total",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("warning", 1),
            "severity counts cannot exceed non-passing results",
        ),
        (
            lambda payload: payload["gate_counts"].__setitem__("skipped", 2),
            "skipped gate count cannot exceed passed count",
        ),
    ],
)
def test_configured_promotion_state_rejects_inconsistent_gate_counts(
    tmp_path: Path,
    mutator: Any,
    match: str,
) -> None:
    state = build_promotion_state_from_evaluation(
        evaluate_promotion_gates(
            run_type="strategy",
            config={
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric_path": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": 1.0,
                    }
                ]
            },
            sources={"metrics": {"sharpe_ratio": 2.0}},
        )
    )
    mutator(state._payload)

    _assert_state_mutation_blocks_serialization_and_write(tmp_path, state, match=match)


def test_write_alpha_evaluation_artifacts_persists_promotion_gate_artifact(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "ts_utc": [
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
                "2025-01-01T00:00:00Z",
                "2025-01-02T00:00:00Z",
            ],
            "timeframe": ["1D", "1D", "1D", "1D", "1D", "1D"],
            "prediction_score": [0.1, 0.6, 0.2, 0.5, 0.3, 0.4],
            "forward_return": [0.2, 0.3, 0.1, 0.4, 0.0, 0.2],
        }
    )
    result = evaluate_alpha_predictions(frame)

    manifest = write_alpha_evaluation_artifacts(
        tmp_path / "alpha" / "run-1",
        result,
        run_id="run-1",
        alpha_name="demo_alpha",
        aligned_frame=frame,
        promotion_gate_config={
            "gates": [
                {
                    "gate_id": "min_valid_timestamps",
                    "source": "qa_summary",
                    "metric": "forecast.valid_timestamps",
                    "comparator": "gte",
                    "threshold": float(result.summary["n_periods"]),
                },
                {
                    "gate_id": "nulls_clean",
                    "source": "qa_summary",
                    "metric": "nulls.prediction_null_rate",
                    "comparator": "lte",
                    "threshold": 0.0,
                },
            ]
        },
    )

    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["evaluation_status"] == "pass"
    payload = json.loads((tmp_path / "alpha" / "run-1" / "promotion_gates.json").read_text(encoding="utf-8"))
    assert payload["promotion_status"] == "eligible"
    assert payload["passed_gate_count"] == 2


def test_save_experiment_persists_strategy_promotion_gate_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")
    results_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "SPY", "SPY"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "signal": [1.0, 1.0, 0.0, 0.0],
            "executed_signal": [0.0, 1.0, 1.0, 0.0],
            "position": [0.0, 1.0, 1.0, 0.0],
            "delta_position": [0.0, 1.0, 0.0, -1.0],
            "abs_delta_position": [0.0, 1.0, 0.0, 1.0],
            "turnover": [0.0, 1.0, 0.0, 1.0],
            "trade_event": [False, True, False, True],
            "gross_strategy_return": [0.0, 0.02, -0.01, 0.0],
            "transaction_cost": [0.0, 0.0, 0.0, 0.0],
            "slippage_cost": [0.0, 0.0, 0.0, 0.0],
            "execution_friction": [0.0, 0.0, 0.0, 0.0],
            "strategy_return": [0.0, 0.02, -0.01, 0.0],
            "equity_curve": [1.0, 1.02, 1.0098, 1.0098],
        }
    )
    metrics = compute_performance_metrics(results_df)
    metrics["sanity_issue_count"] = 0.0
    metrics["sanity_warning_count"] = 0.0
    experiment_dir = save_experiment(
        "demo_strategy",
        results_df,
        metrics,
        {
            "strategy_name": "demo_strategy",
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "min_sharpe",
                        "source": "metrics",
                        "metric": "sharpe_ratio",
                        "comparator": "gte",
                        "threshold": float(metrics["sharpe_ratio"]),
                    },
                    {
                        "gate_id": "sanity_clean",
                        "source": "qa_summary",
                        "metric": "sanity.issue_count",
                        "comparator": "lte",
                        "threshold": 0.0,
                    },
                ]
            },
        },
    )

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["promotion_status"] == "eligible"
    registry_entries = load_registry(experiment_tracker.ARTIFACTS_ROOT / "registry.jsonl")
    assert registry_entries[0]["promotion_status"] == "eligible"


def test_save_walk_forward_experiment_persists_split_stability_promotion_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_tracker, "ARTIFACTS_ROOT", tmp_path / "artifacts" / "strategies")

    def _split_frame(returns: list[float]) -> pd.DataFrame:
        equity = []
        level = 1.0
        for value in returns:
            level *= 1.0 + value
            equity.append(level)
        return pd.DataFrame(
            {
                "ts_utc": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                "symbol": ["SPY", "SPY"],
                "signal": [1.0, 1.0],
                "executed_signal": [1.0, 1.0],
                "position": [1.0, 1.0],
                "delta_position": [1.0, 0.0],
                "abs_delta_position": [1.0, 0.0],
                "turnover": [1.0, 0.0],
                "trade_event": [True, False],
                "gross_strategy_return": returns,
                "transaction_cost": [0.0, 0.0],
                "slippage_cost": [0.0, 0.0],
                "execution_friction": [0.0, 0.0],
                "strategy_return": returns,
                "equity_curve": equity,
            }
        )

    split_a = _split_frame([0.01, 0.02])
    split_b = _split_frame([0.01, 0.01])
    split_b["ts_utc"] = ["2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z"]
    split_results = [
        {
            "split_id": "split_000",
            "split_metadata": {
                "split_id": "split_000",
                "mode": "rolling",
                "train_start": "2024-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-01-03",
            },
            "split_rows": 4,
            "train_rows": 2,
            "test_rows": 2,
            "metrics": compute_performance_metrics(split_a),
            "results_df": split_a,
        },
        {
            "split_id": "split_001",
            "split_metadata": {
                "split_id": "split_001",
                "mode": "rolling",
                "train_start": "2024-01-02",
                "train_end": "2025-01-01",
                "test_start": "2025-01-03",
                "test_end": "2025-01-05",
            },
            "split_rows": 4,
            "train_rows": 2,
            "test_rows": 2,
            "metrics": compute_performance_metrics(split_b),
            "results_df": split_b,
        },
    ]
    aggregate = compute_performance_metrics(pd.concat([split_a, split_b], ignore_index=True))
    aggregate["split_count"] = 2

    experiment_dir = save_walk_forward_experiment(
        "demo_strategy",
        split_results,
        aggregate,
        {
            "strategy_name": "demo_strategy",
            "evaluation_config_path": "configs/evaluation.yml",
            "evaluation": {"mode": "rolling", "timeframe": "1D"},
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "stable_sharpe_floor",
                        "source": "split_metrics",
                        "metric": "sharpe_ratio",
                        "statistic": "min",
                        "comparator": "gte",
                        "threshold": min(
                            float(split_results[0]["metrics"]["sharpe_ratio"]),
                            float(split_results[1]["metrics"]["sharpe_ratio"]),
                        ),
                    }
                ]
            },
        },
    )

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion_gate_summary"]["evaluation_status"] == "pass"
    assert (experiment_dir / "promotion_gates.json").exists()


def test_write_portfolio_artifacts_persists_promotion_gate_artifact(tmp_path: Path) -> None:
    portfolio_output = pd.DataFrame(
        {
            "weight__alpha": [0.5, 0.5],
            "weight__beta": [0.5, 0.5],
            "strategy_return__alpha": [0.00, 0.01],
            "strategy_return__beta": [0.02, 0.03],
            "gross_portfolio_return": [0.01, 0.02],
            "portfolio_weight_change": [1.0, 0.0],
            "portfolio_abs_weight_change": [1.0, 0.0],
            "portfolio_turnover": [1.0, 0.0],
            "portfolio_rebalance_event": [1, 0],
            "portfolio_changed_sleeve_count": [2, 0],
            "portfolio_transaction_cost": [0.0, 0.0],
            "portfolio_fixed_fee": [0.0, 0.0],
            "portfolio_slippage_proxy": [0.01, 0.0],
            "portfolio_slippage_cost": [0.0, 0.0],
            "portfolio_execution_friction": [0.0, 0.0],
            "net_portfolio_return": [0.01, 0.02],
            "portfolio_return": [0.01, 0.02],
            "portfolio_equity_curve": [101.0, 103.02],
            "ts_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
        }
    )
    metrics = compute_portfolio_metrics(portfolio_output, "1D")
    manifest = write_portfolio_artifacts(
        tmp_path / "portfolio-run",
        portfolio_output,
        metrics,
        {
            "portfolio_name": "core_portfolio",
            "allocator": "equal_weight",
            "timeframe": "1D",
            "initial_capital": 100.0,
            "promotion_gates": {
                "gates": [
                    {
                        "gate_id": "max_drawdown",
                        "source": "metrics",
                        "metric": "max_drawdown",
                        "comparator": "lte",
                        "threshold": float(metrics["max_drawdown"]),
                    }
                ]
            },
        },
        [
            {"strategy_name": "alpha", "run_id": "run-a"},
            {"strategy_name": "beta", "run_id": "run-b"},
        ],
    )

    assert "promotion_gates.json" in manifest["artifact_files"]
    assert manifest["promotion_gate_summary"]["promotion_status"] == "eligible"


@pytest.mark.parametrize(
    "unsafe_filename",
    [
        "../escaped.json",
        "../../outside.json",
        "nested/state.json",
        "nested\\state.json",
        "/absolute/state.json",
        "C:\\absolute\\state.json",
        "C:/absolute/state.json",
        ".",
        "..",
        "",
        "   ",
    ],
)
def test_write_promotion_state_artifact_rejects_unsafe_filename(tmp_path: Path, unsafe_filename: str) -> None:
    state = build_unconfigured_promotion_state(
        run_type="review",
        provenance={"object_type": "review", "object_id": "r1", "review_id": "r1", "run_type": "review"},
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    with pytest.raises(PromotionGateError):
        write_promotion_state_artifact(output_dir, state, artifact_filename=unsafe_filename)
    assert not any(output_dir.rglob("*.*"))
    assert not any((tmp_path.parent).rglob("escaped.json"))


def test_write_promotion_state_artifact_accepts_valid_basename(tmp_path: Path) -> None:
    state = build_unconfigured_promotion_state(
        run_type="review",
        provenance={"object_type": "review", "object_id": "r1", "review_id": "r1", "run_type": "review"},
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    result = write_promotion_state_artifact(output_dir, state, artifact_filename="custom_state.json")
    assert result == output_dir / "custom_state.json"
    assert result.exists()
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["artifact_metadata"]["artifact_filename"] == "custom_state.json"


def test_write_promotion_state_artifact_default_filename_unchanged(tmp_path: Path) -> None:
    state = build_unconfigured_promotion_state(
        run_type="review",
        provenance={"object_type": "review", "object_id": "r1", "review_id": "r1", "run_type": "review"},
    )
    result = write_promotion_state_artifact(tmp_path, state)
    assert result.name == "promotion_gates.json"
    assert result.exists()


def test_forged_evaluation_with_arbitrary_matching_status_rejected_at_serialization(tmp_path: Path) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    forged = PromotionGateEvaluation(
        configured=evaluation.configured,
        run_type=evaluation.run_type,
        evaluation_status=evaluation.evaluation_status,
        promotion_status="definitely_promote_everything",
        status_on_pass="definitely_promote_everything",
        status_on_fail=evaluation.status_on_fail,
        gate_count=evaluation.gate_count,
        passed_gate_count=evaluation.passed_gate_count,
        failed_gate_count=evaluation.failed_gate_count,
        missing_gate_count=evaluation.missing_gate_count,
        highest_severity=evaluation.highest_severity,
        severity_counts=evaluation.severity_counts,
        warning_gate_count=evaluation.warning_gate_count,
        review_gate_count=evaluation.review_gate_count,
        rejected_gate_count=evaluation.rejected_gate_count,
        blocked_gate_count=evaluation.blocked_gate_count,
        decision_reason_codes=evaluation.decision_reason_codes,
        artifact_filename=evaluation.artifact_filename,
        definitions=evaluation.definitions,
        results=evaluation.results,
    )
    with pytest.raises(PromotionGateError):
        build_promotion_state_from_evaluation(forged)


def test_forged_evaluation_with_status_on_fail_forgery_rejected(tmp_path: Path) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    forged = PromotionGateEvaluation(
        configured=evaluation.configured,
        run_type=evaluation.run_type,
        evaluation_status=evaluation.evaluation_status,
        promotion_status="forged_status",
        status_on_pass=evaluation.status_on_pass,
        status_on_fail="forged_status",
        gate_count=evaluation.gate_count,
        passed_gate_count=evaluation.passed_gate_count,
        failed_gate_count=evaluation.failed_gate_count,
        missing_gate_count=evaluation.missing_gate_count,
        highest_severity=evaluation.highest_severity,
        severity_counts=evaluation.severity_counts,
        warning_gate_count=evaluation.warning_gate_count,
        review_gate_count=evaluation.review_gate_count,
        rejected_gate_count=evaluation.rejected_gate_count,
        blocked_gate_count=evaluation.blocked_gate_count,
        decision_reason_codes=evaluation.decision_reason_codes,
        artifact_filename=evaluation.artifact_filename,
        definitions=evaluation.definitions,
        results=evaluation.results,
    )
    with pytest.raises(PromotionGateError):
        build_promotion_state_from_evaluation(forged)


def test_unsupported_arbitrary_pass_status_rejected() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "pass"
    forged = PromotionGateEvaluation(
        configured=evaluation.configured,
        run_type=evaluation.run_type,
        evaluation_status=evaluation.evaluation_status,
        promotion_status="custom_unsupported_value",
        status_on_pass="custom_unsupported_value",
        status_on_fail=evaluation.status_on_fail,
        gate_count=evaluation.gate_count,
        passed_gate_count=evaluation.passed_gate_count,
        failed_gate_count=evaluation.failed_gate_count,
        missing_gate_count=evaluation.missing_gate_count,
        highest_severity=evaluation.highest_severity,
        severity_counts=evaluation.severity_counts,
        warning_gate_count=evaluation.warning_gate_count,
        review_gate_count=evaluation.review_gate_count,
        rejected_gate_count=evaluation.rejected_gate_count,
        blocked_gate_count=evaluation.blocked_gate_count,
        decision_reason_codes=evaluation.decision_reason_codes,
        artifact_filename=evaluation.artifact_filename,
        definitions=evaluation.definitions,
        results=evaluation.results,
    )
    with pytest.raises(PromotionGateError, match="not a supported canonical or compatibility value"):
        build_promotion_state_from_evaluation(forged)


def test_unsupported_arbitrary_fail_status_rejected() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    forged = PromotionGateEvaluation(
        configured=evaluation.configured,
        run_type=evaluation.run_type,
        evaluation_status=evaluation.evaluation_status,
        promotion_status="definitely_ignore_everything",
        status_on_pass=evaluation.status_on_pass,
        status_on_fail="definitely_ignore_everything",
        gate_count=evaluation.gate_count,
        passed_gate_count=evaluation.passed_gate_count,
        failed_gate_count=evaluation.failed_gate_count,
        missing_gate_count=evaluation.missing_gate_count,
        highest_severity=evaluation.highest_severity,
        severity_counts=evaluation.severity_counts,
        warning_gate_count=evaluation.warning_gate_count,
        review_gate_count=evaluation.review_gate_count,
        rejected_gate_count=evaluation.rejected_gate_count,
        blocked_gate_count=evaluation.blocked_gate_count,
        decision_reason_codes=evaluation.decision_reason_codes,
        artifact_filename=evaluation.artifact_filename,
        definitions=evaluation.definitions,
        results=evaluation.results,
    )
    with pytest.raises(PromotionGateError, match="not a supported canonical or compatibility value"):
        build_promotion_state_from_evaluation(forged)


def test_genuine_custom_pass_status_remains_valid() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "review_ready",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    assert evaluation.promotion_status == "review_ready"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "review_ready"


def test_genuine_custom_fail_status_remains_valid() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == "manual_review"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "manual_review"


def test_severity_driven_fail_with_custom_status_on_fail_uses_severity() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "manual_review",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                    "severity": "reject",
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == "rejected"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "rejected"


def test_approved_custom_pass_status_valid() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "approved",
            "status_on_fail": "blocked",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    assert evaluation.promotion_status == "approved"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "approved"


def test_needs_work_custom_fail_status_valid() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == "needs_work"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "needs_work"


@pytest.mark.parametrize(
    "unsafe_filename",
    [
        "../escaped.json",
        "nested/state.json",
        "nested\\state.json",
        "/absolute/state.json",
        "C:\\absolute\\state.json",
        ".",
        "..",
    ],
)
def test_build_unconfigured_promotion_state_rejects_unsafe_filename(unsafe_filename: str) -> None:
    with pytest.raises(PromotionGateError):
        build_unconfigured_promotion_state(
            run_type="review",
            provenance={"run_type": "review"},
            artifact_filename=unsafe_filename,
        )


@pytest.mark.parametrize(
    "unsafe_filename",
    [
        "../escaped.json",
        "nested/state.json",
        "nested\\state.json",
        "/absolute/state.json",
        "C:\\absolute\\state.json",
        ".",
        "..",
    ],
)
def test_build_promotion_state_from_evaluation_rejects_unsafe_filename(unsafe_filename: str) -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    with pytest.raises(PromotionGateError):
        build_promotion_state_from_evaluation(evaluation, artifact_filename=unsafe_filename)


def test_build_promotion_state_from_evaluation_rejects_empty_filename() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    with pytest.raises(PromotionGateError):
        build_promotion_state_from_evaluation(evaluation, artifact_filename="")


def test_build_promotion_state_from_evaluation_rejects_whitespace_filename() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    with pytest.raises(PromotionGateError):
        build_promotion_state_from_evaluation(evaluation, artifact_filename="   ")


def test_build_promotion_state_from_evaluation_none_inherits_evaluation_filename() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    state = build_promotion_state_from_evaluation(evaluation, artifact_filename=None)
    payload = state.to_payload()
    assert payload["artifact_metadata"]["artifact_filename"] == "promotion_gates.json"


def test_build_promotion_state_from_evaluation_accepts_valid_custom_basename() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                }
            ],
        },
        sources={"metrics": {"score": 1.0}},
    )
    assert evaluation is not None
    state = build_promotion_state_from_evaluation(evaluation, artifact_filename="custom_state.json")
    payload = state.to_payload()
    assert payload["artifact_metadata"]["artifact_filename"] == "custom_state.json"


def test_severity_driven_failure_cannot_be_overridden_by_custom_fail_status() -> None:
    evaluation = evaluate_promotion_gates(
        run_type="review",
        config={
            "status_on_pass": "eligible",
            "status_on_fail": "needs_work",
            "gates": [
                {
                    "gate_id": "gate_a",
                    "source": "metrics",
                    "metric_path": "score",
                    "comparator": "gte",
                    "threshold": 0.5,
                    "severity": "block",
                }
            ],
        },
        sources={"metrics": {"score": 0.1}},
    )
    assert evaluation is not None
    assert evaluation.evaluation_status == "fail"
    assert evaluation.promotion_status == "blocked"
    state = build_promotion_state_from_evaluation(evaluation)
    payload = state.to_payload()
    assert payload["promotion_status"] == "blocked"
