from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.regime_policy_stress_tests import (
    RegimePolicyStressTestConfig,
    RegimePolicyStressTestConfigError,
    apply_regime_policy_stress_test_overrides,
    load_regime_policy_stress_test_config,
)


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = {
        "stress_test_name": "test_regime_stress",
        "source_review_pack": "configs/regime_stress_tests/fixtures/regime_review_pack",
        "source_policy_candidates": {
            "policy_metrics_path": "configs/regime_stress_tests/fixtures/policy_candidates.csv",
            "baseline_policy": "static_baseline",
            "candidate_policy_names": ["policy_optimized", "gmm_calibrated_overlay"],
        },
        "regime_context": {
            "preferred_regime_source": "calibrated_taxonomy",
            "ml_overlay": "gmm_classifier",
            "transition_window_bars": 5,
            "confidence_floor": 0.55,
            "entropy_ceiling": 1.10,
        },
        "scenarios": [
            {
                "name": "transition_shock_low_to_high_vol",
                "type": "transition_shock",
                "from_regime": "low_vol",
                "to_regime": "high_vol",
                "shock_start": "2026-03-01",
                "shock_length_bars": 10,
                "stress_intensity": 1.0,
            }
        ],
        "metrics": {
            "include_policy_turnover": True,
            "include_drawdown": True,
            "include_transition_windows": True,
            "include_fallback_usage": True,
            "include_adaptive_vs_static": True,
        },
        "stress_gates": {
            "max_policy_turnover": 0.60,
            "max_stress_drawdown": -0.18,
            "min_adaptive_vs_static_drawdown_delta": -0.03,
            "max_fallback_activation_rate": 0.75,
            "max_state_change_count": 40,
        },
        "output_root": "artifacts/regime_stress_tests",
    }
    if overrides:
        payload.update(overrides)
    path = tmp_path / "stress.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def test_load_regime_policy_stress_test_config_valid_config(tmp_path: Path) -> None:
    config = load_regime_policy_stress_test_config(_write_config(tmp_path))

    assert isinstance(config, RegimePolicyStressTestConfig)
    assert config.stress_test_name == "test_regime_stress"
    assert config.source_policy_candidates is not None
    assert config.source_policy_candidates.baseline_policy == "static_baseline"
    assert config.scenarios[0].scenario_name == "transition_shock_low_to_high_vol"


def test_regime_policy_stress_test_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"unexpected": True})

    with pytest.raises(RegimePolicyStressTestConfigError, match="unsupported keys"):
        load_regime_policy_stress_test_config(path)


def test_regime_policy_stress_test_config_rejects_invalid_scenario_type(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "scenarios": [
                {
                    "name": "bad_type",
                    "type": "unknown_type",
                    "stress_intensity": 1.0,
                }
            ]
        },
    )

    with pytest.raises(RegimePolicyStressTestConfigError, match="must be one of"):
        load_regime_policy_stress_test_config(path)


def test_regime_policy_stress_test_config_rejects_invalid_stress_gate_threshold(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"stress_gates": {"max_policy_turnover": 1.5}},
    )

    with pytest.raises(RegimePolicyStressTestConfigError, match="between 0 and 1"):
        load_regime_policy_stress_test_config(path)


def test_regime_policy_stress_test_config_missing_required_policy_metrics_path(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"source_policy_candidates": {"baseline_policy": "static_baseline"}})

    with pytest.raises(RegimePolicyStressTestConfigError, match="policy_metrics_path"):
        load_regime_policy_stress_test_config(path)


def test_regime_policy_stress_test_config_scenario_id_is_deterministic(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {
            "scenarios": [
                {
                    "name": "scenario_a",
                    "type": "confidence_collapse",
                    "confidence_floor_override": 0.2,
                    "collapse_length_bars": 10,
                }
            ]
        },
    )

    first = load_regime_policy_stress_test_config(path)
    second = load_regime_policy_stress_test_config(path)

    assert first.scenarios[0].scenario_id == second.scenarios[0].scenario_id


def test_regime_policy_stress_test_override_handling(tmp_path: Path) -> None:
    config = load_regime_policy_stress_test_config(_write_config(tmp_path))

    updated = apply_regime_policy_stress_test_overrides(
        config,
        source_review_pack="new/review",
        policy_metrics_path="new/policies.csv",
        output_root="new/output",
    )

    assert updated.source_review_pack == "new/review"
    assert updated.source_policy_candidates is not None
    assert updated.source_policy_candidates.policy_metrics_path == "new/policies.csv"
    assert updated.output_root == "new/output"


def test_regime_policy_stress_test_config_preserves_source_candidate_selection(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        {"source_candidate_selection": "artifacts/candidate_selection/sample_selection_run"},
    )

    config = load_regime_policy_stress_test_config(path)

    assert config.source_candidate_selection == "artifacts/candidate_selection/sample_selection_run"
